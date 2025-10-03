from __future__ import annotations

import asyncio
import json
import os
import re
import difflib
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Optional dependency: Gemini (google.genai)
try:
    from google import genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

load_dotenv()

ROUTING_SYSTEM_PROMPT = """You are a tool router. Pick ONE MCP tool and arguments for the user's market/finance query.

Available tools:
- get_stock_price(symbol)
- compare_stocks(symbol1, symbol2)
- get_history(symbol, period="1mo", interval="1d")
- check_data_sources()
- get_company_info(symbol)
- get_top_movers(exchange="LIST", count=5)
- get_dividends(symbol, period="1y")
- get_next_earnings(symbol)
- analyze_trend(symbol, period="6mo")
- get_volatility(symbol, period="3mo")
- portfolio_value(holdings_json)
- get_news(symbol, limit=5)
- analyze_sentiment(text)
- simulate_investment(symbol, amount, start_date)
- compare_to_index(symbol, index_symbol="^GSPC", period="1y")

Return strict JSON with keys: tool, args.
"""

# --- Router helpers ---

import re, difflib, json

# Stopwords: words we should NEVER treat as tickers
_STOPWORDS = {
    "a","an","and","the","of","for","to","vs","versus","compare","stock","stocks","show",
    "price","history","chart","months","month","days","day","info","information","company",
    "trend","volatility","simulate","investment","news","next","earnings","dividends",
    "portfolio","value","benchmark","index","sp500","gainers","losers","movers"
}

_ALIAS_TO_TICKER = {
    "google":"GOOGL","alphabet":"GOOGL","apple":"AAPL","microsoft":"MSFT","amazon":"AMZN",
    "meta":"META","facebook":"META","tesla":"TSLA","netflix":"NFLX","nvidia":"NVDA",
    "amd":"AMD","advanced micro devices":"AMD","intel":"INTC","broadcom":"AVGO",
    "sp500":"^GSPC","s&p500":"^GSPC","s&p 500":"^GSPC","spx":"^GSPC","dow":"^DJI","nasdaq":"^IXIC",
}

_KNOWN_TICKERS = {
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","TSLA","NFLX","NVDA","AMD","INTC","AVGO",
    "^GSPC","^DJI","^IXIC"
}

def _extract_symbol(query: str) -> str | None:
    """
    Extract a likely ticker:
      1) name→ticker alias map (case-insensitive)
      2) UPPERCASE tokens that look like tickers, excluding stopwords
      3) fuzzy match against aliases/known tickers (typo tolerant)
    """
    q = query.strip().lower()

    # 1) aliases
    for name, ticker in _ALIAS_TO_TICKER.items():
        if name in q:
            return ticker

    # 2) UPPERCASE tokens only (avoid turning 'show', 'info', 'of' into tickers)
    upper_tokens = re.findall(r"\b([A-Z]{1,5}(?:\.[A-Z]+)?)\b", query)
    upper_tokens = [t for t in upper_tokens if t.lower() not in _STOPWORDS]
    for tok in upper_tokens:
        if tok in _KNOWN_TICKERS:
            return tok
    if upper_tokens:
        # accept first non-stopword UPPER token as a reasonable guess
        return upper_tokens[0]

    # 3) fuzzy match whole query
    names = list(_ALIAS_TO_TICKER.keys()) + list(_KNOWN_TICKERS)
    best = difflib.get_close_matches(q, names, n=1, cutoff=0.8)
    if best:
        key = best[0]
        if key in _ALIAS_TO_TICKER:
            return _ALIAS_TO_TICKER[key]
        if key in _KNOWN_TICKERS:
            return key
    return None


def _extract_two_symbols(query: str) -> list[str]:
    """Pull up to two symbols for comparisons (ignores stopwords and 'VS')."""
    # prefer UPPERCASE tokens
    uppers = re.findall(r"\b([A-Z]{1,5}(?:\.[A-Z]+)?)\b", query)
    uppers = [u for u in uppers if u.lower() not in _STOPWORDS]
    if len(uppers) >= 2:
        return uppers[:2]

    # fall back to names/aliases
    found = []
    q = query.lower()
    for name, ticker in _ALIAS_TO_TICKER.items():
        if name in q and ticker not in found:
            found.append(ticker)
        if len(found) == 2:
            break
    return found


def _rule_based_router(query: str) -> Dict[str, Any]:
    q = query.lower()
    symbol = _extract_symbol(query) or "AAPL"

    def stock(sym: str = "AAPL") -> Dict[str, Any]:
        return {"tool": "get_stock_price", "args": {"symbol": sym}}

    # two symbols for comparison
    syms = _extract_two_symbols(query)

    if any(w in q for w in ["compare to index", "vs sp500", "benchmark"]):
        return {"tool": "compare_to_index", "args": {"symbol": symbol, "index_symbol": "^GSPC", "period": "1y"}}

    if any(w in q for w in ["compare", "vs", "versus"]) and len(syms) >= 2:
        return {"tool": "compare_stocks", "args": {"symbol1": syms[0], "symbol2": syms[1]}}

    if any(w in q for w in ["history", "historical", "chart", "months", "days"]):
        period = "6mo" if ("6" in q and "month" in q) else "1mo"
        return {"tool": "get_history", "args": {"symbol": symbol, "period": period, "interval": "1d"}}

    if any(w in q for w in ["dividend", "dividends"]):
        return {"tool": "get_dividends", "args": {"symbol": symbol, "period": "1y"}}

    if "earnings" in q:
        return {"tool": "get_next_earnings", "args": {"symbol": symbol}}

    if any(w in q for w in ["trend", "uptrend", "downtrend"]):
        return {"tool": "analyze_trend", "args": {"symbol": symbol, "period": "6mo"}}

    if "volatility" in q or "vol " in q or q.endswith(" vol"):
        return {"tool": "get_volatility", "args": {"symbol": symbol, "period": "3mo"}}

    if "portfolio" in q or "holdings" in q:
        m = re.search(r"\{.*\}", query)
        holdings_json = m.group(0) if m else json.dumps({symbol: 10})
        # ensure string, never dict
        if not isinstance(holdings_json, str):
            holdings_json = json.dumps(holdings_json)
        return {"tool": "portfolio_value", "args": {"holdings_json": holdings_json}}

    if "news" in q:
        return {"tool": "get_news", "args": {"symbol": symbol, "limit": 5}}

    if "sentiment" in q:
        return {"tool": "analyze_sentiment", "args": {"text": query}}

    if "simulate" in q or "if i invested" in q:
        return {"tool": "simulate_investment", "args": {"symbol": symbol, "amount": 1000, "start_date": "2024-01-01"}}

    if "mover" in q or "gainer" in q or "loser" in q:
        return {"tool": "get_top_movers", "args": {"exchange": "LIST", "count": 5}}

    return stock(symbol)


async def _route_with_gemini(query: str) -> Dict[str, Any]:
    """Route query with Gemini if configured, else fallback to rules."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return _rule_based_router(query)

    client = genai.Client(api_key=api_key)
    prompt = f"{ROUTING_SYSTEM_PROMPT}\nUser: {query}\nReturn JSON only:"
    try:
        resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        text = resp.text if hasattr(resp, "text") else (resp.candidates[0].content.parts[0].text if getattr(resp, "candidates", None) else "{}")
        data = json.loads(text)
        if isinstance(data, dict) and "tool" in data and "args" in data:
            return data
    except Exception:
        pass
    return _rule_based_router(query)


def _server_params() -> StdioServerParameters:
    """Launch MCP server with project root as cwd."""
    root = Path(__file__).resolve().parents[1]
    return StdioServerParameters(command="python", args=["-m", "stock_agent.mcp_server"], cwd=str(root))


async def _call_tool(session: ClientSession, spec: Dict[str, Any]) -> str:
    tool = spec.get("tool")
    args = spec.get("args", {})
    if not tool:
        return "No tool selected."
    # ensure portfolio_value receives a string
    if tool == "portfolio_value" and isinstance(args.get("holdings_json"), dict):
        args["holdings_json"] = json.dumps(args["holdings_json"])
    result = await session.call_tool(tool, args)
    return result.content[0].text if result.content else "(no content)"


async def handle_query(query: str) -> str:
    spec = await _route_with_gemini(query)
    async with stdio_client(_server_params()) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            return await _call_tool(session, spec)


def main() -> None:
    try:
        while True:
            q = input("What is your query? → ").strip()
            if not q:
                continue
            print(asyncio.run(handle_query(q)))
    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()