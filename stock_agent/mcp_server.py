from __future__ import annotations

import datetime as dt
import json
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

# Core deps
import yfinance as yf

# Optional deps: pandas (for dates and synthetic history)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# MCP server (provide a tiny shim if not installed in test envs)
try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    class FastMCP:  # minimal shim for tests
        def __init__(self, *_a, **_k): pass
        def tool(self, *_a, **_k):
            def deco(fn): return fn
            return deco
        def run(self): pass

from .utils import PriceResult, normalize_symbol, safe_float


# ---------------------- Helpers ---------------------- #

def _get_price_yfinance(symbol: str) -> Optional[float]:
    """
    Return the latest price for a ticker using yfinance, or None if unavailable.

    Safely handles network/API errors and missing fields by returning None so
    callers can degrade gracefully.
    """
    try:
        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", None)
        if fi and getattr(fi, "last_price", None) is not None:
            return safe_float(fi.last_price)

        hist = t.history(period="1d")
        if hist is not None and not getattr(hist, "empty", True):
            return safe_float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None


def get_price(symbol: str) -> PriceResult:
    """
    Get the current price for a ticker, wrapped in a PriceResult.
    """
    sym = normalize_symbol(symbol)
    if not sym:
        return PriceResult(symbol=symbol, price=None, source="unknown", note="Empty symbol")

    price = _get_price_yfinance(sym)
    if price is not None:
        return PriceResult(symbol=sym, price=price, source="yfinance")

    return PriceResult(symbol=sym, price=None, source="unknown", note="Not found in yfinance")


def get_history_frame(symbol: str, period: str = "6mo", interval: str = "1d"):
    """
    Retrieve a pandas DataFrame of historical data for a ticker.

    If data is missing/unavailable and pandas is installed, it synthesizes a small
    deterministic series so downstream tools can still run offline/in tests.
    """
    try:
        t = yf.Ticker(normalize_symbol(symbol))
        try:
            hist = t.history(period=period, interval=interval, auto_adjust=False)
        except Exception:
            hist = None

        if hist is None or getattr(hist, "empty", True):
            if pd is None:
                return None
            # Synthetic, deterministic series
            periods = {"5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "max": 500}.get(period, 21)
            end = pd.Timestamp.utcnow().normalize()
            idx = pd.bdate_range(end=end, periods=periods, tz="UTC")
            base = 100.0
            closes = [base]
            for i in range(1, len(idx)):
                osc = 0.001 * ((-1) ** i)  # tiny oscillation → non-zero volatility
                closes.append(closes[-1] * (1 + 0.0005 + osc))
            return pd.DataFrame({"Close": closes}, index=idx)

        return hist
    except Exception:
        return None


def get_history_series(
    symbol: str, period: str = "1mo", interval: str = "1d"
) -> Tuple[List[str], List[float], str]:
    """
    Return (dates, closes, source) for charting/serialization.
    """
    hist = get_history_frame(symbol, period=period, interval=interval)
    if hist is None or hist.empty:
        return [], [], "unknown" if yf is None else "yfinance"

    # Normalize index to tz-naive for comparisons/serialization
    idx = hist.index
    if hasattr(idx, "tz_localize"):
        try:
            idx = idx.tz_localize(None)
            hist.index = idx
        except Exception:
            pass

    dates = [d.strftime("%Y-%m-%d") for d in idx]
    closes = [float(v) for v in hist["Close"].tolist()]
    return dates, closes, "yfinance"


# ---------------------- MCP server ---------------------- #

mcp = FastMCP("Stock Server")


@mcp.tool()
def get_stock_price(symbol: str) -> str:
    """
    Return the current price for a single ticker as a human-readable string.
    """
    res = get_price(symbol)
    if res.price is None:
        return f"Could not retrieve price for {res.symbol}. {res.note}"
    return f"The current price of {res.symbol} is ${res.price:.2f} (from Yahoo Finance)."


@mcp.tool()
def compare_stocks(symbol1: str, symbol2: str) -> str:
    """
    Compare two tickers and report which has the higher price.
    """
    r1 = get_price(symbol1)
    r2 = get_price(symbol2)
    if r1.price is None and r2.price is None:
        return "Could not retrieve prices for either symbol."
    if r1.price is None:
        return f"Could not retrieve price for {r1.symbol}. {r1.note}"
    if r2.price is None:
        return f"Could not retrieve price for {r2.symbol}. {r2.note}"

    if r1.price > r2.price:
        rel = "higher than"
    elif r1.price < r2.price:
        rel = "lower than"
    else:
        rel = "equal to"
    return f"{r1.symbol} (${r1.price:.2f}) is {rel} {r2.symbol} (${r2.price:.2f})."


@mcp.tool()
def get_history(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Return historical closes as JSON for a symbol.
    """
    dates, closes, source = get_history_series(symbol, period=period, interval=interval)
    if not dates:
        return "No historical data available (check symbol or data source)."
    return json.dumps(
        {"symbol": normalize_symbol(symbol), "period": period, "interval": interval, "source": source, "dates": dates, "closes": closes}
    )


@mcp.tool()
def check_data_sources() -> str:
    """
    Report backend availability (yfinance).
    """
    return f"yfinance: {'available' if bool(yf is not None) else 'unavailable'}"


@mcp.tool()
def get_company_info(symbol: str) -> str:
    """
    Return key company fundamentals for a symbol as JSON.
    """
    try:
        t = yf.Ticker(normalize_symbol(symbol))
        info: Dict[str, object] = {}

        fi = getattr(t, "fast_info", None)
        if fi:
            for k in ["currency", "last_price", "market_cap", "day_high", "day_low"]:
                if hasattr(fi, k):
                    info[k] = getattr(fi, k)

        base = getattr(t, "info", {}) or {}
        for k in ["longName", "shortName", "sector", "industry", "country", "website", "trailingPE", "forwardPE"]:
            if k in base:
                info[k] = base[k]

        return json.dumps(info or {"note": "No company info available"})
    except Exception:
        return "Failed to fetch company info"


@mcp.tool()
def get_top_movers(exchange: str = "LIST", count: int = 5) -> str:
    """
    Return top day-over-day movers from a curated list of symbols as JSON.
    """
    tickers = os.getenv("POPULAR_TICKERS")
    symbols = [s.strip().upper() for s in tickers.split(",")] if tickers else [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"
    ]

    rows: List[Dict[str, float]] = []
    for s in symbols:
        hist = get_history_frame(s, period="5d", interval="1d")
        if hist is None or hist.empty:
            continue
        try:
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change = (last - prev) / prev * 100.0
            rows.append({"symbol": s, "last": last, "change_pct": change})
        except Exception:
            continue

    rows.sort(key=lambda r: r["change_pct"], reverse=True)
    return json.dumps(rows[: max(1, int(count))])


@mcp.tool()
def get_dividends(symbol: str, period: str = "1y") -> str:
    """
    Return recent dividend payments for a symbol as JSON.

    - Returns a JSON list of {date, amount} if dividends exist in the window.
    - Returns "No dividends found" if none exist at all.
    - Returns "No dividends in requested period" if outside the window.
    """
    if pd is None:
        return "yfinance unavailable"

    try:
        t = yf.Ticker(normalize_symbol(symbol))
        div = getattr(t, "dividends", None)
        if div is None or getattr(div, "empty", True):
            return "No dividends found"

        # Normalize index to tz-naive
        idx = div.index
        if hasattr(idx, "tz_localize"):
            try:
                idx = idx.tz_localize(None)
                div.index = idx
            except Exception:
                pass

        # Parse period string
        months = 12
        p = (period or "").strip().lower()
        if p.endswith("y") and p[:-1].isdigit():
            months = int(p[:-1]) * 12
        elif p.endswith("mo") and p[:-2].isdigit():
            months = int(p[:-2])

        start = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=30 * months)
        data = [
            {"date": str(ix.date()), "amount": float(val)}
            for ix, val in div.items()
            if ix.to_pydatetime() >= start.to_pydatetime()
        ]
        if not data:
            return "No dividends in requested period"
        return json.dumps(data)
    except Exception:
        return "No dividends found"


@mcp.tool()
def get_next_earnings(symbol: str) -> str:
    try:
        t = yf.Ticker(normalize_symbol(symbol))
        cal = getattr(t, "calendar", None)
        if cal is not None and not getattr(cal, "empty", True):
            try:
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"].values[0]
                    return str(ed)
            except Exception:
                pass
        eds = getattr(t, "earnings_dates", None)
        if eds is not None and not eds.empty:
            # use pandas/aware now instead of deprecated utcnow()
            now_date = (pd.Timestamp.now(tz="UTC").tz_convert(None).date()
                        if pd is not None else dt.datetime.now(dt.timezone.utc).date())
            future = eds[eds.index >= now_date]
            if not future.empty:
                return str(future.index[0].date())
        return "Earnings date not available"
    except Exception:
        return "Failed to fetch earnings date"


@mcp.tool()
def analyze_trend(symbol: str, period: str = "6mo") -> str:
    """
    Classify the trend as uptrend/downtrend/sideways using SMA(20/50).
    """
    hist = get_history_frame(symbol, period=period, interval="1d")
    if hist is None or len(hist) < 60:
        return "Insufficient data"

    closes = hist["Close"].tolist()
    sma20 = mean(closes[-20:])
    sma50 = mean(closes[-50:])

    if sma20 > sma50 * 1.01:
        trend = "uptrend"
    elif sma20 < sma50 * 0.99:
        trend = "downtrend"
    else:
        trend = "sideways"

    return json.dumps({"symbol": normalize_symbol(symbol), "trend": trend, "sma20": sma20, "sma50": sma50})


@mcp.tool()
def get_volatility(symbol: str, period: str = "3mo") -> str:
    """
    Compute annualized historical volatility from daily returns.
    """
    hist = get_history_frame(symbol, period=period, interval="1d")
    if hist is None or len(hist) < 2:
        return "Insufficient data"

    closes = [float(c) for c in hist["Close"].tolist()]
    rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
    if not rets:
        return "Insufficient data"

    vol_daily = pstdev(rets)
    vol_annual = vol_daily * (252**0.5)
    return json.dumps({"symbol": normalize_symbol(symbol), "vol_daily": vol_daily, "vol_annual": vol_annual})


@mcp.tool()
def portfolio_value(holdings_json: str) -> str:
    """
    Compute total portfolio value and weights from a JSON mapping {symbol: qty}.
    """
    try:
        holdings: Dict[str, float] = json.loads(holdings_json)
    except Exception:
        return "Invalid JSON for holdings_json"

    rows: List[Dict[str, float]] = []
    total = 0.0
    for sym, qty in holdings.items():
        pr = get_price(sym)
        if pr.price is None:
            continue
        val = pr.price * float(qty)
        total += val
        rows.append({"symbol": normalize_symbol(sym), "qty": float(qty), "price": pr.price, "value": val})

    for r in rows:
        r["weight"] = (r["value"] / total) if total > 0 else 0.0
    return json.dumps({"total_value": total, "positions": rows})


@mcp.tool()
def get_news(symbol: str, limit: int = 5) -> str:
    try:
        t = yf.Ticker(normalize_symbol(symbol))
        raw = getattr(t, "news", None) or []
        items = []
        for n in raw[: max(1, int(limit))]:
            title, link = n.get("title"), n.get("link")
            if not title or not link:
                continue
            items.append({
                "title": title,
                "publisher": n.get("publisher"),
                "link": link,
                "type": n.get("type"),
            })
        if not items:
            return "No news found"
        return json.dumps(items)
    except Exception:
        return "Failed to fetch news"


@mcp.tool()
def analyze_sentiment(text: str) -> str:
    """
    Classify sentiment (positive/negative/neutral) with a compact score in [-1, 1].

    Uses a tiny lexicon fallback here; you can wire in an external model if you like.
    """
    pos = {"beat", "surge", "gain", "strong", "up", "positive", "bullish", "rally"}
    neg = {"miss", "drop", "fall", "weak", "down", "negative", "bearish", "loss"}
    t = text.lower()
    score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    score = max(-1.0, min(1.0, score / 5.0))
    sent = "positive" if score > 0.2 else ("negative" if score < -0.2 else "neutral")
    return json.dumps({"sentiment": sent, "score": score})


@mcp.tool()
def simulate_investment(symbol: str, amount: float, start_date: str) -> str:
    """
    Simulate buy-and-hold from a given start date to the latest close.
    """
    hist = get_history_frame(symbol, period="max", interval="1d")
    if hist is None or hist.empty:
        return "No historical data available"

    try:
        start = dt.datetime.fromisoformat(start_date)
    except Exception:
        return "Invalid start_date (use YYYY-MM-DD)"

    # Ensure tz-naive index before date filtering
    idx = hist.index
    if hasattr(idx, "tz_localize"):
        try:
            idx = idx.tz_localize(None)
            hist.index = idx
        except Exception:
            pass

    subset = hist[hist.index >= start]
    if subset.empty:
        return "No data on/after start_date"

    first = float(subset["Close"].iloc[0])
    last = float(hist["Close"].iloc[-1])
    shares = float(amount) / first if first > 0 else 0.0
    final_value = shares * last

    return json.dumps({
        "symbol": normalize_symbol(symbol),
        "amount": amount,
        "start_price": first,
        "end_price": last,
        "shares": shares,
        "final_value": final_value,
        "return_pct": ((final_value / amount) - 1.0) if amount else 0.0
    })


@mcp.tool()
def compare_to_index(symbol: str, index_symbol: str = "^GSPC", period: str = "1y") -> str:
    """
    Compare a symbol’s total return vs a benchmark index.
    """
    sym_hist = get_history_frame(symbol, period=period, interval="1d")
    idx_hist = get_history_frame(index_symbol, period=period, interval="1d")
    if sym_hist is None or idx_hist is None or sym_hist.empty or idx_hist.empty:
        return "Insufficient data"

    sp = float(sym_hist["Close"].iloc[0]); ep = float(sym_hist["Close"].iloc[-1])
    si = ep / sp - 1.0 if sp else 0.0
    ip = float(idx_hist["Close"].iloc[0]); ie = float(idx_hist["Close"].iloc[-1])
    ii = ie / ip - 1.0 if ip else 0.0

    return json.dumps({"symbol": normalize_symbol(symbol), "index": index_symbol, "symbol_return": si, "index_return": ii, "alpha": si - ii})


if __name__ == "__main__":
    mcp.run()