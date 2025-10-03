"""
Tests for the MCP stock server tools using a fake yfinance backend.

These tests:
- Stub out `yfinance.Ticker` with deterministic data (no network).
- Validate each tool returns well-formed outputs or clear error strings.
"""

from __future__ import annotations

import datetime as dt
import json
import types
from typing import List

import pandas as pd

from stock_agent import mcp_server as srv


class FastInfo:
    """Lightweight stand-in for yfinance's `fast_info` object."""

    def __init__(
        self,
        last_price: float | None = None,
        currency: str = "USD",
        market_cap: int = 1_000_000_000,
    ):
        self.last_price = last_price
        self.currency = currency
        self.market_cap = market_cap
        self.day_high = (last_price or 100) * 1.01
        self.day_low = (last_price or 100) * 0.99


class FakeTicker:
    """Deterministic fake for `yfinance.Ticker` used in unit tests."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._base_price = {
            "AAPL": 180.0,
            "MSFT": 350.0,
            "TSLA": 250.0,
            "^GSPC": 4000.0,
            "AMD": 110.0,
        }.get(symbol, 100.0)

        # Stand-in for yfinance.Ticker.fast_info
        self.fast_info = FastInfo(last_price=self._base_price)

        # Minimal company info mapping
        self.info = {
            "longName": f"Company {symbol}",
            "shortName": symbol,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "US",
            "website": f"https://www.{symbol.lower()}.com",
            "trailingPE": 25.0,
            "forwardPE": 22.0,
        }

        # Quarterly dividends over ~1 year
        today = pd.Timestamp.utcnow().normalize()
        dates = pd.date_range(end=today, periods=4, freq="90D")
        self._dividends = pd.Series([0.2, 0.2, 0.22, 0.22], index=dates)

        # Simple news list
        self.news = [
            {
                "title": f"{symbol} beats expectations",
                "publisher": "MockWire",
                "link": f"https://news/{symbol}/1",
                "type": "STORY",
            },
            {
                "title": f"{symbol} announces new product",
                "publisher": "MockWire",
                "link": f"https://news/{symbol}/2",
                "type": "STORY",
            },
        ]

        # Earnings dates
        next_q = today + pd.Timedelta(days=30)
        self.earnings_dates = pd.DataFrame(index=[next_q], data={"EPS Estimate": [1.2]})

        # Optional calendar (varies by yfinance version)
        self.calendar = pd.DataFrame({"Value": [next_q]}, index=["Earnings Date"])

    def history(
        self, period: str = "1mo", interval: str = "1d", auto_adjust: bool = False
    ) -> pd.DataFrame:
        """Produce a deterministic price series with a tiny oscillation (for non-zero volatility)."""
        periods_map = {"5d": 5, "1mo": 21, "3mo": 63, "6mo": 126, "1y": 252, "max": 500}
        n = periods_map.get(period, 21)

        end = pd.Timestamp.utcnow().normalize()
        idx = pd.bdate_range(end=end, periods=n, tz="UTC")

        drift = 0.0005 if self.symbol != "^GSPC" else 0.0002
        closes: List[float] = [self._base_price]
        for i in range(1, len(idx)):
            osc = 0.001 * ((-1) ** i)  # small alternating oscillation
            closes.append(closes[-1] * (1 + drift + osc))

        return pd.DataFrame({"Close": closes}, index=idx)

    @property
    def dividends(self) -> pd.Series:
        """Expose a deterministic dividend series."""
        return self._dividends


class FakeYFModule(types.SimpleNamespace):
    """Module-like object exposing Ticker, to replace `yfinance` inside the server."""

    def __init__(self):
        super().__init__()
        self.Ticker = lambda sym: FakeTicker(sym)


def use_fake_yf(monkeypatch) -> None:
    """Monkeypatch the server's `yf` module reference with our deterministic fake."""
    fake = FakeYFModule()
    monkeypatch.setattr(srv, "yf", fake, raising=True)


def test_get_stock_price_uses_yfinance_fast_info(monkeypatch):
    """Price tool should return a line containing the symbol and a formatted price."""
    use_fake_yf(monkeypatch)
    out = srv.get_stock_price("AAPL")
    assert "AAPL" in out and "$" in out


def test_compare_stocks_relation(monkeypatch):
    """Compare tool should state whether one price is higher/lower/equal than the other."""
    use_fake_yf(monkeypatch)
    out = srv.compare_stocks("MSFT", "AAPL")
    assert "MSFT" in out and "AAPL" in out
    assert ("higher than" in out) or ("lower than" in out) or ("equal to" in out)


def test_get_history_returns_series(monkeypatch):
    """History tool should return JSON with aligned date/close arrays for the symbol."""
    use_fake_yf(monkeypatch)
    s = json.loads(srv.get_history("TSLA"))
    assert s["symbol"] == "TSLA"
    assert len(s["dates"]) == len(s["closes"]) > 0


def test_analyze_trend_labels(monkeypatch):
    """Trend tool should classify into one of the known labels."""
    use_fake_yf(monkeypatch)
    data = json.loads(srv.analyze_trend("AAPL"))
    assert data["trend"] in {"uptrend", "downtrend", "sideways"}


def test_get_volatility(monkeypatch):
    """Volatility tool should report a positive annualized volatility for the synthetic series."""
    use_fake_yf(monkeypatch)
    data = json.loads(srv.get_volatility("AAPL"))
    assert data["vol_annual"] > 0


def test_portfolio_value(monkeypatch):
    """Portfolio tool should calculate total value and normalized weights that sum to ~1."""
    use_fake_yf(monkeypatch)
    payload = json.dumps({"AAPL": 10, "MSFT": 5})
    data = json.loads(srv.portfolio_value(payload))
    assert data["total_value"] > 0
    assert len(data["positions"]) == 2
    weights = [p["weight"] for p in data["positions"]]
    assert abs(sum(weights) - 1.0) < 1e-6


def test_get_news(monkeypatch):
    """News tool should return at least one item with a title for the fake ticker."""
    use_fake_yf(monkeypatch)
    items = json.loads(srv.get_news("AMD", limit=1))
    assert isinstance(items, list) and items and "title" in items[0]


def test_get_dividends(monkeypatch):
    """Dividends tool should produce JSON list (or a clear 'No dividends' message)."""
    use_fake_yf(monkeypatch)
    res = srv.get_dividends("AAPL", period="1y")
    if res.startswith("No dividends"):
        assert True
    else:
        data = json.loads(res)
        assert isinstance(data, list)


def test_get_next_earnings(monkeypatch):
    """Earnings tool should return a date string or explain unavailability."""
    use_fake_yf(monkeypatch)
    res = srv.get_next_earnings("AAPL")
    assert isinstance(res, str) and res


def test_simulate_investment(monkeypatch):
    """Simulation tool should produce a positive final value for the last year using fake data."""
    use_fake_yf(monkeypatch)
    start = (dt.date.today().replace(year=dt.date.today().year - 1)).isoformat()
    data = json.loads(srv.simulate_investment("AAPL", amount=1000, start_date=start))
    assert data["final_value"] > 0


def test_compare_to_index(monkeypatch):
    """Index comparison should return both symbol and index returns."""
    use_fake_yf(monkeypatch)
    data = json.loads(srv.compare_to_index("AAPL", index_symbol="^GSPC", period="6mo"))
    assert "symbol_return" in data and "index_return" in data
