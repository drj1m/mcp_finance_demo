from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["normalize_symbol", "safe_float", "PriceResult"]

def normalize_symbol(symbol: str) -> str:
    """Normalize a ticker symbol to uppercase, safe against None or whitespace."""
    return (symbol or "").strip().upper()

def safe_float(value: Any) -> Optional[float]:
    """Convert to float, returning None if conversion fails."""
    try:
        return float(value)
    except Exception:
        return None

@dataclass(frozen=True)
class PriceResult:
    """Container for a price lookup result."""
    symbol: str
    price: Optional[float]
    source: str  # e.g. "yfinance" | "unknown"
    note: str = ""