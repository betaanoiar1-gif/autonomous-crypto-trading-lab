from typing import Protocol
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

class MarketDataProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame: ...


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    out = df.copy()
    out = out.sort_index()
    if out.index.has_duplicates:
        raise ValueError("OHLCV index contains duplicate timestamps")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    return out
