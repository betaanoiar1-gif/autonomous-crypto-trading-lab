from typing import Protocol
import pandas as pd

class MarketDataProvider(Protocol):
    def fetch_ohlcv(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Return OHLCV indexed by timestamp with open/high/low/close/volume columns."""
        ...
