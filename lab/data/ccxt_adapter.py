from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
import pandas as pd
import ccxt


@dataclass
class CCXTMarketData:
    exchange_id: str = "binance"

    def _exchange(self):
        cls = getattr(ccxt, self.exchange_id)
        return cls({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        if limit < 10 or limit > 1500:
            raise ValueError("limit must be between 10 and 1500")
        ex = self._exchange()
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            raise RuntimeError(f"No OHLCV returned for {symbol} {timeframe}")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].astype(float)
        if df.index[-1] >= pd.Timestamp(datetime.now(timezone.utc)).floor(timeframe):
            df = df.iloc[:-1]
        return df

    def fetch_multi(self, symbols: list[str], timeframe: str = "1h", limit: int = 1000) -> dict[str, pd.DataFrame]:
        out = {}
        for symbol in symbols:
            out[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
            time.sleep(0.05)
        return out
