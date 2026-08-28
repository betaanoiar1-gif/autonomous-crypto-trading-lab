from __future__ import annotations

import ccxt
import pandas as pd


class CCXTDataSource:
    """Public OHLCV adapter. Authentication is not required for public market data."""

    def __init__(self, exchange_id: str = "binance") -> None:
        if not hasattr(ccxt, exchange_id):
            raise ValueError(f"Unknown CCXT exchange: {exchange_id}")
        self.exchange_id = exchange_id
        self.exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: int | None = None,
        limit: int = 1000,
        market_type: str = "spot",
    ) -> pd.DataFrame:
        params = {"type": "future"} if market_type == "futures" else {"type": "spot"}
        if hasattr(self.exchange, "options"):
            self.exchange.options.update(params)
        rows = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not rows:
            raise ValueError(f"No OHLCV returned for {symbol} {timeframe}")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna()
