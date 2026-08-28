from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time

import ccxt
import pandas as pd


_TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


@dataclass
class CCXTMarketData:
    exchange_id: str = "binance"

    def _exchange(self):
        cls = getattr(ccxt, self.exchange_id)
        return cls({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        if limit < 10 or limit > 1500:
            raise ValueError("limit must be between 10 and 1500")
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported fixed timeframe: {timeframe}")
        ex = self._exchange()
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not rows:
            raise RuntimeError(f"No OHLCV returned for {symbol} {timeframe}")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].astype(float)

        # Never use pandas floor() with calendar frequencies such as MonthEnd.
        # Remove the currently forming candle by comparing its age in milliseconds.
        interval_ms = _TIMEFRAME_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        last_open_ms = int(df.index[-1].timestamp() * 1000)
        if now_ms - last_open_ms < interval_ms:
            df = df.iloc[:-1]

        if len(df) < 10:
            raise RuntimeError(f"Insufficient closed OHLCV rows for {symbol} {timeframe}: {len(df)}")
        return df

    def fetch_multi(self, symbols: list[str], timeframe: str = "1h", limit: int = 1000) -> dict[str, pd.DataFrame]:
        return {symbol: self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) for symbol in symbols}

    def fetch_multi_timeframes(
        self,
        symbols: list[str],
        timeframes: list[str],
        limit: int = 1000,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for timeframe in timeframes:
            for symbol in symbols:
                out[(symbol, timeframe)] = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                time.sleep(0.05)
        return out
