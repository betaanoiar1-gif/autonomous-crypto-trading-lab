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

# Target history sizes used by the research engine. These are deliberately
# larger on faster timeframes so walk-forward windows have enough observations.
_HISTORY_TARGETS = {
    "15m": 10_000,
    "1h": 5_000,
    "4h": 3_000,
}


@dataclass
class CCXTMarketData:
    exchange_id: str = "binance"

    def _exchange(self):
        cls = getattr(ccxt, self.exchange_id)
        return cls({"enableRateLimit": True})

    @staticmethod
    def _frame_to_df(rows) -> pd.DataFrame:
        if not rows:
            raise RuntimeError("Exchange returned no OHLCV rows")
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = (
            df.drop_duplicates("timestamp")
            .sort_values("timestamp")
            .set_index("timestamp")
        )
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].astype(float)
        return df

    @staticmethod
    def _drop_open_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        interval_ms = _TIMEFRAME_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        last_open_ms = int(df.index[-1].timestamp() * 1000)
        if now_ms - last_open_ms < interval_ms:
            return df.iloc[:-1]
        return df

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000) -> pd.DataFrame:
        if limit < 10 or limit > 1500:
            raise ValueError("limit must be between 10 and 1500")
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported fixed timeframe: {timeframe}")
        ex = self._exchange()
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = self._frame_to_df(rows)
        df = self._drop_open_candle(df, timeframe)
        if len(df) < 10:
            raise RuntimeError(
                f"Insufficient closed OHLCV rows for {symbol} {timeframe}: {len(df)}"
            )
        return df

    def fetch_ohlcv_history(
        self,
        symbol: str,
        timeframe: str,
        target_rows: int,
        page_limit: int = 1500,
    ) -> pd.DataFrame:
        """Paginate backward/forward using `since` so research gets real history.

        CCXT exchanges commonly cap one OHLCV request at a relatively small page
        size. We therefore fetch multiple pages, deduplicate timestamps, remove
        the currently forming candle, and keep the most recent target_rows closed
        candles. No synthetic candles are created.
        """
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported fixed timeframe: {timeframe}")
        if target_rows < 100:
            raise ValueError("target_rows must be at least 100")
        if page_limit < 100 or page_limit > 1500:
            raise ValueError("page_limit must be between 100 and 1500")

        ex = self._exchange()
        interval_ms = _TIMEFRAME_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # Start far enough back to cover the requested closed candles plus a
        # small overlap for boundary alignment and the current open candle.
        since_ms = now_ms - interval_ms * (target_rows + page_limit + 5)
        collected = []
        last_first_ts = None

        for _ in range((target_rows // page_limit) + 10):
            rows = ex.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=page_limit,
            )
            if not rows:
                break

            collected.extend(rows)
            first_ts = rows[0][0]
            last_ts = rows[-1][0]
            if last_ts <= since_ms or last_first_ts == first_ts:
                break
            last_first_ts = first_ts

            if len(collected) >= target_rows + page_limit:
                # We have more than enough rows; no need for another request.
                break

            since_ms = int(last_ts) + interval_ms
            time.sleep(0.05)

            if int(datetime.now(timezone.utc).timestamp() * 1000) - last_ts < interval_ms:
                break

        df = self._frame_to_df(collected)
        df = self._drop_open_candle(df, timeframe)
        if len(df) < min(target_rows, 100):
            raise RuntimeError(
                f"Insufficient historical OHLCV rows for {symbol} {timeframe}: "
                f"got {len(df)}, target {target_rows}"
            )
        return df.tail(target_rows)

    def fetch_multi(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> dict[str, pd.DataFrame]:
        return {
            symbol: self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            for symbol in symbols
        }

    def fetch_multi_timeframes(
        self,
        symbols: list[str],
        timeframes: list[str],
        limit: int = 1500,
    ) -> dict[tuple[str, str], pd.DataFrame]:
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for timeframe in timeframes:
            target = _HISTORY_TARGETS.get(timeframe, limit)
            for symbol in symbols:
                out[(symbol, timeframe)] = self.fetch_ohlcv_history(
                    symbol,
                    timeframe=timeframe,
                    target_rows=target,
                    page_limit=min(limit, 1500),
                )
                time.sleep(0.05)
        return out
