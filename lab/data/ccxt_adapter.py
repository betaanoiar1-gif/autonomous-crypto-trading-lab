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

_HISTORY_TARGETS = {"15m": 10_000, "1h": 5_000, "4h": 3_000}


@dataclass
class CCXTMarketData:
    exchange_id: str = "binance"

    def _exchange(self, market_type: str = "spot"):
        market_type = str(market_type).lower().strip()
        if market_type == "futures" and self.exchange_id == "binance":
            return ccxt.binanceusdm({"enableRateLimit": True})
        if market_type == "futures" and self.exchange_id == "kraken":
            return ccxt.krakenfutures({"enableRateLimit": True})
        cls = getattr(ccxt, self.exchange_id)
        return cls({"enableRateLimit": True})

    @staticmethod
    def _contract_symbol(symbol: str) -> str:
        if ":" in symbol:
            return symbol
        quote = symbol.split("/")[-1]
        return f"{symbol}:{quote}"

    @staticmethod
    def _frame_to_df(rows) -> pd.DataFrame:
        if not rows:
            raise RuntimeError("Exchange returned no OHLCV rows")
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        numeric = ["open", "high", "low", "close", "volume"]
        df[numeric] = df[numeric].astype(float)
        return df

    @staticmethod
    def _drop_open_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        interval_ms = _TIMEFRAME_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        last_open_ms = int(df.index[-1].timestamp() * 1000)
        return df.iloc[:-1] if now_ms - last_open_ms < interval_ms else df

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, target_rows: int, page_limit: int = 1500,
                            market_type: str = "spot") -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported fixed timeframe: {timeframe}")
        if target_rows < 100:
            raise ValueError("target_rows must be at least 100")
        ex = self._exchange(market_type)
        api_symbol = self._contract_symbol(symbol) if market_type == "futures" else symbol
        interval_ms = _TIMEFRAME_MS[timeframe]
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since_ms = now_ms - interval_ms * (target_rows + page_limit + 5)
        collected = []
        last_ts = None
        max_pages = max(10, (target_rows // page_limit) + 10)
        for _ in range(max_pages):
            rows = ex.fetch_ohlcv(api_symbol, timeframe=timeframe, since=since_ms, limit=min(page_limit, 1500))
            if not rows:
                break
            collected.extend(rows)
            page_last = int(rows[-1][0])
            if last_ts is not None and page_last <= last_ts:
                break
            last_ts = page_last
            if len(collected) >= target_rows + page_limit:
                break
            since_ms = page_last + interval_ms
            if int(datetime.now(timezone.utc).timestamp() * 1000) - page_last < interval_ms:
                break
            time.sleep(0.05)
        df = self._drop_open_candle(self._frame_to_df(collected), timeframe)
        if len(df) < min(target_rows, 100):
            raise RuntimeError(f"Insufficient historical OHLCV rows for {symbol} {timeframe}: got {len(df)}, target {target_rows}")
        return df.tail(target_rows)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000, market_type: str = "spot") -> pd.DataFrame:
        target = min(limit, _HISTORY_TARGETS.get(timeframe, limit))
        return self.fetch_ohlcv_history(symbol, timeframe, target, page_limit=min(limit, 1500), market_type=market_type)

    def fetch_multi_timeframes(self, symbols: list[str], timeframes: list[str], limit: int = 1500,
                               market_type: str = "spot") -> dict[tuple[str, str], pd.DataFrame]:
        out: dict[tuple[str, str], pd.DataFrame] = {}
        for timeframe in timeframes:
            target = _HISTORY_TARGETS.get(timeframe, limit)
            for symbol in symbols:
                out[(symbol, timeframe)] = self.fetch_ohlcv_history(symbol, timeframe, target, page_limit=min(limit, 1500), market_type=market_type)
                time.sleep(0.05)
        return out

    def fetch_funding_history(self, symbol: str, since_ms: int | None = None, until_ms: int | None = None,
                              target_rows: int = 500) -> pd.Series:
        """Fetch public historical funding rates for a futures contract via CCXT.

        CCXT's unified `fetchFundingRateHistory` returns fundingRate as a decimal
        at the funding timestamp. The method is public market data and needs no
        API key on supported exchanges.
        """
        ex = self._exchange("futures")
        if not ex.has.get("fetchFundingRateHistory"):
            raise RuntimeError(f"{self.exchange_id} does not expose funding-rate history through CCXT")
        api_symbol = self._contract_symbol(symbol)
        params = {}
        if until_ms is not None:
            params["until"] = int(until_ms)
        rows = ex.fetch_funding_rate_history(api_symbol, since=since_ms, limit=min(target_rows, 1000), params=params)
        points = []
        for row in rows:
            ts = row.get("timestamp")
            rate = row.get("fundingRate")
            if ts is None or rate is None:
                continue
            points.append((pd.to_datetime(int(ts), unit="ms", utc=True), float(rate)))
        if not points:
            raise RuntimeError(f"No historical funding rates returned for {symbol}")
        return pd.Series(dict(points)).sort_index()
