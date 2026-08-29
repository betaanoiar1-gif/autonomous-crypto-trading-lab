from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import time
import zipfile

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
_VISION_BASE = "https://data.binance.vision"
_VISION_TIMEOUT_SECONDS = 30


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

    @staticmethod
    def _vision_symbol(symbol: str) -> str:
        return symbol.replace("/", "").replace(":", "")

    @staticmethod
    def _vision_months_back(target_rows: int, timeframe: str) -> int:
        bars_per_month = max(1, int((30 * 24 * 60 * 60 * 1000) / _TIMEFRAME_MS[timeframe]))
        return max(3, int(target_rows / bars_per_month) + 4)

    @staticmethod
    def _vision_month_url(market_type: str, symbol: str, timeframe: str, year: int, month: int) -> str:
        sym = CCXTMarketData._vision_symbol(symbol)
        if market_type == "futures":
            root = f"{_VISION_BASE}/data/futures/um/monthly/klines/{sym}/{timeframe}"
        else:
            root = f"{_VISION_BASE}/data/spot/monthly/klines/{sym}/{timeframe}"
        return f"{root}/{sym}-{timeframe}-{year:04d}-{month:02d}.zip"

    @staticmethod
    def _vision_daily_url(market_type: str, symbol: str, timeframe: str, day: datetime) -> str:
        sym = CCXTMarketData._vision_symbol(symbol)
        if market_type == "futures":
            root = f"{_VISION_BASE}/data/futures/um/daily/klines/{sym}/{timeframe}"
        else:
            root = f"{_VISION_BASE}/data/spot/daily/klines/{sym}/{timeframe}"
        date = day.strftime("%Y-%m-%d")
        return f"{root}/{sym}-{timeframe}-{date}.zip"

    @staticmethod
    def _download(url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "autonomous-crypto-trading-lab/1.0"})
        with urlopen(request, timeout=_VISION_TIMEOUT_SECONDS) as response:
            return response.read()

    @classmethod
    def _read_vision_zip(cls, payload: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError("Binance Vision archive contains no CSV")
            with archive.open(csv_names[0]) as fh:
                frame = pd.read_csv(fh, header=None)

        if frame.empty:
            raise RuntimeError("Binance Vision CSV is empty")
        frame = frame.iloc[:, :6].copy()
        frame.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"].astype("int64"), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna().drop_duplicates("timestamp").set_index("timestamp").sort_index()

    def _fetch_vision_history(self, symbol: str, timeframe: str, target_rows: int, market_type: str = "spot") -> pd.DataFrame:
        if market_type == "futures" and os.getenv("ACL_VISION_FUTURES", "0") != "1":
            raise RuntimeError("Binance Vision futures fallback disabled; use historical futures source explicitly")

        now = datetime.now(timezone.utc)
        frames: list[pd.DataFrame] = []
        months_back = self._vision_months_back(target_rows, timeframe)

        # Monthly archives are compact and deterministic. The current partial month
        # may not be published yet, so it is filled from daily archives below.
        year, month = now.year, now.month
        for offset in range(months_back):
            y, m = year, month - offset
            while m <= 0:
                m += 12
                y -= 1
            url = self._vision_month_url(market_type, symbol, timeframe, y, m)
            try:
                frames.append(self._read_vision_zip(self._download(url)))
            except (HTTPError, URLError, zipfile.BadZipFile, RuntimeError, ValueError):
                continue
            if sum(len(x) for x in frames) >= target_rows + 2_000:
                break

        # Fill the current month from daily archives when the monthly archive is not
        # available yet (or when we still need a few more rows).
        if sum(len(x) for x in frames) < target_rows + 2_000:
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            for day_offset in range(now.day):
                day = month_start.replace(hour=0) + pd.Timedelta(days=day_offset)
                day_dt = day.to_pydatetime() if hasattr(day, "to_pydatetime") else day
                url = self._vision_daily_url(market_type, symbol, timeframe, day_dt)
                try:
                    frames.append(self._read_vision_zip(self._download(url)))
                except (HTTPError, URLError, zipfile.BadZipFile, RuntimeError, ValueError):
                    continue

        if not frames:
            raise RuntimeError(f"Binance Vision returned no historical data for {symbol} {timeframe}")
        df = pd.concat(frames, axis=0).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = self._drop_open_candle(df, timeframe)
        if len(df) < min(target_rows, 100):
            raise RuntimeError(f"Insufficient Vision history for {symbol} {timeframe}: got {len(df)}, target {target_rows}")
        return df.tail(target_rows)

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, target_rows: int, page_limit: int = 1500,
                            market_type: str = "spot") -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"Unsupported fixed timeframe: {timeframe}")
        if target_rows < 100:
            raise ValueError("target_rows must be at least 100")

        source = os.getenv("ACL_DATA_SOURCE", "auto").lower().strip()
        if source not in {"auto", "ccxt", "vision"}:
            raise ValueError("ACL_DATA_SOURCE must be one of: auto, ccxt, vision")

        if source in {"vision", "auto"} and self.exchange_id == "binance" and market_type == "spot":
            try:
                return self._fetch_vision_history(symbol, timeframe, target_rows, market_type=market_type)
            except Exception:
                if source == "vision":
                    raise

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
                out[(symbol, timeframe)] = self.fetch_ohlcv_history(
                    symbol, timeframe, target, page_limit=min(limit, 1500), market_type=market_type
                )
                time.sleep(0.05)
        return out

    def fetch_funding_history(self, symbol: str, since_ms: int | None = None, until_ms: int | None = None,
                              target_rows: int = 500, page_limit: int = 1000) -> pd.Series:
        """Fetch paginated historical funding rates for a futures contract via CCXT."""
        ex = self._exchange("futures")
        if not ex.has.get("fetchFundingRateHistory"):
            raise RuntimeError(f"{self.exchange_id} does not expose funding-rate history through CCXT")
        api_symbol = self._contract_symbol(symbol)
        collected: list[tuple[pd.Timestamp, float]] = []
        cursor = int(since_ms) if since_ms is not None else None
        max_pages = max(4, (target_rows // page_limit) + 8)
        for _ in range(max_pages):
            params = {}
            if until_ms is not None:
                params["until"] = int(until_ms)
            rows = ex.fetch_funding_rate_history(api_symbol, since=cursor, limit=min(page_limit, 1000), params=params)
            if not rows:
                break
            page_points = []
            for row in rows:
                ts = row.get("timestamp")
                rate = row.get("fundingRate")
                if ts is None or rate is None:
                    continue
                ts_i = int(ts)
                if since_ms is not None and ts_i < since_ms:
                    continue
                if until_ms is not None and ts_i > until_ms:
                    continue
                page_points.append((pd.to_datetime(ts_i, unit="ms", utc=True), float(rate)))
            collected.extend(page_points)
            page_last = max((int(ts.timestamp() * 1000) for ts, _ in page_points), default=None)
            if page_last is None:
                break
            if until_ms is not None and page_last >= until_ms:
                break
            if cursor is not None and page_last <= cursor:
                break
            if len(collected) >= target_rows and until_ms is None:
                break
            cursor = page_last + 1
            time.sleep(0.05)
        if not collected:
            raise RuntimeError(f"No historical funding rates returned for {symbol}")
        series = pd.Series(dict(collected)).sort_index()
        if since_ms is not None:
            series = series[series.index >= pd.to_datetime(since_ms, unit="ms", utc=True)]
        if until_ms is not None:
            series = series[series.index <= pd.to_datetime(until_ms, unit="ms", utc=True)]
        if series.empty:
            raise RuntimeError(f"Funding history for {symbol} contains no events in the requested interval")
        return series
