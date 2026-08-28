from __future__ import annotations

import ccxt
import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

def fetch_ohlcv(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 1000, exchange_id: str = "binance") -> pd.DataFrame:
    exchange_cls = getattr(ccxt, exchange_id, None)
    if exchange_cls is None:
        raise ValueError(f"Unknown CCXT exchange: {exchange_id}")
    exchange = exchange_cls({"enableRateLimit": True})
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise RuntimeError("Exchange returned no OHLCV data")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for col in COLUMNS[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame.dropna().drop_duplicates("timestamp").set_index("timestamp").sort_index()
