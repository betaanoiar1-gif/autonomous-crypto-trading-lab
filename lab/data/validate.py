from __future__ import annotations

import pandas as pd


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    out = df.copy()
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    if (out[["open", "high", "low", "close", "volume"]] < 0).any().any():
        raise ValueError("OHLCV data contains negative values")
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError("Invalid high prices detected")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError("Invalid low prices detected")
    if (out["low"] <= 0).any() or (out["close"] <= 0).any():
        raise ValueError("Non-positive market prices detected")
    return out
