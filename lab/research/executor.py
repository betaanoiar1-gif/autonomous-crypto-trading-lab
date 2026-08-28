from __future__ import annotations

import numpy as np
import pandas as pd


def _int_param(params: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def compile_signal(df: pd.DataFrame, family: str, params: dict, directions: list[str]) -> pd.Series:
    close = df["close"].astype(float)
    family = family.lower().strip()
    allow_short = "short" in directions or "both" in directions

    if family == "momentum":
        n = _int_param(params, "lookback", 20, 2, 200)
        score = close.pct_change(n)
        sig = np.where(score > 0, 1.0, np.where((score < 0) & allow_short, -1.0, 0.0))
        return pd.Series(sig, index=df.index)

    if family == "mean_reversion":
        n = _int_param(params, "lookback", 40, 2, 200)
        z_entry = float(params.get("z_entry", 1.5))
        z_exit = float(params.get("z_exit", 0.25))
        mean = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=0).replace(0, np.nan)
        z = (close - mean) / std
        long = z < -abs(z_entry)
        short = (z > abs(z_entry)) & allow_short
        neutral = z.abs() < abs(z_exit)
        sig = pd.Series(0.0, index=df.index)
        sig[long] = 1.0
        sig[short] = -1.0
        sig[neutral] = 0.0
        return sig

    if family == "breakout":
        n = _int_param(params, "lookback", 40, 2, 200)
        high = df["high"].astype(float) if "high" in df else close
        low = df["low"].astype(float) if "low" in df else close
        prior_high = high.shift(1).rolling(n).max()
        prior_low = low.shift(1).rolling(n).min()
        sig = pd.Series(0.0, index=df.index)
        sig[close > prior_high] = 1.0
        if allow_short:
            sig[close < prior_low] = -1.0
        return sig

    if family == "moving_average_cross":
        fast = _int_param(params, "fast", 10, 2, 100)
        slow = _int_param(params, "slow", 40, fast + 1, 300)
        fast_ma = close.ewm(span=fast, adjust=False).mean()
        slow_ma = close.ewm(span=slow, adjust=False).mean()
        sig = pd.Series(np.where(fast_ma > slow_ma, 1.0, 0.0), index=df.index)
        if allow_short:
            sig[fast_ma < slow_ma] = -1.0
        return sig

    raise ValueError(f"Unsupported executable family: {family}")
