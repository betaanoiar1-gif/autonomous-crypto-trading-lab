from __future__ import annotations

import numpy as np
import pandas as pd


def _int_param(params: dict, key: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(params.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _float_param(params: dict, key: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(params.get(key, default))
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
        z_entry = _float_param(params, "z_entry", 1.5, 0.5, 4.0)
        z_exit = _float_param(params, "z_exit", 0.25, 0.0, 2.0)
        mean = close.rolling(n).mean()
        std = close.rolling(n).std(ddof=0).replace(0, np.nan)
        z = (close - mean) / std
        sig = pd.Series(0.0, index=df.index)
        sig[z < -abs(z_entry)] = 1.0
        if allow_short:
            sig[z > abs(z_entry)] = -1.0
        sig[z.abs() < abs(z_exit)] = 0.0
        return sig

    if family == "breakout":
        n = _int_param(params, "lookback", 40, 2, 200)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
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

    if family == "rsi_reversion":
        n = _int_param(params, "rsi_length", 14, 2, 50)
        low_thr = _float_param(params, "rsi_low", 30.0, 5.0, 45.0)
        high_thr = _float_param(params, "rsi_high", 70.0, 55.0, 95.0)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        sig = pd.Series(0.0, index=df.index)
        sig[rsi < low_thr] = 1.0
        if allow_short:
            sig[rsi > high_thr] = -1.0
        sig[((rsi >= 45.0) & (rsi <= 55.0))] = 0.0
        return sig

    if family == "atr_breakout":
        n = _int_param(params, "atr_length", 14, 2, 50)
        mult = _float_param(params, "atr_mult", 1.5, 0.25, 5.0)
        prev_close = close.shift(1)
        tr = pd.concat([
            df["high"].astype(float) - df["low"].astype(float),
            (df["high"].astype(float) - prev_close).abs(),
            (df["low"].astype(float) - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / n, adjust=False).mean().shift(1)
        signal_base = prev_close
        upper = signal_base + mult * atr
        lower = signal_base - mult * atr
        sig = pd.Series(0.0, index=df.index)
        sig[close > upper] = 1.0
        if allow_short:
            sig[close < lower] = -1.0
        return sig

    if family == "trend_pullback":
        n = _int_param(params, "lookback", 40, 5, 200)
        threshold = _float_param(params, "pullback_threshold", 0.01, 0.001, 0.10)
        ema = close.ewm(span=n, adjust=False).mean()
        trend = ema.diff().fillna(0)
        pullback = close / ema - 1.0
        sig = pd.Series(0.0, index=df.index)
        sig[(trend > 0) & (pullback < -threshold)] = 1.0
        if allow_short:
            sig[(trend < 0) & (pullback > threshold)] = -1.0
        return sig

    if family == "channel_reversion":
        n = _int_param(params, "channel_length", 40, 5, 200)
        window = close.rolling(n)
        lo = window.quantile(0.10)
        hi = window.quantile(0.90)
        mid = window.mean()
        sig = pd.Series(0.0, index=df.index)
        sig[close < lo] = 1.0
        if allow_short:
            sig[close > hi] = -1.0
        sig[(close >= mid * 0.995) & (close <= mid * 1.005)] = 0.0
        return sig

    if family == "invented_composite":
        # Safe strategy DSL: the agent can invent a composition of bounded,
        # deterministic components, but cannot execute arbitrary Python.
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        trend_fast = _int_param(params, "trend_fast", 18, 3, 100)
        trend_slow = _int_param(params, "trend_slow", 72, trend_fast + 1, 300)
        momentum_window = _int_param(params, "momentum_window", 24, 2, 200)
        breakout_window = _int_param(params, "breakout_window", 36, 3, 200)
        vol_window = _int_param(params, "vol_window", 24, 5, 100)
        volume_window = _int_param(params, "volume_window", 24, 5, 100)

        w_trend = _float_param(params, "w_trend", 1.0, -3.0, 3.0)
        w_momentum = _float_param(params, "w_momentum", 1.0, -3.0, 3.0)
        w_breakout = _float_param(params, "w_breakout", 0.75, -3.0, 3.0)
        w_candle = _float_param(params, "w_candle", 0.5, -3.0, 3.0)
        w_volume = _float_param(params, "w_volume", 0.5, -3.0, 3.0)
        long_threshold = _float_param(params, "long_threshold", 1.75, 0.25, 6.0)
        short_threshold = _float_param(params, "short_threshold", 1.75, 0.25, 6.0)
        exit_threshold = _float_param(params, "exit_threshold", 0.40, 0.0, 2.0)
        vol_floor = _float_param(params, "vol_floor", 0.002, 0.0, 0.10)
        vol_cap = _float_param(params, "vol_cap", 0.050, 0.001, 0.20)
        volume_mult = _float_param(params, "volume_mult", 1.0, 0.5, 2.5)

        fast = close.ewm(span=trend_fast, adjust=False).mean()
        slow = close.ewm(span=trend_slow, adjust=False).mean()
        trend_component = np.sign(fast.shift(1) - slow.shift(1))

        momentum_component = np.sign(close.pct_change(momentum_window).shift(1))

        prior_high = high.shift(1).rolling(breakout_window).max()
        prior_low = low.shift(1).rolling(breakout_window).min()
        breakout_component = pd.Series(0.0, index=df.index)
        breakout_component[close > prior_high] = 1.0
        breakout_component[close < prior_low] = -1.0
        breakout_component = breakout_component.shift(1).fillna(0.0)

        candle_range = (high - low).replace(0, np.nan)
        body = ((close - open_) / candle_range).clip(-1.0, 1.0).shift(1)

        avg_volume = volume.shift(1).rolling(volume_window).mean()
        volume_component = np.sign(close.pct_change().shift(1))
        volume_component = volume_component.where(volume.shift(1) >= avg_volume * volume_mult, 0.0)

        volatility = close.pct_change().rolling(vol_window).std().shift(1)
        allowed = (volatility >= vol_floor) & (volatility <= vol_cap)

        score = (
            w_trend * pd.Series(trend_component, index=df.index)
            + w_momentum * pd.Series(momentum_component, index=df.index)
            + w_breakout * breakout_component
            + w_candle * body.fillna(0.0)
            + w_volume * volume_component.fillna(0.0)
        ).fillna(0.0)

        sig = pd.Series(0.0, index=df.index)
        sig[(score >= long_threshold) & allowed] = 1.0
        if allow_short:
            sig[(score <= -short_threshold) & allowed] = -1.0
        sig[score.abs() < exit_threshold] = 0.0
        return sig.fillna(0.0)

    raise ValueError(f"Unsupported executable family: {family}")
