from __future__ import annotations

"""Phase 0 v9: standalone research-foundation validation harness.

No dependency on prior Phase 0 versions. Uses cached historical OHLCV and the
canonical backtester/config, then validates data integrity, spot long/flat
semantics, lookahead detection, noise false positives, and a known causal
synthetic edge with low turnover.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_foundation_v9_latest.json"
DEFAULT_CACHE = Path(
    "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"
)
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)


@dataclass
class Audit:
    name: str
    passed: bool
    details: dict


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(OUT)


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _metrics(df: pd.DataFrame, signal: pd.Series, settings, cost_mult: float = 1.0) -> dict:
    pos = pd.Series(signal, index=df.index, dtype=float).clip(0.0, 1.0)
    result = run_ohlcv(
        df,
        pos,
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return dict(result.metrics)


def _load_cache(cache_dir: Path, bars: int, max_symbols: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []
    for symbol in SYMBOLS[:max_symbols]:
        path = cache_dir / f"{symbol.replace('/', '_')}_1h.parquet"
        print(f"CACHE {symbol} 1h ...", flush=True)
        if not path.exists() or path.stat().st_size <= 0:
            errors.append({"symbol": symbol, "error": "cache_missing"})
            print(f"MISSING {symbol}", flush=True)
            continue
        try:
            df = pd.read_parquet(path).sort_index()
            if len(df) < min(bars, 5000):
                raise ValueError(f"only {len(df)} bars")
            data[symbol] = df.iloc[-min(len(df), bars):].copy()
            print(f"OK {symbol}: {len(data[symbol])} bars", flush=True)
        except Exception as exc:
            errors.append({
                "symbol": symbol,
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"ERROR {symbol}: {exc}", flush=True)
    return data, errors


def _align_common_cutoff(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    common_end = min(df.index[-1] for df in data.values())
    common_start = max(df.index[0] for df in data.values())
    aligned = {
        symbol: df.loc[(df.index >= common_start) & (df.index <= common_end)].copy()
        for symbol, df in data.items()
    }
    return aligned, {
        "common_start": str(common_start),
        "common_end": str(common_end),
        "bars": {symbol: len(df) for symbol, df in aligned.items()},
    }


def _integrity(symbol: str, df: pd.DataFrame) -> Audit:
    required = ["open", "high", "low", "close", "volume"]
    values = df[required].to_numpy(dtype=float)
    monotonic = bool(df.index.is_monotonic_increasing)
    unique = bool(df.index.is_unique)
    finite = bool(np.isfinite(values).all())
    valid_ohlc = bool(
        (
            (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["low"] <= df[["open", "close"]].min(axis=1))
            & (df["high"] >= df["low"])
            & (df["volume"] >= 0)
        ).all()
    )
    deltas = df.index.to_series().diff().dropna()
    bad_intervals = int((deltas != pd.Timedelta(hours=1)).sum()) if len(deltas) else 0
    max_gap = str(deltas.max()) if len(deltas) else "0 days 00:00:00"
    return Audit(
        f"integrity:{symbol}",
        monotonic and unique and finite and valid_ohlc,
        {
            "bars": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "monotonic": monotonic,
            "unique": unique,
            "finite": finite,
            "valid_ohlc": valid_ohlc,
            "bad_1h_intervals": bad_intervals,
            "max_gap": max_gap,
            "sha256": _hash_df(df),
        },
    )


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    future = df["close"].pct_change().shift(-1).fillna(0.0)
    metrics = _metrics(df, (future > 0).astype(float), settings)
    detected = bool(
        metrics["total_return"] > 0.10
        and metrics["profit_factor"] > 1.5
    )
    return Audit(
        "lookahead_canary",
        detected,
        {
            "leaked_return": metrics["total_return"],
            "leaked_pf": metrics["profit_factor"],
            "leaked_trades": metrics["trade_count"],
        },
    )


def _noise_floor(
    data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 50
) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns: list[float] = []
    for i in range(trials):
        df = data[keys[i % len(keys)]]
        signal = pd.Series(
            (rng.random(len(df)) > 0.70).astype(float),
            index=df.index,
        )
        metrics = _metrics(df, signal, settings, cost_mult=2.0)
        ok = bool(
            metrics["total_return"] > 0.0
            and metrics["profit_factor"] > 1.0
            and metrics["max_drawdown"] > -0.20
        )
        accepted += int(ok)
        returns.append(float(metrics["total_return"]))
    return {
        "trials": trials,
        "accepted": accepted,
        "false_positive_rate": accepted / max(trials, 1),
        "median_return": float(np.median(returns)),
        "max_return": float(np.max(returns)),
    }


def _synthetic_edge(
    settings,
    seed: int,
    n: int = 12000,
    sigma: float = 0.0012,
    alpha: float = 0.00035,
    switch_prob: float = 0.015,
    hold_bars: int = 48,
) -> dict:
    """Known causal persistent-regime edge with low turnover.

    A latent state persists for many bars and directly affects next-bar returns.
    The observed predictor is the recent return sign. The test strategy smooths
    that predictor and holds positions to reduce turnover. This deliberately
    differs from the market-data noise process used above.
    """
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=float)
    state[0] = 1.0
    switches = 0
    for t in range(1, n):
        if rng.random() < switch_prob:
            state[t] = -state[t - 1]
            switches += 1
        else:
            state[t] = state[t - 1]

    noise = rng.normal(0.0, sigma, n)
    returns = np.clip(state * alpha + noise, -0.02, 0.02)
    close = 100.0 * np.cumprod(1.0 + returns)
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    prev_close = np.r_[close[0], close[:-1]]
    df = pd.DataFrame(
        {
            "open": prev_close,
            "high": np.maximum(prev_close, close),
            "low": np.minimum(prev_close, close),
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )

    observable = pd.Series(returns, index=index).rolling(
        24, min_periods=24
    ).mean().fillna(0.0)
    desired = (observable > 0.0).astype(float)
    position = desired.to_numpy(copy=True)

    # Hysteresis/minimum hold: once a position changes, keep it for hold_bars.
    current = 0.0
    age = hold_bars
    held = np.zeros(n, dtype=float)
    for t in range(n):
        want = float(position[t])
        if want != current and age >= hold_bars:
            current = want
            age = 0
        held[t] = current
        age += 1

    signal = pd.Series(held, index=index)
    normal = _metrics(df, signal, settings, cost_mult=1.0)
    stress = _metrics(df, signal, settings, cost_mult=2.0)
    turnover_ok = bool(normal["trade_count"] <= max(2, switches * 2 + 10))
    detected = bool(
        normal["total_return"] > 0.02
        and stress["total_return"] > 0.01
        and normal["profit_factor"] > 1.05
        and stress["profit_factor"] > 1.02
        and turnover_ok
    )
    return {
        "normal": normal,
        "stress": stress,
        "sigma": sigma,
        "alpha": alpha,
        "switch_probability": switch_prob,
        "regime_switches": switches,
        "prediction_window": 24,
        "hold_bars": hold_bars,
        "detected": detected,
        "turnover_ok": turnover_ok,
        "alignment": "observable past returns at t predict return(t+1); position is held with hysteresis",
    }


def run(
    minutes: float = 30.0,
    bars: int = 50000,
    max_symbols: int = 12,
    seed: int = 20260829,
) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    cache = Path(os.getenv("PHASE0_CACHE_DIR", str(DEFAULT_CACHE)))
    symbols = SYMBOLS[:max_symbols]

    print("=== PHASE 0 FOUNDATION V9 ===", flush=True)
    print(
        "AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED",
        flush=True,
    )
    print(f"Cache: {cache}", flush=True)
    print(f"Target: {bars} 1h bars x {len(symbols)} spot markets", flush=True)
    _save(
        {
            "started_at": started.isoformat(),
            "decision": "LOADING",
            "version": "v9",
            "symbols": list(symbols),
            "bars": bars,
        }
    )

    data, errors = _load_cache(cache, bars, max_symbols)
    required = min(len(symbols), 8)
    if len(data) < required or time.monotonic() >= deadline:
        payload = {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": "PHASE0_DATA_INSUFFICIENT",
            "markets_loaded": list(data),
            "errors": errors,
        }
        _save(payload)
        return payload

    data, alignment = _align_common_cutoff(data)
    audits = [_integrity(symbol, df) for symbol, df in data.items()]
    print("=== DATA INTEGRITY ===", flush=True)
    for audit in audits:
        print(
            f"{audit.name}: {'PASS' if audit.passed else 'FAIL'} | "
            f"gaps={audit.details['bad_1h_intervals']}",
            flush=True,
        )

    probe_df = next(iter(data.values())).iloc[:120]
    alternating = pd.Series(
        np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0),
        index=probe_df.index,
    )
    probe = _metrics(probe_df, alternating, settings)
    spot_ok = bool(probe["short_exposure"] == 0.0)
    print(
        f"SPOT POSITION PROBE: {'PASS' if spot_ok else 'FAIL'} | "
        f"long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}",
        flush=True,
    )

    canary = _lookahead_canary(probe_df, settings)
    print(
        f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | "
        f"return={canary.details['leaked_return']:.2%} "
        f"PF={canary.details['leaked_pf']:.2f}",
        flush=True,
    )

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(
        f"Noise false-positive rate: {noise['false_positive_rate']:.2%} "
        f"({noise['accepted']}/{noise['trials']})",
        flush=True,
    )

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_edge(settings, seed + 1)
    synthetic_ok = bool(synthetic["detected"])
    print(
        f"Synthetic normal return={synthetic['normal']['total_return']:.2%} "
        f"PF={synthetic['normal']['profit_factor']:.2f} "
        f"trades={synthetic['normal']['trade_count']}",
        flush=True,
    )
    print(
        f"Synthetic stress return={synthetic['stress']['total_return']:.2%} "
        f"PF={synthetic['stress']['profit_factor']:.2f} "
        f"trades={synthetic['stress']['trade_count']}",
        flush=True,
    )
    print(
        f"Synthetic detection: {'PASS' if synthetic_ok else 'FAIL'}",
        flush=True,
    )

    gates = {
        "data": len(data) >= required and min(len(df) for df in data.values()) >= min(bars, 5000),
        "integrity": all(a.passed for a in audits),
        "spot_long_flat": spot_ok,
        "lookahead_canary": canary.passed,
        "noise_fp_le_5pct": noise_ok,
        "synthetic_edge_detected": synthetic_ok,
    }
    decision = (
        "PHASE0_READY_FOR_DISCOVERY"
        if all(gates.values())
        else "PHASE0_BLOCKED"
    )
    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (datetime.now(timezone.utc) - started).total_seconds() / 60.0,
        "decision": decision,
        "version": "v9",
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "alignment": alignment,
        "markets_loaded": list(data),
        "bars": {symbol: len(df) for symbol, df in data.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "spot_position_probe": probe,
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": gates,
        "data_dir": str(cache),
        "deadline_seconds": minutes * 60.0,
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("PHASE0_MINUTES", "30")),
        bars=int(os.getenv("PHASE0_BARS", "50000")),
        max_symbols=int(os.getenv("PHASE0_MAX_SYMBOLS", "12")),
        seed=int(os.getenv("PHASE0_SEED", "20260829")),
    )
