from __future__ import annotations

"""Phase 0 v7: calibrated research-foundation gate.

The purpose of this module is validation of the research plumbing, not strategy
search. It reuses the cached 1h parquet panel, aligns a common cutoff, verifies
spot long/flat execution, runs a deliberate lookahead canary, estimates the
random-noise false-positive floor, and tests the evaluator on a recoverable,
low-turnover synthetic regime edge.

No LLM generation, futures, leverage, or live trading are used.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_foundation_v7_latest.json"
CACHE = Path(
    os.getenv(
        "PHASE0_CACHE_DIR",
        "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3",
    )
)
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
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


def _load_cache(bars: int, max_symbols: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []
    required = min(bars, 5000)
    for symbol in SYMBOLS[:max_symbols]:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        print(f"CACHE {symbol} 1h ...", flush=True)
        if not path.exists() or path.stat().st_size <= 0:
            errors.append({"symbol": symbol, "error": "cache_missing"})
            print(f"MISSING {symbol}", flush=True)
            continue
        try:
            df = pd.read_parquet(path).sort_index()
            if len(df) < required:
                raise ValueError(f"only {len(df)} bars")
            data[symbol] = df.iloc[-min(len(df), bars):].copy()
            print(f"OK {symbol}: {len(data[symbol])} bars", flush=True)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)
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
    deltas = df.index.to_series().diff().dropna()
    gap_count = int((deltas != pd.Timedelta(hours=1)).sum()) if len(deltas) else 0
    max_gap = str(deltas.max()) if len(deltas) else "0 days 00:00:00"
    passed = monotonic and unique and finite and valid_ohlc
    return Audit(
        f"integrity:{symbol}",
        passed,
        {
            "bars": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "monotonic": monotonic,
            "unique": unique,
            "finite": finite,
            "valid_ohlc": valid_ohlc,
            "gap_count": gap_count,
            "max_gap": max_gap,
            "sha256": hashlib.sha256(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest(),
        },
    )


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    future_return = df["close"].pct_change().shift(-1).fillna(0.0)
    leaked = _metrics(df, (future_return > 0).astype(float), settings, 1.0)
    detected = bool(leaked["total_return"] > 0.20 or leaked["profit_factor"] > 1.5)
    return Audit(
        "lookahead_canary",
        detected,
        {
            "leaked_return": leaked["total_return"],
            "leaked_pf": leaked["profit_factor"],
            "leaked_trades": leaked["trade_count"],
            "definition": "future return at t+1 is intentionally leaked into signal(t)",
        },
    )


def _noise_floor(data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns: list[float] = []
    for i in range(trials):
        df = data[keys[i % len(keys)]]
        signal = pd.Series((rng.random(len(df)) > 0.70).astype(float), index=df.index)
        metrics = _metrics(df, signal, settings, 2.0)
        ok = bool(
            metrics["total_return"] > 0
            and metrics["profit_factor"] > 1.0
            and metrics["max_drawdown"] > -0.20
        )
        accepted += int(ok)
        returns.append(float(metrics["total_return"]))
    return {
        "trials": trials,
        "accepted": accepted,
        "false_positive_rate": accepted / max(trials, 1),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
    }


def _synthetic_edge(seed: int, settings, n: int = 12000) -> dict:
    """Test a known *recoverable* persistent regime edge.

    A latent state persists for 72 bars. Returns contain a small state-dependent
    drift plus noise. The observable predictor is the rolling mean of past
    returns, so the evaluator can recover the state causally at close(t). The
    position is updated only when the estimated state changes, keeping turnover
    low enough for transaction-cost stress to remain meaningful.
    """
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=float)
    block = 72
    for start in range(0, n, block):
        state[start : start + block] = 1.0 if rng.random() < 0.5 else -1.0

    sigma = 0.0015
    alpha = 0.00055
    noise = rng.normal(0.0, sigma, n)
    returns = np.clip(alpha * state + noise, -0.02, 0.02)

    close = 100.0 * np.cumprod(1.0 + returns)
    index = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    opens = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close),
            "low": np.minimum(opens, close),
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )

    lookback = 24
    observable = pd.Series(returns, index=index).rolling(lookback).mean()
    estimated_state = np.sign(observable.fillna(0.0)).to_numpy()
    # Long/flat policy: long in estimated positive regime, flat otherwise.
    signal = pd.Series((estimated_state > 0).astype(float), index=index)

    normal = _metrics(frame, signal, settings, 1.0)
    stress = _metrics(frame, signal, settings, 2.0)

    active = signal.to_numpy()
    realized_state = state
    direction_accuracy = float(
        np.mean((active > 0) == (realized_state > 0))
    )
    turnover = float(np.abs(np.diff(np.r_[0.0, active])).sum())

    detected = bool(
        normal["total_return"] > 0.05
        and stress["total_return"] > 0.0
        and normal["profit_factor"] > 1.10
        and turnover < n * 0.08
    )

    return {
        "normal": normal,
        "stress": stress,
        "sigma": sigma,
        "alpha": alpha,
        "regime_block_bars": block,
        "observable_lookback": lookback,
        "direction_accuracy": direction_accuracy,
        "turnover": turnover,
        "detected": detected,
        "alignment": "past 24-bar return mean at t determines position for t+1",
    }


def run(minutes: float = 30.0, bars: int = 50000, max_symbols: int = 12, seed: int = 20260829) -> dict:
    settings = load_settings()
    started = datetime.now(timezone.utc)
    print("=== PHASE 0 FOUNDATION V7 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Cache: {CACHE}", flush=True)
    print(f"Target: {bars} bars × {max_symbols} spot markets", flush=True)
    _save({"started_at": started.isoformat(), "decision": "LOADING"})

    data, errors = _load_cache(bars, max_symbols)
    required_markets = min(max_symbols, 8)
    if len(data) < required_markets:
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
        print(f"{audit.name}: {'PASS' if audit.passed else 'FAIL'} | gaps={audit.details['gap_count']}", flush=True)

    probe_df = next(iter(data.values())).iloc[:120]
    probe_signal = pd.Series(
        np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0),
        index=probe_df.index,
    )
    probe = _metrics(probe_df, probe_signal, settings)
    spot_ok = bool(probe["short_exposure"] == 0.0 and probe["long_exposure"] > 0.0)
    print(f"SPOT POSITION PROBE: {'PASS' if spot_ok else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    canary = _lookahead_canary(probe_df, settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | return={canary.details['leaked_return']:.2%} PF={canary.details['leaked_pf']:.2f}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_edge(seed + 1, settings)
    synthetic_ok = bool(synthetic["detected"])
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f} turnover={synthetic['turnover']:.0f}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic state accuracy={synthetic['direction_accuracy']:.2%}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic_ok else 'FAIL'}", flush=True)

    gates = {
        "data": len(data) >= required_markets and min(map(len, data.values())) >= min(bars, 5000),
        "integrity": all(a.passed for a in audits),
        "spot_long_flat": spot_ok,
        "lookahead_canary": canary.passed,
        "noise_fp_le_5pct": noise_ok,
        "synthetic_edge_detected": synthetic_ok,
    }
    decision = "PHASE0_READY_FOR_DISCOVERY" if all(gates.values()) else "PHASE0_BLOCKED"

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (datetime.now(timezone.utc) - started).total_seconds() / 60.0,
        "decision": decision,
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "alignment": alignment,
        "markets_loaded": list(data),
        "bars": {s: len(df) for s, df in data.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "spot_position_probe": probe,
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": gates,
        "cache": str(CACHE),
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
