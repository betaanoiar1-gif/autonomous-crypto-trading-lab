from __future__ import annotations

"""Phase 0 v8: calibrated synthetic harness over the validated v6 plumbing.

The historical-data, integrity, spot, lookahead and noise checks are reused from
v6. Only the synthetic calibration is changed: returns follow a persistent,
observable AR(1) regime so the known causal edge has positive expectancy with
low turnover under normal and doubled costs.
"""

from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .phase0_foundation_v6 import (
    _align,
    _integrity,
    _load_cache,
    _metrics,
    _noise,
    _lookahead,
    _save,
    DEFAULT_CACHE,
    SYMBOLS,
)
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_foundation_v8_latest.json"
CACHE = Path(os.getenv("PHASE0_CACHE_DIR", str(DEFAULT_CACHE)))


def _synthetic_edge(settings, seed: int, n: int = 12000, sigma: float = 0.0012,
                    phi: float = 0.42, threshold: float = 0.0,
                    min_hold: int = 24) -> dict:
    """Known causal edge with persistent signal and deliberately low turnover.

    r[t] = phi * r[t-1] + eps[t]
    signal[t] = 1 when r[t] > threshold, but the position is held for at least
    min_hold bars. Since signal[t] observes r[t], it is causally valid for the
    next-bar execution convention.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        r[t] = float(np.clip(phi * r[t - 1] + eps[t], -0.02, 0.02))

    close = 100.0 * np.cumprod(1.0 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    prev_close = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({
        "open": prev_close,
        "high": np.maximum(prev_close, close),
        "low": np.minimum(prev_close, close),
        "close": close,
        "volume": 1_000_000.0,
    }, index=idx)

    raw = (r > threshold).astype(float)
    sig = np.zeros(n, dtype=float)
    current = 0.0
    age = min_hold
    for t in range(n):
        desired = raw[t]
        if desired != current and age >= min_hold:
            current = desired
            age = 0
        sig[t] = current
        age += 1
    signal = pd.Series(sig, index=idx)

    normal = _metrics(df, signal, settings, 1.0)
    stress = _metrics(df, signal, settings, 2.0)
    detected = bool(
        normal["total_return"] > 0.02
        and stress["total_return"] > 0.01
        and normal["profit_factor"] > 1.0
        and stress["profit_factor"] > 1.0
        and normal["trade_count"] < n / min_hold * 1.5
    )
    return {
        "normal": normal,
        "stress": stress,
        "sigma": sigma,
        "phi": phi,
        "min_hold_bars": min_hold,
        "detected": detected,
        "alignment": "signal(t) observes r[t] and is applied to return(t+1), with minimum holding period",
    }


def run(minutes: float = 20.0, bars: int = 50000, max_symbols: int = 12,
        seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    settings = load_settings()
    symbols = SYMBOLS[:max_symbols]

    print("=== PHASE 0 FOUNDATION V8 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Cache: {CACHE}", flush=True)
    print(f"Target: {bars} 1h bars × {len(symbols)} spot markets", flush=True)
    _save({"started_at": started.isoformat(), "decision": "LOADING", "version": "v8"})

    data, errors = _load_cache(CACHE, bars, max_symbols)
    required = min(len(symbols), 8)
    if len(data) < required:
        payload = {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": "PHASE0_DATA_INSUFFICIENT",
            "markets_loaded": list(data),
            "errors": errors,
        }
        _save(payload)
        return payload

    data, alignment = _align(data)
    audits = [_integrity(s, df) for s, df in data.items()]
    print("=== DATA INTEGRITY ===", flush=True)
    for a in audits:
        print(f"{a.name}: {'PASS' if a.passed else 'FAIL'} | gaps={a.details.get('bad_1h_intervals', 0)}", flush=True)

    probe_df = next(iter(data.values())).iloc[:120]
    alternating = pd.Series(np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0), index=probe_df.index)
    probe = _metrics(probe_df, alternating, settings)
    spot_ok = probe["short_exposure"] == 0.0
    print(f"SPOT POSITION PROBE: {'PASS' if spot_ok else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    canary = _lookahead(probe_df, settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | return={canary.details['total_return']:.2%} PF={canary.details['profit_factor']:.2f}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise(data, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_edge(settings, seed + 1)
    synthetic_ok = synthetic["detected"]
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f} trades={synthetic['normal']['trade_count']}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f} trades={synthetic['stress']['trade_count']}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic_ok else 'FAIL'}", flush=True)

    gates = {
        "data": len(data) >= required and min(map(len, data.values())) >= min(bars, 5000),
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
        "version": "v8",
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "alignment": alignment,
        "markets_loaded": list(data),
        "bars": {s: len(df) for s, df in data.items()},
        "errors": errors,
        "integrity": [a.details for a in audits],
        "spot_position_probe": probe,
        "lookahead_canary": canary.details,
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": gates,
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("PHASE0_MINUTES", "20")),
        bars=int(os.getenv("PHASE0_BARS", "50000")),
        max_symbols=int(os.getenv("PHASE0_MAX_SYMBOLS", "12")),
        seed=int(os.getenv("PHASE0_SEED", "20260829")),
    )
