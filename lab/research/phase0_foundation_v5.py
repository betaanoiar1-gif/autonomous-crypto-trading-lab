from __future__ import annotations

"""Phase 0 v5: clean research-foundation gate.

This module validates the research plumbing before any discovery search.
It is spot-only, long/flat, unlevered, AI-free, cache-first, gap-aware,
and uses explicit causal alignment for synthetic validation.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_foundation_v5_latest.json"
DEFAULT_CACHE = Path("/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3")
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
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _metrics(df: pd.DataFrame, signal: pd.Series, settings, cost_mult: float = 1.0) -> dict:
    pos = pd.Series(signal, index=df.index).astype(float).clip(0.0, 1.0)
    r = run_ohlcv(
        df,
        pos,
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return dict(r.metrics)


def _integrity(symbol: str, df: pd.DataFrame) -> Audit:
    required = ["open", "high", "low", "close", "volume"]
    values = df[required].to_numpy(dtype=float)
    monotonic = bool(df.index.is_monotonic_increasing)
    unique = bool(df.index.is_unique)
    finite = bool(np.isfinite(values).all())
    valid_ohlc = bool((
        (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df["low"])
        & (df["volume"] >= 0)
    ).all())
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
                errors.append({"symbol": symbol, "error": f"only {len(df)} bars"})
                continue
            data[symbol] = df.iloc[-min(len(df), bars):].copy()
            print(f"OK {symbol}: {len(data[symbol])} bars", flush=True)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
    return data, errors


def _align(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    common_end = min(df.index[-1] for df in data.values())
    common_start = max(df.index[0] for df in data.values())
    aligned = {s: df.loc[(df.index >= common_start) & (df.index <= common_end)].copy() for s, df in data.items()}
    return aligned, {
        "common_start": str(common_start),
        "common_end": str(common_end),
        "bars": {s: len(df) for s, df in aligned.items()},
    }


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    # Deliberately illegal: at close(t) it uses r[t+1], so a correct next-bar evaluator must show a strong result.
    future = df["close"].pct_change().shift(-1).fillna(0.0)
    leaked = _metrics(df, (future > 0).astype(float), settings, 1.0)
    detected = bool(leaked["total_return"] > 0.20 or leaked["profit_factor"] > 1.5)
    return Audit("lookahead_canary", detected, {
        "leaked_return": leaked["total_return"],
        "leaked_pf": leaked["profit_factor"],
        "leaked_trades": leaked["trade_count"],
    })


def _noise_floor(data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    symbols = list(data)
    accepted = 0
    returns = []
    for i in range(trials):
        df = data[symbols[i % len(symbols)]]
        sig = pd.Series((rng.random(len(df)) > 0.65).astype(float), index=df.index)
        m = _metrics(df, sig, settings, 2.0)
        ok = bool(m["total_return"] > 0 and m["profit_factor"] > 1.0 and m["max_drawdown"] > -0.20)
        accepted += int(ok)
        returns.append(float(m["total_return"]))
    return {
        "trials": trials,
        "accepted": accepted,
        "false_positive_rate": accepted / max(trials, 1),
        "median_return": float(np.median(returns)),
        "max_return": float(np.max(returns)),
    }


def _synthetic_edge(settings, seed: int, n: int = 6000, sigma: float = 0.0015, alpha: float = 0.00030) -> dict:
    # Known causal edge: r[t+1] = noise[t+1] + alpha*sign(r[t]).
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, n)
    for t in range(n - 1):
        r[t + 1] = float(np.clip(noise[t + 1] + alpha * np.sign(r[t]), -0.02, 0.02))
    close = 100.0 * np.cumprod(1.0 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "high": np.maximum(np.r_[close[0], close[:-1]], close),
        "low": np.minimum(np.r_[close[0], close[:-1]], close),
        "close": close,
        "volume": 1_000_000.0,
    }, index=idx)
    signal = pd.Series((r > 0).astype(float), index=idx)
    normal = _metrics(df, signal, settings, 1.0)
    stress = _metrics(df, signal, settings, 2.0)
    detected = bool(normal["total_return"] > 0 and stress["total_return"] > 0 and normal["profit_factor"] > 1.0)
    return {
        "normal": normal,
        "stress": stress,
        "sigma": sigma,
        "alpha": alpha,
        "detected": detected,
        "alignment": "signal(t)=sign(r[t]) applied to return(t+1)",
    }


def run(minutes: float = 30.0, bars: int = 50000, max_symbols: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    cache = Path(os.getenv("PHASE0_CACHE_DIR", str(DEFAULT_CACHE)))
    symbols = SYMBOLS[:max_symbols]
    print("=== PHASE 0 FOUNDATION V5 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Cache: {cache}", flush=True)
    _save({"started_at": started.isoformat(), "decision": "LOADING", "symbols": list(symbols), "bars": bars})

    data, errors = _load_cache(cache, bars, max_symbols)
    if len(data) < min(len(symbols), 8) or time.monotonic() >= deadline:
        payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": "PHASE0_DATA_INSUFFICIENT", "markets_loaded": list(data), "errors": errors}
        _save(payload)
        return payload

    aligned, alignment = _align(data)
    audits = [_integrity(s, df) for s, df in aligned.items()]
    print("=== DATA INTEGRITY ===", flush=True)
    for a in audits:
        print(f"{a.name}: {'PASS' if a.passed else 'FAIL'}", flush=True)

    probe_df = next(iter(aligned.values())).iloc[:120]
    probe = _metrics(probe_df, pd.Series(np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0), index=probe_df.index), settings)
    spot_ok = bool(probe["short_exposure"] == 0.0)
    print(f"SPOT POSITION PROBE: {'PASS' if spot_ok else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    canary = _lookahead_canary(probe_df, settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | leaked_return={canary.details['leaked_return']:.2%} | PF={canary.details['leaked_pf']:.2f}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(aligned, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_edge(settings, seed + 1)
    synthetic_ok = synthetic["detected"]
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic_ok else 'FAIL'}", flush=True)

    gates = {
        "data": len(aligned) >= min(len(symbols), 8) and min(len(df) for df in aligned.values()) >= min(bars, 5000),
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
        "markets_loaded": list(aligned),
        "bars": {s: len(df) for s, df in aligned.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": gates,
        "data_dir": str(cache),
        "data_files": [str(cache / f"{s.replace('/', '_')}_1h.parquet") for s in aligned],
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
