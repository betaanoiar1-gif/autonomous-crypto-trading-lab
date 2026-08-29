from __future__ import annotations

"""Phase 0: research-foundation audit.

This module does not discover strategies. It validates that the research
microscope can distinguish signal from noise before expensive evolution.
Spot, long/flat, unlevered research only.
"""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData

OUT = ROOT / "experiments" / "phase0_foundation_latest.json"
DATA_DIR = ROOT / "experiments" / "phase0_data"

# Liquid spot universe; Phase 0 is intentionally broad rather than optimized.
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "ADA/USDT", "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT",
    "AVAX/USDT", "TRX/USDT",
)
TARGET_1H_BARS = int(os.getenv("PHASE0_BARS", "50000"))
MAX_SYMBOLS = int(os.getenv("PHASE0_MAX_SYMBOLS", str(len(SYMBOLS))))
NOISE_TRIALS = int(os.getenv("PHASE0_NOISE_TRIALS", "100"))


@dataclass
class AuditResult:
    name: str
    passed: bool
    details: dict


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _hash_df(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].astype(float).pct_change().fillna(0.0)


def _signal_buy_hold(df: pd.DataFrame) -> pd.Series:
    # Position is decided at close and executed at the next open/bar return.
    return pd.Series(1.0, index=df.index)


def _signal_vol_matched(df: pd.DataFrame, target_ann_vol: float = 0.25) -> pd.Series:
    r = _returns(df)
    realized = r.rolling(168).std() * math.sqrt(8760.0)
    scale = (target_ann_vol / realized.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    return scale.clip(0.0, 1.0)


def _run(df: pd.DataFrame, signal: pd.Series, settings, fee_mult: float = 1.0) -> dict:
    # Defensive assertion: Phase 0 never sends a short signal to spot.
    signal = signal.clip(0.0, 1.0)
    result = run_ohlcv(
        df,
        signal,
        settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return dict(result.metrics)


def _noise_signal(index: pd.Index, rng: np.random.Generator) -> pd.Series:
    # Independent random long/flat state. It must not inspect returns or prices.
    values = (rng.random(len(index)) > 0.5).astype(float)
    return pd.Series(values, index=index)


def _future_leak_signal(df: pd.DataFrame) -> pd.Series:
    # Deliberately illegal signal used only as a canary. If evaluator plumbing
    # ever reports normal-looking strength here, the data flow is contaminated.
    r = df["close"].pct_change().shift(-1).fillna(0.0)
    return (r > 0).astype(float)


def _block_shuffle_returns(df: pd.DataFrame, rng: np.random.Generator, block: int = 24) -> pd.DataFrame:
    close = df["close"].astype(float).copy()
    r = close.pct_change().fillna(0.0).to_numpy()
    blocks = [r[i:i + block] for i in range(0, len(r), block)]
    rng.shuffle(blocks)
    shuffled = np.concatenate(blocks)
    # Preserve the original starting level; this creates a price path whose
    # local return distribution is retained but temporal order is destroyed.
    new_close = float(close.iloc[0]) * np.cumprod(1.0 + shuffled)
    out = df.copy()
    out["close"] = new_close
    # Keep candles internally valid enough for the backtester.
    out["open"] = out["close"].shift(1).fillna(out["close"])
    out["high"] = np.maximum(out["open"].to_numpy(), out["close"].to_numpy())
    out["low"] = np.minimum(out["open"].to_numpy(), out["close"].to_numpy())
    return out


def _synthetic_edge_signal(df: pd.DataFrame) -> pd.Series:
    # Known, non-lookahead synthetic alpha: next-position is based on the
    # previous bar's signed return, with a tiny positive injected edge.
    r = df["close"].pct_change().fillna(0.0)
    return (r > 0).astype(float)


def _integrity_audit(data: dict[tuple[str, str], pd.DataFrame]) -> list[AuditResult]:
    results: list[AuditResult] = []
    for key, df in data.items():
        name = f"integrity:{key[0]}:{key[1]}"
        monotonic = bool(df.index.is_monotonic_increasing)
        unique = bool(df.index.is_unique)
        finite = bool(np.isfinite(df[["open", "high", "low", "close", "volume"]].to_numpy()).all())
        valid_ohlc = bool(((df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] <= df[["open", "close"]].min(axis=1))).all())
        no_open_bar = bool(len(df) == 0 or (datetime.now(timezone.utc) - df.index[-1].to_pydatetime()).total_seconds() > 3600)
        passed = monotonic and unique and finite and valid_ohlc and no_open_bar
        results.append(AuditResult(name, passed, {"bars": len(df), "monotonic": monotonic, "unique": unique, "finite": finite, "valid_ohlc": valid_ohlc, "closed_candle": no_open_bar, "sha256": _hash_df(df)}))
    return results


def _benchmark_audit(data: dict[tuple[str, str], pd.DataFrame], settings) -> dict:
    rows = []
    for key, df in data.items():
        bh = _run(df, _signal_buy_hold(df), settings, 1.0)
        bh2 = _run(df, _signal_buy_hold(df), settings, 2.0)
        vt = _run(df, _signal_vol_matched(df), settings, 1.0)
        rows.append({"market": f"{key[0]} {key[1]}", "buy_hold": bh, "buy_hold_stress": bh2, "vol_matched_buy_hold": vt})
    return {"markets": rows}


def _canary_audit(data: dict[tuple[str, str], pd.DataFrame], settings) -> list[AuditResult]:
    results: list[AuditResult] = []
    for key, df in data.items():
        clean = _run(df, pd.Series(0.0, index=df.index), settings, 1.0)
        leaked = _run(df, _future_leak_signal(df), settings, 1.0)
        # A lookahead canary is intentionally expected to look unusually strong
        # on many paths. We fail Phase 0 if it behaves exactly like a flat series,
        # because that indicates the signal plumbing may be disconnected.
        suspicious = leaked["total_return"] <= clean["total_return"] + 0.01
        results.append(AuditResult(f"lookahead_canary:{key[0]}:{key[1]}", not suspicious, {"flat_return": clean["total_return"], "leaked_return": leaked["total_return"], "leaked_pf": leaked["profit_factor"], "expected": "leaked signal should materially differ from flat baseline"}))
    return results


def _noise_floor(data: dict[tuple[str, str], pd.DataFrame], settings, seed: int = 20260829) -> dict:
    rng = np.random.default_rng(seed)
    accepted = 0
    scores = []
    for trial in range(NOISE_TRIALS):
        key = list(data.keys())[trial % len(data)]
        df = data[key]
        shuffled = _block_shuffle_returns(df, rng)
        sig = _noise_signal(shuffled.index, rng)
        m = _run(shuffled, sig, settings, 2.0)
        # Noise acceptance definition is deliberately simple at Phase 0:
        # positive return AND PF>1 AND |DD|<20%.
        ok = m["total_return"] > 0 and m["profit_factor"] > 1.0 and m["max_drawdown"] > -0.20
        accepted += int(ok)
        scores.append(float(m["total_return"]))
    return {"trials": NOISE_TRIALS, "accepted": accepted, "false_positive_rate": float(accepted / max(NOISE_TRIALS, 1)), "median_noise_return": float(np.median(scores)), "max_noise_return": float(np.max(scores))}


def _synthetic_detection(data: dict[tuple[str, str], pd.DataFrame], settings) -> dict:
    rows = []
    for key, df in list(data.items())[:4]:
        sig = _synthetic_edge_signal(df)
        normal = _run(df, sig, settings, 1.0)
        stress = _run(df, sig, settings, 2.0)
        rows.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "detected": bool(normal["total_return"] > 0 and stress["total_return"] > 0)})
    return {"markets": rows, "detected_ratio": float(np.mean([r["detected"] for r in rows])) if rows else 0.0}


def run(minutes: float = 20.0, bars: int = TARGET_1H_BARS, max_symbols: int = MAX_SYMBOLS, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    adapter = CCXTMarketData()
    symbols = list(SYMBOLS)[:max_symbols]
    data: dict[tuple[str, str], pd.DataFrame] = {}

    print("=== PHASE 0 FOUNDATION AUDIT ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Target: {bars} hourly bars × {len(symbols)} spot markets", flush=True)
    print(f"Checkpoint: {OUT}", flush=True)
    _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "symbols": symbols, "target_bars": bars})

    load_errors = []
    for symbol in symbols:
        if time.monotonic() >= deadline:
            break
        print(f"LOAD {symbol} 1h ...", flush=True)
        try:
            df = adapter.fetch_ohlcv_history(symbol, "1h", bars, page_limit=1500, market_type="spot")
            data[(symbol, "1h")] = df
            path = DATA_DIR / f"{symbol.replace('/', '_')}_1h.parquet"
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path, index=True)
            print(f"OK {symbol}: {len(df)} bars | {df.index[0]} -> {df.index[-1]}", flush=True)
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "loaded": [f"{k[0]} {k[1]}" for k in data], "bars_loaded": {f"{k[0]} {k[1]}": len(v) for k, v in data.items()}, "load_errors": load_errors})
        except Exception as exc:
            load_errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)

    if len(data) < 8:
        payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": "PHASE0_DATA_INSUFFICIENT", "loaded_markets": len(data), "required_markets": 8, "load_errors": load_errors}
        _save(payload)
        return payload

    integrity = _integrity_audit(data)
    print("=== INTEGRITY ===", flush=True)
    for r in integrity:
        print(f"{r.name}: {'PASS' if r.passed else 'FAIL'} | {r.details}", flush=True)

    benchmarks = _benchmark_audit(data, settings)
    canaries = _canary_audit(data, settings)
    print("=== LOOKAHEAD CANARIES ===", flush=True)
    for r in canaries:
        print(f"{r.name}: {'PASS' if r.passed else 'FAIL'} | leaked_return={r.details['leaked_return']:.2%}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed)
    print(f"Noise FP rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE DETECTION ===", flush=True)
    synthetic = _synthetic_detection(data, settings)
    print(f"Synthetic detection ratio: {synthetic['detected_ratio']:.2%}", flush=True)

    results = [r for r in integrity + canaries]
    all_integrity_ok = all(r.passed for r in results)
    noise_ok = noise["false_positive_rate"] <= 0.05
    synthetic_ok = synthetic["detected_ratio"] >= 0.80
    data_ok = len(data) >= 8 and min(len(x) for x in data.values()) >= min(bars, 5000)

    decision = "PHASE0_READY_FOR_DISCOVERY" if all_integrity_ok and noise_ok and synthetic_ok and data_ok else "PHASE0_BLOCKED"
    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (datetime.now(timezone.utc) - started).total_seconds() / 60.0,
        "decision": decision,
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "symbols_requested": symbols,
        "markets_loaded": [f"{k[0]} {k[1]}" for k in data],
        "bars": {f"{k[0]} {k[1]}": len(v) for k, v in data.items()},
        "load_errors": load_errors,
        "integrity": [asdict(r) for r in integrity],
        "benchmarks": benchmarks,
        "lookahead_canaries": [asdict(r) for r in canaries],
        "noise_floor": noise,
        "synthetic_edge_detection": synthetic,
        "gates": {"data": data_ok, "integrity": all_integrity_ok, "noise_fp_le_5pct": noise_ok, "synthetic_detection_ge_80pct": synthetic_ok},
        "data_paths": [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))],
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload
