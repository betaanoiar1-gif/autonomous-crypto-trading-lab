from __future__ import annotations

"""Phase 0 v2: research-foundation validation before discovery.

Goals:
- backfill a broad spot universe with multi-year 1h data;
- verify OHLCV integrity and closed candles;
- verify spot execution is long/flat only;
- verify a deliberately leaked signal is detectable by a canary;
- estimate a false-positive rate on shuffled noise;
- verify recovery of a known synthetic edge with positive expected value;
- save data in parquet and a durable JSON report.

This module does NOT discover strategies and contains no LLM generation.
"""

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData

OUT = ROOT / "experiments" / "phase0_foundation_v2_latest.json"
DATA_DIR = ROOT / "experiments" / "phase0_data_v2"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT",
    "ADA/USDT", "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT",
    "AVAX/USDT", "TRX/USDT",
)
TARGET_BARS = int(os.getenv("PHASE0_BARS", "50000"))
MAX_SYMBOLS = int(os.getenv("PHASE0_MAX_SYMBOLS", "12"))
NOISE_TRIALS = int(os.getenv("PHASE0_NOISE_TRIALS", "100"))


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
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(df: pd.DataFrame, signal: pd.Series, settings, cost_mult: float = 1.0) -> dict:
    # Phase 0 is spot-only: defensive clamp protects against accidental short signals.
    signal = pd.Series(signal, index=df.index).astype(float).clip(0.0, 1.0)
    result = run_ohlcv(
        df,
        signal,
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return dict(result.metrics)


def _flat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def _buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def _vol_matched(df: pd.DataFrame, target_vol: float = 0.25) -> pd.Series:
    r = df["close"].pct_change().fillna(0.0)
    vol = r.rolling(168).std() * math.sqrt(8760.0)
    return (target_vol / vol.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    flat = _metrics(df, _flat(df), settings)
    future = df["close"].pct_change().shift(-1).fillna(0.0)
    leaked = _metrics(df, (future > 0).astype(float), settings)
    # The canary should produce a materially different / stronger result than flat.
    distinguishable = (
        leaked["total_return"] > flat["total_return"] + 0.05
        or leaked["profit_factor"] > max(2.0, flat["profit_factor"] + 1.0)
    )
    return Audit(
        "lookahead_canary",
        distinguishable,
        {
            "flat_return": flat["total_return"],
            "leaked_return": leaked["total_return"],
            "leaked_pf": leaked["profit_factor"],
            "leaked_trades": leaked["trade_count"],
            "expected": "future-dependent signal must be distinguishable; if not, signal plumbing is suspect",
        },
    )


def _noise_floor(data: dict[str, pd.DataFrame], settings, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns = []
    for i in range(NOISE_TRIALS):
        df = data[keys[i % len(keys)]]
        sig = pd.Series((rng.random(len(df)) > 0.65).astype(float), index=df.index)
        m = _metrics(df, sig, settings, 2.0)
        # Conservative acceptance proxy. Phase 0 calibrates the evaluator, not a final alpha gate.
        ok = bool(m["total_return"] > 0 and m["profit_factor"] > 1.0 and m["max_drawdown"] > -0.20)
        accepted += int(ok)
        returns.append(float(m["total_return"]))
    return {
        "trials": NOISE_TRIALS,
        "accepted": accepted,
        "false_positive_rate": accepted / max(NOISE_TRIALS, 1),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
    }


def _synthetic_edge_frame(seed: int, n: int = 5000, base_sigma: float = 0.003, alpha: float = 0.0012) -> tuple[pd.DataFrame, pd.Series]:
    """Create a known causal edge: r[t] contains alpha * sign(r[t-1])."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, base_sigma, size=n)
    for i in range(1, n):
        r[i] = noise[i] + alpha * np.sign(r[i - 1])
    close = 100.0 * np.cumprod(1.0 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    previous = np.r_[0.0, r[:-1]]
    frame = pd.DataFrame(index=idx)
    frame["close"] = close
    frame["open"] = np.r_[close[0], close[:-1]]
    frame["high"] = np.maximum(frame["open"].to_numpy(), frame["close"].to_numpy())
    frame["low"] = np.minimum(frame["open"].to_numpy(), frame["close"].to_numpy())
    frame["volume"] = 1_000_000.0
    signal = pd.Series((previous > 0).astype(float), index=idx)
    return frame, signal


def _synthetic_detection(settings, seed: int) -> dict:
    frame, signal = _synthetic_edge_frame(seed)
    normal = _metrics(frame, signal, settings, 1.0)
    stress = _metrics(frame, signal, settings, 2.0)
    detected = bool(normal["total_return"] > 0 and stress["total_return"] > 0 and normal["profit_factor"] > 1.0)
    return {"normal": normal, "stress": stress, "alpha_design": 0.0012, "noise_sigma": 0.003, "detected": detected}


def _data_integrity(data: dict[str, pd.DataFrame]) -> list[Audit]:
    audits = []
    for key, df in data.items():
        values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        monotonic = bool(df.index.is_monotonic_increasing)
        unique = bool(df.index.is_unique)
        finite = bool(np.isfinite(values).all())
        valid_ohlc = bool(((df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] <= df[["open", "close"]].min(axis=1))).all())
        recent = df.index[-1].to_pydatetime()
        closed = bool((datetime.now(timezone.utc) - recent).total_seconds() >= 3600)
        passed = monotonic and unique and finite and valid_ohlc and closed
        audits.append(Audit(
            f"integrity:{key}",
            passed,
            {"bars": len(df), "monotonic": monotonic, "unique": unique, "finite": finite, "valid_ohlc": valid_ohlc, "closed_candle": closed, "sha256": _hash_df(df)},
        ))
    return audits


def run(minutes: float = 60.0, bars: int = TARGET_BARS, max_symbols: int = MAX_SYMBOLS, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    adapter = CCXTMarketData()
    symbols = list(SYMBOLS)[:max_symbols]
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []

    print("=== PHASE 0 FOUNDATION V2 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Target: {bars} 1h bars × {len(symbols)} spot markets", flush=True)
    print(f"Checkpoint: {OUT}", flush=True)
    _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "target_bars": bars, "symbols": symbols})

    for symbol in symbols:
        if time.monotonic() >= deadline:
            errors.append({"symbol": symbol, "error": "deadline_before_load"})
            break
        print(f"LOAD {symbol} 1h ...", flush=True)
        try:
            df = adapter.fetch_ohlcv_history(symbol, "1h", bars, page_limit=1500, market_type="spot")
            data[symbol] = df
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{symbol.replace('/', '_')}_1h.parquet"
            df.to_parquet(path, index=True)
            print(f"OK {symbol}: {len(df)} bars | {df.index[0]} -> {df.index[-1]}", flush=True)
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "loaded": list(data), "bars": {k: len(v) for k, v in data.items()}, "errors": errors})
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)

    if len(data) < 8 or min(map(len, data.values())) < min(bars, 5000):
        payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": "PHASE0_DATA_INSUFFICIENT", "loaded_markets": len(data), "required_markets": 8, "required_bars": min(bars, 5000), "errors": errors}
        _save(payload)
        return payload

    audits = _data_integrity(data)
    print("=== DATA INTEGRITY ===", flush=True)
    for a in audits:
        print(f"{a.name}: {'PASS' if a.passed else 'FAIL'}", flush=True)

    spot_position_probe = pd.Series(np.where(np.arange(100) % 3 == 0, -1.0, 1.0), index=next(iter(data.values())).index[:100])
    probe_df = next(iter(data.values())).iloc[:100]
    probe = _metrics(probe_df, spot_position_probe, settings, 1.0)
    spot_safe = bool(probe["short_exposure"] == 0.0 and probe["long_exposure"] > 0.0)
    spot_audit = Audit("spot_long_flat_enforcement", spot_safe, probe)
    print(f"SPOT POSITION PROBE: {'PASS' if spot_safe else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    benchmark = {}
    for symbol, df in list(data.items())[:4]:
        benchmark[symbol] = {
            "buy_hold": _metrics(df, _buy_hold(df), settings, 1.0),
            "buy_hold_2x_cost": _metrics(df, _buy_hold(df), settings, 2.0),
            "vol_matched": _metrics(df, _vol_matched(df), settings, 1.0),
        }

    canary = _lookahead_canary(next(iter(data.values())), settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | leaked_return={canary.details['leaked_return']:.2%}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed)
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%}", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_detection(settings, seed + 1)
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic['detected'] else 'FAIL'}", flush=True)

    all_integrity = all(a.passed for a in audits)
    data_ok = len(data) >= 8 and min(len(v) for v in data.values()) >= min(bars, 5000)
    noise_ok = noise['false_positive_rate'] <= 0.05
    synthetic_ok = synthetic['detected']
    decision = "PHASE0_READY_FOR_DISCOVERY" if all((all_integrity, data_ok, spot_safe, canary.passed, noise_ok, synthetic_ok)) else "PHASE0_BLOCKED"

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
        "markets_loaded": list(data),
        "bars": {k: len(v) for k, v in data.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "spot_position_probe": asdict(spot_audit),
        "benchmarks": benchmark,
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": {"data": data_ok, "integrity": all_integrity, "spot_long_flat": spot_safe, "lookahead_canary": canary.passed, "noise_fp_le_5pct": noise_ok, "synthetic_edge_detected": synthetic_ok},
        "data_dir": str(DATA_DIR),
        "data_files": [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))],
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload
