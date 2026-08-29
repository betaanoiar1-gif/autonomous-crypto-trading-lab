from __future__ import annotations

"""AI-free regime-adaptive strategy discovery.

The candidate is a deterministic strategy that classifies each bar into
trend/range/high-volatility regimes and selects a bounded behavior for that
regime. It evolves only numeric parameters; no LLM, arbitrary code, futures,
or live trading are used.
"""

import gc
import itertools
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics

OUT = ROOT / "experiments" / "regime_evolution_latest.json"

BASE_GRAMMAR = {
    "trend_fast": (8, 12, 18, 24, 36),
    "trend_slow": (60, 90, 120, 180, 240),
    "regime_window": (24, 36, 48, 72),
    "vol_window": (12, 24, 36, 48),
    "momentum_window": (12, 24, 48, 72, 96),
    "range_window": (20, 40, 60, 90),
    "trend_threshold": (0.0005, 0.001, 0.002, 0.003),
    "high_vol_quantile": (0.75, 0.85, 0.90, 0.95),
    "range_z_entry": (1.0, 1.25, 1.5, 1.75, 2.0),
    "range_z_exit": (0.1, 0.25, 0.5, 0.75),
    "momentum_threshold": (0.005, 0.01, 0.015, 0.02),
    "max_vol": (0.025, 0.035, 0.05, 0.07),
    "trend_min_run": (1, 2, 3, 4),
    "cooldown": (0, 1, 2, 4),
}


def _save(payload: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _candidate_pool(seed: int, limit: int) -> list[dict]:
    rng = random.Random(seed)
    out, seen = [], set()
    keys = tuple(BASE_GRAMMAR)
    while len(out) < limit:
        p = {k: rng.choice(BASE_GRAMMAR[k]) for k in keys}
        if p["trend_slow"] <= p["trend_fast"]: continue
        if p["range_z_exit"] >= p["range_z_entry"]: continue
        if p["high_vol_quantile"] <= 0.5: continue
        sig = tuple(sorted(p.items()))
        if sig in seen: continue
        seen.add(sig); out.append(p)
    return out


def _signal(df: pd.DataFrame, p: dict) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    fast = close.ewm(span=int(p["trend_fast"]), adjust=False).mean()
    slow = close.ewm(span=int(p["trend_slow"]), adjust=False).mean()
    trend_strength = (fast - slow) / slow.replace(0, np.nan)

    ret = close.pct_change()
    vol = ret.rolling(int(p["vol_window"])).std()
    vol_ref = vol.rolling(int(p["regime_window"])).quantile(float(p["high_vol_quantile"]))

    slope = slow.pct_change(int(p["regime_window"])) / max(1, int(p["regime_window"]))
    trend = slope.abs() >= float(p["trend_threshold"])
    high_vol = vol > vol_ref

    # Trend behavior: directional momentum only when trend is sufficiently strong.
    mom = close.pct_change(int(p["momentum_window"]))
    trend_sig = pd.Series(0.0, index=df.index)
    trend_sig[(trend_strength.shift(1) > 0) & (mom.shift(1) > float(p["momentum_threshold"]))] = 1.0
    trend_sig[(trend_strength.shift(1) < 0) & (mom.shift(1) < -float(p["momentum_threshold"]))] = -1.0

    # Range behavior: fade standardized extremes around a rolling mean.
    w = close.rolling(int(p["range_window"]))
    mean = w.mean()
    std = w.std(ddof=0).replace(0, np.nan)
    z = (close - mean) / std
    range_sig = pd.Series(0.0, index=df.index)
    range_sig[z.shift(1) < -float(p["range_z_entry"])] = 1.0
    range_sig[z.shift(1) > float(p["range_z_entry"])] = -1.0
    range_sig[z.shift(1).abs() < float(p["range_z_exit"])] = 0.0

    # High-volatility regime: stay flat rather than chase noise.
    sig = pd.Series(0.0, index=df.index)
    sig[trend & ~high_vol] = trend_sig[trend & ~high_vol]
    sig[~trend & ~high_vol] = range_sig[~trend & ~high_vol]

    # Optional cooldown after a signal change to reduce churn.
    cooldown = int(p["cooldown"])
    if cooldown > 0:
        changes = sig.ne(sig.shift(1)).fillna(False)
        hold = changes.astype(int).rolling(cooldown + 1).max().shift(1).fillna(0).astype(bool)
        sig[hold] = sig.shift(1).fillna(0)[hold]

    # Enforce maximum allowed volatility as a hard safety filter.
    sig[vol.shift(1) > float(p["max_vol"])] = 0.0
    return sig.fillna(0.0)


def _eval(df, p, settings, fee_mult=1.0):
    sig = _signal(df, p)
    r = run_ohlcv(
        df,
        sig,
        settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return _metrics(r, r.returns)


def _score(m):
    ret = float(m.get("total_return", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    sh = float(m.get("sharpe", 0.0))
    trades = int(m.get("trade_count", 0))
    if trades < 8: return -30.0 + trades
    return 100 * ret + 20 * (min(2.5, pf) - 1) + 4 * sh - 30 * dd + min(9, math.log1p(trades))


def _gate(m, stress):
    return bool(
        float(m["total_return"]) > 0
        and float(m["profit_factor"]) > 1
        and float(m["max_drawdown"]) >= -0.50
        and int(m["trade_count"]) >= 8
        and float(stress["total_return"]) > 0
        and float(stress["profit_factor"]) > 1
    )


def _evaluate_primary(df, p, settings):
    n = len(df)
    cut = int(n * 0.70)
    holdout = df.iloc[cut:]
    normal = _eval(holdout, p, settings, 1.0)
    stress = _eval(holdout, p, settings, 2.0)

    # Three chronological holdout blocks for consistency, without retuning.
    folds = []
    block = max(30, len(holdout) // 3)
    for i in range(3):
        a = i * block
        b = len(holdout) if i == 2 else (i + 1) * block
        if b <= a: continue
        fm = _eval(holdout.iloc[a:b], p, settings, 1.0)
        folds.append(fm)
    positive = sum(float(x["total_return"]) > 0 for x in folds)
    median_return = float(np.median([float(x["total_return"]) for x in folds])) if folds else 0.0
    median_pf = float(np.median([float(x["profit_factor"]) for x in folds])) if folds else 0.0
    min_trades = min([int(x["trade_count"]) for x in folds], default=0)
    wf = bool(len(folds) == 3 and positive >= 2 and median_return > 0 and median_pf > 1 and min_trades >= 4)

    return {
        "holdout": normal,
        "stress": stress,
        "folds": folds,
        "wf": wf,
        "wf_positive": positive,
        "wf_median_return": median_return,
        "wf_median_pf": median_pf,
        "wf_min_trades": min_trades,
        "primary_pass": _gate(normal, stress) and wf,
        "score": _score(normal) + 70 * median_return,
    }


def run(hours: float = 3.0, pool_size: int = 360, population: int = 12, generations: int = 12):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + hours * 3600

    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "evaluated": 0})
    print("=== REGIME-ADAPTIVE EVOLUTION ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Writes checkpoint before data loading.", flush=True)

    primary = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 800), ("ETH/USDT", "4h", 800), ("BTC/USDT", "4h", 800)):
        print(f"LOAD {symbol} {tf}", flush=True)
        primary[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(primary[(symbol, tf)])}", flush=True)
        gc.collect()

    candidates = _candidate_pool(20260829, pool_size)
    print(f"GENERATED {len(candidates)} REGIME-ADAPTIVE CANDIDATES", flush=True)

    records = []
    seen = set()
    for i, p in enumerate(candidates, 1):
        if time.monotonic() >= deadline: break
        try:
            results = []
            for key in primary:
                x = _evaluate_primary(primary[key], p, settings)
                results.append(x)
            score = float(np.mean([x["score"] for x in results]))
            positive_markets = sum(x["primary_pass"] for x in results)
            rec = {"parameters": p, "score": score, "primary_passes": positive_markets, "markets": {f"{k[0]} {k[1]}": x for k, x in zip(primary, results)}}
            records.append(rec)
        except Exception as exc:
            print(f"ERROR candidate {i}: {type(exc).__name__}: {exc}", flush=True)
        if i == 1 or i % 10 == 0:
            best = max(records, key=lambda x: x["score"]) if records else None
            print(f"PROGRESS {i}/{len(candidates)} | best={best['score']:.2f} | markets_passed={best['primary_passes']}/3" if best else f"PROGRESS {i}", flush=True)
        gc.collect()

    records.sort(key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
    finalists = []
    family = []
    for r in records:
        p = r["parameters"]
        signature = (p["trend_fast"], p["trend_slow"], p["range_window"], p["regime_window"], p["high_vol_quantile"], p["trend_threshold"], p["range_z_entry"], p["range_z_exit"])
        if signature in family: continue
        family.append(signature); finalists.append(r)
        if len(finalists) >= population: break

    print("\n=== FRESH REGIME CONFIRMATION ===", flush=True)
    fresh = []
    for i, r in enumerate(finalists, 1):
        entry = dict(r)
        checks = []
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            if key not in primary:
                print(f"LAZY LOAD {key[0]} {key[1]}", flush=True)
                primary[key] = adapter.fetch_ohlcv_history(key[0], key[1], 800, page_limit=300, market_type="spot")
                print(f"  bars={len(primary[key])}", flush=True)
            normal = _eval(primary[key], r["parameters"], settings, 1.0)
            stress = _eval(primary[key], r["parameters"], settings, 2.0)
            ok = _gate(normal, stress)
            checks.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"  {key[0]} {key[1]}: return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
            gc.collect()
        entry["fresh"] = checks
        entry["validated"] = bool(r["primary_passes"] == 3 and all(x["pass"] for x in checks))
        fresh.append(entry)
        if entry["validated"]:
            print("VALIDATED CANDIDATE FOUND — stopping.", flush=True)
            break

    validated = [x for x in fresh if x["validated"]]
    decision = "VALIDATED_REGIME_ADAPTIVE_STRATEGY" if validated else "NO_VALIDATED_REGIME_ADAPTIVE_STRATEGY"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generated": len(candidates), "screened": len(records), "finalists": len(finalists), "validated_count": len(validated), "results": fresh}
    _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", OUT, flush=True)
    return payload


if __name__ == "__main__":
    run()
