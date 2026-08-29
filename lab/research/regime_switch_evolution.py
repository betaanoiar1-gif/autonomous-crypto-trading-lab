from __future__ import annotations

"""AI-free regime-switch strategy evolution.

A candidate is a policy over market states, not a single indicator:
- TREND regime -> one bounded trend family
- RANGE regime -> one bounded mean-reversion family
- HIGH VOL regime -> either defensive flat or breakout family

Regime boundaries and family parameters are evolved deterministically.
No LLM, futures, live trading, or arbitrary code execution.
"""

import gc
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..metrics import sharpe
from .executor import compile_signal


@dataclass(frozen=True)
class Candidate:
    params: dict
    title: str


TREND_FAMILIES = ("momentum", "breakout", "moving_average_cross")
RANGE_FAMILIES = ("mean_reversion", "rsi_reversion", "channel_reversion")
HIGH_FAMILIES = ("flat", "breakout", "momentum")


def _bounded(name, value, lo, hi, integer=False):
    try:
        x = int(value) if integer else float(value)
    except (TypeError, ValueError):
        x = lo
    return max(lo, min(hi, x))


def regime_labels(df: pd.DataFrame, vol_window: int, trend_window: int, vol_low: float, vol_high: float, trend_threshold: float):
    close = df["close"].astype(float)
    logret = np.log(close / close.shift(1))
    vol = logret.rolling(vol_window).std()
    ema = close.ewm(span=trend_window, adjust=False).mean()
    trend_strength = (ema.diff(trend_window).abs() / close).replace([np.inf, -np.inf], np.nan)

    # Quantile thresholds are calculated only from the history available at each point.
    low_q = vol.rolling(max(vol_window * 8, 40), min_periods=max(vol_window * 4, 20)).quantile(vol_low)
    high_q = vol.rolling(max(vol_window * 8, 40), min_periods=max(vol_window * 4, 20)).quantile(vol_high)

    labels = pd.Series("warmup", index=df.index)
    trend = trend_strength >= trend_threshold
    high = vol >= high_q
    low = vol <= low_q
    labels[high] = "high_vol"
    labels[(~high) & trend] = "trend"
    labels[(~high) & (~trend) & low] = "range"
    labels[(~high) & (~trend) & (~low)] = "mixed"
    return labels


def policy_signal(df: pd.DataFrame, p: dict) -> pd.Series:
    labels = regime_labels(
        df,
        _bounded("vol_window", p.get("vol_window", 24), 8, 96, True),
        _bounded("trend_window", p.get("trend_window", 72), 12, 240, True),
        _bounded("vol_low_q", p.get("vol_low_q", 0.20), 0.05, 0.45),
        _bounded("vol_high_q", p.get("vol_high_q", 0.80), 0.55, 0.95),
        _bounded("trend_threshold", p.get("trend_threshold", 0.02), 0.002, 0.20),
    )

    trend_family = p["trend_family"]
    range_family = p["range_family"]
    high_family = p["high_family"]

    trend_params = dict(p["trend_params"])
    range_params = dict(p["range_params"])
    high_params = dict(p["high_params"])

    trend_sig = compile_signal(df, trend_family, trend_params, ["both"])
    range_sig = compile_signal(df, range_family, range_params, ["both"])
    if high_family == "flat":
        high_sig = pd.Series(0.0, index=df.index)
    else:
        high_sig = compile_signal(df, high_family, high_params, ["both"])

    out = pd.Series(0.0, index=df.index)
    out[labels == "trend"] = trend_sig[labels == "trend"]
    out[labels == "range"] = range_sig[labels == "range"]
    out[labels == "high_vol"] = high_sig[labels == "high_vol"]
    # Mixed and warmup stay flat, forcing the candidate to prove itself in clear regimes.
    return out.fillna(0.0)


def evaluate(df, candidate: Candidate, settings, fee_mult=1.0):
    fee = settings.execution.commission_bps * fee_mult
    slip = settings.execution.slippage_bps * fee_mult
    result = run_ohlcv(
        df,
        policy_signal(df, candidate.params),
        settings.capital.initial_usd,
        fee,
        slip,
        market_type="spot",
        leverage=1.0,
    )
    active = result.trades["position"].diff().abs().fillna(result.trades["position"].abs()) > 0
    m = dict(result.metrics)
    m["sharpe"] = float(sharpe(result.returns))
    m["trade_count"] = int(active.sum())
    return m


def score(m):
    ret = float(m.get("total_return", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    trades = int(m.get("trade_count", 0))
    sh = float(m.get("sharpe", 0.0))
    if trades < 6:
        return -15.0
    return 100 * ret + 18 * (min(2.5, pf) - 1) + 4 * sh - 35 * dd + min(8, math.log1p(trades))


def mutate_param(rng, family, params):
    p = dict(params)
    if family in ("momentum", "breakout", "channel_reversion"):
        key = "lookback" if family != "channel_reversion" else "channel_length"
        p[key] = max(5, min(240, int(round(p[key] * rng.choice((0.8, 0.9, 1.1, 1.2))))))
    elif family == "moving_average_cross":
        p["fast"] = max(2, min(90, int(round(p["fast"] * rng.choice((0.8, 0.9, 1.1, 1.2))))))
        p["slow"] = max(p["fast"] + 3, min(260, int(round(p["slow"] * rng.choice((0.85, 1.0, 1.15))))))
    elif family == "mean_reversion":
        p["lookback"] = max(10, min(200, int(round(p["lookback"] * rng.choice((0.8, 0.9, 1.1, 1.2))))))
        p["z_entry"] = round(max(0.8, min(3.5, p["z_entry"] * rng.choice((0.85, 1.0, 1.15)))), 3)
        p["z_exit"] = round(max(0.05, min(1.5, min(p["z_exit"] * rng.choice((0.8, 1.0, 1.2)), p["z_entry"] * 0.75))), 3)
    elif family == "rsi_reversion":
        p["rsi_length"] = max(3, min(50, int(round(p["rsi_length"] * rng.choice((0.8, 0.9, 1.1, 1.2))))))
        p["rsi_low"] = max(5, min(45, p["rsi_low"] + rng.choice((-5, 0, 5))))
        p["rsi_high"] = max(55, min(95, p["rsi_high"] + rng.choice((-5, 0, 5))))
    elif family == "atr_breakout":
        p["atr_length"] = max(2, min(50, int(round(p["atr_length"] * rng.choice((0.8, 0.9, 1.1, 1.2))))))
        p["atr_mult"] = round(max(0.25, min(5.0, p["atr_mult"] * rng.choice((0.8, 1.0, 1.2)))), 3)
    return p


def seed_candidate(rng):
    tf = rng.choice(TREND_FAMILIES)
    rf = rng.choice(RANGE_FAMILIES)
    hf = rng.choice(HIGH_FAMILIES)

    tmap = {
        "momentum": {"lookback": rng.choice((20, 40, 60, 100, 140))},
        "breakout": {"lookback": rng.choice((20, 40, 60, 100, 140))},
        "moving_average_cross": {"fast": rng.choice((5, 10, 20, 30)), "slow": rng.choice((60, 90, 120, 180))},
    }
    rmap = {
        "mean_reversion": {"lookback": rng.choice((20, 40, 60, 100)), "z_entry": rng.choice((1.25, 1.75, 2.25, 2.75)), "z_exit": rng.choice((0.25, 0.5, 0.75))},
        "rsi_reversion": {"rsi_length": rng.choice((7, 14, 21, 28)), "rsi_low": rng.choice((20, 25, 30, 35)), "rsi_high": rng.choice((65, 70, 75, 80))},
        "channel_reversion": {"channel_length": rng.choice((20, 40, 60, 90, 120))},
    }
    if hf == "flat":
        hmap = {}
    elif hf == "breakout":
        hmap = {"lookback": rng.choice((10, 20, 40, 60, 100))}
    else:
        hmap = {"lookback": rng.choice((20, 40, 60, 100, 140))}

    p = {
        "vol_window": rng.choice((12, 24, 36, 48)),
        "trend_window": rng.choice((24, 48, 72, 120, 180)),
        "vol_low_q": rng.choice((0.15, 0.20, 0.25, 0.30)),
        "vol_high_q": rng.choice((0.70, 0.75, 0.80, 0.85, 0.90)),
        "trend_threshold": rng.choice((0.008, 0.012, 0.02, 0.03, 0.05)),
        "trend_family": tf,
        "range_family": rf,
        "high_family": hf,
        "trend_params": tmap[tf],
        "range_params": rmap[rf],
        "high_params": hmap,
    }
    p["title"] = f"RegimeSwitch trend={tf} range={rf} high={hf}"
    return Candidate(p, p["title"])


def mutate_candidate(rng, c: Candidate):
    p = json.loads(json.dumps(c.params))
    field = rng.choice(("regime", "trend", "range", "high"))
    if field == "regime":
        p["vol_window"] = rng.choice((12, 18, 24, 36, 48, 72))
        p["trend_window"] = rng.choice((24, 36, 48, 72, 120, 180, 240))
        p["vol_low_q"] = rng.choice((0.10, 0.15, 0.20, 0.25, 0.30, 0.35))
        p["vol_high_q"] = rng.choice((0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95))
        p["trend_threshold"] = rng.choice((0.006, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08))
    elif field == "trend":
        fam = rng.choice(TREND_FAMILIES)
        p["trend_family"] = fam
        p["trend_params"] = seed_candidate(rng).params["trend_params"]
    elif field == "range":
        fam = rng.choice(RANGE_FAMILIES)
        p["range_family"] = fam
        p["range_params"] = seed_candidate(rng).params["range_params"]
    else:
        fam = rng.choice(HIGH_FAMILIES)
        p["high_family"] = fam
        p["high_params"] = seed_candidate(rng).params["high_params"]

    if rng.random() < 0.65:
        tf = p["trend_family"]; p["trend_params"] = mutate_param(rng, tf, p["trend_params"])
    if rng.random() < 0.65:
        rf = p["range_family"]; p["range_params"] = mutate_param(rng, rf, p["range_params"])
    if p["high_family"] != "flat" and rng.random() < 0.50:
        hf = p["high_family"]; p["high_params"] = mutate_param(rng, hf, p["high_params"])

    p["title"] = f"RegimeSwitch trend={p['trend_family']} range={p['range_family']} high={p['high_family']}"
    return Candidate(p, p["title"])


def _save(payload):
    out = ROOT / "experiments" / "regime_switch_evolution_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)
    return out


def run(minutes: int = 120, population: int = 8, generations: int = 12):
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60
    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0})

    print("=== REGIME-SWITCH EVOLUTION ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Checkpoint-first | population", population, "| generations", generations, flush=True)

    adapter = CCXTMarketData(exchange_id="binance")
    data = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 700), ("ETH/USDT", "4h", 700), ("BTC/USDT", "4h", 700)):
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "market": f"{symbol} {tf}", "bars": len(data[(symbol, tf)])})

    rng = random.Random(20260829)
    pop = []
    seen = set()
    while len(pop) < population:
        c = seed_candidate(rng)
        key = json.dumps(c.params, sort_keys=True)
        if key not in seen:
            seen.add(key); pop.append(c)

    evaluated = 0
    all_results = []
    primary_keys = (("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h"))

    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} ===", flush=True)
        genres = []
        for i, c in enumerate(pop, 1):
            if time.monotonic() >= deadline: break
            ms = []
            for key in primary_keys:
                m = evaluate(data[key], c, settings, 1.0)
                ms.append(m)
            vals = [float(m["total_return"]) for m in ms]
            score_value = float(np.mean([score(m) for m in ms]) - 20 * (max(vals) - min(vals)))
            rec = {"generation": gen, "candidate": c.params, "title": c.title, "score": score_value, "markets": ms}
            gen_results.append(rec); all_results.append(rec); evaluated += 1
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "RUNNING", "generation": gen, "evaluated": evaluated, "latest": rec})
            print(f"eval {i}/{len(pop)} | score={score_value:.2f} | returns=" + ",".join(f"{x:.2%}" for x in vals), flush=True)
            gc.collect()

        if not gen_results: break
        gen_results.sort(key=lambda x: x["score"], reverse=True)
        elites = gen_results[:max(2, population // 2)]
        print(f"GENERATION RESULT best={elites[0]['score']:.2f}", flush=True)

        next_pop = [Candidate(x["candidate"], x["title"]) for x in elites]
        while len(next_pop) < population and time.monotonic() < deadline:
            parent = rng.choice(next_pop)
            child = mutate_candidate(rng, parent)
            key = json.dumps(child.params, sort_keys=True)
            if key in seen:
                continue
            seen.add(key); next_pop.append(child)
        pop = next_pop

    all_results.sort(key=lambda x: x["score"], reverse=True)
    finalists = []
    family_shapes = set()
    for r in all_results:
        shape = (r["candidate"]["trend_family"], r["candidate"]["range_family"], r["candidate"]["high_family"])
        if shape in family_shapes: continue
        family_shapes.add(shape); finalists.append(r)
        if len(finalists) >= 8: break

    print("\n=== FRESH CONFIRMATION ===", flush=True)
    fresh = []
    for idx, r in enumerate(finalists, 1):
        confirmations = []
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            df = adapter.fetch_ohlcv_history(key[0], key[1], 700, page_limit=300, market_type="spot")
            n = evaluate(df, Candidate(r["candidate"], r["title"]), settings, 1.0)
            s = evaluate(df, Candidate(r["candidate"], r["title"]), settings, 2.0)
            passed = bool(n["total_return"] > 0 and n["profit_factor"] > 1 and n["max_drawdown"] >= -0.50 and n["trade_count"] >= 8 and s["total_return"] > 0 and s["profit_factor"] > 1)
            confirmations.append({"market": f"{key[0]} {key[1]}", "normal": n, "stress": s, "pass": passed})
            print(f"#{idx} {key[0]} {key[1]} return={n['total_return']:.2%} PF={n['profit_factor']:.2f} DD={n['max_drawdown']:.2%} trades={n['trade_count']} stress={s['total_return']:.2%} pass={passed}", flush=True)
            del df; gc.collect()
        fresh.append({**r, "fresh": confirmations, "validated": all(x["pass"] for x in confirmations)})
        if fresh[-1]["validated"]:
            break

    winners = [x for x in fresh if x["validated"]]
    decision = "VALIDATED_REGIME_SWITCH_STRATEGY" if winners else "NO_VALIDATED_REGIME_SWITCH_STRATEGY"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "evaluated": evaluated, "finalists": len(finalists), "validated_count": len(winners), "winner": winners[0] if winners else (fresh[0] if fresh else None), "finalists_detail": fresh}
    out = _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(winners), flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
