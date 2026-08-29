from __future__ import annotations

"""Checkpoint-first AI-free evolutionary search."""

import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics, _run

FAMILIES = (
    "momentum", "breakout", "trend_pullback", "mean_reversion",
    "moving_average_cross", "rsi_reversion", "atr_breakout", "channel_reversion",
)

SEEDS = {
    "momentum": [{"lookback": x} for x in (20, 40, 60, 100, 140, 180, 220)],
    "breakout": [{"lookback": x} for x in (10, 20, 40, 60, 100, 140, 180)],
    "trend_pullback": [{"lookback": x, "pullback_threshold": y} for x in (8, 12, 20, 40, 60) for y in (0.0035, 0.005, 0.0075, 0.01, 0.02)],
    "mean_reversion": [{"lookback": x, "z_entry": z, "z_exit": e} for x in (20, 40, 60, 100, 140) for z in (1.25, 1.75, 2.25, 2.75, 3.25) for e in (0.25, 0.5, 0.75, 1.0) if e < z],
    "moving_average_cross": [{"fast": f, "slow": s} for f in (5, 10, 15, 20, 30, 45, 60) for s in (40, 60, 90, 120, 180, 240) if s > f],
    "rsi_reversion": [{"rsi_length": n, "rsi_low": lo, "rsi_high": hi} for n in (7, 14, 21, 28) for lo in (20, 25, 30, 35, 40) for hi in (60, 65, 70, 75, 80, 85) if lo < 50 < hi],
    "atr_breakout": [{"atr_length": n, "atr_mult": m} for n in (3, 5, 8, 14, 21) for m in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)],
    "channel_reversion": [{"channel_length": x} for x in (20, 40, 60, 90, 120, 180)],
}


def _save(payload: dict) -> Path:
    out = ROOT / "experiments" / "adaptive_evolution_safe_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)
    return out


def _mutate(rng: random.Random, family: str, params: dict) -> dict:
    p = dict(params)
    mult = rng.choice((0.8, 0.9, 1.0, 1.1, 1.2))
    if family in {"momentum", "breakout"}:
        p["lookback"] = max(5, min(240, int(round(p["lookback"] * mult))))
    elif family == "channel_reversion":
        p["channel_length"] = max(10, min(240, int(round(p["channel_length"] * mult))))
    elif family == "trend_pullback":
        p["lookback"] = max(5, min(200, int(round(p["lookback"] * mult))))
        p["pullback_threshold"] = round(max(0.001, min(0.05, p["pullback_threshold"] * rng.choice((0.8, 1.0, 1.2)))), 4)
    elif family == "mean_reversion":
        p["lookback"] = max(10, min(200, int(round(p["lookback"] * mult))))
        p["z_entry"] = round(max(0.8, min(3.5, p["z_entry"] * rng.choice((0.85, 1.0, 1.15)))), 3)
        p["z_exit"] = round(max(0.1, min(1.5, min(p["z_exit"] * rng.choice((0.8, 1.0, 1.2)), p["z_entry"] * 0.75))), 3)
    elif family == "moving_average_cross":
        p["fast"] = max(2, min(100, int(round(p["fast"] * mult))))
        p["slow"] = max(p["fast"] + 3, min(300, int(round(p["slow"] * mult))))
    elif family == "rsi_reversion":
        p["rsi_length"] = max(3, min(50, int(round(p["rsi_length"] * mult))))
        p["rsi_low"] = max(5, min(45, p["rsi_low"] + rng.choice((-5, 0, 5))))
        p["rsi_high"] = max(55, min(95, p["rsi_high"] + rng.choice((-5, 0, 5))))
    elif family == "atr_breakout":
        p["atr_length"] = max(2, min(50, int(round(p["atr_length"] * mult))))
        p["atr_mult"] = round(max(0.25, min(5.0, p["atr_mult"] * rng.choice((0.8, 1.0, 1.2)))), 3)
    return p


def _eval(df, family, params, settings, fee_mult=1.0):
    r = _run(df, family, dict(params), ["both"], settings.capital.initial_usd,
             settings.execution.commission_bps * fee_mult,
             settings.execution.slippage_bps * fee_mult)
    return _metrics(r, r.returns)


def _score(m: dict) -> float:
    ret = float(m.get("total_return", 0.0)); pf = float(m.get("profit_factor", 0.0))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0)))); trades = int(m.get("trade_count", 0))
    if trades < 4:
        return -20.0
    return 100 * ret + 20 * (min(2.5, pf) - 1) - 35 * dd + min(8.0, math.log1p(trades))


def run(hours: float = 3.0, population: int = 16, generations: int = 8):
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + hours * 3600

    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0, "seen_candidates": 0, "generator": "deterministic_evolution", "ai_generation": False, "futures": False, "live_trading": False})
    print("=== SAFE ADAPTIVE EVOLUTION ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("START CHECKPOINT WRITTEN", flush=True)

    adapter = CCXTMarketData(exchange_id="binance")
    print("SMOKE LOAD ETH/USDT 1h", flush=True)
    smoke_df = adapter.fetch_ohlcv_history("ETH/USDT", "1h", 220, page_limit=220, market_type="spot")
    _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "SMOKE", "smoke_bars": len(smoke_df), "seen_candidates": 0})
    smoke = _eval(smoke_df, "momentum", {"lookback": 40}, settings)
    print(f"SMOKE RESULT return={smoke['total_return']:.2%} PF={smoke['profit_factor']:.2f} trades={smoke['trade_count']}", flush=True)
    del smoke_df
    gc.collect()

    data = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 1800), ("ETH/USDT", "4h", 1200), ("BTC/USDT", "4h", 1200), ("BTC/USDT", "1h", 1800), ("ETH/USDT", "15m", 1800)):
        if time.monotonic() >= deadline: break
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1200, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADING", "loaded_market": f"{symbol} {tf}", "loaded_bars": len(data[(symbol, tf)])})
        gc.collect()

    rng = random.Random(20260829)
    pop = []
    seen = set()
    per_family = max(2, population // len(FAMILIES))
    for fam in FAMILIES:
        for par in rng.sample(SEEDS[fam], min(per_family, len(SEEDS[fam]))):
            key = (fam, tuple(sorted(par.items())))
            if key not in seen:
                seen.add(key); pop.append((fam, dict(par)))
            if len(pop) >= population: break
        if len(pop) >= population: break

    results = []
    total_evaluated = 0
    primary_keys = (("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h"))

    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} ===", flush=True)
        gen_results = []
        for i, (fam, params) in enumerate(pop, 1):
            if time.monotonic() >= deadline: break
            market_scores = []
            ok = 0
            for key in primary_keys:
                m = _eval(data[key], fam, params, settings, 1.0)
                market_scores.append(m)
                ok += int(m["total_return"] > 0 and m["profit_factor"] > 1 and m["trade_count"] >= 8)
            rets = [float(m["total_return"]) for m in market_scores]
            value = float(np.mean([_score(m) for m in market_scores]) + 20 * ok - 20 * (max(rets) - min(rets)))
            rec = {"family": fam, "params": dict(params), "title": fam + " | " + str(params), "score": value, "ok": ok, "markets": market_scores}
            gen_results.append(rec); results.append(rec); total_evaluated += 1
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "RUNNING", "generation": gen, "evaluated": total_evaluated, "seen_candidates": len(seen), "latest": rec})
            print(f"eval {i}/{len(pop)} | score={value:.2f} | ok={ok}/3 | {rec['title']}", flush=True)
            gc.collect()
        if not gen_results: break
        gen_results.sort(key=lambda x: (x["ok"], x["score"]), reverse=True)
        best = gen_results[0]
        print(f"GENERATION RESULT: best={best['score']:.2f} | ok={best['ok']}/3", flush=True)
        elite_n = max(2, min(6, population // 3)); elites = gen_results[:elite_n]
        next_pop = [(x["family"], dict(x["params"])) for x in elites]
        while len(next_pop) < population and time.monotonic() < deadline:
            e = rng.choice(elites)
            child = _mutate(rng, e["family"], e["params"])
            key = (e["family"], tuple(sorted(child.items())))
            if key not in seen:
                seen.add(key); next_pop.append((e["family"], child))
            elif rng.random() < 0.15:
                fam2 = rng.choice(FAMILIES); par2 = dict(rng.choice(SEEDS[fam2])); key2 = (fam2, tuple(sorted(par2.items())))
                if key2 not in seen:
                    seen.add(key2); next_pop.append((fam2, par2))
        pop = next_pop

    results.sort(key=lambda x: (x["ok"], x["score"]), reverse=True)
    finalists = []; family_count = {}
    for r in results:
        if family_count.get(r["family"], 0) >= 2: continue
        finalists.append(r); family_count[r["family"]] = family_count.get(r["family"], 0) + 1
        if len(finalists) >= 10: break

    print("\n=== FROZEN FRESH CONFIRMATION ===", flush=True)
    confirmations = []
    for idx, r in enumerate(finalists, 1):
        fresh = []
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            normal = _eval(data[key], r["family"], r["params"], settings, 1.0)
            stress = _eval(data[key], r["family"], r["params"], settings, 2.0)
            ok = normal["total_return"] > 0 and normal["profit_factor"] > 1 and normal["max_drawdown"] >= -0.50 and normal["trade_count"] >= 8 and stress["total_return"] > 0 and stress["profit_factor"] > 1
            fresh.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": bool(ok)})
            print(f"  {key[0]} {key[1]}: return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
        validated = r["ok"] == 3 and all(x["pass"] for x in fresh)
        confirmations.append({**r, "fresh": fresh, "validated": bool(validated)})
        print(f"FINALIST {idx}: screen={r['ok']}/3 | fresh={sum(x['pass'] for x in fresh)}/2 | validated={validated}", flush=True)
        if validated: break
        gc.collect()

    validated = [x for x in confirmations if x["validated"]]
    decision = "VALIDATED_ALGORITHMIC_STRATEGY" if validated else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generation": generations, "evaluated": total_evaluated, "seen_candidates": len(seen), "finalists": len(finalists), "validated_count": len(validated), "results": confirmations}
    out = _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Checkpoint:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
