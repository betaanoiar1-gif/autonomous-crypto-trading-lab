from __future__ import annotations

"""Safe adaptive evolution runner.

Designed to prove progress immediately: writes a STARTING checkpoint before
loading any market data, then evaluates a tiny smoke candidate before entering
larger generations. No LLM, futures, live trading, or arbitrary code.
"""

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
from .adaptive_evolution import Candidate, FAMILIES, SEEDS, _initial_population, _mutate
from .evaluator import _metrics, _run

OUT = ROOT / "experiments" / "adaptive_evolution_safe_latest.json"


def save(payload: dict):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def metric(df, c, settings, fee_mult=1.0):
    r = _run(df, c.family, dict(c.params), ["both"], settings.capital.initial_usd,
             settings.execution.commission_bps * fee_mult,
             settings.execution.slippage_bps * fee_mult)
    return _metrics(r, r.returns)


def score(m):
    ret = float(m.get("total_return", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    trades = int(m.get("trade_count", 0))
    if trades < 4:
        return -20.0 + trades
    return 100.0 * ret + 18.0 * (min(2.5, pf) - 1.0) - 30.0 * dd + min(8.0, math.log1p(trades))


def run(hours: float = 3.0, population: int = 16, generations: int = 8, max_finalists: int = 10):
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + hours * 3600

    payload = {
        "started_at": started.isoformat(),
        "updated_at": started.isoformat(),
        "decision": "STARTING",
        "generation": 0,
        "evaluated": 0,
        "seen_candidates": 0,
    }
    save(payload)

    print("=== SAFE ADAPTIVE EVOLUTION ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("START CHECKPOINT:", OUT, flush=True)

    adapter = CCXTMarketData(exchange_id="binance")
    data = {}
    for symbol, tf, bars in [
        ("ETH/USDT", "1h", 1200),
        ("ETH/USDT", "4h", 900),
        ("BTC/USDT", "4h", 900),
        ("BTC/USDT", "1h", 1200),
        ("ETH/USDT", "15m", 1200),
    ]:
        if time.monotonic() >= deadline:
            break
        print(f"LOAD {symbol} {tf} ...", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1000, market_type="spot")
        print(f"LOADED {symbol} {tf}: {len(data[(symbol, tf)])} bars", flush=True)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        save(payload)

    # Smoke test: one deterministic candidate, one market, immediately persisted.
    pop = _initial_population(20260829, population)
    seen = {c.key for c in pop}
    payload.update({"decision": "SMOKE_TEST", "seen_candidates": len(seen), "updated_at": datetime.now(timezone.utc).isoformat()})
    save(payload)

    if not data:
        payload.update({"decision": "DATA_ERROR", "error": "No market data loaded"})
        save(payload)
        print("DATA_ERROR: no market data", flush=True)
        return payload

    smoke = pop[0]
    smoke_key = next(iter(data.keys()))
    print(f"SMOKE TEST: {smoke.title} on {smoke_key[0]} {smoke_key[1]}", flush=True)
    try:
        sm = metric(data[smoke_key], smoke, settings, 1.0)
        print(f"SMOKE RESULT: return={sm['total_return']:.2%} PF={sm['profit_factor']:.2f} DD={sm['max_drawdown']:.2%} trades={sm['trade_count']}", flush=True)
    except Exception as exc:
        payload.update({"decision": "SMOKE_ERROR", "error": f"{type(exc).__name__}: {exc}"})
        save(payload)
        raise

    results = []
    primary_keys = [("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h")]
    rng = random.Random(20260829)

    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline:
            break
        print(f"\n=== GENERATION {gen}/{generations} ===", flush=True)
        gen_results = []
        for i, c in enumerate(pop, 1):
            if time.monotonic() >= deadline:
                break
            market_scores = []
            ok = 0
            for key in primary_keys:
                m = metric(data[key], c, settings, 1.0)
                market_scores.append(m)
                ok += int(float(m["total_return"]) > 0 and float(m["profit_factor"]) > 1 and int(m["trade_count"]) >= 8)
            rets = [float(m["total_return"]) for m in market_scores]
            value = float(np.mean([score(m) for m in market_scores]) + 20 * ok - 20 * (max(rets) - min(rets)))
            gen_results.append({"candidate": c, "score": value, "ok": ok, "markets": market_scores})
            payload.update({"decision": "RUNNING", "generation": gen, "evaluated": payload.get("evaluated", 0) + 1, "seen_candidates": len(seen), "updated_at": datetime.now(timezone.utc).isoformat()})
            save(payload)
            if i % 2 == 0 or i == 1:
                print(f"eval {i}/{len(pop)} | score={value:.2f} | ok={ok}/3 | {c.title}", flush=True)
            gc.collect()

        if not gen_results:
            break
        gen_results.sort(key=lambda x: (x["ok"], x["score"]), reverse=True)
        results.extend(gen_results)
        best = gen_results[0]
        print(f"GENERATION RESULT: best={best['score']:.2f} | ok={best['ok']}/3", flush=True)

        elite_n = max(2, min(6, population // 3))
        elites = [x["candidate"] for x in gen_results[:elite_n]]
        next_pop = list(elites)
        while len(next_pop) < population and time.monotonic() < deadline:
            parent = rng.choice(elites)
            child = _mutate(rng, parent)
            if child.key not in seen:
                seen.add(child.key)
                next_pop.append(child)
            elif rng.random() < 0.15:
                fam = rng.choice(FAMILIES)
                child = Candidate(fam, dict(rng.choice(SEEDS[fam])))
                if child.key not in seen:
                    seen.add(child.key)
                    next_pop.append(child)
        pop = next_pop

    # Unique, diverse finalists from all generations.
    results.sort(key=lambda x: (x["ok"], x["score"]), reverse=True)
    finalists = []
    family_count = {}
    for x in results:
        c = x["candidate"]
        if family_count.get(c.family, 0) >= 2:
            continue
        finalists.append(x)
        family_count[c.family] = family_count.get(c.family, 0) + 1
        if len(finalists) >= max_finalists:
            break

    print("\n=== FROZEN FRESH CONFIRMATION ===", flush=True)
    confirmed = []
    for i, x in enumerate(finalists, 1):
        c = x["candidate"]
        fresh = []
        for key in [("BTC/USDT", "1h"), ("ETH/USDT", "15m")]:
            normal = metric(data[key], c, settings, 1.0)
            stress = metric(data[key], c, settings, 2.0)
            ok = bool(normal["total_return"] > 0 and normal["profit_factor"] > 1 and normal["max_drawdown"] >= -0.50 and normal["trade_count"] >= 8 and stress["total_return"] > 0 and stress["profit_factor"] > 1)
            fresh.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"  {key[0]} {key[1]}: return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
        passes = sum(x["pass"] for x in fresh)
        rec = {"title": c.title, "family": c.family, "parameters": dict(c.params), "screen_score": x["score"], "screen_ok": x["ok"], "fresh": fresh, "fresh_pass": passes, "validated": bool(x["ok"] == 3 and passes == 2)}
        confirmed.append(rec)
        print(f"FINALIST {i}: {c.title} | screen={x['ok']}/3 | fresh={passes}/2 | validated={rec['validated']}", flush=True)
        if rec["validated"]:
            break
        gc.collect()

    validated = [x for x in confirmed if x["validated"]]
    decision = "VALIDATED_ALGORITHMIC_STRATEGY" if validated else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "generation": min(generations, int(payload.get("generation", 0))),
        "evaluated": payload.get("evaluated", 0),
        "seen_candidates": len(seen),
        "finalists": len(finalists),
        "validated_count": len(validated),
        "results": confirmed,
    }
    save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Checkpoint:", OUT, flush=True)
    return payload


if __name__ == "__main__":
    run()
