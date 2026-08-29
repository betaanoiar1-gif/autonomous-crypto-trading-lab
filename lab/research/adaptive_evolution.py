from __future__ import annotations

"""Adaptive evolutionary strategy discovery without LLM generation.

The engine starts from multiple implemented strategy families, evaluates a
small diverse population, keeps elites, mutates them, and applies early
rejection. Parameters remain frozen during independent confirmation. Futures
and live trading are intentionally disabled.
"""

import gc
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics, _run

FAMILIES = (
    "momentum", "breakout", "trend_pullback", "mean_reversion",
    "moving_average_cross", "rsi_reversion", "atr_breakout", "channel_reversion",
)

SEEDS = {
    "momentum": [{"lookback": x} for x in [20, 40, 60, 100, 140, 180]],
    "breakout": [{"lookback": x} for x in [10, 20, 40, 60, 100, 140]],
    "trend_pullback": [
        {"lookback": x, "pullback_threshold": y}
        for x in [10, 20, 40, 60]
        for y in [0.005, 0.01, 0.02, 0.03]
    ],
    "mean_reversion": [
        {"lookback": x, "z_entry": z, "z_exit": e}
        for x in [20, 40, 60, 100, 140]
        for z in [1.25, 1.75, 2.25, 2.75, 3.25]
        for e in [0.25, 0.5, 0.75, 1.0]
        if e < z
    ],
    "moving_average_cross": [
        {"fast": f, "slow": s}
        for f in [5, 10, 15, 20, 30, 45, 60]
        for s in [40, 60, 90, 120, 180, 240]
        if s > f
    ],
    "rsi_reversion": [
        {"rsi_length": n, "rsi_low": lo, "rsi_high": hi}
        for n in [7, 14, 21, 28]
        for lo in [20, 25, 30, 35, 40]
        for hi in [60, 65, 70, 75, 80, 85]
        if lo < 50 < hi
    ],
    "atr_breakout": [
        {"atr_length": n, "atr_mult": m}
        for n in [3, 5, 8, 14, 21]
        for m in [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    ],
    "channel_reversion": [{"channel_length": x} for x in [20, 40, 60, 90, 120, 180]],
}


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict

    @property
    def key(self):
        return self.family, tuple(sorted(self.params.items()))

    @property
    def title(self):
        return self.family + " | " + ", ".join(f"{k}={v}" for k, v in self.params.items())


def _mutate(rng: random.Random, c: Candidate) -> Candidate:
    p = dict(c.params)
    fam = c.family

    if fam in {"momentum", "breakout", "channel_reversion"}:
        key = "channel_length" if fam == "channel_reversion" else "lookback"
        base = max(2, int(p[key]))
        p[key] = max(2, min(240, int(round(base * rng.choice([0.65, 0.8, 0.9, 1.0, 1.1, 1.25, 1.45])))))
    elif fam == "trend_pullback":
        p["lookback"] = max(2, min(200, int(round(p["lookback"] * rng.choice([0.7, 0.85, 1.0, 1.15, 1.35])))))
        p["pullback_threshold"] = round(max(0.001, min(0.10, p["pullback_threshold"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.4]))), 4)
    elif fam == "mean_reversion":
        p["lookback"] = max(10, min(200, int(round(p["lookback"] * rng.choice([0.7, 0.85, 1.0, 1.15, 1.35])))))
        p["z_entry"] = round(max(0.8, min(3.5, p["z_entry"] * rng.choice([0.8, 0.9, 1.0, 1.1, 1.2]))), 3)
        p["z_exit"] = round(max(0.05, min(1.5, p["z_exit"] * rng.choice([0.7, 0.85, 1.0, 1.15, 1.35]))), 3)
        p["z_exit"] = min(p["z_exit"], round(max(0.05, p["z_entry"] * 0.75), 3))
    elif fam == "moving_average_cross":
        f = int(p["fast"]); s = int(p["slow"])
        f = max(2, min(100, int(round(f * rng.choice([0.7, 0.85, 1.0, 1.15, 1.3])))))
        s = max(f + 2, min(300, int(round(s * rng.choice([0.75, 0.9, 1.0, 1.15, 1.35])))))
        p["fast"], p["slow"] = f, s
    elif fam == "rsi_reversion":
        p["rsi_length"] = max(2, min(50, int(round(p["rsi_length"] * rng.choice([0.75, 0.9, 1.0, 1.15, 1.3])))))
        p["rsi_low"] = float(max(5, min(45, p["rsi_low"] + rng.choice([-5, 0, 5]))))
        p["rsi_high"] = float(max(55, min(95, p["rsi_high"] + rng.choice([-5, 0, 5]))))
    elif fam == "atr_breakout":
        p["atr_length"] = max(2, min(50, int(round(p["atr_length"] * rng.choice([0.7, 0.85, 1.0, 1.15, 1.35])))))
        p["atr_mult"] = round(max(0.25, min(5.0, p["atr_mult"] * rng.choice([0.7, 0.85, 1.0, 1.15, 1.4]))), 3)

    return Candidate(fam, p)


def _initial_population(seed: int, size: int) -> list[Candidate]:
    rng = random.Random(seed)
    pop, seen = [], set()
    for fam in FAMILIES:
        base_seeds = SEEDS[fam]
        take = max(2, size // len(FAMILIES))
        for base in rng.sample(base_seeds, min(take, len(base_seeds))):
            for _ in range(2):
                c = Candidate(fam, dict(base)) if not pop or rng.random() < 0.45 else _mutate(rng, Candidate(fam, dict(base)))
                if c.key not in seen:
                    seen.add(c.key); pop.append(c)
                if len(pop) >= size:
                    return pop
    return pop


def _run_metrics(df, c, settings, fee_mult=1.0):
    r = _run(df, c.family, dict(c.params), ["both"], settings.capital.initial_usd,
             settings.execution.commission_bps * fee_mult,
             settings.execution.slippage_bps * fee_mult)
    return _metrics(r, r.returns)


def _score(m: dict) -> float:
    ret = float(m.get("total_return", 0.0)); pf = min(2.5, float(m.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0)))); trades = int(m.get("trade_count", 0)); sh = float(m.get("sharpe", 0.0))
    if trades < 4: return -20.0 + 2.0 * trades
    return 100 * ret + 18 * (pf - 1) + 4 * sh - 30 * dd + min(8, math.log1p(trades))


def _screen(df, c, settings):
    n = len(df); a = df.iloc[: int(n * 0.55)]; b = df.iloc[int(n * 0.55):]
    x = _run_metrics(a, c, settings); y = _run_metrics(b, c, settings)
    ys = _run_metrics(b, c, settings, 2.0)
    return x, y, ys


def _cross_score(records):
    rets = [float(x[1]["total_return"]) for x in records]; pfs = [float(x[1]["profit_factor"]) for x in records]
    stress = [float(x[2]["total_return"]) for x in records]; dds = [abs(min(0, float(x[1]["max_drawdown"]))) for x in records]
    positive = sum(r > 0 and pf > 1 and s > 0 for r, pf, s in zip(rets, pfs, stress))
    dispersion = max(rets) - min(rets)
    return float(np.mean([_score(x[1]) for x in records]) + 18 * positive - 22 * dispersion - 22 * max(dds))


def _save(payload):
    out = ROOT / "experiments" / "adaptive_evolution_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)
    return out


def run(hours: float = 3.0, population: int = 24, generations: int = 10, elites: int = 8):
    settings = load_settings(); adapter = CCXTMarketData(exchange_id="binance")
    deadline = time.monotonic() + hours * 3600
    started = datetime.now(timezone.utc)
    print("=== ADAPTIVE EVOLUTION SEARCH ===", flush=True)
    print("AI generation: DISABLED | Futures: DISABLED | Live trading: DISABLED", flush=True)
    print("Population:", population, "Generations:", generations, "Elites:", elites, flush=True)

    data = {}
    for symbol, tf, bars in [("ETH/USDT", "1h", 2500), ("ETH/USDT", "4h", 1600), ("BTC/USDT", "4h", 1600), ("BTC/USDT", "1h", 2500), ("ETH/USDT", "15m", 2500)]:
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1200, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)

    rng = random.Random(20260829)
    pop = _initial_population(20260829, population)
    seen = {c.key for c in pop}; all_records = []

    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} ===", flush=True)
        generation = []
        for i, c in enumerate(pop, 1):
            if time.monotonic() >= deadline: break
            try:
                records = []
                for key in [("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h")]:
                    records.append((key, *_screen(data[key], c, settings)[1:]))
                score = _cross_score(records)
                generation.append({"candidate": c, "score": score, "records": records})
                if i % 4 == 0 or i == 1:
                    print(f"  eval {i}/{len(pop)} | score={score:.2f} | {c.title}", flush=True)
            except Exception as exc:
                print(f"  ERROR {c.title}: {type(exc).__name__}: {exc}", flush=True)
            gc.collect()

        generation.sort(key=lambda x: x["score"], reverse=True)
        all_records.extend(generation)
        best = generation[0] if generation else None
        if not best: break
        passed = 0
        for x in generation:
            ok = all(float(rec[1]["total_return"]) > 0 and float(rec[1]["profit_factor"]) > 1 and float(rec[2]["total_return"]) > 0 for rec in x["records"])
            passed += int(ok)
        print(f"GENERATION RESULT: best={best['score']:.2f} | candidates={len(generation)} | cross_market_positive={passed}", flush=True)

        elite = [x["candidate"] for x in generation[:elites]]
        next_pop = list(elite)
        while len(next_pop) < population and time.monotonic() < deadline:
            parent = rng.choice(elite)
            child = _mutate(rng, parent)
            if child.key not in seen:
                seen.add(child.key); next_pop.append(child)
            elif rng.random() < 0.2:
                fam = rng.choice(FAMILIES); child = Candidate(fam, dict(rng.choice(SEEDS[fam])))
                if child.key not in seen:
                    seen.add(child.key); next_pop.append(child)
        pop = next_pop

        checkpoint = {
            "started_at": started.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "generation": gen,
            "decision": "RUNNING",
            "seen_candidates": len(seen),
            "best": {"title": best["candidate"].title, "family": best["candidate"].family, "parameters": best["candidate"].params, "score": best["score"]},
        }
        path = _save(checkpoint)
        print("  checkpoint:", path, flush=True)

    all_records.sort(key=lambda x: x["score"], reverse=True)
    finalists = []
    family_count = {}
    for x in all_records:
        c = x["candidate"]
        if family_count.get(c.family, 0) >= 3: continue
        finalists.append(x); family_count[c.family] = family_count.get(c.family, 0) + 1
        if len(finalists) >= 10: break

    print("\n=== FROZEN FRESH CONFIRMATION ===", flush=True)
    validated = []
    confirmations = []
    for i, x in enumerate(finalists, 1):
        c = x["candidate"]; print(f"FINALIST {i}/{len(finalists)}: {c.title}", flush=True)
        fresh = []
        for key in [("BTC/USDT", "1h"), ("ETH/USDT", "15m")]:
            normal = _run_metrics(data[key], c, settings, 1.0); stress = _run_metrics(data[key], c, settings, 2.0)
            ok = (normal["total_return"] > 0 and normal["profit_factor"] > 1 and normal["max_drawdown"] >= -0.50 and normal["trade_count"] >= 8 and stress["total_return"] > 0 and stress["profit_factor"] > 1)
            fresh.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"  {key[0]} {key[1]}: R={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} T={normal['trade_count']} stress={stress['total_return']:.2%} PASS={ok}", flush=True)
        rec = {"title": c.title, "family": c.family, "parameters": dict(c.params), "screen_score": x["score"], "fresh": fresh, "validated": all(v["pass"] for v in fresh)}
        confirmations.append(rec)
        if rec["validated"]:
            validated.append(rec); break
        gc.collect()

    decision = "VALIDATED_ALGORITHMIC_STRATEGY" if validated else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generation_count": gen if 'gen' in locals() else 0, "seen_candidates": len(seen), "finalists": len(finalists), "validated_count": len(validated), "confirmations": confirmations}
    out = _save(payload)
    print("\n=== EVOLUTION FINISHED ===", flush=True)
    print("Decision:", decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
