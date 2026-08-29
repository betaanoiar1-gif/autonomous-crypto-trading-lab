from __future__ import annotations

"""Autonomous AI-free strategy discovery.

Generates candidates only from strategy families actually implemented by the
project evaluator. Uses cross-market screening before selection, then frozen
fresh confirmation. No LLM, no futures, no live trading.
"""

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics, _run


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict
    title: str


SEEDS = {
    "momentum": [
        {"lookback": x} for x in [20, 30, 40, 60, 80, 100, 120, 140, 160, 180]
    ],
    "breakout": [
        {"lookback": x} for x in [10, 20, 30, 40, 60, 80, 100, 120]
    ],
    "trend_pullback": [
        {"lookback": x, "pullback_threshold": y}
        for x in [10, 15, 20, 30, 40, 60, 80]
        for y in [0.005, 0.01, 0.02, 0.03, 0.05]
    ],
    "mean_reversion": [
        {"lookback": x, "z_entry": z, "z_exit": e}
        for x in [20, 30, 40, 60, 80, 120, 160]
        for z in [1.0, 1.5, 2.0, 2.5, 3.0]
        for e in [0.25, 0.5, 0.75, 1.0]
        if e < z
    ],
    "moving_average_cross": [
        {"fast": f, "slow": s}
        for f in [5, 8, 12, 18, 24, 30, 40, 50, 60]
        for s in [40, 60, 80, 100, 120, 150, 180, 240]
        if s > f
    ],
    "rsi_reversion": [
        {"rsi_length": n, "rsi_low": lo, "rsi_high": hi}
        for n in [7, 10, 14, 21, 25, 30]
        for lo in [20, 25, 30, 35, 40]
        for hi in [60, 65, 70, 75, 80, 85]
        if lo < 50 < hi
    ],
    "atr_breakout": [
        {"atr_length": n, "atr_mult": m}
        for n in [3, 5, 7, 10, 14, 18, 21]
        for m in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    ],
    "channel_reversion": [
        {"channel_length": x} for x in [10, 20, 30, 40, 60, 80, 100, 120, 150, 180]
    ],
}

FAMILIES = tuple(SEEDS.keys())


def _key(c: Candidate) -> tuple:
    return (c.family, tuple(sorted(c.params.items())))


def _title(c: Candidate) -> str:
    return c.family + " | " + ", ".join(f"{k}={v}" for k, v in c.params.items())


def _mutate(rng: random.Random, family: str, base: dict) -> dict:
    p = dict(base)
    if family in {"momentum", "breakout", "channel_reversion"}:
        b = max(2, int(p["lookback"])) if family != "channel_reversion" else max(5, int(p["channel_length"]))
        vals = sorted({max(2, min(240, int(round(b * f)))) for f in [0.6, 0.8, 1.0, 1.25, 1.5, 1.8]})
        if family == "channel_reversion":
            p["channel_length"] = rng.choice(vals)
        else:
            p["lookback"] = rng.choice(vals)
    elif family == "moving_average_cross":
        f = int(p["fast"]); s = int(p["slow"])
        p["fast"] = max(2, min(100, int(round(f * rng.choice([0.7, 0.85, 1.0, 1.15, 1.35])))))
        p["slow"] = max(p["fast"] + 2, min(300, int(round(s * rng.choice([0.75, 0.9, 1.0, 1.15, 1.4])))))
    elif family == "trend_pullback":
        p["lookback"] = max(2, min(200, int(round(p["lookback"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.45])))))
        p["pullback_threshold"] = round(max(0.001, min(0.1, p["pullback_threshold"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.5]))), 4)
    elif family == "mean_reversion":
        p["lookback"] = max(10, min(200, int(round(p["lookback"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.45])))))
        p["z_entry"] = round(max(0.8, min(3.5, p["z_entry"] * rng.choice([0.8, 0.9, 1.0, 1.1, 1.25]))), 3)
        p["z_exit"] = round(max(0.05, min(1.5, p["z_exit"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.4]))), 3)
        if p["z_exit"] >= p["z_entry"]:
            p["z_exit"] = round(max(0.05, p["z_entry"] * 0.5), 3)
    elif family == "rsi_reversion":
        p["rsi_length"] = max(2, min(50, int(round(p["rsi_length"] * rng.choice([0.75, 0.9, 1.0, 1.15, 1.3])))))
        p["rsi_low"] = float(max(5, min(45, p["rsi_low"] + rng.choice([-5, 0, 5]))))
        p["rsi_high"] = float(max(55, min(95, p["rsi_high"] + rng.choice([-5, 0, 5]))))
        if p["rsi_low"] >= 50: p["rsi_low"] = 45.0
        if p["rsi_high"] <= 50: p["rsi_high"] = 55.0
    elif family == "atr_breakout":
        p["atr_length"] = max(2, min(50, int(round(p["atr_length"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.4])))))
        p["atr_mult"] = round(max(0.25, min(5.0, p["atr_mult"] * rng.choice([0.7, 0.85, 1.0, 1.2, 1.45]))), 3)
    return p


def _make_candidates(seed: int, limit: int) -> list[Candidate]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    seen: set[tuple] = set()

    for family in FAMILIES:
        for seed_params in SEEDS[family]:
            for attempt in range(5):
                p = dict(seed_params) if attempt == 0 else _mutate(rng, family, seed_params)
                c = Candidate(family, p, "")
                c = Candidate(c.family, c.params, _title(c))
                k = _key(c)
                if k not in seen:
                    seen.add(k); out.append(c)
                if len(out) >= limit:
                    return out

    while len(out) < limit:
        family = rng.choice(FAMILIES)
        base = rng.choice(SEEDS[family])
        p = _mutate(rng, family, base)
        c = Candidate(family, p, _title(Candidate(family, p, "")))
        if _key(c) not in seen:
            seen.add(_key(c)); out.append(c)
    return out


def _score(m: dict) -> float:
    ret = float(m.get("total_return", 0.0))
    pf = min(3.0, float(m.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    trades = int(m.get("trade_count", 0))
    sharpe = float(m.get("sharpe", 0.0))
    return 100 * ret + 18 * (pf - 1) + 4 * sharpe - 30 * dd + min(8, math.log1p(max(0, trades)))


def _run_one(df, c, settings, fee_mult=1.0):
    r = _run(
        df, c.family, dict(c.params), ["both"],
        settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
    )
    return _metrics(r, r.returns)


def _market_record(df, c, settings):
    normal = _run_one(df, c, settings, 1.0)
    stress = _run_one(df, c, settings, 2.0)
    return {
        "normal": normal,
        "stress": stress,
    }


def _pass(market: dict, require_stress: bool = True) -> bool:
    n = market["normal"]
    s = market["stress"]
    ok = (
        float(n["total_return"]) > 0
        and float(n["profit_factor"]) > 1
        and float(n["max_drawdown"]) >= -0.50
        and int(n["trade_count"]) >= 8
    )
    if require_stress:
        ok = ok and float(s["total_return"]) > 0 and float(s["profit_factor"]) > 1
    return bool(ok)


def run(hours: float = 3.0, candidate_limit: int = 600, shortlist: int = 20) -> dict:
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + max(0.05, hours * 3600)
    adapter = CCXTMarketData(exchange_id="binance")

    print("=== AI-FREE AUTONOMOUS SEARCH v2 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Real strategy families: " + ", ".join(FAMILIES), flush=True)

    datasets = {}
    for symbol, tf, bars in [
        ("ETH/USDT", "1h", 5000),
        ("ETH/USDT", "4h", 3000),
        ("BTC/USDT", "4h", 3000),
        ("BTC/USDT", "1h", 5000),
        ("ETH/USDT", "15m", 5000),
    ]:
        datasets[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1500, market_type="spot")
        print(f"Loaded {symbol} {tf}: {len(datasets[(symbol, tf)])} bars", flush=True)

    candidates = _make_candidates(20260829, candidate_limit)
    print(f"Generated {len(candidates)} unique candidates", flush=True)

    primary_keys = [("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h")]
    scored = []

    for i, c in enumerate(candidates, 1):
        if time.monotonic() >= deadline:
            break
        try:
            markets = {f"{k[0]} {k[1]}": _market_record(datasets[k], c, settings) for k in primary_keys}
            raw = []
            for rec in markets.values():
                n = rec["normal"]; s = rec["stress"]
                raw.append(_score(n) + 50 * float(s["total_return"]) + 10 * (float(s["profit_factor"]) - 1))
            cross = float(np.mean(raw))
            eligible_markets = sum(_pass(v, True) for v in markets.values())
            # Strong preference for generalization; one-market miracles rank poorly.
            cross += 20 * eligible_markets
            cross -= 25 * (max(float(v["normal"]["total_return"]) for v in markets.values()) - min(float(v["normal"]["total_return"]) for v in markets.values()))
            scored.append({"candidate": c, "score": cross, "markets": markets, "eligible_markets": eligible_markets})
        except Exception as exc:
            print(f"Candidate {i} ERROR: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0:
            best = max(scored, key=lambda x: x["score"]) if scored else None
            print(f"Progress {i}/{len(candidates)} | best={best['score']:.2f}" if best else f"Progress {i}", flush=True)
        gc.collect()

    scored.sort(key=lambda x: (x["eligible_markets"], x["score"]), reverse=True)

    # Diversity: different family first, then distinct parameter signatures.
    selected = []
    sigs = set()
    family_counts = {}
    for r in scored:
        c = r["candidate"]
        sig = (c.family, tuple(sorted(c.params.items())))
        if sig in sigs:
            continue
        if family_counts.get(c.family, 0) >= max(3, shortlist // 4):
            continue
        sigs.add(sig); selected.append(r); family_counts[c.family] = family_counts.get(c.family, 0) + 1
        if len(selected) >= shortlist:
            break

    print("\n=== FROZEN FRESH CONFIRMATION ===", flush=True)
    fresh_keys = [("BTC/USDT", "1h"), ("ETH/USDT", "15m")]
    confirmed = []

    for i, r in enumerate(selected, 1):
        if time.monotonic() >= deadline:
            break
        c = r["candidate"]
        fresh = {f"{k[0]} {k[1]}": _market_record(datasets[k], c, settings) for k in fresh_keys}
        fresh_pass = sum(_pass(v, True) for v in fresh.values())
        all_pass = r["eligible_markets"] == 3 and fresh_pass == 2
        r["fresh"] = fresh
        r["fresh_pass"] = fresh_pass
        r["validated"] = bool(all_pass)
        confirmed.append(r)
        print(f"#{i} {c.title} | screen={r['eligible_markets']}/3 | fresh={fresh_pass}/2 | VALIDATED={all_pass}", flush=True)
        for name, v in fresh.items():
            n=v["normal"]; s=v["stress"]
            print(f"  {name}: R={n['total_return']:.2%} PF={n['profit_factor']:.2f} DD={n['max_drawdown']:.2%} T={n['trade_count']} stress={s['total_return']:.2%}", flush=True)
        if all_pass:
            break
        gc.collect()

    validated = [x for x in confirmed if x.get("validated")]
    decision = "VALIDATED_ALGORITHMIC_STRATEGY" if validated else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    winner = validated[0] if validated else (confirmed[0] if confirmed else None)

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_hours": (datetime.now(timezone.utc) - started).total_seconds() / 3600,
        "decision": decision,
        "generated": len(candidates),
        "screened": len(scored),
        "shortlisted": len(selected),
        "validated_count": len(validated),
        "winner": {
            "family": winner["candidate"].family,
            "title": winner["candidate"].title,
            "parameters": winner["candidate"].params,
            "screen_score": winner["score"],
            "screen_markets": winner["markets"],
            "fresh": winner.get("fresh", {}),
            "fresh_pass": winner.get("fresh_pass", 0),
        } if winner else None,
        "top_confirmed": [
            {
                "family": x["candidate"].family,
                "title": x["candidate"].title,
                "parameters": x["candidate"].params,
                "score": x["score"],
                "fresh_pass": x.get("fresh_pass", 0),
                "validated": x.get("validated", False),
            } for x in confirmed
        ],
    }

    out = ROOT / "experiments" / "ai_free_autosearch_latest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
