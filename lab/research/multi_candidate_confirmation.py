from __future__ import annotations

"""AI-free multi-candidate confirmation.

Selects a diverse set of candidates using only the deterministic algorithmic
search engine, then evaluates all finalists on fresh symbol/timeframe checks.
No LLM generation, futures, or live trading are used.
"""

import gc
import json
import math
from datetime import datetime, timezone

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .algorithmic_discovery import _candidate_pool, _frozen_eval
from .evaluator import _metrics, _run


def _compact(m: dict) -> dict:
    return {
        "return": float(m.get("total_return", 0.0)),
        "pf": float(m.get("profit_factor", 0.0)),
        "dd": float(m.get("max_drawdown", 0.0)),
        "trades": int(m.get("trade_count", 0)),
        "sharpe": float(m.get("sharpe", 0.0)),
    }


def _cross_score(parts: list[dict]) -> float:
    rets = [float(x["holdout"]["total_return"]) for x in parts]
    pfs = [float(x["holdout"]["profit_factor"]) for x in parts]
    dds = [abs(min(0.0, float(x["holdout"]["max_drawdown"]))) for x in parts]
    trades = [int(x["holdout"]["trade_count"]) for x in parts]
    stress = [float(x["stress"]["total_return"]) for x in parts]

    positive = sum(r > 0 for r in rets)
    bad = sum(
        r <= 0 or pf <= 1 or s <= 0 or t < 8 or dd > 0.50
        for r, pf, s, t, dd in zip(rets, pfs, stress, trades, dds)
    )

    return (
        120.0 * float(np.mean(rets))
        + 30.0 * (float(np.median(pfs)) - 1.0)
        + 60.0 * float(np.mean(stress))
        + 10.0 * positive
        + 25.0 * float(np.min(rets))
        - 25.0 * (float(np.max(rets)) - float(np.min(rets)))
        - 30.0 * float(np.max(dds))
        - 20.0 * bad
        - (25.0 if min(trades) < 8 else 0.0)
    )


def _fresh_confirmation(df, record: dict, settings) -> dict:
    p = dict(record["parameters"])
    capital = settings.capital.initial_usd
    fee = settings.execution.commission_bps
    slip = settings.execution.slippage_bps

    normal = _run(df, record["family"], p, ["both"], capital, fee, slip)
    stress = _run(df, record["family"], p, ["both"], capital, fee * 2, slip * 2)

    nm = _metrics(normal, normal.returns)
    sm = _metrics(stress, stress.returns)

    passed = bool(
        nm["total_return"] > 0
        and nm["profit_factor"] > 1
        and nm["max_drawdown"] >= -0.50
        and nm["trade_count"] >= 8
        and sm["total_return"] > 0
        and sm["profit_factor"] > 1
    )

    return {
        "normal": _compact(nm),
        "stress": _compact(sm),
        "passed": passed,
    }


def run(limit: int = 180, finalists: int = 12) -> dict:
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)

    print("=== MULTI-CANDIDATE FINAL CONFIRMATION ===", flush=True)
    print("AI generation: DISABLED", flush=True)
    print("Futures: DISABLED | Live trading: DISABLED", flush=True)
    print("Selection: cross-market robustness before final confirmation", flush=True)

    datasets = {}
    for symbol, timeframe, bars in [
        ("ETH/USDT", "1h", 5000),
        ("ETH/USDT", "4h", 3000),
        ("BTC/USDT", "4h", 3000),
        ("BTC/USDT", "1h", 5000),
        ("ETH/USDT", "15m", 5000),
    ]:
        datasets[(symbol, timeframe)] = adapter.fetch_ohlcv_history(
            symbol,
            timeframe,
            bars,
            page_limit=1500,
            market_type="spot",
        )
        print(
            f"Loaded {symbol} {timeframe}: {len(datasets[(symbol, timeframe)])} bars",
            flush=True,
        )

    candidates = _candidate_pool(seed=20260829, limit=limit)
    print(f"Generated: {len(candidates)} unique candidates", flush=True)

    scored = []
    for i, candidate in enumerate(candidates, 1):
        try:
            parts = [
                _frozen_eval(datasets[key], candidate, settings)
                for key in (
                    ("ETH/USDT", "1h"),
                    ("ETH/USDT", "4h"),
                    ("BTC/USDT", "4h"),
                )
            ]
            scored.append({
                "title": candidate.title,
                "family": candidate.family,
                "parameters": dict(candidate.params),
                "score": _cross_score(parts),
                "screen": parts,
            })
        except Exception as exc:
            print(f"Screen error {i}: {type(exc).__name__}: {exc}", flush=True)
        gc.collect()
        if i % 15 == 0 and scored:
            best = max(scored, key=lambda x: x["score"])
            print(f"Screened {i}/{limit} | best={best['score']:.2f}", flush=True)

    scored.sort(key=lambda x: x["score"], reverse=True)

    # Diversity filter: do not send near-identical candidates to confirmation.
    finalist_records = []
    seen = set()
    for record in scored:
        p = record["parameters"]
        signature = (
            record["family"],
            p["trend_fast"],
            p["trend_slow"],
            p["momentum_window"],
            p["breakout_window"],
            round(p["w_trend"], 2),
            round(p["w_momentum"], 2),
            round(p["w_breakout"], 2),
            round(p["w_candle"], 2),
            round(p["w_volume"], 2),
        )
        if signature in seen:
            continue
        seen.add(signature)
        finalist_records.append(record)
        if len(finalist_records) >= finalists:
            break

    print(f"\nFinalists for fresh confirmation: {len(finalist_records)}", flush=True)

    fresh_keys = (("BTC/USDT", "1h"), ("ETH/USDT", "15m"))
    confirmed = []

    for i, record in enumerate(finalist_records, 1):
        fresh = {}
        for key in fresh_keys:
            fresh[f"{key[0]} {key[1]}"] = _fresh_confirmation(
                datasets[key], record, settings
            )
        record["fresh_confirmation"] = fresh
        record["fresh_pass_count"] = sum(x["passed"] for x in fresh.values())
        confirmed.append(record)

        print(f"\n#{i} {record['title']}", flush=True)
        print(
            f"screen={record['score']:.2f} | "
            f"fresh={record['fresh_pass_count']}/{len(fresh_keys)}",
            flush=True,
        )
        for market, result in fresh.items():
            n = result["normal"]
            s = result["stress"]
            print(
                f"  {market}: Return={n['return']:.2%} PF={n['pf']:.2f} "
                f"DD={n['dd']:.2%} Trades={n['trades']} "
                f"Stress={s['return']:.2%} PASS={result['passed']}",
                flush=True,
            )
        gc.collect()

    confirmed.sort(
        key=lambda x: (x["fresh_pass_count"], x["score"]),
        reverse=True,
    )
    eligible = [x for x in confirmed if x["fresh_pass_count"] == len(fresh_keys)]

    decision = (
        "VALIDATED_ALGORITHMIC_STRATEGY"
        if eligible
        else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    )

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "generated": len(candidates),
        "screened": len(scored),
        "finalists": len(finalist_records),
        "validated_count": len(eligible),
        "winner": eligible[0] if eligible else (confirmed[0] if confirmed else None),
        "top_confirmed": confirmed,
    }

    out = ROOT / "experiments" / "multi_candidate_confirmation_latest.json"
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(eligible), flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
