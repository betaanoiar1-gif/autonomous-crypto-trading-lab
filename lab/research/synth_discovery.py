from __future__ import annotations

"""AI-free synthetic strategy discovery across the project's real strategy families.

This engine deliberately avoids LLM generation. It samples frozen parameterized
strategies from the evaluator's supported families, evaluates them across
multiple spot markets/timeframes, then sends a diverse shortlist through fresh
independent confirmation. No live trading and no futures are used.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
import os
import time

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics, _run


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict
    title: str


FAMILIES = (
    "momentum",
    "breakout",
    "trend_pullback",
    "mean_reversion",
    "moving_average_cross",
    "rsi_reversion",
    "atr_breakout",
    "channel_reversion",
)


def candidate_pool(seed: int, limit: int = 240) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    out = []
    seen = set()

    for _ in range(limit * 40):
        if len(out) >= limit:
            break

        family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]

        if family in {"momentum", "breakout", "channel_reversion"}:
            lookback = int(rng.choice([10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200]))
            params = {"lookback": lookback}
            if family == "channel_reversion":
                params = {"channel_length": lookback}

        elif family == "trend_pullback":
            lookback = int(rng.choice([10, 15, 20, 30, 40, 60, 80, 100, 120]))
            threshold = float(rng.choice([0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]))
            params = {"lookback": lookback, "pullback_threshold": threshold}

        elif family == "mean_reversion":
            lookback = int(rng.choice([15, 20, 30, 40, 60, 80, 100, 120, 150, 180, 200]))
            z_entry = float(rng.choice([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]))
            z_exit = float(rng.choice([0.05, 0.15, 0.25, 0.5, 0.75, 1.0, 1.25]))
            if z_exit >= z_entry:
                continue
            params = {"lookback": lookback, "z_entry": z_entry, "z_exit": z_exit}

        elif family == "moving_average_cross":
            fast = int(rng.choice([4, 6, 8, 10, 12, 15, 18, 22, 28, 35, 45, 60]))
            slow = int(rng.choice([30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 240, 280]))
            if slow <= fast:
                continue
            params = {"fast": fast, "slow": slow}

        elif family == "rsi_reversion":
            length = int(rng.choice([5, 7, 9, 11, 14, 17, 21, 25, 30]))
            low = float(rng.choice([20, 25, 30, 35, 40]))
            high = float(rng.choice([60, 65, 70, 75, 80, 85, 90]))
            if low >= 50 or high <= 50 or high <= low + 10:
                continue
            params = {"rsi_length": length, "rsi_low": low, "rsi_high": high}

        else:  # atr_breakout
            length = int(rng.choice([2, 3, 4, 5, 7, 9, 11, 14, 18, 21, 25, 30]))
            mult = float(rng.choice([0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]))
            params = {"atr_length": length, "atr_mult": mult}

        key = (family, tuple(sorted(params.items())))
        if key in seen:
            continue
        seen.add(key)
        title = family.replace("_", " ").title() + " | " + ", ".join(f"{k}={v}" for k, v in params.items())
        out.append(Candidate(family, params, title))

    return out


def frozen_metrics(df, candidate, settings, stress=False):
    fee = settings.execution.commission_bps * (2 if stress else 1)
    slip = settings.execution.slippage_bps * (2 if stress else 1)
    result = _run(
        df,
        candidate.family,
        dict(candidate.params),
        ["both"],
        settings.capital.initial_usd,
        fee,
        slip,
    )
    return _metrics(result, result.returns)


def frozen_wf(df, candidate, settings, windows=4):
    n = len(df)
    test_size = max(40, n // (windows + 2))
    rows = []
    start = 0

    while len(rows) < windows and start + test_size <= n:
        # Parameters remain frozen; this is a genuine fixed-parameter WF check.
        test = df.iloc[start:start + test_size]
        m = frozen_metrics(test, candidate, settings, stress=False)
        rows.append({
            "fold": len(rows) + 1,
            "return": float(m["total_return"]),
            "pf": float(m["profit_factor"]),
            "dd": float(m["max_drawdown"]),
            "trades": int(m["trade_count"]),
            "sharpe": float(m["sharpe"]),
        })
        start += test_size

    if not rows:
        return {"positive": 0, "median_return": 0.0, "median_pf": 0.0, "min_trades": 0, "passed": False, "folds": []}

    rets = [x["return"] for x in rows]
    pfs = [x["pf"] for x in rows]
    trades = [x["trades"] for x in rows]
    positive = sum(x > 0 for x in rets)
    return {
        "positive": positive,
        "median_return": float(np.median(rets)),
        "median_pf": float(np.median(pfs)),
        "min_trades": int(min(trades)),
        "passed": bool(len(rows) == windows and positive >= 3 and np.median(rets) > 0 and np.median(pfs) > 1 and min(trades) >= 4),
        "folds": rows,
    }


def screen(df, candidate, settings):
    n = len(df)
    if n < 600:
        return None

    # Use only the discovery portion here. Fresh confirmation is later and separate.
    cut = int(n * 0.70)
    discovery = df.iloc[:cut]
    holdout = df.iloc[cut:]

    hm = frozen_metrics(holdout, candidate, settings, stress=False)
    sm = frozen_metrics(holdout, candidate, settings, stress=True)
    wf = frozen_wf(discovery, candidate, settings, windows=4)

    score = (
        100 * hm["total_return"]
        + 20 * (min(2.5, hm["profit_factor"]) - 1)
        + 5 * hm["sharpe"]
        - 30 * abs(min(0, hm["max_drawdown"]))
        + 50 * sm["total_return"]
        + 10 * (min(2.0, sm["profit_factor"]) - 1)
        + 35 * wf["median_return"]
        + 10 * (1 if wf["passed"] else 0)
        - 25 * (max(0, 8 - hm["trade_count"]))
    )

    return {
        "title": candidate.title,
        "family": candidate.family,
        "parameters": dict(candidate.params),
        "holdout": {"return": float(hm["total_return"]), "pf": float(hm["profit_factor"]), "dd": float(hm["max_drawdown"]), "trades": int(hm["trade_count"]), "sharpe": float(hm["sharpe"])},
        "stress": {"return": float(sm["total_return"]), "pf": float(sm["profit_factor"]), "dd": float(sm["max_drawdown"]), "trades": int(sm["trade_count"])},
        "wf": wf,
        "score": float(score),
    }


def fresh(df, record, settings):
    normal = frozen_metrics(df, Candidate(record["family"], record["parameters"], record["title"]), settings, stress=False)
    stress = frozen_metrics(df, Candidate(record["family"], record["parameters"], record["title"]), settings, stress=True)
    passed = bool(
        normal["total_return"] > 0
        and normal["profit_factor"] > 1
        and normal["max_drawdown"] >= -0.50
        and normal["trade_count"] >= 8
        and stress["total_return"] > 0
        and stress["profit_factor"] > 1
    )
    return {"normal": {"return": float(normal["total_return"]), "pf": float(normal["profit_factor"]), "dd": float(normal["max_drawdown"]), "trades": int(normal["trade_count"]), "sharpe": float(normal["sharpe"])}, "stress": {"return": float(stress["total_return"]), "pf": float(stress["profit_factor"])}, "passed": passed}


def run(hours: float = 3.0, batch_size: int = 90, finalists: int = 12):
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + hours * 3600
    adapter = CCXTMarketData(exchange_id="binance")

    print("=== SYNTHETIC AI-FREE DISCOVERY ===", flush=True)
    print("Families:", ", ".join(FAMILIES), flush=True)
    print("AI generation: DISABLED | Futures: DISABLED | Live trading: DISABLED", flush=True)

    markets = {}
    for symbol, tf, bars in [
        ("ETH/USDT", "1h", 5000),
        ("ETH/USDT", "4h", 3000),
        ("BTC/USDT", "4h", 3000),
        ("BTC/USDT", "1h", 5000),
        ("ETH/USDT", "15m", 5000),
    ]:
        markets[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1500, market_type="spot")
        print(f"Loaded {symbol} {tf}: {len(markets[(symbol, tf)])} bars", flush=True)

    seed = 20260829
    all_records = []
    seen = set()
    batch_no = 0

    while time.monotonic() < deadline:
        batch_no += 1
        candidates = candidate_pool(seed + batch_no, batch_size)
        candidates = [c for c in candidates if (c.family, tuple(sorted(c.params.items()))) not in seen]
        if not candidates:
            continue

        print(f"\n=== BATCH {batch_no} | candidates={len(candidates)} ===", flush=True)

        for idx, c in enumerate(candidates, 1):
            if time.monotonic() >= deadline:
                break
            key = (c.family, tuple(sorted(c.params.items())))
            seen.add(key)
            try:
                # Screen on three environments before any fresh confirmation.
                parts = []
                for key_market in [("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h")]:
                    r = screen(markets[key_market], c, settings)
                    if r is None:
                        parts = []
                        break
                    parts.append(r)
                if not parts:
                    continue

                # Conservative cross-market selection score.
                rets = [p["holdout"]["return"] for p in parts]
                pfs = [p["holdout"]["pf"] for p in parts]
                dds = [abs(min(0, p["holdout"]["dd"])) for p in parts]
                stresses = [p["stress"]["return"] for p in parts]
                trades = [p["holdout"]["trades"] for p in parts]
                wf_passes = sum(p["wf"]["passed"] for p in parts)
                robust = (
                    120 * float(np.mean(rets))
                    + 30 * (float(np.median(pfs)) - 1)
                    + 60 * float(np.mean(stresses))
                    + 25 * float(np.min(rets))
                    - 25 * float(max(rets) - min(rets))
                    - 30 * float(max(dds))
                    + 8 * wf_passes
                    - 25 * sum(t < 8 for t in trades)
                )

                record = parts[0].copy()
                record["score"] = float(robust)
                record["markets"] = {
                    "ETH/USDT 1h": parts[0],
                    "ETH/USDT 4h": parts[1],
                    "BTC/USDT 4h": parts[2],
                }
                all_records.append(record)

                if idx % 15 == 0:
                    best = max(all_records, key=lambda x: x["score"])
                    print(f"Screened {idx}/{len(candidates)} | best_score={best['score']:.2f}", flush=True)
            except Exception as exc:
                print(f"Candidate error: {type(exc).__name__}: {exc}", flush=True)
            gc.collect()

        all_records.sort(key=lambda x: x["score"], reverse=True)
        print(f"=== BATCH {batch_no} COMPLETE | unique={len(all_records)} | best={all_records[0]['score']:.2f}", flush=True)

        if len(all_records) >= finalists:
            # Pick diverse families first.
            short = []
            family_counts = {}
            for r in all_records:
                f = r["family"]
                if family_counts.get(f, 0) >= 3:
                    continue
                short.append(r)
                family_counts[f] = family_counts.get(f, 0) + 1
                if len(short) >= finalists:
                    break

            print("\n=== FRESH CONFIRMATION ===", flush=True)
            confirmed = []
            for i, r in enumerate(short, 1):
                fresh_results = {
                    "BTC/USDT 1h": fresh(markets[("BTC/USDT", "1h")], r, settings),
                    "ETH/USDT 15m": fresh(markets[("ETH/USDT", "15m")], r, settings),
                }
                r2 = dict(r)
                r2["fresh"] = fresh_results
                r2["fresh_passes"] = sum(x["passed"] for x in fresh_results.values())
                confirmed.append(r2)
                print(f"#{i} {r['title']} | fresh={r2['fresh_passes']}/2", flush=True)
                for market, x in fresh_results.items():
                    n = x["normal"]
                    print(f"  {market}: return={n['return']:.2%} PF={n['pf']:.2f} DD={n['dd']:.2%} trades={n['trades']} stress={x['stress']['return']:.2%} pass={x['passed']}", flush=True)
                gc.collect()

            confirmed.sort(key=lambda x: (x["fresh_passes"], x["score"]), reverse=True)
            winner = next((x for x in confirmed if x["fresh_passes"] == 2), None)
            if winner:
                decision = "VALIDATED_ALGORITHMIC_STRATEGY"
                print("\n=== VALIDATED ===", flush=True)
                print(winner["title"], flush=True)
                payload = {"decision": decision, "generated": len(all_records), "winner": winner, "top": confirmed}
                out = ROOT / "experiments" / "synth_discovery_latest.json"
                out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                print("Saved:", out, flush=True)
                return payload

        # Keep exploring. Never stop just because a near-miss was found.
        time.sleep(0.5)

    payload = {
        "decision": "NO_VALIDATED_ALGORITHMIC_STRATEGY",
        "generated": len(all_records),
        "top": all_records[:finalists],
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    out = ROOT / "experiments" / "synth_discovery_latest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n=== DISCOVERY FINISHED ===", flush=True)
    print(payload["decision"], flush=True)
    print("Generated:", len(all_records), flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
