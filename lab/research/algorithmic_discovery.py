from __future__ import annotations

"""AI-free bounded strategy discovery.

Generates deterministic composite strategies from a safe executable DSL,
screens them on a validation slice, then performs frozen OOS/WF/stress and
independent-market confirmation. No LLM generation, no arbitrary code, no
futures, and no live trading.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd

from ..config import ROOT, load_settings
from .run import _fetch_markets_cached
from .evaluator import _run, _metrics


@dataclass(frozen=True)
class Candidate:
    family: str
    params: dict
    title: str


FAMILIES = ("trend_blend", "momentum_blend", "breakout_blend", "reversion_blend")


def _candidate_pool(seed: int, limit: int = 240) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    out: list[Candidate] = []
    seen: set[tuple] = set()

    fasts = [8, 12, 18, 24, 30, 36, 45, 60]
    slows = [40, 60, 80, 100, 120, 150, 180, 240]
    moms = [8, 12, 18, 24, 36, 48, 72, 96, 120]
    brks = [12, 20, 30, 40, 60, 90, 120]
    vols = [12, 18, 24, 36, 48]
    volume_windows = [12, 24, 36, 48, 72]
    floors = [0.002, 0.004, 0.006, 0.008, 0.010]
    caps = [0.018, 0.025, 0.030, 0.035, 0.045, 0.060]

    weight_templates = [
        (1.5, 0.5, 0.0, 0.0, 0.0),
        (1.0, 1.0, 0.5, 0.0, 0.5),
        (0.5, 1.5, 0.5, 0.0, 0.5),
        (1.0, 0.5, 1.0, 0.5, 0.5),
        (0.5, 1.0, 1.0, 0.5, 0.5),
        (0.0, 1.5, 1.0, 0.5, 1.0),
        (1.0, -0.5, 1.0, 0.5, 0.5),
        (0.5, 1.0, -0.5, 1.0, 0.5),
    ]

    threshold_sets = [
        (1.5, 1.5, 0.25),
        (2.0, 2.0, 0.50),
        (2.5, 2.5, 0.75),
        (3.0, 2.5, 0.50),
        (1.75, 2.25, 0.40),
        (2.25, 1.75, 0.40),
    ]

    # Seeded deterministic exploration. Each candidate is generated without
    # observing performance, so parameters are not fitted to the final OOS set.
    attempts = 0
    while len(out) < limit and attempts < limit * 20:
        attempts += 1
        family = FAMILIES[int(rng.integers(0, len(FAMILIES)))]
        fast = int(rng.choice(fasts))
        slow = int(rng.choice([x for x in slows if x > fast]))
        momentum = int(rng.choice(moms))
        breakout = int(rng.choice(brks))
        vol_window = int(rng.choice(vols))
        volume_window = int(rng.choice(volume_windows))
        floor = float(rng.choice(floors))
        cap = float(rng.choice([x for x in caps if x > floor]))
        weights = tuple(float(x) for x in weight_templates[int(rng.integers(0, len(weight_templates)))])
        long_threshold, short_threshold, exit_threshold = threshold_sets[int(rng.integers(0, len(threshold_sets)))]
        volume_mult = float(rng.choice([0.75, 0.90, 1.00, 1.10, 1.25, 1.40]))

        # Give each family a slightly different compositional bias.
        if family == "trend_blend":
            weights = (max(1.0, abs(weights[0]) + 0.5), weights[1], weights[2], weights[3], weights[4])
        elif family == "momentum_blend":
            weights = (weights[0], max(1.0, abs(weights[1]) + 0.5), weights[2], weights[3], weights[4])
        elif family == "breakout_blend":
            weights = (weights[0], weights[1], max(1.0, abs(weights[2]) + 0.5), weights[3], weights[4])
        else:
            weights = (weights[0], weights[1], weights[2], weights[3] + 0.75, weights[4])

        params = {
            "trend_fast": fast,
            "trend_slow": slow,
            "momentum_window": momentum,
            "breakout_window": breakout,
            "vol_window": vol_window,
            "volume_window": volume_window,
            "w_trend": round(weights[0], 3),
            "w_momentum": round(weights[1], 3),
            "w_breakout": round(weights[2], 3),
            "w_candle": round(weights[3], 3),
            "w_volume": round(weights[4], 3),
            "long_threshold": float(long_threshold),
            "short_threshold": float(short_threshold),
            "exit_threshold": float(exit_threshold),
            "vol_floor": round(floor, 4),
            "vol_cap": round(cap, 4),
            "volume_mult": round(volume_mult, 2),
        }
        key = (family, tuple(sorted(params.items())))
        if key in seen:
            continue
        seen.add(key)
        title = family.replace("_", " ").title() + " | " + ", ".join(f"{k}={v}" for k, v in params.items())
        out.append(Candidate("invented_composite", params, title))

    return out


def _score(m: dict) -> float:
    ret = float(m.get("total_return", 0.0))
    pf = min(3.0, float(m.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    trades = int(m.get("trade_count", 0))
    sharpe = float(m.get("sharpe", 0.0))
    trade_bonus = min(8.0, math.log1p(max(trades, 0)))
    return 100.0 * ret + 15.0 * (pf - 1.0) + 3.0 * sharpe - 25.0 * dd + trade_bonus


def _frozen_eval(df: pd.DataFrame, candidate: Candidate, settings: object) -> dict:
    n = len(df)
    train_end = int(n * 0.55)
    valid_end = int(n * 0.70)
    train = df.iloc[:train_end]
    valid = df.iloc[train_end:valid_end]
    holdout = df.iloc[valid_end:]

    p = dict(candidate.params)
    directions = ["both"]
    capital = settings.capital.initial_usd
    fee = settings.execution.commission_bps
    slip = settings.execution.slippage_bps

    # Fast selection slice: candidate is already frozen; this is only ranking.
    vr = _run(valid, candidate.family, p, directions, capital, fee, slip)
    vm = _metrics(vr, vr.returns)

    # Final holdout: parameters remain frozen.
    hr = _run(holdout, candidate.family, p, directions, capital, fee, slip)
    hm = _metrics(hr, hr.returns)

    # Frozen walk-forward on the holdout itself: no retuning.
    folds = []
    block = max(40, len(holdout) // 4)
    for i in range(4):
        a = i * block
        b = (i + 1) * block if i < 3 else len(holdout)
        if b <= a:
            continue
        fr = _run(holdout.iloc[a:b], candidate.family, p, directions, capital, fee, slip)
        fm = _metrics(fr, fr.returns)
        folds.append({
            "fold": i + 1,
            "return": float(fm["total_return"]),
            "profit_factor": float(fm["profit_factor"]),
            "max_drawdown": float(fm["max_drawdown"]),
            "trade_count": int(fm["trade_count"]),
            "sharpe": float(fm["sharpe"]),
        })

    returns = [x["return"] for x in folds]
    pfs = [x["profit_factor"] for x in folds]
    trades = [x["trade_count"] for x in folds]
    positive = sum(x > 0 for x in returns)
    wf_median = float(np.median(returns)) if returns else 0.0
    wf_pf = float(np.median(pfs)) if pfs else 0.0
    wf_min_trades = min(trades) if trades else 0
    wf_pass = bool(len(folds) == 4 and positive >= 3 and wf_median > 0 and wf_pf > 1 and wf_min_trades >= 4)

    stress = _run(holdout, candidate.family, p, directions, capital, fee * 2, slip * 2)
    sm = _metrics(stress, stress.returns)

    primary_pass = bool(
        float(hm["total_return"]) > 0
        and float(hm["profit_factor"]) > 1
        and float(hm["max_drawdown"]) >= -0.50
        and int(hm["trade_count"]) >= 8
        and float(sm["total_return"]) > 0
        and wf_pass
    )

    result = {
        "title": candidate.title,
        "family": candidate.family,
        "parameters": p,
        "validation": {"metrics": vm},
        "holdout": {k: float(hm[k]) if k not in {"trade_count"} else int(hm[k]) for k in ("total_return", "profit_factor", "max_drawdown", "sharpe", "trade_count")},
        "stress": {"total_return": float(sm["total_return"]), "profit_factor": float(sm["profit_factor"]), "max_drawdown": float(sm["max_drawdown"]), "trade_count": int(sm["trade_count"])},
        "walk_forward": {"folds": folds, "positive_folds": positive, "median_return": wf_median, "median_pf": wf_pf, "min_trade_count": wf_min_trades, "passed": wf_pass},
        "primary_pass": primary_pass,
        "rank_score": _score(hm) + 75.0 * wf_median + 10.0 * (1 if wf_pass else 0),
    }
    del train, valid, holdout, vr, vm, hr, hm, stress, sm, folds
    gc.collect()
    return result


def _confirmation(df: pd.DataFrame, record: dict, settings: object) -> dict:
    result = _run(
        df,
        record["family"],
        dict(record["parameters"]),
        ["both"],
        settings.capital.initial_usd,
        settings.execution.commission_bps,
        settings.execution.slippage_bps,
    )
    m = _metrics(result, result.returns)
    passed = bool(
        float(m["total_return"]) > 0
        and float(m["profit_factor"]) > 1
        and float(m["max_drawdown"]) >= -0.50
        and int(m["trade_count"]) >= 8
    )
    return {
        "metrics": {k: float(m[k]) if k != "trade_count" else int(m[k]) for k in ("total_return", "profit_factor", "max_drawdown", "sharpe", "trade_count")},
        "passed": passed,
        "reasons": [] if passed else [
            r for r, bad in (
                ("non_positive_return", float(m["total_return"]) <= 0),
                ("profit_factor_le_1", float(m["profit_factor"]) <= 1),
                ("drawdown_over_50pct", float(m["max_drawdown"]) < -0.50),
                ("too_few_trades", int(m["trade_count"]) < 8),
            ) if bad
        ],
    }


def run(hours: float = 3.0, pool_size: int = 240, finalists: int = 12) -> dict:
    settings = load_settings()
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + max(0.05, float(hours)) * 3600.0
    ledger_path = ROOT / "experiments" / "algorithmic_discovery_latest.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    spot, _, _, _, snapshot = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]

    history_path = ROOT / "experiments" / "algorithmic_discovery_history.jsonl"
    seen: set[tuple] = set()
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines()[-5000:]:
            try:
                rec = json.loads(line)
                seen.add((rec.get("family"), tuple(sorted((rec.get("parameters") or {}).items()))))
            except json.JSONDecodeError:
                continue

    all_finalists: list[dict] = []
    batches = 0
    seed = int(started.timestamp())

    print("=== AI-FREE ALGORITHMIC DISCOVERY ===", flush=True)
    print("Generator: deterministic stochastic grammar | LLM: DISABLED | Futures: DISABLED | Live trading: DISABLED", flush=True)
    print("Primary: ETH/USDT 1h | Independent: BTC/USDT 4h", flush=True)
    print(f"Candidate pool per batch: {pool_size} | finalists per batch: {finalists}", flush=True)
    print("Parameters are generated before validation; final OOS and confirmation are frozen.", flush=True)

    while time.monotonic() < deadline:
        batches += 1
        candidates = [c for c in _candidate_pool(seed + batches, pool_size) if (c.family, tuple(sorted(c.params.items()))) not in seen]
        if not candidates:
            continue

        ranked: list[dict] = []
        for idx, cand in enumerate(candidates, 1):
            if time.monotonic() >= deadline:
                break
            try:
                rec = _frozen_eval(primary, cand, settings)
                ranked.append(rec)
                seen.add((cand.family, tuple(sorted(cand.params.items()))))
                if idx % 25 == 0:
                    print(f"Screened {idx}/{len(candidates)} | batch={batches}", flush=True)
            except Exception as exc:
                print(f"Candidate error: {type(exc).__name__}: {exc}", flush=True)
            gc.collect()

        ranked.sort(key=lambda r: r["rank_score"], reverse=True)
        top = ranked[:finalists]
        all_finalists.extend(top)
        all_finalists.sort(key=lambda r: r["rank_score"], reverse=True)
        all_finalists = all_finalists[:20]

        # Independent confirmation only for primary-pass finalists.
        for rec in list(all_finalists):
            if not rec.get("primary_pass") or rec.get("confirmation") is not None:
                continue
            conf = _confirmation(independent, rec, settings)
            rec["confirmation"] = {**conf, "market": "BTC/USDT", "timeframe": "4h"}
            rec["validated"] = bool(rec["primary_pass"] and conf["passed"])
            rec["status"] = "VALIDATED" if rec["validated"] else "REJECTED"
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            gc.collect()
            if rec["validated"]:
                final = {
                    "started_at": started.isoformat(),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "duration_hours": (datetime.now(timezone.utc) - started).total_seconds() / 3600.0,
                    "decision": "VALIDATED_ALGORITHMIC_STRATEGY",
                    "generator": "ai_free_algorithmic_grammar",
                    "primary_market": "ETH/USDT@1h",
                    "independent_market": "BTC/USDT@4h",
                    "candidate_batches": batches,
                    "validated": [rec],
                    "top": all_finalists[:10],
                    "snapshot": snapshot,
                }
                ledger_path.write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                print("=== VALIDATED ALGORITHMIC STRATEGY FOUND ===", flush=True)
                print(json.dumps(rec, indent=2, ensure_ascii=False, default=str), flush=True)
                return final

        ledger = {
            "started_at": started.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "deadline_seconds_remaining": max(0.0, deadline - time.monotonic()),
            "decision": "RUNNING",
            "generator": "ai_free_algorithmic_grammar",
            "candidate_batches": batches,
            "pool_size": pool_size,
            "screened_top": all_finalists[:10],
            "snapshot": snapshot,
        }
        ledger_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"=== BATCH {batches} COMPLETE | top_score={all_finalists[0]['rank_score'] if all_finalists else 0:.2f} ===", flush=True)

    final = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_hours": (datetime.now(timezone.utc) - started).total_seconds() / 3600.0,
        "decision": "NO_VALIDATED_ALGORITHMIC_STRATEGY",
        "generator": "ai_free_algorithmic_grammar",
        "candidate_batches": batches,
        "screened_count": len(seen),
        "top": all_finalists[:10],
        "snapshot": snapshot,
    }
    ledger_path.write_text(json.dumps(final, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("=== ALGORITHMIC DISCOVERY FINISHED ===", flush=True)
    print("Decision:", final["decision"], flush=True)
    print("Batches:", batches, "| Unique candidates seen:", len(seen), flush=True)
    print("Checkpoint:", ledger_path, flush=True)
    return final


if __name__ == "__main__":
    run(hours=float(os.getenv("ACL_DISCOVERY_HOURS", "3")))
