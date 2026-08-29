from __future__ import annotations

"""Analyze the latest autonomous loop without rerunning research."""

import json
from pathlib import Path
from collections import defaultdict

from ..config import ROOT

LEDGER = ROOT / "experiments" / "autonomous_loop_latest.json"
OUT = ROOT / "experiments" / "autonomous_postmortem_latest.json"


def _candidate_score(r: dict) -> float:
    oos = r.get("out_of_sample", {}) or {}
    rob = r.get("robustness", {}) or {}
    wf = rob.get("walk_forward", {}) or {}
    ret = float(oos.get("total_return", 0.0))
    pf = float(oos.get("profit_factor", 0.0))
    dd = float(oos.get("max_drawdown", 0.0))
    trades = int(oos.get("trade_count", 0))
    positive = int(wf.get("positive_windows", 0))
    folds = len(wf.get("windows", []))
    stress = float(rob.get("stressed_total_return", 0.0))
    return (
        ret * 100.0
        + (pf - 1.0) * 30.0
        + stress * 40.0
        + positive * 4.0
        + (2.0 if folds and positive >= 3 else 0.0)
        - max(abs(dd) - 0.40, 0.0) * 30.0
        + min(trades, 40) * 0.05
    )


def _reasons(r: dict) -> list[str]:
    oos = r.get("out_of_sample", {}) or {}
    rob = r.get("robustness", {}) or {}
    wf = rob.get("walk_forward", {}) or {}
    reasons = []
    if float(oos.get("total_return", 0.0)) <= 0:
        reasons.append("oos_non_positive")
    if float(oos.get("profit_factor", 0.0)) <= 1:
        reasons.append("oos_pf_le_1")
    if float(oos.get("max_drawdown", 0.0)) < -0.50:
        reasons.append("oos_dd_over_50pct")
    if int(oos.get("trade_count", 0)) < 8:
        reasons.append("oos_too_few_trades")
    if not bool(wf.get("passed", False)):
        reasons.append("walk_forward_failed")
    if float(rob.get("stressed_total_return", 0.0)) <= 0:
        reasons.append("stress_failed")
    if not bool(rob.get("parameter_stability", True)):
        reasons.append("parameter_stability_failed")
    return reasons


def run(top_n: int = 8) -> dict:
    if not LEDGER.exists():
        raise FileNotFoundError(f"Missing ledger: {LEDGER}")

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    candidates = []

    for cycle in ledger.get("cycles", []):
        for r in cycle.get("leaderboard", []) or []:
            # Leaderboard entries are compact and sufficient for first-pass ranking.
            status = r.get("status", "")
            if status == "REJECTED":
                candidates.append({
                    "cycle": cycle.get("cycle"),
                    "run_id": cycle.get("run_id"),
                    "title": r.get("title"),
                    "symbol": r.get("symbol"),
                    "timeframe": r.get("timeframe"),
                    "market_type": r.get("market_type"),
                    "score": float(r.get("score", 0.0)),
                    "walk_forward_passed": bool(r.get("walk_forward_passed", False)),
                    "confirmation_passed": bool(r.get("confirmation_passed", False)),
                })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Aggregate by strategy title to reveal recurring near-misses.
    by_title = defaultdict(list)
    for c in candidates:
        by_title[c["title"]].append(c)

    recurring = []
    for title, items in by_title.items():
        recurring.append({
            "title": title,
            "occurrences": len(items),
            "best_score": max(x["score"] for x in items),
            "best": max(items, key=lambda x: x["score"]),
        })
    recurring.sort(key=lambda x: (x["occurrences"], x["best_score"]), reverse=True)

    result = {
        "source": str(LEDGER),
        "cycles_completed": ledger.get("cycles_completed", len(ledger.get("cycles", []))),
        "decision": ledger.get("decision"),
        "top_near_misses": candidates[:top_n],
        "recurring_near_misses": recurring[:top_n],
        "recommendation": (
            "Do not rerun the broad grid. Re-test only recurring near-misses using frozen parameters,"
            " independent symbols/timeframes, and unchanged validation gates."
        ),
    }

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("=== AUTONOMOUS POSTMORTEM ===")
    print(f"Cycles analyzed: {result['cycles_completed']}")
    print(f"Previous decision: {result['decision']}")
    print()
    print("=== TOP NEAR-MISSES ===")
    for i, c in enumerate(result["top_near_misses"], 1):
        print(
            f"#{i} {c['title']} | {c['symbol']} | {c['timeframe']} | "
            f"score={c['score']:.2f} | WF={c['walk_forward_passed']} | CONF={c['confirmation_passed']}"
        )
    print()
    print("=== RECURRING IDEAS ===")
    for i, c in enumerate(result["recurring_near_misses"], 1):
        print(
            f"#{i} occurrences={c['occurrences']} | best_score={c['best_score']:.2f} | {c['title']}"
        )
    print()
    print(result["recommendation"])
    print(f"Saved: {OUT}")
    return result


if __name__ == "__main__":
    run()
