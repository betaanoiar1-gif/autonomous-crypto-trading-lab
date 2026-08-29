from __future__ import annotations

"""Resilient one-command refinement runner.

Uses the existing idea engine's signal/evaluation primitives but avoids its
broken refinement display path. Refines only the strongest trend_volatility
candidate, freezes the winner, and performs an independent BTC 4h check.
Research/paper-only; no orders are created.
"""

import json
from pathlib import Path

from ..config import ROOT, load_settings
from .run import _fetch_markets_cached
from .idea_engine import Idea, _evaluate_idea, _independent, _gate_reasons


BASE = Idea(
    "trend_volatility",
    {
        "fast": 30,
        "slow": 120,
        "vol_window": 24,
        "vol_floor": 0.007,
        "vol_cap": 0.035,
    },
)


def _ring() -> list[Idea]:
    p = BASE.params
    specs = [
        {**p, "fast": 25, "slow": 110},
        {**p, "fast": 28, "slow": 116},
        {**p, "fast": 32, "slow": 124},
        {**p, "fast": 35, "slow": 130},
        {**p, "vol_floor": 0.006},
        {**p, "vol_floor": 0.008},
        {**p, "vol_cap": 0.030},
        {**p, "vol_cap": 0.040},
    ]
    return [Idea("trend_volatility", x) for x in specs]


def _gate(record: dict) -> bool:
    return not _gate_reasons(record)


def run() -> dict:
    settings = load_settings()
    spot, _, _, _, _ = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]

    print("=== AUTONOMOUS REFINEMENT ===", flush=True)
    print("Base: trend_volatility 30/120 | ETH/USDT 1h", flush=True)
    print("Refinement: bounded ring of 8 | parameters frozen per test", flush=True)
    print("Futures: disabled | Live trading: disabled", flush=True)
    print()

    candidates: list[dict] = []
    ideas = [BASE] + _ring()

    for i, idea in enumerate(ideas, 1):
        try:
            record = _evaluate_idea(primary, idea, settings)
            record["gate_reasons"] = _gate_reasons(record)
            candidates.append(record)
            h = record["holdout"]
            print(
                f"[{i:02d}] {idea.name} {idea.params} | "
                f"OOS={h['total_return']:.2%} | PF={h['profit_factor']:.2f} | "
                f"DD={h['max_drawdown']:.2%} | trades={h['trade_count']} | "
                f"WF={record['positive_folds']}/4 | "
                f"GATE={'PASS' if not record['gate_reasons'] else 'FAIL:' + ','.join(record['gate_reasons'])}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{i:02d}] ERROR {type(exc).__name__}: {exc}", flush=True)

    if not candidates:
        raise RuntimeError("No refinement candidate was evaluated.")

    # Prefer candidates that actually pass the full primary gate; otherwise
    # retain the strongest score strictly for diagnosis, never promotion.
    passing = [x for x in candidates if _gate(x)]
    pool = passing if passing else candidates
    pool.sort(key=lambda x: x["score"], reverse=True)
    best = pool[0]
    frozen = Idea(best["idea"], best["parameters"])

    print()
    print("=== BEST REFINED CANDIDATE ===", flush=True)
    print(f"Idea: {frozen.name}", flush=True)
    print(f"Parameters: {frozen.params}", flush=True)
    print(
        f"OOS={best['holdout']['total_return']:.2%} | "
        f"PF={best['holdout']['profit_factor']:.2f} | "
        f"DD={best['holdout']['max_drawdown']:.2%} | "
        f"trades={best['holdout']['trade_count']} | "
        f"WF={best['positive_folds']}/4",
        flush=True,
    )
    print("Gate reasons:", best["gate_reasons"] or "NONE", flush=True)

    print()
    print("=== FROZEN INDEPENDENT TEST ===", flush=True)
    independent_result = _independent(independent, frozen, settings)
    print(
        f"BTC/USDT 4h | return={independent_result['total_return']:.2%} | "
        f"PF={independent_result['profit_factor']:.2f} | "
        f"DD={independent_result['max_drawdown']:.2%} | "
        f"trades={independent_result['trade_count']} | "
        f"Sharpe={independent_result['sharpe']:.2f} | "
        f"PASS={independent_result['passed']}",
        flush=True,
    )

    decision = "PROMOTE_TO_PAPER" if _gate(best) and independent_result["passed"] else "REJECT_REFINEMENT"

    output = {
        "base": {"idea": BASE.name, "parameters": BASE.params},
        "candidates": candidates,
        "best": best,
        "independent_result": independent_result,
        "decision": decision,
    }

    path = ROOT / "experiments" / "autonomous_refinement_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    print("=== FINAL DECISION ===", flush=True)
    print("Decision:", decision, flush=True)
    print("Saved:", path, flush=True)
    return output


if __name__ == "__main__":
    run()
