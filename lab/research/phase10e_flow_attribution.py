from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .phase10d_spot_flow_multiday import run as run_multiday


@dataclass(frozen=True)
class AttributionRow:
    source: str
    target: str
    horizon: int
    raw_ic: float
    residual_ic: float
    reversal_ic: float
    observations: int


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _corr_adjust(raw_ic: float, beta_proxy: float) -> float:
    if not np.isfinite(raw_ic) or not np.isfinite(beta_proxy):
        return float("nan")
    return float(raw_ic - beta_proxy)


def run(minutes: float = 15.0, seed: int = 20260829, days: int = 14) -> dict[str, Any]:
    started = time.perf_counter()
    days = max(7, min(int(days), 14))

    print("=== PHASE 10E FLOW ATTRIBUTION ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"DAYS: {days}", flush=True)
    print("BASE ENGINE: PHASE 10D MULTI-DAY FLOW", flush=True)
    print("ATTRIBUTION: reversal + residual beta adjustment", flush=True)

    base = run_multiday(
        minutes=minutes,
        seed=seed,
        days=days,
        sample_symbols=6,
    )

    daily = base.get("daily", [])
    pair_edges = base.get("pair_edges", [])

    # The 10D engine already produces direction-reversed ICs. For attribution,
    # use the market-aggregate direct IC as the baseline and conservatively
    # treat the market beta adjustment as the median same-horizon pair effect.
    horizon_buckets: dict[int, list[float]] = {}
    for row in pair_edges:
        h = int(row.get("horizon", 0))
        ic = _safe_float(row.get("ic"))
        if np.isfinite(ic):
            horizon_buckets.setdefault(h, []).append(ic)

    median_pair_ic = {
        h: float(np.median(v)) for h, v in horizon_buckets.items() if v
    }

    direct: list[AttributionRow] = []
    symbols = sorted({str(r.get("symbol")) for r in daily if r.get("symbol")})

    for symbol in symbols:
        rows = [r for r in daily if r.get("symbol") == symbol]
        ic1 = _safe_float(np.nanmedian([_safe_float(r.get("ic_1h")) for r in rows]))
        ic6 = _safe_float(np.nanmedian([_safe_float(r.get("ic_6h")) for r in rows]))
        rev1 = -ic1 if np.isfinite(ic1) else float("nan")
        rev6 = -ic6 if np.isfinite(ic6) else float("nan")

        beta1 = median_pair_ic.get(60, 0.0)
        beta6 = median_pair_ic.get(360, 0.0)

        direct.append(
            AttributionRow(
                source=symbol,
                target=symbol,
                horizon=60,
                raw_ic=ic1,
                residual_ic=_corr_adjust(ic1, beta1),
                reversal_ic=rev1,
                observations=sum(int(r.get("trades", 0)) for r in rows),
            )
        )
        direct.append(
            AttributionRow(
                source=symbol,
                target=symbol,
                horizon=360,
                raw_ic=ic6,
                residual_ic=_corr_adjust(ic6, beta6),
                reversal_ic=rev6,
                observations=sum(int(r.get("trades", 0)) for r in rows),
            )
        )

    persistent = []
    for row in direct:
        if np.isfinite(row.reversal_ic) and abs(row.reversal_ic) >= 0.03:
            persistent.append(asdict(row))

    top_pairs = []
    for row in pair_edges[:30]:
        ic = _safe_float(row.get("ic"))
        if not np.isfinite(ic):
            continue
        h = int(row.get("horizon", 0))
        beta = median_pair_ic.get(h, 0.0)
        residual = _corr_adjust(ic, beta)
        top_pairs.append({
            "source": row.get("source"),
            "target": row.get("target"),
            "horizon": h,
            "raw_ic": ic,
            "residual_ic": residual,
            "reverse_ic": _safe_float(row.get("reverse_ic")),
            "observations": int(row.get("observations", 0)),
        })

    decision = (
        "PHASE10E_ATTRIBUTION_SIGNAL_FOUND"
        if persistent
        else "PHASE10E_NO_ATTRIBUTED_FLOW_EDGE"
    )

    out = Path("experiments/phase10e_flow")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "phase10e_flow_reversal_residual_latest.json"

    result: dict[str, Any] = {
        "version": "phase10e_flow_attribution",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "symbols": symbols,
        "base_decision": base.get("decision"),
        "direct_attribution": [asdict(x) for x in direct],
        "top_pairs": top_pairs,
        "persistent_candidates": persistent,
        "median_pair_ic": median_pair_ic,
        "decision": decision,
        "elapsed_sec": time.perf_counter() - started,
        "next": "Only persistent residual/reversal effects should enter the next validation gate.",
    }

    checkpoint.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")

    print("=== PHASE 10E COMPLETE ===", flush=True)
    print("DECISION:", decision, flush=True)
    print("SYMBOLS:", len(symbols), flush=True)
    print("PAIR EDGES:", len(pair_edges), flush=True)
    print("PERSISTENT:", len(persistent), flush=True)
    print("CHECKPOINT:", checkpoint, flush=True)
    print("ELAPSED_SEC:", round(time.perf_counter() - started, 2), flush=True)

    return result
