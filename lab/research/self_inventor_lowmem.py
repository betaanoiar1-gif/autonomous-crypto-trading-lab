from __future__ import annotations

import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ROOT, load_settings
from ..local_agent import LocalAgent
from .evaluator import evaluate, frozen_confirmation
from .run import _fetch_markets_cached
from .json_utils import extract_json_object


# Hard bounds: deliberately small to keep Colab RAM predictable.
BATCH_SIZE = 4
MAX_PRIMARY_CANDIDATES = 4
MIN_TRADES = 8


def _hash_spec(spec: dict[str, Any]) -> str:
    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str).encode()
    import hashlib
    return hashlib.sha256(raw).hexdigest()[:16]


def _int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _normalize(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    p = dict(raw.get("parameters") or {})
    p["trend_fast"] = _int(p.get("trend_fast"), 3, 80, 18)
    p["trend_slow"] = _int(p.get("trend_slow"), p["trend_fast"] + 1, 240, 90)
    p["trend_slow"] = max(p["trend_fast"] + 1, p["trend_slow"])
    for k, lo, hi, d in [
        ("momentum_window", 2, 160, 24),
        ("breakout_window", 3, 160, 36),
        ("vol_window", 8, 72, 24),
        ("volume_window", 8, 72, 24),
    ]:
        p[k] = _int(p.get(k), lo, hi, d)
    for k, lo, hi, d in [
        ("w_trend", -3, 3, 1.0),
        ("w_momentum", -3, 3, 1.0),
        ("w_breakout", -3, 3, 0.75),
        ("w_candle", -3, 3, 0.5),
        ("w_volume", -3, 3, 0.5),
        ("long_threshold", 0.5, 5.0, 1.75),
        ("short_threshold", 0.5, 5.0, 1.75),
        ("exit_threshold", 0.05, 1.5, 0.4),
        ("vol_floor", 0.0, 0.08, 0.005),
        ("vol_cap", 0.005, 0.15, 0.05),
        ("volume_mult", 0.6, 2.0, 1.0),
    ]:
        p[k] = round(_float(p.get(k), lo, hi, d), 6)
    if p["vol_cap"] <= p["vol_floor"]:
        p["vol_cap"] = min(0.15, p["vol_floor"] + 0.01)
    return {
        "title": str(raw.get("title") or f"Invention {idx}")[:120],
        "thesis": str(raw.get("thesis") or "Novel composition")[:1000],
        "parameters": p,
        "falsifiers": list(raw.get("falsifiers") or [])[:6],
        "candidate_id": f"INV-{_hash_spec(p)}",
    }


def _release_agent(agent: LocalAgent | None) -> None:
    if agent is None:
        return
    model = getattr(agent, "model", None)
    tok = getattr(agent, "tokenizer", None)
    try:
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
    finally:
        del model, tok, agent
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _invent(batch: int) -> list[dict[str, Any]]:
    agent: LocalAgent | None = None
    try:
        agent = LocalAgent(max_new_tokens=384, temperature=0.0)
        prompt = f"""
Invent exactly {batch} genuinely different crypto trading mechanisms.
Do not copy named textbook strategies and do not use performance claims.
Use only these executable primitives: trend relationship, windowed momentum,
prior-range position, candle-body pressure, relative volume, realized-volatility
regime, weighted score, asymmetric thresholds, and neutral/exit state.
Each result must be a NEW composition, not a parameter tweak of a known strategy.
Return ONLY JSON in this shape:
{{"strategies":[{{"title":"...","thesis":"...","parameters":{{
"trend_fast":18,"trend_slow":90,"momentum_window":24,"breakout_window":36,
"vol_window":24,"volume_window":24,"w_trend":1.0,"w_momentum":1.0,
"w_breakout":0.75,"w_candle":0.5,"w_volume":0.5,"long_threshold":1.75,
"short_threshold":1.75,"exit_threshold":0.4,"vol_floor":0.005,
"vol_cap":0.05,"volume_mult":1.0}},"falsifiers":["..."]}}]}}
""".strip()
        raw = agent.chat(prompt)
        obj = extract_json_object(raw)
        arr = obj.get("strategies", [])
        return arr if isinstance(arr, list) else []
    except Exception as exc:
        print(f"Inventor fallback: {type(exc).__name__}: {exc}")
        return []
    finally:
        _release_agent(agent)


def _gate(ev: Any) -> tuple[bool, list[str]]:
    h = ev.out_of_sample
    wf = ev.robustness.get("walk_forward", {})
    reasons: list[str] = []
    if int(h.get("trade_count", 0)) < MIN_TRADES:
        reasons.append("oos_too_few_trades")
    if float(h.get("total_return", 0.0)) <= 0:
        reasons.append("non_positive_oos_return")
    if float(h.get("profit_factor", 0.0)) <= 1:
        reasons.append("oos_pf_le_1")
    if float(h.get("max_drawdown", 0.0)) < -0.50:
        reasons.append("oos_drawdown_over_50pct")
    if not bool(ev.robustness.get("parameter_stability", True)):
        reasons.append("parameter_stability_failed")
    if float(ev.robustness.get("stressed_total_return", 0.0)) <= 0:
        reasons.append("doubled_cost_stress_failed")
    if not bool(wf.get("passed", False)):
        reasons.append("walk_forward_failed")
    return not reasons, reasons


def _slim(spec: dict[str, Any], ev: Any, primary_ok: bool, reasons: list[str], conf: dict[str, Any] | None) -> dict[str, Any]:
    h = ev.out_of_sample
    w = ev.robustness.get("walk_forward", {})
    c = conf or {}
    cm = c.get("metrics", {})
    return {
        "candidate_id": spec["candidate_id"],
        "title": spec["title"],
        "thesis": spec["thesis"],
        "parameters": dict(spec["parameters"]),
        "holdout": {
            "total_return": float(h.get("total_return", 0.0)),
            "profit_factor": float(h.get("profit_factor", 0.0)),
            "max_drawdown": float(h.get("max_drawdown", 0.0)),
            "trade_count": int(h.get("trade_count", 0)),
            "sharpe": float(h.get("sharpe", 0.0)),
            "research_score": float(h.get("research_score", 0.0)),
        },
        "walk_forward": {
            "positive_windows": int(w.get("positive_windows", 0)),
            "median_return": float(w.get("median_return", 0.0)),
            "worst_return": float(w.get("worst_return", 0.0)),
            "median_sharpe": float(w.get("median_sharpe", 0.0)),
            "min_trade_count": int(w.get("min_trade_count", 0)),
            "passed": bool(w.get("passed", False)),
        },
        "stressed_total_return": float(ev.robustness.get("stressed_total_return", 0.0)),
        "primary_gate": primary_ok,
        "primary_reasons": reasons,
        "confirmation": {
            "passed": bool(c.get("passed", False)),
            "total_return": float(cm.get("total_return", 0.0)),
            "profit_factor": float(cm.get("profit_factor", 0.0)),
            "max_drawdown": float(cm.get("max_drawdown", 0.0)),
            "trade_count": int(cm.get("trade_count", 0)),
            "sharpe": float(cm.get("sharpe", 0.0)),
            "rejection_reasons": list(c.get("rejection_reasons", [])),
        },
        "status": "VALIDATED" if primary_ok and bool(c.get("passed", False)) else "REJECTED",
    }


def run(count: int = BATCH_SIZE) -> dict[str, Any]:
    settings = load_settings()
    count = max(2, min(BATCH_SIZE, int(count)))
    print("=== LOW-MEMORY SELF INVENTOR ===")
    print(f"New inventions per batch: {count}")
    print("Futures: DISABLED | Live trading: DISABLED | Arbitrary code: DISABLED")

    # Load historical market data once, then release everything except the two frames used.
    spot, _, _, _, snapshot = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]

    raw_specs = _invent(count)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_specs, 1):
        if not isinstance(raw, dict):
            continue
        spec = _normalize(raw, i)
        if spec["candidate_id"] in seen:
            continue
        seen.add(spec["candidate_id"])
        normalized.append(spec)
        if len(normalized) >= count:
            break

    if not normalized:
        print("No executable inventions returned by the local inventor.")
        return {"decision": "NO_INVENTIONS", "validated": []}

    run_id = datetime.now(timezone.utc).strftime("INVENT-LM-%Y%m%dT%H%M%SZ")
    ledger = ROOT / "experiments" / run_id
    ledger.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    for i, spec in enumerate(normalized, 1):
        try:
            ev = evaluate(
                primary,
                "invented_composite",
                dict(spec["parameters"]),
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )
            primary_ok, reasons = _gate(ev)

            # Expensive independent test only for primary survivors.
            conf = None
            if primary_ok:
                conf = frozen_confirmation(
                    independent,
                    "invented_composite",
                    dict(spec["parameters"]),
                    ["both"],
                    settings.capital.initial_usd,
                    settings.execution.commission_bps,
                    settings.execution.slippage_bps,
                    market_type="spot",
                    leverage=1.0,
                    funding_rates=None,
                )

            row = _slim(spec, ev, primary_ok, reasons, conf)
            rows.append(row)
            (ledger / f"candidate_{i:02d}.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{i:02d}] {spec['title']} | OOS={row['holdout']['total_return']:.2%} | PF={row['holdout']['profit_factor']:.2f} | DD={row['holdout']['max_drawdown']:.2%} | WF={row['walk_forward']['passed']} | CONF={row['confirmation']['passed']} | {row['status']}")
            del ev, conf, row
            gc.collect()
        except Exception as exc:
            print(f"[{i:02d}] {spec['title']} | ERROR={type(exc).__name__}: {exc}")
            gc.collect()

    rows.sort(key=lambda r: float(r.get("holdout", {}).get("research_score", -1e9)), reverse=True)
    validated = [r for r in rows if r.get("status") == "VALIDATED"]
    decision = "VALIDATED_INVENTION" if validated else "NO_VALIDATED_INVENTION"
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "primary": {"symbol": "ETH/USDT", "timeframe": "1h", "market_type": "spot", "bars": len(primary)},
        "independent": {"symbol": "BTC/USDT", "timeframe": "4h", "market_type": "spot", "bars": len(independent)},
        "memory_mode": "low_memory_sequential",
        "generator": "local_llm_safe_dsl",
        "candidate_count": len(rows),
        "validated_count": len(validated),
        "validated": validated,
        "leaderboard": rows,
        "market_snapshot": snapshot,
    }
    latest = ROOT / "experiments" / "self_inventor_latest.json"
    latest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ledger / "result.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=== FINAL DECISION ===")
    print("Decision:", decision)
    print("Validated inventions:", len(validated))
    print("Saved:", latest)
    return manifest


if __name__ == "__main__":
    run()
