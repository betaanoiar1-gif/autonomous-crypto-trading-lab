from __future__ import annotations

"""Memory-bounded autonomous strategy inventor.

The local model only proposes safe-DSL strategy specifications. The model is
released before numerical evaluation so Colab RAM stays bounded. Numerical
candidates are evaluated sequentially and only scalar results are persisted.
No arbitrary model-generated Python is executed.
"""

from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import ROOT, load_settings
from ..local_agent import LocalAgent
from .evaluator import evaluate, frozen_confirmation
from .run import _fetch_markets_cached
from .json_utils import extract_json_object

MIN_TRADES = 8
MAX_CANDIDATES = 4


def _hash_spec(params: dict[str, Any]) -> str:
    raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except (TypeError, ValueError):
        x = default
    return max(lo, min(hi, x))


def _float(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = default
    return max(lo, min(hi, x))


def normalize(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    p = dict(raw.get("parameters") or {})
    p["trend_fast"] = _int(p.get("trend_fast"), 3, 100, 18)
    p["trend_slow"] = _int(p.get("trend_slow"), p["trend_fast"] + 1, 300, max(72, p["trend_fast"] + 1))
    if p["trend_slow"] <= p["trend_fast"]:
        p["trend_slow"] = min(300, p["trend_fast"] + 1)

    for k, lo, hi, default in (
        ("momentum_window", 2, 200, 24),
        ("breakout_window", 3, 200, 36),
        ("vol_window", 5, 100, 24),
        ("volume_window", 5, 100, 24),
    ):
        p[k] = _int(p.get(k), lo, hi, default)

    for k, lo, hi, default in (
        ("w_trend", -3.0, 3.0, 1.0),
        ("w_momentum", -3.0, 3.0, 1.0),
        ("w_breakout", -3.0, 3.0, 0.75),
        ("w_candle", -3.0, 3.0, 0.5),
        ("w_volume", -3.0, 3.0, 0.5),
        ("long_threshold", 0.25, 6.0, 1.75),
        ("short_threshold", 0.25, 6.0, 1.75),
        ("exit_threshold", 0.0, 2.0, 0.40),
        ("vol_floor", 0.0, 0.10, 0.002),
        ("vol_cap", 0.001, 0.20, 0.05),
        ("volume_mult", 0.5, 2.5, 1.0),
    ):
        p[k] = round(_float(p.get(k), lo, hi, default), 6)

    if p["vol_cap"] <= p["vol_floor"]:
        p["vol_cap"] = min(0.20, p["vol_floor"] + 0.01)

    falsifiers = raw.get("falsifiers")
    if not isinstance(falsifiers, list):
        falsifiers = ["Fails OOS/WF/stress or independent confirmation."]

    return {
        "title": str(raw.get("title") or f"Invented Composite {idx}").strip()[:120],
        "thesis": str(raw.get("thesis") or "Autonomous safe feature composition.").strip()[:1000],
        "parameters": p,
        "falsifiers": [str(x)[:300] for x in falsifiers[:8]],
        "candidate_id": f"INV-{_hash_spec(p)}",
    }


def _fallback_specs() -> list[dict[str, Any]]:
    return [
        {
            "title": "Volatility Gated Consensus",
            "parameters": {
                "trend_fast": 13, "trend_slow": 89, "momentum_window": 21,
                "breakout_window": 55, "vol_window": 24, "volume_window": 30,
                "w_trend": 1.25, "w_momentum": 0.75, "w_breakout": 1.10,
                "w_candle": 0.35, "w_volume": 0.55, "long_threshold": 2.10,
                "short_threshold": 2.10, "exit_threshold": 0.45,
                "vol_floor": 0.006, "vol_cap": 0.040, "volume_mult": 1.05,
            }
        },
        {
            "title": "Asymmetric Pressure Composite",
            "parameters": {
                "trend_fast": 21, "trend_slow": 144, "momentum_window": 34,
                "breakout_window": 72, "vol_window": 36, "volume_window": 18,
                "w_trend": 1.0, "w_momentum": 1.35, "w_breakout": 0.65,
                "w_candle": 0.80, "w_volume": 0.40, "long_threshold": 2.30,
                "short_threshold": 1.70, "exit_threshold": 0.35,
                "vol_floor": 0.004, "vol_cap": 0.055, "volume_mult": 1.15,
            }
        },
        {
            "title": "Slow Trend Shock Filter",
            "parameters": {
                "trend_fast": 34, "trend_slow": 200, "momentum_window": 18,
                "breakout_window": 40, "vol_window": 18, "volume_window": 36,
                "w_trend": 1.50, "w_momentum": 0.55, "w_breakout": 0.80,
                "w_candle": 0.25, "w_volume": 0.70, "long_threshold": 2.25,
                "short_threshold": 2.25, "exit_threshold": 0.50,
                "vol_floor": 0.008, "vol_cap": 0.045, "volume_mult": 1.10,
            }
        },
    ]


def _ask_agent(agent: LocalAgent, count: int) -> list[dict[str, Any]]:
    prompt = f"""
Invent {count} genuinely different crypto trading strategy hypotheses.
Do not return plain moving-average cross, plain momentum, simple breakout,
plain RSI reversion, ATR breakout, or simple trend pullback.
Use only these safe components: trend relationship, momentum window,
prior-range state, candle-body pressure, relative volume, realized volatility,
weighted scoring, asymmetric long/short thresholds, and neutral/exit rules.
Each invention must be executable by the safe invented_composite family.
Return ONLY JSON with key strategies, a list of objects containing title,
thesis, parameters, falsifiers. Do not invent performance numbers.
Allowed parameter keys: trend_fast, trend_slow, momentum_window, breakout_window,
vol_window, volume_window, w_trend, w_momentum, w_breakout, w_candle, w_volume,
long_threshold, short_threshold, exit_threshold, vol_floor, vol_cap, volume_mult.
""".strip()
    raw = agent.chat(prompt)
    obj = extract_json_object(raw)
    strategies = obj.get("strategies", [])
    return strategies if isinstance(strategies, list) else []


def _release_agent(agent: LocalAgent | None) -> None:
    if agent is None:
        return
    try:
        model = getattr(agent, "model", None)
        tokenizer = getattr(agent, "tokenizer", None)
        try:
            if model is not None:
                model.cpu()
        except Exception:
            pass
        del model, tokenizer
    finally:
        del agent
        gc.collect()


def _gate(ev: Any) -> tuple[bool, list[str]]:
    h = ev.out_of_sample
    wf = ev.robustness.get("walk_forward", {})
    reasons: list[str] = []
    if int(h.get("trade_count", 0)) < MIN_TRADES:
        reasons.append("oos_too_few_trades")
    if float(h.get("total_return", 0)) <= 0:
        reasons.append("non_positive_oos_return")
    if float(h.get("profit_factor", 0)) <= 1:
        reasons.append("oos_pf_le_1")
    if float(h.get("max_drawdown", 0)) < -0.50:
        reasons.append("oos_drawdown_over_50pct")
    if not bool(ev.robustness.get("parameter_stability", False)):
        reasons.append("parameter_stability_failed")
    if float(ev.robustness.get("stressed_total_return", 0)) <= 0:
        reasons.append("doubled_cost_stress_failed")
    if not bool(wf.get("passed", False)):
        reasons.append("walk_forward_failed")
    return not reasons, reasons


def _slim(spec: dict[str, Any], ev: Any, primary_pass: bool, reasons: list[str], conf: dict[str, Any]) -> dict[str, Any]:
    h = ev.out_of_sample
    w = ev.robustness.get("walk_forward", {})
    cm = conf.get("metrics", {})
    return {
        "candidate_id": spec["candidate_id"],
        "title": spec["title"],
        "thesis": spec["thesis"],
        "parameters": dict(spec["parameters"]),
        "holdout": {
            "total_return": float(h.get("total_return", 0)),
            "profit_factor": float(h.get("profit_factor", 0)),
            "max_drawdown": float(h.get("max_drawdown", 0)),
            "trade_count": int(h.get("trade_count", 0)),
            "sharpe": float(h.get("sharpe", 0)),
            "sortino": float(h.get("sortino", 0)),
            "research_score": float(h.get("research_score", 0)),
        },
        "robustness": {
            "parameter_stability": bool(ev.robustness.get("parameter_stability", False)),
            "stressed_total_return": float(ev.robustness.get("stressed_total_return", 0)),
            "stressed_max_drawdown": float(ev.robustness.get("stressed_max_drawdown", 0)),
            "walk_forward": {
                "positive_windows": int(w.get("positive_windows", 0)),
                "median_return": float(w.get("median_return", 0)),
                "worst_return": float(w.get("worst_return", 0)),
                "median_sharpe": float(w.get("median_sharpe", 0)),
                "min_trade_count": int(w.get("min_trade_count", 0)),
                "passed": bool(w.get("passed", False)),
            },
        },
        "primary_gate": bool(primary_pass),
        "primary_reasons": reasons,
        "independent_confirmation": {
            "passed": bool(conf.get("passed", False)),
            "metrics": {
                "total_return": float(cm.get("total_return", 0)),
                "profit_factor": float(cm.get("profit_factor", 0)),
                "max_drawdown": float(cm.get("max_drawdown", 0)),
                "trade_count": int(cm.get("trade_count", 0)),
                "sharpe": float(cm.get("sharpe", 0)),
            },
            "rejection_reasons": list(conf.get("rejection_reasons", [])),
        },
        "status": "VALIDATED" if primary_pass and conf.get("passed", False) else "REJECTED",
    }


def run(count: int = MAX_CANDIDATES) -> dict[str, Any]:
    settings = load_settings()
    count = max(3, min(MAX_CANDIDATES, int(count)))

    print("=== LOW-MEMORY AUTONOMOUS STRATEGY INVENTOR ===", flush=True)
    print(f"Candidate batch: {count}", flush=True)
    print("Futures: DISABLED | Live trading: DISABLED", flush=True)

    spot, _, _, _, snapshot = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]

    specs: list[dict[str, Any]] = []
    agent: LocalAgent | None = None
    try:
        agent = LocalAgent()
        health = agent.healthcheck()
        print(f"Inventor model: {health.get('model')} | device={health.get('device')} | healthy={health.get('ok')}", flush=True)
        specs = _ask_agent(agent, count)
    except Exception as exc:
        print(f"LLM invention unavailable: {type(exc).__name__}: {exc}", flush=True)
    finally:
        _release_agent(agent)

    # Keep only what is needed for this batch and add deterministic fallbacks.
    specs.extend(_fallback_specs())
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(specs, 1):
        if not isinstance(raw, dict):
            continue
        item = normalize(raw, i)
        if item["candidate_id"] in seen:
            continue
        seen.add(item["candidate_id"])
        normalized.append(item)
        if len(normalized) >= count:
            break
    del specs
    gc.collect()

    run_id = datetime.now(timezone.utc).strftime("INVENT-%Y%m%dT%H%M%SZ")
    ledger = ROOT / "experiments" / run_id
    ledger.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    print(f"Novel inventions accepted for evaluation: {len(normalized)}", flush=True)

    for i, spec in enumerate(normalized, 1):
        try:
            ev = evaluate(
                primary,
                "invented_composite",
                spec["parameters"],
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )
            primary_pass, reasons = _gate(ev)

            # Only run independent confirmation when the primary gate passes;
            # this cuts peak memory and avoids expensive work on bad inventions.
            if primary_pass:
                conf = frozen_confirmation(
                    independent,
                    "invented_composite",
                    spec["parameters"],
                    ["both"],
                    settings.capital.initial_usd,
                    settings.execution.commission_bps,
                    settings.execution.slippage_bps,
                    market_type="spot",
                    leverage=1.0,
                    funding_rates=None,
                )
            else:
                conf = {"passed": False, "metrics": {}, "rejection_reasons": ["primary_gate_failed"]}

            row = _slim(spec, ev, primary_pass, reasons, conf)
            rows.append(row)
            (ledger / f"candidate_{i:02d}.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            print(
                f"[{i:02d}] {spec['title']} | "
                f"OOS={row['holdout']['total_return']:.2%} | "
                f"PF={row['holdout']['profit_factor']:.2f} | "
                f"DD={row['holdout']['max_drawdown']:.2%} | "
                f"WF={row['robustness']['walk_forward']['passed']} | "
                f"CONF={row['independent_confirmation']['passed']} | "
                f"{row['status']}",
                flush=True,
            )

            del ev, conf, row
            gc.collect()

            if rows[-1]["status"] == "VALIDATED":
                break

        except Exception as exc:
            error_row = {
                "candidate_id": spec["candidate_id"],
                "title": spec["title"],
                "parameters": dict(spec["parameters"]),
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(error_row)
            (ledger / f"candidate_{i:02d}.json").write_text(
                json.dumps(error_row, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[{i:02d}] {spec['title']} | ERROR={error_row['error']}", flush=True)
        finally:
            gc.collect()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "generator": "local_llm_safe_dsl",
        "memory_mode": "low_memory_sequential",
        "arbitrary_code_execution": False,
        "primary": {"symbol": "ETH/USDT", "timeframe": "1h", "market_type": "spot"},
        "independent": {"symbol": "BTC/USDT", "timeframe": "4h", "market_type": "spot"},
        "candidate_count": len(rows),
        "validated": [r for r in rows if r.get("status") == "VALIDATED"],
        "leaderboard": sorted(
            rows,
            key=lambda r: float(r.get("holdout", {}).get("research_score", -1e9)),
            reverse=True,
        ),
        "market_snapshot": snapshot,
    }

    latest = ROOT / "experiments" / "self_inventor_latest.json"
    latest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ledger / "result.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    decision = "STOP_VALIDATED" if manifest["validated"] else "NO_VALIDATED_INVENTION"
    print("=== FINAL DECISION ===", flush=True)
    print("Decision:", decision, flush=True)
    print("Validated inventions:", len(manifest["validated"]), flush=True)
    print("Saved:", latest, flush=True)

    manifest["decision"] = decision
    return manifest


if __name__ == "__main__":
    run()
