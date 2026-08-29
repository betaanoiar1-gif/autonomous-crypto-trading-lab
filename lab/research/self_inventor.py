from __future__ import annotations

"""Autonomous strategy inventor.

The local language model proposes *new compositions* of a small, safe feature
DSL. The research engine then evaluates those inventions with the existing
costed backtest, walk-forward, stability, stress and independent-confirmation
gates. No arbitrary model-generated Python is executed.
"""

from datetime import datetime, timezone
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


def _hash_spec(spec: dict[str, Any]) -> str:
    raw = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = lo
    return max(lo, min(hi, x))


def _int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
    except (TypeError, ValueError):
        x = default
    return max(lo, min(hi, x))


def _normalize(spec: dict[str, Any], idx: int) -> dict[str, Any]:
    p = dict(spec.get("parameters") or {})
    p["trend_fast"] = _int(p.get("trend_fast"), 3, 100, 18)
    p["trend_slow"] = _int(p.get("trend_slow"), p["trend_fast"] + 1, 300, max(72, p["trend_fast"] + 1))
    if p["trend_slow"] <= p["trend_fast"]:
        p["trend_slow"] = min(300, p["trend_fast"] + 1)
    for key, lo, hi, default in (
        ("momentum_window", 2, 200, 24),
        ("breakout_window", 3, 200, 36),
        ("vol_window", 5, 100, 24),
        ("volume_window", 5, 100, 24),
    ):
        p[key] = _int(p.get(key), lo, hi, default)
    for key, lo, hi, default in (
        ("w_trend", -3, 3, 1.0),
        ("w_momentum", -3, 3, 1.0),
        ("w_breakout", -3, 3, .75),
        ("w_candle", -3, 3, .5),
        ("w_volume", -3, 3, .5),
        ("long_threshold", .25, 6, 1.75),
        ("short_threshold", .25, 6, 1.75),
        ("exit_threshold", 0, 2, .40),
        ("vol_floor", 0, .10, .002),
        ("vol_cap", .001, .20, .05),
        ("volume_mult", .5, 2.5, 1.0),
    ):
        p[key] = round(_clamp(p.get(key), lo, hi), 6)
    if p["vol_cap"] <= p["vol_floor"]:
        p["vol_cap"] = min(.20, p["vol_floor"] + .01)

    title = str(spec.get("title") or f"Invented Composite {idx}").strip()[:120]
    thesis = str(spec.get("thesis") or "Autonomously invented feature composition.").strip()[:1000]
    falsifiers = spec.get("falsifiers")
    if not isinstance(falsifiers, list):
        falsifiers = ["Fails OOS/WF/stress or independent confirmation."]
    return {"title": title, "thesis": thesis, "parameters": p,
            "falsifiers": [str(x)[:300] for x in falsifiers[:8]],
            "candidate_id": f"INV-{_hash_spec(p)}"}


def _fallback_specs() -> list[dict[str, Any]]:
    # Diverse seeds are deliberately not copies of the old named strategies.
    return [
        {"title": "Volatility Gated Consensus", "parameters": {"trend_fast": 13, "trend_slow": 89, "momentum_window": 21, "breakout_window": 55, "vol_window": 24, "volume_window": 30, "w_trend": 1.25, "w_momentum": .75, "w_breakout": 1.10, "w_candle": .35, "w_volume": .55, "long_threshold": 2.10, "short_threshold": 2.10, "exit_threshold": .45, "vol_floor": .006, "vol_cap": .040, "volume_mult": 1.05}},
        {"title": "Asymmetric Pressure Composite", "parameters": {"trend_fast": 21, "trend_slow": 144, "momentum_window": 34, "breakout_window": 72, "vol_window": 36, "volume_window": 18, "w_trend": 1.00, "w_momentum": 1.35, "w_breakout": .65, "w_candle": .80, "w_volume": .40, "long_threshold": 2.30, "short_threshold": 1.70, "exit_threshold": .35, "vol_floor": .004, "vol_cap": .055, "volume_mult": 1.15}},
        {"title": "Slow Trend Shock Filter", "parameters": {"trend_fast": 34, "trend_slow": 200, "momentum_window": 18, "breakout_window": 40, "vol_window": 18, "volume_window": 36, "w_trend": 1.50, "w_momentum": .55, "w_breakout": .80, "w_candle": .25, "w_volume": .70, "long_threshold": 2.25, "short_threshold": 2.25, "exit_threshold": .50, "vol_floor": .008, "vol_cap": .045, "volume_mult": 1.10}},
    ]


def _ask_agent(agent: LocalAgent, count: int) -> list[dict[str, Any]]:
    prompt = f"""
Invent {count} genuinely different crypto trading strategy hypotheses.
Do NOT return known strategies such as plain moving-average cross, plain momentum,
breakout, RSI reversion, ATR breakout, or simple trend pullback.
Instead invent NEW COMPOSITIONS using only this safe feature vocabulary:
trend relationship, momentum over a window, prior-range breakout state, candle-body pressure,
relative volume, realized volatility regime, weighted scoring, asymmetric long/short thresholds,
and neutral/exit threshold.

Each invention must be executable by the lab's `invented_composite` strategy family.
Return ONLY one JSON object with key `strategies` containing a list. Each item must contain:
`title`, `thesis`, `parameters`, `falsifiers`.
Parameter keys allowed:
trend_fast, trend_slow, momentum_window, breakout_window, vol_window, volume_window,
w_trend, w_momentum, w_breakout, w_candle, w_volume,
long_threshold, short_threshold, exit_threshold, vol_floor, vol_cap, volume_mult.
Use bounded sensible numeric values. Do not invent performance numbers.
""".strip()
    try:
        raw = agent.chat(prompt)
        obj = extract_json_object(raw)
        strategies = obj.get("strategies", [])
        return strategies if isinstance(strategies, list) else []
    except Exception:
        return []


def _gate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    h = record["holdout"]
    wf = record["robustness"]["walk_forward"]
    reasons: list[str] = []
    if h["trade_count"] < MIN_TRADES:
        reasons.append("oos_too_few_trades")
    if h["total_return"] <= 0:
        reasons.append("non_positive_oos_return")
    if h["profit_factor"] <= 1:
        reasons.append("oos_pf_le_1")
    if h["max_drawdown"] < -0.50:
        reasons.append("oos_drawdown_over_50pct")
    if not record["robustness"].get("parameter_stability", True):
        reasons.append("parameter_stability_failed")
    if record["robustness"].get("stressed_total_return", 0) <= 0:
        reasons.append("doubled_cost_stress_failed")
    if not wf.get("passed", False):
        reasons.append("walk_forward_failed")
    return not reasons, reasons


def run(count: int = 8) -> dict[str, Any]:
    settings = load_settings()
    count = max(3, min(12, int(count)))
    spot, _, _, _, snapshot = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent_market = spot[("BTC/USDT", "4h")]

    agent: LocalAgent | None = None
    specs: list[dict[str, Any]] = []
    try:
        agent = LocalAgent()
        health = agent.healthcheck()
        print(f"Inventor model: {health.get('model')} | device={health.get('device')} | healthy={health.get('ok')}")
        specs.extend(_ask_agent(agent, count))
    except Exception as exc:
        print(f"LLM inventor unavailable: {type(exc).__name__}: {exc}")

    specs.extend(_fallback_specs())
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(specs, 1):
        if not isinstance(raw, dict):
            continue
        item = _normalize(raw, i)
        if item["candidate_id"] in seen:
            continue
        seen.add(item["candidate_id"])
        normalized.append(item)
        if len(normalized) >= count:
            break

    ledger_dir = ROOT / "experiments" / datetime.now(timezone.utc).strftime("INVENT-%Y%m%dT%H%M%SZ")
    ledger_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []

    print("=== AUTONOMOUS STRATEGY INVENTOR ===")
    print(f"Primary: ETH/USDT | 1h | spot | bars={len(primary)}")
    print("Independent: BTC/USDT | 4h | spot")
    print(f"Novel candidates: {len(normalized)}")
    print("Arbitrary code execution: DISABLED")
    print()

    for i, spec in enumerate(normalized, 1):
        params = spec["parameters"]
        try:
            ev = evaluate(
                primary,
                "invented_composite",
                params,
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )
            record = {
                "candidate_id": spec["candidate_id"],
                "title": spec["title"],
                "thesis": spec["thesis"],
                "parameters": params,
                "in_sample": ev.in_sample,
                "holdout": ev.out_of_sample,
                "robustness": ev.robustness,
            }
            primary_pass, reasons = _gate(record)
            confirmation = frozen_confirmation(
                independent_market,
                "invented_composite",
                params,
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )
            passed = bool(primary_pass and confirmation["passed"])
            record["primary_gate"] = primary_pass
            record["primary_reasons"] = reasons
            record["independent_confirmation"] = confirmation
            record["status"] = "VALIDATED" if passed else "REJECTED"
            record["score"] = float(record["holdout"].get("research_score", 0.0))
            rows.append(record)
            print(f"[{i:02d}] {spec['title']} | OOS={record['holdout']['total_return']:.2%} | PF={record['holdout']['profit_factor']:.2f} | DD={record['holdout']['max_drawdown']:.2%} | WF={record['robustness']['walk_forward'].get('passed', False)} | CONF={confirmation['passed']} | {record['status']}")
        except Exception as exc:
            rows.append({"candidate_id": spec["candidate_id"], "title": spec["title"], "parameters": params, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{i:02d}] {spec['title']} | ERROR={type(exc).__name__}: {exc}")

    rows.sort(key=lambda r: float(r.get("score", -1e9)), reverse=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary": {"symbol": "ETH/USDT", "timeframe": "1h", "market_type": "spot"},
        "independent": {"symbol": "BTC/USDT", "timeframe": "4h", "market_type": "spot"},
        "market_snapshot": snapshot,
        "generator": "local_llm_safe_dsl",
        "arbitrary_code_execution": False,
        "candidate_count": len(rows),
        "validated": [r for r in rows if r.get("status") == "VALIDATED"],
        "leaderboard": rows,
    }
    (ledger_dir / "result.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    latest = ROOT / "experiments" / "self_inventor_latest.json"
    latest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    if manifest["validated"]:
        print("=== VALIDATED INVENTION FOUND ===")
        for r in manifest["validated"]:
            print(r["title"], r["parameters"])
        decision = "STOP_VALIDATED"
    else:
        print("=== NO VALIDATED INVENTION ===")
        decision = "NO_VALIDATED_INVENTION"
    print("Decision:", decision)
    print("Ledger:", latest)
    manifest["decision"] = decision
    return manifest


if __name__ == "__main__":
    run()
