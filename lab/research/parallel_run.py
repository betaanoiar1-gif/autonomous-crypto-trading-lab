from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..local_agent import LocalAgent
from ..pine_factory import build_pine
from ..schemas import MarketType
from .prompt import build_prompt, SYSTEM as RESEARCH_SYSTEM
from .parser import parse_hypotheses
from .run import (
    DIVERSITY_SLOTS,
    SUPPORTED_TIMEFRAMES,
    SYMBOLS,
    _enforce_slot,
    _fetch_markets_cached,
    _load_failures,
    _next_timeframe,
    _print_diagnostics,
    _safe_one_hypothesis,
    _score,
    _status,
    _direction_value,
)
from .evaluator import evaluate, frozen_confirmation


MAX_WORKERS = 4


def _evaluate_one(h, idx, settings, spot_market, futures_market, futures_funding, futures_error):
    symbol = h.symbols[0]
    timeframe = h.timeframes[0]
    directions = [_direction_value(x) for x in h.directions]
    market_type = h.market_types[0].value if h.market_types else "spot"
    if market_type == "futures" and (not isinstance(futures_market, dict) or symbol not in SYMBOLS):
        return {
            "run_error": str(futures_error or "Historical futures data unavailable"),
            "symbol": symbol,
            "timeframe": timeframe,
            "market_type": market_type,
        }

    active_market = spot_market if market_type == "spot" else futures_market
    funding = None if market_type == "spot" else futures_funding.get(symbol)
    leverage = 2.0 if market_type == "futures" else 1.0
    evaluation = evaluate(
        active_market[(symbol, timeframe)],
        h.executable_family,
        h.executable_parameters,
        directions,
        settings.capital.initial_usd,
        settings.execution.commission_bps,
        settings.execution.slippage_bps,
        settings.validation.holdout_ratio,
        market_type=market_type,
        leverage=leverage,
        funding_rates=funding,
    )

    confirmations = []
    initial_status = _status(evaluation, confirmations=[])
    if initial_status in {"VALIDATION_CANDIDATE", "VALIDATED"}:
        frozen_params = dict(evaluation.out_of_sample.get("selected_parameters", h.executable_parameters))
        other_symbol = next((s for s in SYMBOLS if s != symbol), None)
        if other_symbol is not None:
            confirmation_funding = None if market_type == "spot" else futures_funding.get(other_symbol)
            confirmations.append({
                **frozen_confirmation(
                    active_market[(other_symbol, timeframe)],
                    h.executable_family,
                    frozen_params,
                    directions,
                    settings.capital.initial_usd,
                    settings.execution.commission_bps,
                    settings.execution.slippage_bps,
                    market_type=market_type,
                    leverage=leverage,
                    funding_rates=confirmation_funding,
                ),
                "market": {"symbol": other_symbol, "timeframe": timeframe, "type": "cross_market"},
            })
        neighbor_tf = _next_timeframe(timeframe)
        if neighbor_tf is not None:
            confirmation_funding = None if market_type == "spot" else futures_funding.get(symbol)
            confirmations.append({
                **frozen_confirmation(
                    active_market[(symbol, neighbor_tf)],
                    h.executable_family,
                    frozen_params,
                    directions,
                    settings.capital.initial_usd,
                    settings.execution.commission_bps,
                    settings.execution.slippage_bps,
                    market_type=market_type,
                    leverage=leverage,
                    funding_rates=confirmation_funding,
                ),
                "market": {"symbol": symbol, "timeframe": neighbor_tf, "type": "cross_timeframe"},
            })

    status = "VALIDATED" if evaluation.passed and confirmations and all(c.get("passed", False) for c in confirmations) else initial_status
    rejection_reasons = list(evaluation.rejection_reasons)
    if initial_status in {"VALIDATION_CANDIDATE", "VALIDATED"} and confirmations and not all(c.get("passed", False) for c in confirmations):
        rejection_reasons.append("Independent generalization failed")

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "market_type": market_type,
        "status": status,
        "in_sample": evaluation.in_sample,
        "out_of_sample": evaluation.out_of_sample,
        "robustness": evaluation.robustness,
        "rejection_reasons": rejection_reasons,
        "confirmations": confirmations,
        "directions": directions,
    }


def run(max_hypotheses: int = 12, agent=None) -> dict:
    settings = load_settings()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("RUN-%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)
    memory_path = ROOT / "experiments" / "memory.jsonl"
    prior_failures = _load_failures(memory_path)

    spot_market, futures_market, futures_funding, futures_error, snapshot = _fetch_markets_cached()
    print(f"Data source: binance | timeframes={SUPPORTED_TIMEFRAMES} | symbols={SYMBOLS}")
    print(f"Spot observations: {sum(snapshot['observations_spot'].values())} total rows across pairs/factors")
    if isinstance(futures_market, dict) and not futures_error:
        print(f"Futures data: connected | historical funding loaded | events={sum(len(v) for v in futures_funding.values())} | cache=active")
    elif futures_error:
        print(f"Futures data: unavailable -> {futures_error}")
    else:
        print("Futures data: unavailable | cache=active")

    agent = agent or LocalAgent()
    hypotheses = []
    target = min(max_hypotheses, len(DIVERSITY_SLOTS))
    print(f"Research slots requested: {target}")
    for i in range(target):
        slot = DIVERSITY_SLOTS[i]
        try:
            h = _enforce_slot(
                _safe_one_hypothesis(agent, snapshot, prior_failures, [x.model_dump(mode='json') for x in hypotheses], slot),
                slot,
            )
            hypotheses.append(h)
            print(f"  Slot {i + 1}: {h.executable_family} | {h.market_types[0].value} | {h.symbols[0]} | {h.timeframes[0]} | {_direction_value(h.directions[0])}")
        except Exception as exc:
            print(f"Hypothesis generation {i + 1} failed: {exc}")

    print(f"Hypotheses generated: {len(hypotheses)}")
    print(f"Parallel evaluation workers: {min(MAX_WORKERS, max(1, len(hypotheses)))}")
    records = [None] * len(hypotheses)

    def task(args):
        idx, h = args
        try:
            result = _evaluate_one(h, idx, settings, spot_market, futures_market, futures_funding, futures_error)
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": result.get("symbol", h.symbols[0]),
                "timeframe": result.get("timeframe", h.timeframes[0]),
                "market_type": result.get("market_type", h.market_types[0].value if h.market_types else "spot"),
                "hypothesis": h.model_dump(mode="json"),
            }
            if result.get("run_error"):
                record.update({"status": "REJECTED", "error": result["run_error"]})
            else:
                record.update({k: v for k, v in result.items() if k not in {"symbol", "timeframe", "market_type", "directions"}})
            return idx, record
        except Exception as exc:
            return idx, {
                "run_id": run_id,
                "index": idx,
                "symbol": h.symbols[0],
                "timeframe": h.timeframes[0],
                "market_type": h.market_types[0].value if h.market_types else "spot",
                "hypothesis": h.model_dump(mode="json"),
                "status": "REJECTED",
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(task, (idx, h)) for idx, h in enumerate(hypotheses)]
        for future in as_completed(futures):
            idx, record = future.result()
            records[idx] = record

    for idx, record in enumerate(records):
        if record is None:
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": hypotheses[idx].symbols[0],
                "timeframe": hypotheses[idx].timeframes[0],
                "market_type": hypotheses[idx].market_types[0].value if hypotheses[idx].market_types else "spot",
                "hypothesis": hypotheses[idx].model_dump(mode="json"),
                "status": "REJECTED",
                "error": "Parallel worker returned no result",
            }
            records[idx] = record

        if record.get("status") == "VALIDATED" and settings.output.generate_pine and record.get("market_type") == "spot":
            h = hypotheses[idx]
            pine = build_pine(
                f"ACL {h.title[:50]}",
                h.executable_family,
                h.executable_parameters,
                allow_short=any(_direction_value(d) in {"short", "both"} for d in h.directions),
            )
            path = out / f"validated_candidate_{idx + 1}.pine"
            path.write_text(pine, encoding="utf-8")
            record["pine_path"] = str(path)

        print(f"[{idx + 1}/{len(hypotheses)}] {hypotheses[idx].title} | {record['symbol']} | {record['timeframe']} -> {record['status']}")
        _print_diagnostics(record)

    ranked = sorted(records, key=_score, reverse=True)
    for rank, record in enumerate(ranked, 1):
        record["rank"] = rank
    leaderboard = [
        {
            "rank": r["rank"],
            "title": r["hypothesis"]["title"],
            "status": r["status"],
            "score": r.get("out_of_sample", {}).get("research_score", 0.0),
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "market_type": r["market_type"],
            "walk_forward_passed": r.get("robustness", {}).get("walk_forward", {}).get("passed", False),
            "confirmation_passed": bool(r.get("confirmations")) and all(c.get("passed", False) for c in r.get("confirmations", [])),
        }
        for r in ranked
    ]
    print("Leaderboard:")
    for r in leaderboard:
        print(f"  #{r['rank']} | {r['status']} | score={float(r['score']):.2f} | {r['title']} | {r['symbol']} | {r['timeframe']} | {r['market_type']} | WF={r['walk_forward_passed']} | CONF={r['confirmation_passed']}")

    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "capital": settings.capital.model_dump(),
        "market_snapshot": snapshot,
        "hypothesis_count": len(hypotheses),
        "records": ranked,
        "leaderboard": leaderboard,
        "parallel_workers": MAX_WORKERS,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as f:
        for record in ranked:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return manifest
