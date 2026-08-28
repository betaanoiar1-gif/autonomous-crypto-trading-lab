from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..pine_factory import build_pine
from .fast_generator import generate
from .run import (
    DIVERSITY_SLOTS, SUPPORTED_TIMEFRAMES, SYMBOLS,
    _load_failures, _next_timeframe, _print_diagnostics,
    _direction_value, _fetch_markets_cached, _score, _status,
)
from .evaluator import evaluate, frozen_confirmation

MAX_WORKERS = 4


def _eval_one(idx, h, spot, futures, funding, futures_error, settings, run_id):
    symbol, timeframe = h.symbols[0], h.timeframes[0]
    dirs = [_direction_value(x) for x in h.directions]
    market = h.market_types[0].value if h.market_types else "spot"
    try:
        if market == "futures" and (not isinstance(futures, dict) or futures_error):
            raise RuntimeError(str(futures_error or "Historical futures data unavailable"))
        data = spot if market == "spot" else futures
        fr = None if market == "spot" else funding.get(symbol)
        lev = 2.0 if market == "futures" else 1.0
        ev = evaluate(data[(symbol, timeframe)], h.executable_family, h.executable_parameters,
                      dirs, settings.capital.initial_usd, settings.execution.commission_bps,
                      settings.execution.slippage_bps, settings.validation.holdout_ratio,
                      market_type=market, leverage=lev, funding_rates=fr)
        status = _status(ev, confirmations=[])
        confirmations = []
        if status in {"VALIDATION_CANDIDATE", "VALIDATED"}:
            params = dict(ev.out_of_sample.get("selected_parameters", h.executable_parameters))
            other = next((s for s in SYMBOLS if s != symbol), None)
            if other:
                confirmations.append({**frozen_confirmation(
                    data[(other, timeframe)], h.executable_family, params, dirs,
                    settings.capital.initial_usd, settings.execution.commission_bps,
                    settings.execution.slippage_bps, market_type=market, leverage=lev,
                    funding_rates=None if market == "spot" else funding.get(other)),
                    "market": {"symbol": other, "timeframe": timeframe, "type": "cross_market"}})
            ntf = _next_timeframe(timeframe)
            if ntf:
                confirmations.append({**frozen_confirmation(
                    data[(symbol, ntf)], h.executable_family, params, dirs,
                    settings.capital.initial_usd, settings.execution.commission_bps,
                    settings.execution.slippage_bps, market_type=market, leverage=lev,
                    funding_rates=fr),
                    "market": {"symbol": symbol, "timeframe": ntf, "type": "cross_timeframe"}})
        final_status = "VALIDATED" if ev.passed and confirmations and all(c.get("passed", False) for c in confirmations) else status
        reasons = list(ev.rejection_reasons)
        if status in {"VALIDATION_CANDIDATE", "VALIDATED"} and confirmations and not all(c.get("passed", False) for c in confirmations):
            reasons.append("Independent generalization failed")
        return {"run_id": run_id, "index": idx, "symbol": symbol, "timeframe": timeframe,
                "market_type": market, "hypothesis": h.model_dump(mode="json"), "status": final_status,
                "in_sample": ev.in_sample, "out_of_sample": ev.out_of_sample,
                "robustness": ev.robustness, "rejection_reasons": reasons,
                "confirmations": confirmations}
    except Exception as exc:
        return {"run_id": run_id, "index": idx, "symbol": symbol, "timeframe": timeframe,
                "market_type": market, "hypothesis": h.model_dump(mode="json"),
                "status": "REJECTED", "error": str(exc)}


def run(max_hypotheses: int = 12, agent=None) -> dict:
    settings = load_settings()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("RUN-%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)
    memory = ROOT / "experiments" / "memory.jsonl"
    prior = _load_failures(memory)
    spot, futures, funding, futures_error, snapshot = _fetch_markets_cached()
    print(f"Data source: binance | timeframes={SUPPORTED_TIMEFRAMES} | symbols={SYMBOLS}")
    print(f"Spot observations: {sum(snapshot['observations_spot'].values())} total rows across pairs/factors")
    if isinstance(futures, dict) and not futures_error:
        print(f"Futures data: connected | historical funding loaded | events={sum(len(v) for v in funding.values())} | cache=active")
    elif futures_error:
        print(f"Futures data: unavailable -> {futures_error}")
    target = min(max_hypotheses, len(DIVERSITY_SLOTS))
    print(f"Research slots requested: {target}")
    print("Fast local hypothesis generation: active")
    hypotheses = generate(prior, target)
    for i, h in enumerate(hypotheses, 1):
        print(f"  Slot {i}: {h.executable_family} | {h.market_types[0].value} | {h.symbols[0]} | {h.timeframes[0]} | {_direction_value(h.directions[0])}")
    print(f"Hypotheses generated: {len(hypotheses)}")
    print(f"Parallel evaluation workers: {min(MAX_WORKERS, max(1, len(hypotheses)))}")
    records = [None] * len(hypotheses)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="acl-eval") as pool:
        pending = {pool.submit(_eval_one, i, h, spot, futures, funding, futures_error, settings, run_id): i for i, h in enumerate(hypotheses)}
        for fut, idx in ((f, i) for f, i in pending.items()):
            records[idx] = fut.result()
    for i, (h, r) in enumerate(zip(hypotheses, records), 1):
        print(f"[{i}/{len(records)}] {h.title} | {h.symbols[0]} | {h.timeframes[0]} -> {r['status']}")
        _print_diagnostics(r)
        if r.get("status") == "VALIDATED" and settings.output.generate_pine and r.get("market_type") == "spot":
            p = out / f"validated_candidate_{i}.pine"
            p.write_text(build_pine(f"ACL {h.title[:50]}", h.executable_family, h.executable_parameters,
                                    allow_short=any(_direction_value(d) in {"short", "both"} for d in h.directions)), encoding="utf-8")
            r["pine_path"] = str(p)
    records.sort(key=_score, reverse=True)
    for rank, r in enumerate(records, 1): r["rank"] = rank
    leaderboard = [{"rank":r["rank"],"title":r["hypothesis"]["title"],"status":r["status"],
                    "score":r.get("out_of_sample",{}).get("research_score",0.0),"symbol":r["symbol"],
                    "timeframe":r["timeframe"],"market_type":r["market_type"],
                    "walk_forward_passed":r.get("robustness",{}).get("walk_forward",{}).get("passed",False),
                    "confirmation_passed":bool(r.get("confirmations")) and all(c.get("passed",False) for c in r.get("confirmations",[]))} for r in records]
    print("Leaderboard:")
    for r in leaderboard: print(f"  #{r['rank']} | {r['status']} | score={float(r['score']):.2f} | {r['title']} | {r['symbol']} | {r['timeframe']} | {r['market_type']} | WF={r['walk_forward_passed']} | CONF={r['confirmation_passed']}")
    manifest={"run_id":run_id,"created_at":now.isoformat(),"capital":settings.capital.model_dump(),"market_snapshot":snapshot,
              "hypothesis_count":len(hypotheses),"records":records,"leaderboard":leaderboard,"evaluation_workers":MAX_WORKERS,"fast_generation":True}
    (out/"run.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    with memory.open("a",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False,default=str)+"\n")
    return manifest
