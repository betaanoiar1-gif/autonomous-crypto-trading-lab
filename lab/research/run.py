from __future__ import annotations

from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..kimi_agent import SYSTEM_PROMPT
from ..local_agent import LocalAgent
from ..pine_factory import build_pine
from .parser import parse_hypotheses
from .prompt import build_prompt
from .evaluator import evaluate


def _safe_agent_hypotheses(agent, snapshot, max_hypotheses, prior_failures):
    prompt = build_prompt(snapshot, prior_failures, max_hypotheses)
    text = agent.chat(prompt, system=SYSTEM_PROMPT)
    try:
        return parse_hypotheses(text)
    except ValueError:
        retry = "Return only valid JSON matching the requested schema. No markdown.\n\n" + prompt
        return parse_hypotheses(agent.chat(retry, system=SYSTEM_PROMPT))


def _load_failures(path):
    if not path.exists():
        return []
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines()[-50:]:
        try:
            rec = json.loads(line)
            if rec.get("status") == "REJECTED":
                failures.append(rec)
        except json.JSONDecodeError:
            continue
    return failures


def _fetch_market(data, symbols, timeframe):
    last_error = None
    for exchange_id in ("binance", "kraken"):
        try:
            adapter = CCXTMarketData(exchange_id=exchange_id)
            market = adapter.fetch_multi(symbols, timeframe=timeframe, limit=1000)
            return exchange_id, market
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Public market data unavailable on fallback exchanges: {last_error}")


def run(max_hypotheses: int = 4) -> dict:
    settings = load_settings()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("RUN-%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)

    memory_path = ROOT / "experiments" / "memory.jsonl"
    prior_failures = _load_failures(memory_path)

    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1h"
    exchange_id, market = _fetch_market(CCXTMarketData(), symbols, timeframe)
    snapshot = {
        "exchange": exchange_id,
        "timeframe": timeframe,
        "symbols": symbols,
        "observations": {k: len(v) for k, v in market.items()},
        "latest": {k: {"close": float(v["close"].iloc[-1])} for k, v in market.items()},
    }

    print(f"Data source: {exchange_id} | {timeframe} | {symbols}")
    agent = LocalAgent()
    hypotheses = _safe_agent_hypotheses(agent, snapshot, max_hypotheses, prior_failures)
    print(f"Hypotheses generated: {len(hypotheses)}")

    records = []
    for idx, hypothesis in enumerate(hypotheses):
        symbol = next((s for s in hypothesis.symbols if s in market), symbols[0])
        df = market[symbol]
        directions = [x.value for x in hypothesis.directions]
        family = hypothesis.executable_family or "momentum"
        try:
            evaluation = evaluate(
                df, family, hypothesis.executable_parameters, directions,
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
            )
            passed = evaluation.passed
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "hypothesis": hypothesis.model_dump(),
                "status": "VALIDATION_CANDIDATE" if passed else "REJECTED",
                "in_sample": evaluation.in_sample,
                "out_of_sample": evaluation.out_of_sample,
                "robustness": evaluation.robustness,
                "rejection_reasons": evaluation.rejection_reasons,
            }
            if passed and settings.output.generate_pine:
                allow_short = any(d in {"short", "both"} for d in directions)
                pine = build_pine(
                    f"ACL {hypothesis.title[:50]}",
                    family,
                    hypothesis.executable_parameters,
                    allow_short=allow_short,
                )
                (out / f"candidate_{idx+1}.pine").write_text(pine, encoding="utf-8")
                record["pine_path"] = str(out / f"candidate_{idx+1}.pine")
        except Exception as exc:
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "hypothesis": hypothesis.model_dump(),
                "status": "REJECTED",
                "error": str(exc),
            }
        records.append(record)
        print(f"[{idx+1}/{len(hypotheses)}] {hypothesis.title} -> {record['status']}")

    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "capital": settings.capital.model_dump(),
        "market_snapshot": snapshot,
        "hypothesis_count": len(hypotheses),
        "records": records,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    with memory_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return manifest
