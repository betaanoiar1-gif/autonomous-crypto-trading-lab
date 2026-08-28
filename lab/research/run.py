from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..kimi_agent import SYSTEM_PROMPT
from ..local_agent import LocalAgent
from .parser import parse_hypotheses
from .prompt import build_prompt
from .evaluator import evaluate


def _safe_agent_hypotheses(agent: LocalAgent, snapshot: dict, max_hypotheses: int, prior_failures: list[dict]):
    prompt = build_prompt(snapshot, prior_failures, max_hypotheses)
    text = agent.chat(prompt, system=SYSTEM_PROMPT)
    try:
        return parse_hypotheses(text)
    except ValueError:
        retry = (
            "Your previous response was not valid JSON. Return ONLY valid JSON matching "
            "the requested schema. Do not use markdown fences.\n\n" + prompt
        )
        return parse_hypotheses(agent.chat(retry, system=SYSTEM_PROMPT))


def run(max_hypotheses: int = 4) -> dict:
    settings = load_settings()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("RUN-%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)

    memory_path = ROOT / "experiments" / "memory.jsonl"
    prior_failures = []
    if memory_path.exists():
        for line in memory_path.read_text(encoding="utf-8").splitlines()[-50:]:
            try:
                rec = json.loads(line)
                if rec.get("status") == "REJECTED":
                    prior_failures.append(rec)
            except json.JSONDecodeError:
                pass

    data = CCXTMarketData(exchange_id="binance")
    symbols = ["BTC/USDT", "ETH/USDT"]
    timeframe = "1h"
    market = data.fetch_multi(symbols, timeframe=timeframe, limit=1000)
    snapshot = {
        "exchange": "binance",
        "timeframe": timeframe,
        "symbols": symbols,
        "observations": {k: len(v) for k, v in market.items()},
        "latest": {k: {"close": float(v["close"].iloc[-1])} for k, v in market.items()},
    }

    agent = LocalAgent()
    hypotheses = _safe_agent_hypotheses(agent, snapshot, max_hypotheses, prior_failures)

    records = []
    for idx, hypothesis in enumerate(hypotheses):
        symbol = hypothesis.symbols[0] if hypothesis.symbols and hypothesis.symbols[0] in market else symbols[0]
        timeframe_choice = hypothesis.timeframes[0] if hypothesis.timeframes else timeframe
        if timeframe_choice != timeframe:
            timeframe_choice = timeframe
        df = market[symbol]
        directions = [x.value for x in hypothesis.directions]
        family = hypothesis.executable_family or "momentum"
        try:
            evaluation = evaluate(
                df=df,
                family=family,
                params=hypothesis.executable_parameters,
                directions=directions,
                initial_capital=settings.capital.initial_usd,
                fee_bps=settings.execution.commission_bps,
                slippage_bps=settings.execution.slippage_bps,
                holdout_ratio=settings.validation.holdout_ratio,
            )
            status = "VALIDATION_CANDIDATE" if evaluation.passed else "REJECTED"
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe_choice,
                "hypothesis": hypothesis.model_dump(),
                "status": status,
                "in_sample": evaluation.in_sample,
                "out_of_sample": evaluation.out_of_sample,
                "robustness": evaluation.robustness,
                "rejection_reasons": evaluation.rejection_reasons,
            }
        except Exception as exc:
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe_choice,
                "hypothesis": hypothesis.model_dump(),
                "status": "REJECTED",
                "error": str(exc),
            }
        records.append(record)

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
