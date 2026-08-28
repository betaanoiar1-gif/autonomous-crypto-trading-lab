from __future__ import annotations

from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..local_agent import LocalAgent
from ..pine_factory import build_pine
from .parser import parse_hypotheses
from .prompt import build_prompt, SYSTEM as RESEARCH_SYSTEM
from .evaluator import evaluate

DIVERSITY_SLOTS = [
    {"preferred_family": "momentum", "preferred_timeframe": "15m", "preferred_symbol": "BTC/USDT", "preferred_direction": "long"},
    {"preferred_family": "mean_reversion", "preferred_timeframe": "1h", "preferred_symbol": "ETH/USDT", "preferred_direction": "both"},
    {"preferred_family": "breakout", "preferred_timeframe": "4h", "preferred_symbol": "BTC/USDT", "preferred_direction": "both"},
    {"preferred_family": "moving_average_cross", "preferred_timeframe": "1h", "preferred_symbol": "ETH/USDT", "preferred_direction": "short"},
]

SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h"]
SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _safe_one_hypothesis(agent, snapshot, prior_failures, prior_hypotheses, slot):
    prompt = build_prompt(snapshot, prior_failures, 1, prior_hypotheses, diversity_slot=slot)
    text = agent.chat(prompt, system=RESEARCH_SYSTEM)
    try:
        return parse_hypotheses(text)[0]
    except (ValueError, IndexError):
        retry = (
            "Return exactly ONE hypothesis block using the required line format. "
            "End with END. No JSON, markdown, analysis, or commentary. "
            f"Follow this diversity slot: {slot}\n\n" + prompt
        )
        return parse_hypotheses(agent.chat(retry, system=RESEARCH_SYSTEM))[0]


def _enforce_slot(hypothesis, slot):
    """Keep the model responsible for the thesis while enforcing experimental diversity."""
    family = slot["preferred_family"]
    direction = slot["preferred_direction"]
    timeframe = slot["preferred_timeframe"]
    symbol = slot["preferred_symbol"]
    params = dict(hypothesis.executable_parameters or {})
    if family in {"momentum", "breakout", "mean_reversion"}:
        params["lookback"] = int(params.get("lookback", 20))
    if family == "mean_reversion":
        params["z_entry"] = float(params.get("z_entry", 1.5))
        params["z_exit"] = float(params.get("z_exit", 0.25))
    if family == "moving_average_cross":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 40))
        params["fast"] = max(2, min(100, fast))
        params["slow"] = max(params["fast"] + 1, min(300, slow))
    return hypothesis.model_copy(update={
        "executable_family": family,
        "executable_parameters": params,
        "directions": [direction],
        "timeframes": [timeframe],
        "symbols": [symbol],
    })


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


def _fetch_market(symbols, timeframes):
    """Fetch each unique symbol/timeframe pair once, with exchange fallback."""
    last_error = None
    for exchange_id in ("binance", "kraken"):
        try:
            adapter = CCXTMarketData(exchange_id=exchange_id)
            market = {}
            for timeframe in timeframes:
                per_tf = adapter.fetch_multi(symbols, timeframe=timeframe, limit=1000)
                for symbol, df in per_tf.items():
                    market[(symbol, timeframe)] = df
            return exchange_id, market
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Public market data unavailable on fallback exchanges: {last_error}")


def run(max_hypotheses: int = 4, agent=None) -> dict:
    settings = load_settings()
    now = datetime.now(timezone.utc)
    run_id = now.strftime("RUN-%Y%m%dT%H%M%SZ")
    out = ROOT / "experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)

    memory_path = ROOT / "experiments" / "memory.jsonl"
    prior_failures = _load_failures(memory_path)
    exchange_id, market = _fetch_market(SYMBOLS, SUPPORTED_TIMEFRAMES)
    snapshot = {
        "exchange": exchange_id,
        "available_timeframes": SUPPORTED_TIMEFRAMES,
        "symbols": SYMBOLS,
        "observations": {f"{symbol}@{tf}": len(df) for (symbol, tf), df in market.items()},
        "latest": {f"{symbol}@{tf}": {"close": float(df["close"].iloc[-1])} for (symbol, tf), df in market.items()},
    }

    print(f"Data source: {exchange_id} | timeframes={SUPPORTED_TIMEFRAMES} | symbols={SYMBOLS}")
    agent = agent or LocalAgent()
    hypotheses = []
    target = min(max_hypotheses, len(DIVERSITY_SLOTS))
    for attempt in range(target):
        slot = DIVERSITY_SLOTS[attempt]
        try:
            raw = _safe_one_hypothesis(
                agent,
                snapshot,
                prior_failures,
                [h.model_dump() for h in hypotheses],
                slot,
            )
            hypothesis = _enforce_slot(raw, slot)
            signature = (
                hypothesis.executable_family,
                tuple(hypothesis.directions),
                tuple(hypothesis.symbols),
                tuple(hypothesis.timeframes),
                hypothesis.thesis.strip().lower(),
            )
            if any(signature == (h.executable_family, tuple(h.directions), tuple(h.symbols), tuple(h.timeframes), h.thesis.strip().lower()) for h in hypotheses):
                hypothesis = hypothesis.model_copy(update={"thesis": hypothesis.thesis + f" | diversity slot {attempt + 1}"})
            hypotheses.append(hypothesis)
        except Exception as exc:
            print(f"Hypothesis generation {attempt + 1} failed: {exc}")

    print(f"Hypotheses generated: {len(hypotheses)}")
    records = []
    for idx, hypothesis in enumerate(hypotheses):
        requested_symbol = hypothesis.symbols[0] if hypothesis.symbols else SYMBOLS[0]
        requested_timeframe = hypothesis.timeframes[0] if hypothesis.timeframes else "1h"
        symbol = requested_symbol if requested_symbol in SYMBOLS else SYMBOLS[0]
        timeframe = requested_timeframe if requested_timeframe in SUPPORTED_TIMEFRAMES else "1h"
        df = market[(symbol, timeframe)]
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
            record = {
                "run_id": run_id, "index": idx, "symbol": symbol, "timeframe": timeframe,
                "hypothesis": hypothesis.model_dump(),
                "status": "VALIDATION_CANDIDATE" if evaluation.passed else "REJECTED",
                "in_sample": evaluation.in_sample, "out_of_sample": evaluation.out_of_sample,
                "robustness": evaluation.robustness, "rejection_reasons": evaluation.rejection_reasons,
            }
            if evaluation.passed and settings.output.generate_pine:
                pine = build_pine(
                    f"ACL {hypothesis.title[:50]}", family, hypothesis.executable_parameters,
                    allow_short=any(d in {"short", "both"} for d in directions),
                )
                (out / f"candidate_{idx + 1}.pine").write_text(pine, encoding="utf-8")
                record["pine_path"] = str(out / f"candidate_{idx + 1}.pine")
        except Exception as exc:
            record = {
                "run_id": run_id, "index": idx, "symbol": symbol, "timeframe": timeframe,
                "hypothesis": hypothesis.model_dump(), "status": "REJECTED", "error": str(exc),
            }
        records.append(record)
        print(f"[{idx + 1}/{len(hypotheses)}] {hypothesis.title} | {symbol} | {timeframe} -> {record['status']}")

    manifest = {
        "run_id": run_id, "created_at": now.isoformat(),
        "capital": settings.capital.model_dump(), "market_snapshot": snapshot,
        "hypothesis_count": len(hypotheses), "records": records,
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return manifest
