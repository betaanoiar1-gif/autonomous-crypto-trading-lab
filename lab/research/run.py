from __future__ import annotations

from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..local_agent import LocalAgent
from ..pine_factory import build_pine
from ..schemas import Direction
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


def _direction_value(direction) -> str:
    if isinstance(direction, Direction):
        return direction.value
    value = str(direction).strip().lower()
    return value if value in {"long", "short", "both"} else "long"


def _enforce_slot(hypothesis, slot):
    family = slot["preferred_family"]
    params = dict(hypothesis.executable_parameters or {})
    # Drop parameters that do not belong to this executable family.
    if family in {"momentum", "breakout"}:
        params = {"lookback": int(params.get("lookback", 20))}
    elif family == "mean_reversion":
        params = {
            "lookback": int(params.get("lookback", 40)),
            "z_entry": float(params.get("z_entry", 1.5)),
            "z_exit": float(params.get("z_exit", 0.25)),
        }
    elif family == "moving_average_cross":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 40))
        params = {"fast": max(2, min(100, fast)), "slow": max(max(2, min(100, fast)) + 1, min(300, slow))}
    return hypothesis.model_copy(update={
        "executable_family": family,
        "executable_parameters": params,
        "directions": [Direction(_direction_value(slot["preferred_direction"]))],
        "timeframes": [slot["preferred_timeframe"]],
        "symbols": [slot["preferred_symbol"]],
    })


def _load_failures(path):
    if not path.exists():
        return []
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines()[-100:]:
        try:
            rec = json.loads(line)
            if rec.get("status") == "REJECTED":
                failures.append(rec)
        except json.JSONDecodeError:
            continue
    return failures


def _fetch_market(symbols, timeframes):
    last_error = None
    for exchange_id in ("binance", "kraken"):
        try:
            adapter = CCXTMarketData(exchange_id=exchange_id)
            return exchange_id, adapter.fetch_multi_timeframes(symbols, timeframes, limit=1500)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Public market data unavailable on fallback exchanges: {last_error}")


def _score(record):
    if record.get("status") == "REJECTED":
        return float("-inf")
    oos = record.get("out_of_sample", {})
    robust = record.get("robustness", {})
    walk = robust.get("walk_forward", {})
    return float(oos.get("research_score", 0.0)) + (50.0 if walk.get("passed") else 0.0)


def _status(evaluation):
    if evaluation.passed:
        return "VALIDATED"
    oos = evaluation.out_of_sample
    robust = evaluation.robustness
    if (
        float(oos.get("total_return", 0.0)) > 0
        and float(oos.get("profit_factor", 0.0)) > 1.0
        and float(oos.get("max_drawdown", 0.0)) >= -0.50
        and float(robust.get("stressed_total_return", 0.0)) > 0
    ):
        return "VALIDATION_CANDIDATE"
    return "REJECTED"


def _print_diagnostics(record):
    oos = record.get("out_of_sample", {})
    robust = record.get("robustness", {})
    walk = robust.get("walk_forward", {})
    print(
        "    OOS: "
        f"return={float(oos.get('total_return', 0.0)):.2%} | "
        f"PF={float(oos.get('profit_factor', 0.0)):.2f} | "
        f"DD={float(oos.get('max_drawdown', 0.0)):.2%} | "
        f"trades={int(oos.get('trade_count', 0))} | "
        f"Sharpe={float(oos.get('sharpe', 0.0)):.2f}"
    )
    print(
        "    Robustness: "
        f"WF={'PASS' if walk.get('passed') else 'FAIL'} | "
        f"positive_folds={walk.get('positive_windows', 0)}/{len(walk.get('windows', []))} | "
        f"median_WF_return={float(walk.get('median_return', 0.0)):.2%} | "
        f"stress_return={float(robust.get('stressed_total_return', 0.0)):.2%} | "
        f"stability={'PASS' if robust.get('parameter_stability') else 'FAIL'}"
    )
    reasons = record.get("rejection_reasons") or []
    if reasons:
        print("    Reasons: " + "; ".join(reasons))
    if "error" in record:
        print("    Error: " + str(record["error"]))


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
        "observations": {f"{s}@{tf}": len(df) for (s, tf), df in market.items()},
    }
    print(f"Data source: {exchange_id} | timeframes={SUPPORTED_TIMEFRAMES} | symbols={SYMBOLS}")
    agent = agent or LocalAgent()

    hypotheses = []
    target = min(max_hypotheses, len(DIVERSITY_SLOTS))
    print(f"Research slots requested: {target}")
    for attempt in range(target):
        slot = DIVERSITY_SLOTS[attempt]
        try:
            raw = _safe_one_hypothesis(
                agent, snapshot, prior_failures,
                [h.model_dump(mode="json") for h in hypotheses], slot
            )
            h = _enforce_slot(raw, slot)
            signature = (
                h.executable_family,
                tuple(_direction_value(x) for x in h.directions),
                tuple(h.symbols),
                tuple(h.timeframes),
                h.thesis.strip().lower(),
            )
            if any(
                signature == (
                    x.executable_family,
                    tuple(_direction_value(y) for y in x.directions),
                    tuple(x.symbols),
                    tuple(x.timeframes),
                    x.thesis.strip().lower(),
                )
                for x in hypotheses
            ):
                h = h.model_copy(update={"thesis": h.thesis + f" | diversity slot {attempt + 1}"})
            hypotheses.append(h)
            print(
                f"  Slot {attempt + 1}: {h.executable_family} | {h.symbols[0]} | "
                f"{h.timeframes[0]} | {_direction_value(h.directions[0])}"
            )
        except Exception as exc:
            print(f"Hypothesis generation {attempt + 1} failed: {exc}")

    print(f"Hypotheses generated: {len(hypotheses)}")
    records = []
    for idx, h in enumerate(hypotheses):
        symbol = h.symbols[0] if h.symbols and h.symbols[0] in SYMBOLS else SYMBOLS[0]
        timeframe = h.timeframes[0] if h.timeframes and h.timeframes[0] in SUPPORTED_TIMEFRAMES else "1h"
        df = market[(symbol, timeframe)]
        directions = [_direction_value(x) for x in h.directions]
        try:
            evaluation = evaluate(
                df,
                h.executable_family or "momentum",
                h.executable_parameters,
                directions,
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
            )
            status = _status(evaluation)
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "hypothesis": h.model_dump(mode="json"),
                "status": status,
                "in_sample": evaluation.in_sample,
                "out_of_sample": evaluation.out_of_sample,
                "robustness": evaluation.robustness,
                "rejection_reasons": evaluation.rejection_reasons,
            }
            if status == "VALIDATED" and settings.output.generate_pine:
                pine = build_pine(
                    f"ACL {h.title[:50]}",
                    h.executable_family or "momentum",
                    h.executable_parameters,
                    allow_short=any(d in {"short", "both"} for d in directions),
                )
                path = out / f"validated_candidate_{idx + 1}.pine"
                path.write_text(pine, encoding="utf-8")
                record["pine_path"] = str(path)
        except Exception as exc:
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "hypothesis": h.model_dump(mode="json"),
                "status": "REJECTED",
                "error": str(exc),
            }
        records.append(record)
        print(f"[{idx + 1}/{len(hypotheses)}] {h.title} | {symbol} | {timeframe} -> {record['status']}")
        _print_diagnostics(record)

    records.sort(key=_score, reverse=True)
    for rank, record in enumerate(records, 1):
        record["rank"] = rank
    leaderboard = [
        {
            "rank": r["rank"],
            "title": r["hypothesis"]["title"],
            "status": r["status"],
            "score": r.get("out_of_sample", {}).get("research_score", 0.0),
            "symbol": r["symbol"],
            "timeframe": r["timeframe"],
            "walk_forward_passed": r.get("robustness", {}).get("walk_forward", {}).get("passed", False),
        }
        for r in records
    ]
    print("Leaderboard:")
    for row in leaderboard:
        print(
            f"  #{row['rank']} | {row['status']} | score={float(row['score']):.2f} | "
            f"{row['title']} | {row['symbol']} | {row['timeframe']} | WF={row['walk_forward_passed']}"
        )

    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "capital": settings.capital.model_dump(),
        "market_snapshot": snapshot,
        "hypothesis_count": len(hypotheses),
        "records": records,
        "leaderboard": leaderboard,
    }
    (out / "run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return manifest
