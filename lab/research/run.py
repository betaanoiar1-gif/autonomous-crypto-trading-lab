from __future__ import annotations

from datetime import datetime, timezone
import json

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from ..local_agent import LocalAgent
from ..pine_factory import build_pine
from ..schemas import Direction, MarketType
from .parser import parse_hypotheses
from .prompt import build_prompt, SYSTEM as RESEARCH_SYSTEM
from .evaluator import evaluate

DIVERSITY_SLOTS = [
    {"preferred_family": "momentum", "preferred_market": "spot", "preferred_timeframe": "15m", "preferred_symbol": "BTC/USDT", "preferred_direction": "long"},
    {"preferred_family": "mean_reversion", "preferred_market": "spot", "preferred_timeframe": "1h", "preferred_symbol": "ETH/USDT", "preferred_direction": "both"},
    {"preferred_family": "breakout", "preferred_market": "spot", "preferred_timeframe": "4h", "preferred_symbol": "BTC/USDT", "preferred_direction": "both"},
    {"preferred_family": "moving_average_cross", "preferred_market": "spot", "preferred_timeframe": "1h", "preferred_symbol": "ETH/USDT", "preferred_direction": "short"},
    {"preferred_family": "rsi_reversion", "preferred_market": "spot", "preferred_timeframe": "1h", "preferred_symbol": "BTC/USDT", "preferred_direction": "both"},
    {"preferred_family": "atr_breakout", "preferred_market": "spot", "preferred_timeframe": "15m", "preferred_symbol": "ETH/USDT", "preferred_direction": "both"},
    {"preferred_family": "trend_pullback", "preferred_market": "spot", "preferred_timeframe": "4h", "preferred_symbol": "ETH/USDT", "preferred_direction": "long"},
    {"preferred_family": "channel_reversion", "preferred_market": "spot", "preferred_timeframe": "1h", "preferred_symbol": "BTC/USDT", "preferred_direction": "both"},
]
SUPPORTED_TIMEFRAMES = ["15m", "1h", "4h"]
SYMBOLS = ["BTC/USDT", "ETH/USDT"]


def _safe_one_hypothesis(agent, snapshot, prior_failures, prior_hypotheses, slot):
    prompt = build_prompt(snapshot, prior_failures, 1, prior_hypotheses, diversity_slot=slot)
    try:
        return parse_hypotheses(agent.chat(prompt, system=RESEARCH_SYSTEM))[0]
    except (ValueError, IndexError):
        retry = (
            "Return exactly ONE hypothesis block. No JSON, markdown, analysis, or commentary. "
            f"Use this slot exactly: {slot}\n\n" + prompt
        )
        return parse_hypotheses(agent.chat(retry, system=RESEARCH_SYSTEM))[0]


def _direction_value(direction) -> str:
    if isinstance(direction, Direction):
        return direction.value
    value = str(direction).strip().lower()
    return value if value in {"long", "short", "both"} else "long"


def _to_int(value, default, lo, hi):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _to_float(value, default, lo, hi):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _enforce_slot(hypothesis, slot):
    family = slot["preferred_family"]
    raw = dict(hypothesis.executable_parameters or {})
    if family in {"momentum", "breakout"}:
        params = {"lookback": _to_int(raw.get("lookback", 20), 20, 2, 200)}
        label = "Momentum" if family == "momentum" else "Breakout"
        title = f"{label} | lookback={params['lookback']}"
    elif family == "mean_reversion":
        entry = _to_float(raw.get("z_entry", 1.5), 1.5, 0.8, 3.5)
        exit_ = min(_to_float(raw.get("z_exit", 0.25), 0.25, 0.05, 1.5), max(0.05, entry - 0.05))
        params = {"lookback": _to_int(raw.get("lookback", 40), 40, 10, 200), "z_entry": entry, "z_exit": exit_}
        title = f"Mean Reversion | lookback={params['lookback']} | z_entry={entry:.2f} | z_exit={exit_:.2f}"
    elif family == "moving_average_cross":
        fast = _to_int(raw.get("fast", 10), 10, 2, 100)
        slow = max(fast + 1, _to_int(raw.get("slow", 40), 40, 3, 300))
        params = {"fast": fast, "slow": slow}
        title = f"Moving Average Cross | fast={fast} | slow={slow}"
    elif family == "rsi_reversion":
        params = {
            "rsi_length": _to_int(raw.get("rsi_length", 14), 14, 2, 50),
            "rsi_low": _to_float(raw.get("rsi_low", 30), 30, 5, 45),
            "rsi_high": _to_float(raw.get("rsi_high", 70), 70, 55, 95),
        }
        title = f"RSI Reversion | len={params['rsi_length']} | low={params['rsi_low']:.1f} | high={params['rsi_high']:.1f}"
    elif family == "atr_breakout":
        params = {
            "atr_length": _to_int(raw.get("atr_length", 14), 14, 2, 50),
            "atr_mult": _to_float(raw.get("atr_mult", 1.5), 1.5, 0.25, 5),
        }
        title = f"ATR Breakout | len={params['atr_length']} | mult={params['atr_mult']:.2f}"
    elif family == "trend_pullback":
        params = {
            "lookback": _to_int(raw.get("lookback", 40), 40, 5, 200),
            "pullback_threshold": _to_float(raw.get("pullback_threshold", 0.01), 0.01, 0.001, 0.10),
        }
        title = f"Trend Pullback | lookback={params['lookback']} | threshold={params['pullback_threshold']:.3f}"
    elif family == "channel_reversion":
        params = {"channel_length": _to_int(raw.get("channel_length", 40), 40, 5, 200)}
        title = f"Channel Reversion | length={params['channel_length']}"
    else:
        params = {}
        title = hypothesis.title
    return hypothesis.model_copy(update={
        "title": title,
        "executable_family": family,
        "executable_parameters": params,
        "market_types": [MarketType.SPOT],
        "directions": [Direction(_direction_value(slot["preferred_direction"]))],
        "timeframes": [slot["preferred_timeframe"]],
        "symbols": [slot["preferred_symbol"]],
    })


def _load_failures(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-100:]:
        try:
            rec = json.loads(line)
            if rec.get("status") == "REJECTED":
                out.append(rec)
        except json.JSONDecodeError:
            pass
    return out


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
    walk = record.get("robustness", {}).get("walk_forward", {})
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
        f"    OOS: return={float(oos.get('total_return', 0)):.2%} | "
        f"PF={float(oos.get('profit_factor', 0)):.2f} | "
        f"DD={float(oos.get('max_drawdown', 0)):.2%} | "
        f"trades={int(oos.get('trade_count', 0))} | "
        f"Sharpe={float(oos.get('sharpe', 0)):.2f}"
    )
    print(
        f"    Robustness: WF={'PASS' if walk.get('passed') else 'FAIL'} | "
        f"positive_folds={walk.get('positive_windows', 0)}/{len(walk.get('windows', []))} | "
        f"median_WF_return={float(walk.get('median_return', 0)):.2%} | "
        f"stress_return={float(robust.get('stressed_total_return', 0)):.2%} | "
        f"stability={'PASS' if robust.get('parameter_stability') else 'FAIL'}"
    )
    if record.get("rejection_reasons"):
        print("    Reasons: " + "; ".join(record["rejection_reasons"]))
    if record.get("error"):
        print("    Error: " + str(record["error"]))


def run(max_hypotheses: int = 8, agent=None) -> dict:
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

    records = []
    for idx, h in enumerate(hypotheses):
        symbol = h.symbols[0] if h.symbols and h.symbols[0] in SYMBOLS else SYMBOLS[0]
        timeframe = h.timeframes[0] if h.timeframes and h.timeframes[0] in SUPPORTED_TIMEFRAMES else "1h"
        directions = [_direction_value(x) for x in h.directions]
        market_type = h.market_types[0].value if h.market_types else "spot"
        if market_type == "futures":
            record = {
                "run_id": run_id,
                "index": idx,
                "symbol": symbol,
                "timeframe": timeframe,
                "market_type": market_type,
                "hypothesis": h.model_dump(mode="json"),
                "status": "REJECTED",
                "error": "Futures validation is gated until historical funding-rate data is connected; refusing to score Futures as Spot.",
            }
        else:
            try:
                evaluation = evaluate(
                    market[(symbol, timeframe)], h.executable_family, h.executable_parameters, directions,
                    settings.capital.initial_usd, settings.execution.commission_bps,
                    settings.execution.slippage_bps, settings.validation.holdout_ratio,
                )
                status = _status(evaluation)
                record = {
                    "run_id": run_id,
                    "index": idx,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "market_type": market_type,
                    "hypothesis": h.model_dump(mode="json"),
                    "status": status,
                    "in_sample": evaluation.in_sample,
                    "out_of_sample": evaluation.out_of_sample,
                    "robustness": evaluation.robustness,
                    "rejection_reasons": evaluation.rejection_reasons,
                }
                if status == "VALIDATED" and settings.output.generate_pine:
                    pine = build_pine(
                        f"ACL {h.title[:50]}", h.executable_family, h.executable_parameters,
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
                    "market_type": market_type,
                    "hypothesis": h.model_dump(mode="json"),
                    "status": "REJECTED",
                    "error": str(exc),
                }
        records.append(record)
        print(f"[{idx + 1}/{len(hypotheses)}] {h.title} | {symbol} | {timeframe} -> {record['status']}")
        _print_diagnostics(record)

    records.sort(key=_score, reverse=True)
    for rank, r in enumerate(records, 1):
        r["rank"] = rank
    leaderboard = [
        {
            "rank": r["rank"], "title": r["hypothesis"]["title"], "status": r["status"],
            "score": r.get("out_of_sample", {}).get("research_score", 0.0), "symbol": r["symbol"],
            "timeframe": r["timeframe"], "market_type": r["market_type"],
            "walk_forward_passed": r.get("robustness", {}).get("walk_forward", {}).get("passed", False),
        }
        for r in records
    ]
    print("Leaderboard:")
    for r in leaderboard:
        print(
            f"  #{r['rank']} | {r['status']} | score={float(r['score']):.2f} | "
            f"{r['title']} | {r['symbol']} | {r['timeframe']} | {r['market_type']} | WF={r['walk_forward_passed']}"
        )

    manifest = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "capital": settings.capital.model_dump(),
        "market_snapshot": snapshot,
        "hypothesis_count": len(hypotheses),
        "records": records,
        "leaderboard": leaderboard,
        "futures_validation": "gated_until_funding_history_adapter",
    }
    (out / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return manifest
