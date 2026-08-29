from __future__ import annotations

"""Targeted re-test of recurring near-miss strategies.

Uses historical evidence from the postmortem as a seed, but does not run the
broad grid again. Parameters are frozen for every evaluation and each candidate
is checked on independent symbol/timeframe combinations.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import ROOT, load_settings
from .evaluator import frozen_confirmation
from .fast_evaluator import evaluate_fixed
from .run import _fetch_markets_cached


NEAR_MISS_SEEDS = [
    ("momentum", {"lookback": 143}, "ETH/USDT", "1h"),
    ("momentum", {"lookback": 158}, "ETH/USDT", "1h"),
    ("moving_average_cross", {"fast": 22, "slow": 88}, "BTC/USDT", "1h"),
    ("rsi_reversion", {"rsi_length": 25, "rsi_low": 40, "rsi_high": 85}, "BTC/USDT", "1h"),
    ("atr_breakout", {"atr_length": 5, "atr_mult": 1.0}, "ETH/USDT", "15m"),
    ("trend_pullback", {"lookback": 15, "pullback_threshold": 0.01}, "ETH/USDT", "4h"),
    ("channel_reversion", {"channel_length": 40}, "BTC/USDT", "1h"),
]


def _independent_pairs(symbol: str, timeframe: str) -> list[tuple[str, str]]:
    pairs = []
    other = "BTC/USDT" if symbol != "BTC/USDT" else "ETH/USDT"
    if timeframe != "4h":
        pairs.append((symbol, "4h"))
    if timeframe != "1h":
        pairs.append((symbol, "1h"))
    pairs.append((other, "1h"))
    if (other, "4h") not in pairs:
        pairs.append((other, "4h"))
    return pairs


def _score(oos: dict, wf: dict) -> float:
    return (
        float(oos.get("total_return", 0.0)) * 100
        + (float(oos.get("profit_factor", 0.0)) - 1.0) * 20
        + float(oos.get("sharpe", 0.0)) * 2
        + int(wf.get("positive_windows", 0)) * 3
        + float(wf.get("median_return", 0.0)) * 20
    )


def run() -> dict:
    settings = load_settings()
    spot, _, _, futures_error, _ = _fetch_markets_cached()

    print("=== TARGETED NEAR-MISS RESEARCH ===", flush=True)
    print("Broad grid: DISABLED", flush=True)
    print("Futures: DISABLED", flush=True)
    print("Live trading: DISABLED", flush=True)
    print(f"Seeds: {len(NEAR_MISS_SEEDS)}", flush=True)
    print()

    primary_rows = []

    for family, params, symbol, timeframe in NEAR_MISS_SEEDS:
        key = (symbol, timeframe)
        if key not in spot:
            print(f"SKIP {family} {params}: missing {symbol} {timeframe}", flush=True)
            continue

        try:
            ev = evaluate_fixed(
                spot[key],
                family,
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
            oos = ev.out_of_sample
            wf = ev.robustness.get("walk_forward", {})
            row = {
                "family": family,
                "parameters": params,
                "symbol": symbol,
                "timeframe": timeframe,
                "oos": oos,
                "robustness": ev.robustness,
                "passed_primary": bool(ev.passed),
                "score": _score(oos, wf),
            }
            primary_rows.append(row)
            print(
                f"{family} {params} | {symbol} {timeframe} | "
                f"OOS={float(oos.get('total_return',0)):.2%} | "
                f"PF={float(oos.get('profit_factor',0)):.2f} | "
                f"DD={float(oos.get('max_drawdown',0)):.2%} | "
                f"trades={int(oos.get('trade_count',0))} | "
                f"WF={wf.get('positive_windows',0)}/{len(wf.get('windows',[]))} | "
                f"PRIMARY={bool(ev.passed)}",
                flush=True,
            )
        except Exception as exc:
            print(f"ERROR {family} {params}: {type(exc).__name__}: {exc}", flush=True)

    primary_rows.sort(key=lambda r: r["score"], reverse=True)

    print()
    print("=== TOP NEAR-MISS CANDIDATES ===", flush=True)
    for i, row in enumerate(primary_rows[:5], 1):
        o = row["oos"]
        w = row["robustness"].get("walk_forward", {})
        print(
            f"#{i} {row['family']} {row['parameters']} | {row['symbol']} {row['timeframe']} | "
            f"OOS={float(o.get('total_return',0)):.2%} | PF={float(o.get('profit_factor',0)):.2f} | "
            f"WF={w.get('positive_windows',0)}/{len(w.get('windows',[]))} | PRIMARY={row['passed_primary']}",
            flush=True,
        )

    confirmations = []
    winner = None

    for row in primary_rows[:5]:
        if winner is not None:
            break
        if not row["passed_primary"] and float(row["oos"].get("total_return", 0.0)) <= 0:
            continue

        family = row["family"]
        params = row["parameters"]
        symbol = row["symbol"]
        timeframe = row["timeframe"]

        print()
        print(f"=== INDEPENDENT TEST: {family} {params} ===", flush=True)

        passed_all = True
        row_confs = []
        for cs, ct in _independent_pairs(symbol, timeframe):
            if (cs, ct) not in spot:
                passed_all = False
                continue
            try:
                c = frozen_confirmation(
                    spot[(cs, ct)], family, dict(params), ["both"],
                    settings.capital.initial_usd,
                    settings.execution.commission_bps,
                    settings.execution.slippage_bps,
                    market_type="spot", leverage=1.0, funding_rates=None,
                )
                m = c.get("metrics", c)
                item = {
                    "symbol": cs,
                    "timeframe": ct,
                    "result": c,
                }
                row_confs.append(item)
                print(
                    f"{cs} {ct}: return={float(m.get('total_return',0)):.2%} | "
                    f"PF={float(m.get('profit_factor',0)):.2f} | "
                    f"DD={float(m.get('max_drawdown',0)):.2%} | "
                    f"trades={int(m.get('trade_count',0))} | "
                    f"PASS={bool(c.get('passed',False))}",
                    flush=True,
                )
                passed_all &= bool(c.get("passed", False))
            except Exception as exc:
                passed_all = False
                print(f"{cs} {ct}: ERROR {type(exc).__name__}: {exc}", flush=True)

        confirmations.append({"candidate": row, "confirmations": row_confs, "all_passed": passed_all})
        if row["passed_primary"] and passed_all:
            winner = row

    decision = "PROMOTE_TO_PAPER" if winner else "REJECT_NEAR_MISSES"

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": [
            {"family": f, "parameters": p, "symbol": s, "timeframe": t}
            for f, p, s, t in NEAR_MISS_SEEDS
        ],
        "primary_results": primary_rows,
        "confirmations": confirmations,
        "winner": winner,
        "decision": decision,
        "futures_error": str(futures_error) if futures_error else None,
    }

    out = ROOT / "experiments" / "targeted_near_misses_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    print("=== FINAL DECISION ===", flush=True)
    print("Decision:", decision, flush=True)
    print("Saved:", out, flush=True)
    return result


if __name__ == "__main__":
    run()
