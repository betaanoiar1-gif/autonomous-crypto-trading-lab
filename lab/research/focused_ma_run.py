from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, load_settings
from .evaluator import _metrics, _run, frozen_confirmation
from .run import _fetch_markets_cached

# Small, deliberate grid around the region that repeatedly appeared in prior runs.
MA_PAIRS = [
    (28, 84), (30, 90), (32, 96), (34, 102), (36, 108),
    (38, 114), (40, 120), (42, 126), (44, 132), (45, 135),
    (50, 150), (55, 165), (60, 180), (70, 210), (80, 240),
]


def _run_pair(df, fast: int, slow: int, settings):
    params = {"fast": fast, "slow": slow}
    result = _run(
        df,
        "moving_average_cross",
        params,
        ["both"],
        settings.capital.initial_usd,
        settings.execution.commission_bps,
        settings.execution.slippage_bps,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return _metrics(result, result.returns)


def _window_metrics(df, params, settings, windows=4):
    n = len(df)
    size = n // windows
    out = []
    for i in range(windows):
        start = i * size
        end = n if i == windows - 1 else (i + 1) * size
        chunk = df.iloc[start:end].copy()
        m = _run_pair(chunk, params["fast"], params["slow"], settings)
        out.append({
            "window": i + 1,
            "start": str(chunk.index[0]),
            "end": str(chunk.index[-1]),
            "total_return": float(m["total_return"]),
            "profit_factor": float(m["profit_factor"]),
            "max_drawdown": float(m["max_drawdown"]),
            "trade_count": int(m["trade_count"]),
            "sharpe": float(m["sharpe"]),
        })
    return out


def _score(row):
    # Reward consistency, not one spectacular period.
    return (
        100.0 * row["median_return"]
        + 40.0 * row["worst_return"]
        + 8.0 * (row["median_pf"] - 1.0)
        + 2.0 * row["median_sharpe"]
        - 15.0 * row["worst_dd_abs"]
    )


def run() -> dict:
    settings = load_settings()
    spot, _futures, _funding, _futures_error, snapshot = _fetch_markets_cached()
    df = spot[("ETH/USDT", "4h")].copy()
    if len(df) < 1000:
        raise RuntimeError(f"Need at least 1000 ETH/USDT 4h bars, got {len(df)}")

    print("=== FOCUSED MA RESEARCH ===", flush=True)
    print("Market: ETH/USDT | timeframe=4h | spot", flush=True)
    print(f"Bars: {len(df)}", flush=True)
    print(f"Pairs tested: {len(MA_PAIRS)}", flush=True)
    print("Parameters are evaluated as frozen pairs; no tuning inside a test window.", flush=True)

    rows = []
    for fast, slow in MA_PAIRS:
        windows = _window_metrics(df, {"fast": fast, "slow": slow}, settings, windows=4)
        recent_cut = int(len(df) * 0.25)
        recent = df.iloc[-recent_cut:].copy()
        recent_m = _run_pair(recent, fast, slow, settings)
        returns = [x["total_return"] for x in windows]
        pfs = [x["profit_factor"] for x in windows]
        dds = [x["max_drawdown"] for x in windows]
        sharpes = [x["sharpe"] for x in windows]
        row = {
            "parameters": {"fast": fast, "slow": slow},
            "windows": windows,
            "median_return": float(sorted(returns)[len(returns)//2]),
            "worst_return": float(min(returns)),
            "positive_windows": int(sum(r > 0 for r in returns)),
            "median_pf": float(sorted(pfs)[len(pfs)//2]),
            "worst_dd_abs": float(abs(min(dds))),
            "median_sharpe": float(sorted(sharpes)[len(sharpes)//2]),
            "recent": {
                "total_return": float(recent_m["total_return"]),
                "profit_factor": float(recent_m["profit_factor"]),
                "max_drawdown": float(recent_m["max_drawdown"]),
                "trade_count": int(recent_m["trade_count"]),
                "sharpe": float(recent_m["sharpe"]),
            },
        }
        row["score"] = _score(row)
        rows.append(row)
        print(
            f"MA {fast}/{slow}: median={row['median_return']:.2%} | worst={row['worst_return']:.2%} "
            f"| +windows={row['positive_windows']}/4 | medianPF={row['median_pf']:.2f} "
            f"| worstDD={-row['worst_dd_abs']:.2%} | recent={row['recent']['total_return']:.2%}",
            flush=True,
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    best = rows[0]
    params = best["parameters"]
    print("\n=== TOP 5 ===", flush=True)
    for i, row in enumerate(rows[:5], 1):
        print(
            f"#{i} {row['parameters']} score={row['score']:.2f} median={row['median_return']:.2%} "
            f"worst={row['worst_return']:.2%} +windows={row['positive_windows']}/4 "
            f"recent={row['recent']['total_return']:.2%}", flush=True,
        )

    # One frozen cross-timeframe check only for the best pair.
    next_tf = spot[("ETH/USDT", "1h")].copy()
    confirmation = frozen_confirmation(
        next_tf,
        "moving_average_cross",
        dict(params),
        ["both"],
        settings.capital.initial_usd,
        settings.execution.commission_bps,
        settings.execution.slippage_bps,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    cm = confirmation["metrics"]
    print("\n=== FROZEN 1h CHECK FOR BEST PAIR ===", flush=True)
    print(
        f"params={params} | return={cm['total_return']:.2%} | PF={cm['profit_factor']:.2f} "
        f"| DD={cm['max_drawdown']:.2%} | trades={cm['trade_count']} | Sharpe={cm['sharpe']:.2f} "
        f"| PASS={confirmation['passed']}",
        flush=True,
    )

    decision = "PROMOTE_TO_PAPER" if (
        best["positive_windows"] >= 3
        and best["median_return"] > 0
        and best["median_pf"] > 1.0
        and best["worst_dd_abs"] < 0.50
        and best["recent"]["total_return"] > 0
        and confirmation["passed"]
    ) else "REJECT_MA_HYPOTHESIS"

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market": {"symbol": "ETH/USDT", "timeframe": "4h", "market_type": "spot"},
        "bars": len(df),
        "pairs_tested": MA_PAIRS,
        "results": rows,
        "best": best,
        "frozen_1h_confirmation": confirmation,
        "decision": decision,
        "snapshot": snapshot,
    }
    out = ROOT / "experiments" / "focused_ma_latest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nDECISION: {decision}", flush=True)
    print(f"Saved: {out}", flush=True)
    return payload


if __name__ == "__main__":
    run()
