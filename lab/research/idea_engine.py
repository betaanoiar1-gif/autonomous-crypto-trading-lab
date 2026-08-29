from __future__ import annotations

"""Focused autonomous research with bounded refinement.

The engine evaluates a small set of genuinely different research ideas, ranks
them on a holdout split with frozen parameters inside four validation blocks,
then freezes the best candidate for an independent market/timeframe test.

When a promising candidate narrowly misses a gate, the engine performs one
small refinement ring around that candidate instead of restarting a broad
search. Gates themselves are never loosened.

Research/paper-only. No exchange orders are created here.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from .run import _fetch_markets_cached
from .evaluator import _metrics


@dataclass(frozen=True)
class Idea:
    name: str
    params: dict


def _costed_run(df, signal: pd.Series, settings):
    return run_ohlcv(
        df,
        signal,
        settings.capital.initial_usd,
        settings.execution.commission_bps,
        settings.execution.slippage_bps,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )


def _volatility(close: pd.Series, window: int = 24) -> pd.Series:
    return close.pct_change().rolling(window).std()


def _signal(df: pd.DataFrame, idea: Idea) -> pd.Series:
    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)
    name = idea.name
    p = idea.params

    out = pd.Series(0.0, index=df.index)

    if name == "trend_volatility":
        fast = int(p["fast"])
        slow = int(p["slow"])
        vol_window = int(p["vol_window"])
        vol_floor = float(p["vol_floor"])
        vol_cap = float(p["vol_cap"])
        fast_ma = close.ewm(span=fast, adjust=False).mean()
        slow_ma = close.ewm(span=slow, adjust=False).mean()
        vol = _volatility(close, vol_window)
        allowed = (vol >= vol_floor) & (vol <= vol_cap)
        out[(fast_ma > slow_ma) & allowed] = 1.0
        out[(fast_ma < slow_ma) & allowed] = -1.0

    elif name == "breakout_volume":
        lookback = int(p["lookback"])
        volume_window = int(p["volume_window"])
        volume_mult = float(p["volume_mult"])
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        prior_high = high.shift(1).rolling(lookback).max()
        prior_low = low.shift(1).rolling(lookback).min()
        avg_volume = volume.shift(1).rolling(volume_window).mean()
        liquid = volume.shift(1) >= avg_volume * volume_mult
        out[(close > prior_high) & liquid] = 1.0
        out[(close < prior_low) & liquid] = -1.0

    elif name == "momentum_regime":
        lookback = int(p["lookback"])
        ema = int(p["ema"])
        vol_window = int(p["vol_window"])
        vol_cap = float(p["vol_cap"])
        momentum = close.pct_change(lookback)
        trend = close > close.ewm(span=ema, adjust=False).mean()
        vol = _volatility(close, vol_window)
        allowed = vol <= vol_cap
        out[(momentum > 0) & trend & allowed] = 1.0
        out[(momentum < 0) & (~trend) & allowed] = -1.0

    elif name == "rsi_trend":
        length = int(p["rsi_length"])
        low = float(p["rsi_low"])
        high = float(p["rsi_high"])
        trend = int(p["trend"])
        ema = close.ewm(span=trend, adjust=False).mean()
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        out[(rsi < low) & (close > ema)] = 1.0
        out[(rsi > high) & (close < ema)] = -1.0

    elif name == "channel_volatility":
        window = int(p["window"])
        vol_window = int(p["vol_window"])
        vol_cap = float(p["vol_cap"])
        q_low = float(p["q_low"])
        q_high = float(p["q_high"])
        roll = close.rolling(window)
        low_band = roll.quantile(q_low)
        high_band = roll.quantile(q_high)
        vol = _volatility(close, vol_window)
        allowed = vol <= vol_cap
        out[(close < low_band) & allowed] = 1.0
        out[(close > high_band) & allowed] = -1.0

    else:
        raise ValueError(f"Unknown research idea: {name}")

    return out.shift(1).fillna(0.0)


def _evaluate_idea(df: pd.DataFrame, idea: Idea, settings):
    split = int(len(df) * 0.70)
    train = df.iloc[:split]
    test = df.iloc[split:]

    train_result = _costed_run(train, _signal(train, idea), settings)
    test_result = _costed_run(test, _signal(test, idea), settings)

    train_metrics = _metrics(train_result, train_result.returns)
    test_metrics = _metrics(test_result, test_result.returns)

    block_size = len(test) // 4
    folds = []
    for i in range(4):
        a = i * block_size
        b = (i + 1) * block_size if i < 3 else len(test)
        block = test.iloc[a:b]
        result = _costed_run(block, _signal(block, idea), settings)
        m = _metrics(result, result.returns)
        folds.append({
            "fold": i + 1,
            "return": float(m["total_return"]),
            "profit_factor": float(m["profit_factor"]),
            "drawdown": float(m["max_drawdown"]),
            "trades": int(m["trade_count"]),
            "sharpe": float(m["sharpe"]),
        })

    fold_returns = [x["return"] for x in folds]
    positive = sum(x > 0 for x in fold_returns)
    median_fold = float(np.median(fold_returns))
    worst_fold = float(min(fold_returns))
    median_pf = float(np.median([x["profit_factor"] for x in folds]))
    min_trades = min(x["trades"] for x in folds)

    score = (
        float(test_metrics["total_return"]) * 100
        + (float(test_metrics["profit_factor"]) - 1.0) * 30
        + float(test_metrics["sharpe"]) * 3
        + median_fold * 75
        + (positive * 4)
        - max(abs(float(test_metrics["max_drawdown"])) - 0.40, 0) * 40
    )

    return {
        "idea": idea.name,
        "parameters": dict(idea.params),
        "train": {
            k: train_metrics[k]
            for k in (
                "total_return",
                "profit_factor",
                "max_drawdown",
                "trade_count",
                "sharpe",
            )
        },
        "holdout": {
            k: test_metrics[k]
            for k in (
                "total_return",
                "profit_factor",
                "max_drawdown",
                "trade_count",
                "sharpe",
            )
        },
        "frozen_folds": folds,
        "positive_folds": positive,
        "median_fold_return": median_fold,
        "worst_fold_return": worst_fold,
        "median_fold_pf": median_pf,
        "min_fold_trades": min_trades,
        "score": float(score),
    }


def _gate_reasons(record: dict) -> list[str]:
    h = record["holdout"]
    reasons = []

    if float(h["total_return"]) <= 0:
        reasons.append("non_positive_oos_return")
    if float(h["profit_factor"]) <= 1:
        reasons.append("oos_profit_factor_le_1")
    if float(h["max_drawdown"]) < -0.50:
        reasons.append("oos_drawdown_over_50pct")
    if int(h["trade_count"]) < 8:
        reasons.append("oos_too_few_trades")
    if int(record["positive_folds"]) < 3:
        reasons.append("wf_positive_folds_lt_3")
    if float(record["median_fold_return"]) <= 0:
        reasons.append("wf_median_return_non_positive")
    if float(record["median_fold_pf"]) <= 1:
        reasons.append("wf_median_pf_le_1")
    if int(record["min_fold_trades"]) < 4:
        reasons.append("wf_min_fold_trades_lt_4")

    return reasons


def _primary_gate(record: dict) -> bool:
    return not _gate_reasons(record)


def _independent(df: pd.DataFrame, idea: Idea, settings):
    result = _costed_run(df, _signal(df, idea), settings)
    m = _metrics(result, result.returns)
    passed = bool(
        float(m["total_return"]) > 0
        and float(m["profit_factor"]) > 1
        and float(m["max_drawdown"]) >= -0.50
        and int(m["trade_count"]) >= 8
    )
    return {
        "total_return": float(m["total_return"]),
        "profit_factor": float(m["profit_factor"]),
        "max_drawdown": float(m["max_drawdown"]),
        "trade_count": int(m["trade_count"]),
        "sharpe": float(m["sharpe"]),
        "passed": passed,
        "reasons": []
        if passed
        else [
            reason
            for reason, bad in (
                ("non_positive_return", float(m["total_return"]) <= 0),
                ("profit_factor_le_1", float(m["profit_factor"]) <= 1),
                ("drawdown_over_50pct", float(m["max_drawdown"]) < -0.50),
                ("too_few_trades", int(m["trade_count"]) < 8),
            )
            if bad
        ],
    }


def _refinement_ideas(best: dict) -> list[Idea]:
    """Generate one bounded refinement ring around a promising winner."""
    if best["idea"] != "trend_volatility":
        return []

    p = best["parameters"]
    fast = int(p["fast"])
    slow = int(p["slow"])
    vol_window = int(p["vol_window"])
    floor = float(p["vol_floor"])
    cap = float(p["vol_cap"])

    specs = [
        (max(10, fast - 5), max(fast + 20, slow - 10), floor, cap),
        (fast, slow, floor * 0.9, cap),
        (fast, slow, floor * 1.1, cap),
        (fast, slow, floor, min(0.050, cap + 0.005)),
        (fast, slow, floor, max(floor + 0.010, cap - 0.005)),
        (max(10, fast + 5), slow + 10, floor, cap),
        (max(10, fast - 5), max(fast + 20, slow - 5), floor * 0.9, cap + 0.005),
        (fast + 5, slow + 5, floor * 1.1, max(floor + 0.010, cap - 0.005)),
    ]

    ideas = []
    seen = set()
    for f, s, fl, cp in specs:
        if s <= f or fl <= 0 or cp <= fl:
            continue
        params = {
            "fast": int(f),
            "slow": int(s),
            "vol_window": int(vol_window),
            "vol_floor": round(float(fl), 6),
            "vol_cap": round(float(cp), 6),
        }
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        ideas.append(Idea("trend_volatility", params))

    return ideas


def run() -> dict:
    settings = load_settings()
    spot, _, _, _, _ = _fetch_markets_cached()

    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]

    ideas = [
        Idea("trend_volatility", {"fast": 20, "slow": 80, "vol_window": 24, "vol_floor": 0.005, "vol_cap": 0.035}),
        Idea("trend_volatility", {"fast": 30, "slow": 120, "vol_window": 24, "vol_floor": 0.007, "vol_cap": 0.035}),
        Idea("breakout_volume", {"lookback": 20, "volume_window": 24, "volume_mult": 1.10}),
        Idea("breakout_volume", {"lookback": 40, "volume_window": 48, "volume_mult": 1.20}),
        Idea("breakout_volume", {"lookback": 60, "volume_window": 72, "volume_mult": 1.25}),
        Idea("momentum_regime", {"lookback": 30, "ema": 80, "vol_window": 24, "vol_cap": 0.030}),
        Idea("momentum_regime", {"lookback": 60, "ema": 120, "vol_window": 24, "vol_cap": 0.035}),
        Idea("momentum_regime", {"lookback": 120, "ema": 180, "vol_window": 24, "vol_cap": 0.040}),
        Idea("rsi_trend", {"rsi_length": 7, "rsi_low": 25, "rsi_high": 75, "trend": 80}),
        Idea("rsi_trend", {"rsi_length": 14, "rsi_low": 30, "rsi_high": 70, "trend": 120}),
        Idea("rsi_trend", {"rsi_length": 21, "rsi_low": 35, "rsi_high": 65, "trend": 180}),
        Idea("channel_volatility", {"window": 30, "vol_window": 24, "vol_cap": 0.025, "q_low": 0.10, "q_high": 0.90}),
    ]

    print("=== AUTONOMOUS IDEA RESEARCH ===", flush=True)
    print("Primary: ETH/USDT | 1h | spot", flush=True)
    print("Independent: BTC/USDT | 4h | spot", flush=True)
    print("Ideas: 12 + bounded refinement when justified | Futures: disabled | Live trading: disabled", flush=True)
    print(f"Primary bars: {len(primary)}", flush=True)
    print()

    results = []

    for idx, idea in enumerate(ideas, 1):
        try:
            r = _evaluate_idea(primary, idea, settings)
            r["gate_reasons"] = _gate_reasons(r)
            results.append(r)

            print(
                f"[{idx:02d}] {r['idea']} {r['parameters']} | "
                f"OOS={r['holdout']['total_return']:.2%} | "
                f"PF={r['holdout']['profit_factor']:.2f} | "
                f"DD={r['holdout']['max_drawdown']:.2%} | "
                f"trades={r['holdout']['trade_count']} | "
                f"WF={r['positive_folds']}/4 | "
                f"score={r['score']:.2f}",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[{idx:02d}] {idea.name} ERROR: {type(exc).__name__}: {exc}",
                flush=True,
            )

    if not results:
        raise RuntimeError("No research idea could be evaluated.")

    results.sort(key=lambda x: x["score"], reverse=True)

    best = results[0]

    # If the winner is promising but misses gates, inspect it and run only one
    # tight refinement ring. This is not a relaxation of the validation rules.
    refinement_candidates = _refinement_ideas(best)
    refinement_results = []

    print()
    print("=== BEST INITIAL IDEA ===", flush=True)
    print(f"Idea: {best['idea']}", flush=True)
    print(f"Parameters: {best['parameters']}", flush=True)
    print(
        f"OOS={best['holdout']['total_return']:.2%} | "
        f"PF={best['holdout']['profit_factor']:.2f} | "
        f"DD={best['holdout']['max_drawdown']:.2%} | "
        f"trades={best['holdout']['trade_count']} | "
        f"WF={best['positive_folds']}/4",
        flush=True,
    )
    print(
        "Initial gate reasons: "
        + ("NONE" if not best["gate_reasons"] else ", ".join(best["gate_reasons"])),
        flush=True,
    )

    # Only refine when primary metrics are plausibly promising.
    promising_for_refinement = bool(
        best["holdout"]["total_return"] > 0.05
        and best["holdout"]["profit_factor"] > 1.05
        and best["holdout"]["max_drawdown"] >= -0.40
        and best["holdout"]["trade_count"] >= 10
        and best["positive_folds"] >= 3
    )

    if refinement_candidates and promising_for_refinement and not _primary_gate(best):
        print()
        print(
            f"=== BOUNDED REFINEMENT ({len(refinement_candidates)} candidates) ===",
            flush=True,
        )

        for idx, idea in enumerate(refinement_candidates, 1):
            try:
                r = _evaluate_idea(primary, idea, settings)
                r["gate_reasons"] = _gate_reasons(r)
                refinement_results.append(r)

                print(
                    f"[R{idx:02d}] {idea.parameters} | "
                    f"OOS={r['holdout']['total_return']:.2%} | "
                    f"PF={r['holdout']['profit_factor']:.2f} | "
                    f"DD={r['holdout']['max_drawdown']:.2%} | "
                    f"trades={r['holdout']['trade_count']} | "
                    f"WF={r['positive_folds']}/4 | "
                    f"score={r['score']:.2f}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"[R{idx:02d}] ERROR: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    if refinement_results:
        all_candidates = results + refinement_results
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        best = all_candidates[0]

    frozen = Idea(best["idea"], best["parameters"])

    print()
    print("=== FINAL SHORTLIST ===", flush=True)
    for i, r in enumerate(
        sorted(
            results + refinement_results,
            key=lambda x: x["score"],
            reverse=True,
        )[:5],
        1,
    ):
        print(
            f"#{i} {r['idea']} {r['parameters']} | "
            f"OOS={r['holdout']['total_return']:.2%} | "
            f"PF={r['holdout']['profit_factor']:.2f} | "
            f"DD={r['holdout']['max_drawdown']:.2%} | "
            f"trades={r['holdout']['trade_count']} | "
            f"WF={r['positive_folds']}/4 | "
            f"gates={'PASS' if _primary_gate(r) else 'FAIL'}",
            flush=True,
        )

    primary_ok = _primary_gate(best)

    print()
    print("=== FROZEN INDEPENDENT TEST ===", flush=True)

    independent_result = _independent(
        independent,
        frozen,
        settings,
    )

    print(
        f"Frozen {frozen.name} {frozen.params} | "
        f"BTC/USDT 4h return={independent_result['total_return']:.2%} | "
        f"PF={independent_result['profit_factor']:.2f} | "
        f"DD={independent_result['max_drawdown']:.2%} | "
        f"trades={independent_result['trade_count']} | "
        f"Sharpe={independent_result['sharpe']:.2f} | "
        f"PASS={independent_result['passed']}",
        flush=True,
    )

    if independent_result["reasons"]:
        print(
            "Independent gate reasons: "
            + ", ".join(independent_result["reasons"]),
            flush=True,
        )

    decision = (
        "PROMOTE_TO_PAPER"
        if primary_ok and independent_result["passed"]
        else "REJECT_IDEA_SET"
    )

    output = {
        "primary": {
            "market": "ETH/USDT",
            "timeframe": "1h",
            "market_type": "spot",
        },
        "independent": {
            "market": "BTC/USDT",
            "timeframe": "4h",
            "market_type": "spot",
        },
        "best": best,
        "refinement": {
            "triggered": bool(refinement_results),
            "candidate_count": len(refinement_candidates),
            "evaluated_count": len(refinement_results),
            "results": refinement_results,
        },
        "independent_result": independent_result,
        "decision": decision,
        "research_count": len(results),
    }

    path = ROOT / "experiments" / "autonomous_ideas_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print()
    print("=== FINAL DECISION ===", flush=True)
    print(f"Decision: {decision}", flush=True)
    print(f"Best idea: {best['idea']}", flush=True)
    print(f"Best parameters: {best['parameters']}", flush=True)
    print(f"Primary gate: {primary_ok}", flush=True)
    print(f"Independent gate: {independent_result['passed']}", flush=True)
    print(f"Saved: {path}", flush=True)

    return output


if __name__ == "__main__":
    run()
