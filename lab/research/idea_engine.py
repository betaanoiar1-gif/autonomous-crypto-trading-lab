from __future__ import annotations

"""Focused, autonomous research of genuinely different signal combinations.

This module intentionally avoids the broad 10-cycle loop. It evaluates a small
set of hand-designed research ideas, ranks them on an in-sample/holdout split,
then freezes the winning rule and checks independent market/timeframe data.

It is research/paper-only. No exchange orders are created here.
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
        # Buy pullbacks only while trend is up; short pullbacks only while trend is down.
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

    # Decisions use only information known before the bar being traded.
    return out.shift(1).fillna(0.0)


def _evaluate_idea(df: pd.DataFrame, idea: Idea, settings):
    split = int(len(df) * 0.70)
    train = df.iloc[:split]
    test = df.iloc[split:]

    train_result = _costed_run(train, _signal(train, idea), settings)
    test_result = _costed_run(test, _signal(test, idea), settings)

    train_metrics = _metrics(train_result, train_result.returns)
    test_metrics = _metrics(test_result, test_result.returns)

    # Four contiguous holdout-like blocks, with the same frozen idea.
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
        "train": {k: train_metrics[k] for k in ("total_return", "profit_factor", "max_drawdown", "trade_count", "sharpe")},
        "holdout": {k: test_metrics[k] for k in ("total_return", "profit_factor", "max_drawdown", "trade_count", "sharpe")},
        "frozen_folds": folds,
        "positive_folds": positive,
        "median_fold_return": median_fold,
        "worst_fold_return": worst_fold,
        "median_fold_pf": median_pf,
        "min_fold_trades": min_trades,
        "score": float(score),
    }


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
    }


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
    print("Ideas: 12 | Futures: disabled | Live trading: disabled", flush=True)
    print(f"Primary bars: {len(primary)}", flush=True)
    print()

    results = []
    for idx, idea in enumerate(ideas, 1):
        try:
            r = _evaluate_idea(primary, idea, settings)
            results.append(r)
            print(
                f"[{idx:02d}] {r['idea']} {r['parameters']} | "
                f"OOS={r['holdout']['total_return']:.2%} | "
                f"PF={r['holdout']['profit_factor']:.2f} | "
                f"DD={r['holdout']['max_drawdown']:.2%} | "
                f"trades={r['holdout']['trade_count']} | "
                f"WF={r['positive_folds']}/4 | "
                f"stress-free score={r['score']:.2f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{idx:02d}] {idea.name} ERROR: {type(exc).__name__}: {exc}", flush=True)

    if not results:
        raise RuntimeError("No research idea could be evaluated.")

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    frozen = Idea(best["idea"], best["parameters"])

    print()
    print("=== TOP 3 ===", flush=True)
    for i, r in enumerate(results[:3], 1):
        print(
            f"#{i} {r['idea']} {r['parameters']} | "
            f"OOS={r['holdout']['total_return']:.2%} | PF={r['holdout']['profit_factor']:.2f} | "
            f"DD={r['holdout']['max_drawdown']:.2%} | trades={r['holdout']['trade_count']} | WF={r['positive_folds']}/4",
            flush=True,
        )

    primary_ok = bool(
        best["holdout"]["total_return"] > 0
        and best["holdout"]["profit_factor"] > 1
        and best["holdout"]["max_drawdown"] >= -0.50
        and best["holdout"]["trade_count"] >= 8
        and best["positive_folds"] >= 3
        and best["median_fold_return"] > 0
        and best["median_fold_pf"] > 1
        and best["min_fold_trades"] >= 4
    )

    print()
    print("=== FROZEN INDEPENDENT TEST ===", flush=True)
    independent_result = _independent(independent, frozen, settings)
    print(
        f"Frozen {frozen.idea} {frozen.params} | "
        f"BTC/USDT 4h return={independent_result['total_return']:.2%} | "
        f"PF={independent_result['profit_factor']:.2f} | "
        f"DD={independent_result['max_drawdown']:.2%} | "
        f"trades={independent_result['trade_count']} | "
        f"Sharpe={independent_result['sharpe']:.2f} | "
        f"PASS={independent_result['passed']}",
        flush=True,
    )

    decision = "PROMOTE_TO_PAPER" if primary_ok and independent_result["passed"] else "REJECT_IDEA_SET"

    output = {
        "primary": {"market": "ETH/USDT", "timeframe": "1h", "market_type": "spot"},
        "independent": {"market": "BTC/USDT", "timeframe": "4h", "market_type": "spot"},
        "best": best,
        "independent_result": independent_result,
        "decision": decision,
        "research_count": len(results),
    }

    path = ROOT / "experiments" / "autonomous_ideas_latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print()
    print("=== FINAL DECISION ===", flush=True)
    print(f"Decision: {decision}", flush=True)
    print(f"Best idea: {best['idea']}", flush=True)
    print(f"Best parameters: {best['parameters']}", flush=True)
    print(f"Saved: {path}", flush=True)

    return output


if __name__ == "__main__":
    run()
