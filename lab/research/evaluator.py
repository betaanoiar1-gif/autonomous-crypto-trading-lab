from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..metrics import sharpe, sortino
from .executor import compile_signal


@dataclass
class Evaluation:
    in_sample: dict
    out_of_sample: dict
    robustness: dict
    passed: bool
    rejection_reasons: list[str]


def _metrics(result, returns: pd.Series) -> dict:
    active = result.trades["position"].diff().abs().fillna(result.trades["position"].abs()) > 0
    trades = int(active.sum())
    return {
        **result.metrics,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
        "trade_count": trades,
    }


def _run(df, family, params, directions, capital, fee_bps, slippage_bps):
    sig = compile_signal(df, family, params, directions)
    return run_ohlcv(df, sig, capital, fee_bps, slippage_bps)


def _walk_forward(df, family, params, directions, capital, fee_bps, slippage_bps, windows=4):
    n = len(df)
    if n < 240:
        return {"windows": [], "positive_windows": 0, "median_return": 0.0, "positive_ratio": 0.0, "median_sharpe": 0.0, "min_trade_count": 0, "passed": False}
    train_size = max(120, int(n * 0.45))
    test_size = max(40, int((n - train_size) / windows))
    rows = []
    start = 0
    while len(rows) < windows and start + train_size + test_size <= n:
        test = df.iloc[start + train_size:start + train_size + test_size]
        result = _run(test, family, params, directions, capital, fee_bps, slippage_bps)
        m = _metrics(result, result.returns)
        rows.append({
            "fold": len(rows) + 1,
            "start": str(test.index[0]),
            "end": str(test.index[-1]),
            "total_return": float(m["total_return"]),
            "max_drawdown": float(m["max_drawdown"]),
            "profit_factor": float(m["profit_factor"]),
            "sharpe": float(m["sharpe"]),
            "trade_count": int(m["trade_count"]),
        })
        start += test_size
    if not rows:
        return {"windows": [], "positive_windows": 0, "median_return": 0.0, "positive_ratio": 0.0, "median_sharpe": 0.0, "min_trade_count": 0, "passed": False}
    rets = np.array([r["total_return"] for r in rows], dtype=float)
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    trades = np.array([r["trade_count"] for r in rows], dtype=float)
    positive = int(np.sum(rets > 0))
    positive_ratio = float(positive / len(rows))
    median_return = float(np.median(rets))
    min_trades = int(np.min(trades)) if len(trades) else 0
    passed = bool(len(rows) >= 3 and positive_ratio >= 0.75 and median_return > 0 and min_trades >= 4)
    return {
        "windows": rows,
        "positive_windows": positive,
        "positive_ratio": positive_ratio,
        "median_return": median_return,
        "worst_return": float(np.min(rets)),
        "median_sharpe": float(np.median(sharpes)),
        "min_trade_count": min_trades,
        "passed": passed,
    }


def _research_score(oos: dict, walk: dict) -> float:
    ret = float(oos.get("total_return", 0.0))
    sh = float(oos.get("sharpe", 0.0))
    dd = abs(min(0.0, float(oos.get("max_drawdown", 0.0))))
    wf = float(walk.get("median_return", 0.0))
    wf_sh = float(walk.get("median_sharpe", 0.0))
    return float(100.0 * ret + 5.0 * sh + 75.0 * wf + 2.0 * wf_sh - 20.0 * dd)


def evaluate(df: pd.DataFrame, family: str, params: dict, directions: list[str], initial_capital: float,
             fee_bps: float, slippage_bps: float, holdout_ratio: float = 0.30) -> Evaluation:
    if len(df) < 240:
        raise ValueError("Not enough observations for robust evaluation; need at least 240")

    split = int(len(df) * (1 - holdout_ratio))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    a = _run(train, family, params, directions, initial_capital, fee_bps, slippage_bps)
    b = _run(test, family, params, directions, initial_capital, fee_bps, slippage_bps)

    variants = []
    for key in ("lookback", "fast", "slow"):
        if key not in params:
            continue
        try:
            base = int(params[key])
            delta = max(1, int(base * 0.20))
            for value in (base - delta, base + delta):
                p = dict(params)
                p[key] = max(2, value)
                if key == "slow" and "fast" in p:
                    p[key] = max(int(p["fast"]) + 1, p[key])
                if key == "fast" and "slow" in p:
                    p[key] = min(p[key], int(p["slow"]) - 1)
                rr = _run(train, family, p, directions, initial_capital, fee_bps, slippage_bps)
                variants.append(rr.metrics["total_return"])
        except (TypeError, ValueError):
            continue

    base_ret = a.metrics["total_return"]
    stability = bool(not variants or min(variants) > base_ret - 0.25)
    stressed = _run(test, family, params, directions, initial_capital, fee_bps * 2.0, slippage_bps * 2.0)
    walk = _walk_forward(df, family, params, directions, initial_capital, fee_bps, slippage_bps)
    in_metrics = _metrics(a, a.returns)
    out_metrics = _metrics(b, b.returns)
    out_metrics["research_score"] = _research_score(out_metrics, walk)

    robust = {
        "parameter_stability": stability,
        "stressed_total_return": float(stressed.metrics["total_return"]),
        "stressed_max_drawdown": float(stressed.metrics["max_drawdown"]),
        "stressed_trade_count": _metrics(stressed, stressed.returns)["trade_count"],
        "walk_forward": walk,
    }

    reasons: list[str] = []
    if out_metrics["trade_count"] < 8:
        reasons.append("Too few out-of-sample trades")
    if out_metrics["total_return"] <= 0:
        reasons.append("Non-positive out-of-sample return")
    if out_metrics["profit_factor"] <= 1:
        reasons.append("Out-of-sample profit factor <= 1")
    if out_metrics["max_drawdown"] < -0.50:
        reasons.append("Out-of-sample drawdown exceeds 50%")
    if not stability:
        reasons.append("Parameter stability failed")
    if stressed.metrics["total_return"] <= 0:
        reasons.append("Fails doubled-cost stress test")
    if not walk["passed"]:
        reasons.append("Walk-forward consistency failed")

    return Evaluation(in_metrics, out_metrics, robust, not reasons, reasons)
