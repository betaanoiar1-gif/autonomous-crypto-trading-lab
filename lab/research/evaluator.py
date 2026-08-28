from __future__ import annotations

from dataclasses import dataclass
import itertools
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


def _param_grid(family: str, base: dict) -> list[dict]:
    """Small, bounded grids to limit tuning overfit while allowing re-selection per fold."""
    family = family.lower().strip()
    if family in {"momentum", "breakout"}:
        b = int(base.get("lookback", 20))
        vals = sorted({max(2, min(200, int(b * f))) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
        return [{"lookback": v} for v in vals]
    if family == "mean_reversion":
        b = int(base.get("lookback", 40))
        ze = float(base.get("z_entry", 1.5))
        zx = float(base.get("z_exit", 0.25))
        lookbacks = sorted({max(10, min(200, int(b * f))) for f in (0.75, 1.0, 1.25)})
        entries = sorted({round(max(0.8, min(3.5, ze * f)), 3) for f in (0.85, 1.0, 1.15)})
        exits = sorted({round(max(0.05, min(1.5, zx * f)), 3) for f in (0.75, 1.0, 1.25)})
        grid = []
        for n, e, x in itertools.product(lookbacks, entries, exits):
            if x < e:
                grid.append({"lookback": n, "z_entry": e, "z_exit": x})
        return grid[:15]
    if family == "moving_average_cross":
        fast = int(base.get("fast", 10))
        slow = int(base.get("slow", 40))
        fasts = sorted({max(2, min(100, int(fast * f))) for f in (0.75, 1.0, 1.25)})
        slows = sorted({max(3, min(300, int(slow * f))) for f in (0.75, 1.0, 1.25, 1.5)})
        grid = []
        for f, s in itertools.product(fasts, slows):
            if s > f:
                grid.append({"fast": f, "slow": s})
        return grid[:12]
    return [dict(base)]


def _training_objective(metrics: dict) -> float:
    """Reward return and quality, penalize drawdown and tiny samples."""
    trades = int(metrics.get("trade_count", 0))
    if trades < 4:
        return -10.0 + trades * 0.25
    ret = float(metrics.get("total_return", 0.0))
    pf = min(3.0, float(metrics.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
    sh = float(metrics.get("sharpe", 0.0))
    return 100.0 * ret + 3.0 * pf + 2.0 * sh - 15.0 * dd


def _select_params(train, family, base_params, directions, capital, fee_bps, slippage_bps):
    best = None
    leaderboard = []
    for candidate in _param_grid(family, base_params):
        result = _run(train, family, candidate, directions, capital, fee_bps, slippage_bps)
        metrics = _metrics(result, result.returns)
        score = _training_objective(metrics)
        leaderboard.append({"parameters": candidate, "score": float(score), "metrics": metrics})
        if best is None or score > best[0]:
            best = (score, candidate, metrics)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    if best is None:
        return dict(base_params), {"selected": dict(base_params), "candidates": []}
    return dict(best[1]), {"selected": dict(best[1]), "selected_score": float(best[0]), "candidates": leaderboard[:5]}


def _walk_forward(df, family, base_params, directions, capital, fee_bps, slippage_bps, windows=4):
    n = len(df)
    if n < 240:
        return {"windows": [], "positive_windows": 0, "median_return": 0.0, "positive_ratio": 0.0,
                "median_sharpe": 0.0, "min_trade_count": 0, "passed": False, "mode": "tuned"}

    train_size = max(160, int(n * 0.45))
    test_size = max(40, int((n - train_size) / windows))
    rows = []
    start = 0
    while len(rows) < windows and start + train_size + test_size <= n:
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        selected, tuning = _select_params(train, family, base_params, directions, capital, fee_bps, slippage_bps)
        result = _run(test, family, selected, directions, capital, fee_bps, slippage_bps)
        m = _metrics(result, result.returns)
        rows.append({
            "fold": len(rows) + 1,
            "train_start": str(train.index[0]),
            "train_end": str(train.index[-1]),
            "test_start": str(test.index[0]),
            "test_end": str(test.index[-1]),
            "selected_parameters": selected,
            "tuning": tuning,
            "total_return": float(m["total_return"]),
            "max_drawdown": float(m["max_drawdown"]),
            "profit_factor": float(m["profit_factor"]),
            "sharpe": float(m["sharpe"]),
            "trade_count": int(m["trade_count"]),
        })
        start += test_size

    if not rows:
        return {"windows": [], "positive_windows": 0, "median_return": 0.0, "positive_ratio": 0.0,
                "median_sharpe": 0.0, "min_trade_count": 0, "passed": False, "mode": "tuned"}

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
        "mode": "tuned",
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
