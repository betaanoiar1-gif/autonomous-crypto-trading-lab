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
    return {
        **result.metrics,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
    }


def evaluate(df: pd.DataFrame, family: str, params: dict, directions: list[str], initial_capital: float,
             fee_bps: float, slippage_bps: float, holdout_ratio: float = 0.30) -> Evaluation:
    if len(df) < 100:
        raise ValueError("Not enough observations for evaluation")
    split = int(len(df) * (1 - holdout_ratio))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    train_sig = compile_signal(train, family, params, directions)
    test_sig = compile_signal(test, family, params, directions)
    a = run_ohlcv(train, train_sig, initial_capital, fee_bps, slippage_bps)
    b = run_ohlcv(test, test_sig, initial_capital, fee_bps, slippage_bps)

    variants = []
    for key in ("lookback", "fast", "slow"):
        if key in params:
            try:
                base = int(params[key])
                for delta in (-max(1, int(base * 0.2)), max(1, int(base * 0.2))):
                    p = dict(params)
                    p[key] = max(2, base + delta)
                    if key == "slow" and "fast" in p:
                        p[key] = max(int(p["fast"]) + 1, p[key])
                    sig = compile_signal(train, family, p, directions)
                    rr = run_ohlcv(train, sig, initial_capital, fee_bps, slippage_bps)
                    variants.append(rr.metrics["total_return"])
            except (TypeError, ValueError):
                pass
    base_ret = a.metrics["total_return"]
    stability = bool(not variants or min(variants) > base_ret - 0.25)

    stressed = run_ohlcv(test, test_sig, initial_capital, fee_bps * 2.0, slippage_bps * 2.0)
    robust = {
        "parameter_stability": stability,
        "stressed_total_return": stressed.metrics["total_return"],
        "stressed_max_drawdown": stressed.metrics["max_drawdown"],
    }
    reasons = []
    if b.metrics["total_return"] <= 0:
        reasons.append("Non-positive out-of-sample return")
    if b.metrics["profit_factor"] <= 1:
        reasons.append("Out-of-sample profit factor <= 1")
    if b.metrics["max_drawdown"] < -0.50:
        reasons.append("Out-of-sample drawdown exceeds 50%")
    if not stability:
        reasons.append("Parameter stability failed")
    if stressed.metrics["total_return"] <= 0:
        reasons.append("Fails doubled-cost stress test")
    return Evaluation(
        _metrics(a, a.returns), _metrics(b, b.returns), robust, not reasons, reasons
    )
