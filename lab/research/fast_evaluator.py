from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

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
    return {
        **result.metrics,
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
        "trade_count": int(active.sum()),
    }


def _run(df, family, params, directions, capital, fee_bps, slippage_bps,
         market_type="spot", leverage=1.0, funding_rates=None):
    sig = compile_signal(df, family, params, directions)
    return run_ohlcv(
        df, sig, capital, fee_bps, slippage_bps,
        market_type=market_type, leverage=leverage, funding_rates=funding_rates,
    )


def _walk_forward_fixed(df, family, params, directions, capital, fee_bps, slippage_bps,
                        windows=4, market_type="spot", leverage=1.0, funding_rates=None):
    n = len(df)
    if n < 240:
        return {"windows": [], "positive_windows": 0, "median_return": 0.0,
                "positive_ratio": 0.0, "median_sharpe": 0.0, "min_trade_count": 0,
                "passed": False, "mode": "frozen"}
    train_size = max(160, int(n * 0.45))
    test_size = max(40, int((n - train_size) / windows))
    rows = []
    start = 0
    while len(rows) < windows and start + train_size + test_size <= n:
        test = df.iloc[start + train_size:start + train_size + test_size]
        result = _run(test, family, params, directions, capital, fee_bps, slippage_bps,
                      market_type, leverage, funding_rates)
        m = _metrics(result, result.returns)
        row = {
            "fold": len(rows) + 1,
            "test_start": str(test.index[0]),
            "test_end": str(test.index[-1]),
            "selected_parameters": dict(params),
            "total_return": float(m["total_return"]),
            "max_drawdown": float(m["max_drawdown"]),
            "profit_factor": float(m["profit_factor"]),
            "sharpe": float(m["sharpe"]),
            "trade_count": int(m["trade_count"]),
            "funding_source": m.get("funding_source", "none"),
            "liquidation_events": int(m.get("liquidation_events", 0)),
        }
        rows.append(row)
        print(
            f"      WF Fold {row['fold']}: params={row['selected_parameters']} | "
            f"return={row['total_return']:.2%} | PF={row['profit_factor']:.2f} | "
            f"DD={row['max_drawdown']:.2%} | trades={row['trade_count']} | "
            f"Sharpe={row['sharpe']:.2f}", flush=True,
        )
        start += test_size
    rets = np.array([r["total_return"] for r in rows])
    sharpes = np.array([r["sharpe"] for r in rows])
    trades = np.array([r["trade_count"] for r in rows])
    positive = int((rets > 0).sum()) if len(rows) else 0
    return {
        "windows": rows,
        "positive_windows": positive,
        "positive_ratio": float(positive / len(rows)) if rows else 0.0,
        "median_return": float(np.median(rets)) if rows else 0.0,
        "worst_return": float(np.min(rets)) if rows else 0.0,
        "median_sharpe": float(np.median(sharpes)) if rows else 0.0,
        "min_trade_count": int(np.min(trades)) if rows else 0,
        "passed": bool(
            len(rows) >= 3 and positive / len(rows) >= 0.75
            and float(np.median(rets)) > 0 and int(np.min(trades)) >= 4
        ) if rows else False,
        "mode": "frozen",
    }


def evaluate_fixed(df, family, params, directions, initial_capital, fee_bps, slippage_bps,
                   holdout_ratio=0.30, market_type="spot", leverage=1.0, funding_rates=None):
    """Fast evaluation of a frozen hypothesis.

    Parameter search is intentionally disabled here because the fast hypothesis
    generator already explores the parameter grid across autonomous cycles.
    All validation gates remain active: OOS, stability, doubled-cost stress,
    frozen-parameter walk-forward, funding/liquidation checks.
    """
    if len(df) < 240:
        raise ValueError("Not enough observations for robust evaluation; need at least 240")

    params = dict(params)
    split = int(len(df) * (1 - holdout_ratio))
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()

    a = _run(train, family, params, directions, initial_capital, fee_bps, slippage_bps,
             market_type, leverage, funding_rates)
    b = _run(test, family, params, directions, initial_capital, fee_bps, slippage_bps,
             market_type, leverage, funding_rates)
    im = _metrics(a, a.returns)
    om = _metrics(b, b.returns)

    # Small local perturbation stability check, kept bounded for speed.
    variants = []
    for key, base in params.items():
        try:
            if isinstance(base, int):
                delta = max(1, int(abs(base) * 0.20))
                vals = (max(1, base - delta), base + delta)
            else:
                delta = max(0.01, abs(float(base)) * 0.20)
                vals = (float(base) - delta, float(base) + delta)
            for value in vals:
                p = dict(params)
                p[key] = value
                if key == "slow" and "fast" in p:
                    p[key] = max(int(p["fast"]) + 1, int(value))
                if key == "fast" and "slow" in p:
                    p[key] = min(int(value), int(p["slow"]) - 1)
                rr = _run(train, family, p, directions, initial_capital, fee_bps, slippage_bps,
                          market_type, leverage, funding_rates)
                variants.append(float(rr.metrics["total_return"]))
        except (TypeError, ValueError):
            continue
    stability = bool(not variants or min(variants) > float(im["total_return"]) - 0.25)

    stressed = _run(test, family, params, directions, initial_capital, fee_bps * 2, slippage_bps * 2,
                    market_type, leverage, funding_rates)
    stressed_m = _metrics(stressed, stressed.returns)
    walk = _walk_forward_fixed(
        df, family, params, directions, initial_capital, fee_bps, slippage_bps,
        market_type=market_type, leverage=leverage, funding_rates=funding_rates,
    )

    research_score = float(
        100 * om["total_return"]
        + 5 * om["sharpe"]
        + 75 * walk.get("median_return", 0.0)
        + 2 * walk.get("median_sharpe", 0.0)
        - 20 * abs(min(0.0, float(om["max_drawdown"])))
    )
    om["research_score"] = research_score
    om["selected_parameters"] = dict(params)

    robust = {
        "parameter_stability": stability,
        "selected_parameters": dict(params),
        "main_training_tuning": {"selected": dict(params), "mode": "fast_frozen"},
        "stressed_total_return": float(stressed_m["total_return"]),
        "stressed_max_drawdown": float(stressed_m["max_drawdown"]),
        "stressed_trade_count": int(stressed_m["trade_count"]),
        "walk_forward": walk,
    }

    reasons = []
    if om["trade_count"] < 8:
        reasons.append("Too few out-of-sample trades")
    if om["total_return"] <= 0:
        reasons.append("Non-positive out-of-sample return")
    if om["profit_factor"] <= 1:
        reasons.append("Out-of-sample profit factor <= 1")
    if om["max_drawdown"] < -0.50:
        reasons.append("Out-of-sample drawdown exceeds 50%")
    if not stability:
        reasons.append("Parameter stability failed")
    if stressed_m["total_return"] <= 0:
        reasons.append("Fails doubled-cost stress test")
    if not walk["passed"]:
        reasons.append("Walk-forward consistency failed")
    if market_type == "futures" and om.get("funding_source") != "historical":
        reasons.append("Missing historical funding rates")
    if market_type == "futures" and om.get("liquidation_events", 0) > 0:
        reasons.append("Liquidation event detected")

    return Evaluation(im, om, robust, not reasons, reasons)
