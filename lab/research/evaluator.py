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
    return {**result.metrics, "sharpe": sharpe(returns), "sortino": sortino(returns),
            "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
            "trade_count": int(active.sum())}


def _run(df, family, params, directions, capital, fee_bps, slippage_bps, market_type="spot", leverage=1.0, funding_rates=None):
    sig = compile_signal(df, family, params, directions)
    return run_ohlcv(df, sig, capital, fee_bps, slippage_bps, market_type=market_type,
                     leverage=leverage, funding_rates=funding_rates)


def _param_grid(family: str, base: dict) -> list[dict]:
    family = family.lower().strip()
    if family in {"momentum", "breakout", "trend_pullback"}:
        b = int(base.get("lookback", 20)); vals = sorted({max(2, min(200, int(b * f))) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
        if family == "trend_pullback":
            t = float(base.get("pullback_threshold", 0.01)); ts = sorted({round(max(0.001, min(0.10, t * f)), 4) for f in (0.75, 1.0, 1.25)})
            return [{"lookback": n, "pullback_threshold": x} for n, x in itertools.product(vals, ts)][:15]
        return [{"lookback": v} for v in vals]
    if family == "mean_reversion":
        b = int(base.get("lookback", 40)); ze = float(base.get("z_entry", 1.5)); zx = float(base.get("z_exit", 0.25))
        lbs = sorted({max(10, min(200, int(b * f))) for f in (0.75, 1.0, 1.25)})
        ens = sorted({round(max(0.8, min(3.5, ze * f)), 3) for f in (0.85, 1.0, 1.15)})
        exs = sorted({round(max(0.05, min(1.5, zx * f)), 3) for f in (0.75, 1.0, 1.25)})
        return [{"lookback": n, "z_entry": e, "z_exit": x} for n, e, x in itertools.product(lbs, ens, exs) if x < e][:15]
    if family == "moving_average_cross":
        fast = int(base.get("fast", 10)); slow = int(base.get("slow", 40)); fs = sorted({max(2, min(100, int(fast * f))) for f in (0.75, 1.0, 1.25)}); ss = sorted({max(3, min(300, int(slow * f))) for f in (0.75, 1.0, 1.25, 1.5)})
        return [{"fast": f, "slow": s} for f, s in itertools.product(fs, ss) if s > f][:12]
    if family == "rsi_reversion":
        n = int(base.get("rsi_length", 14)); lo = float(base.get("rsi_low", 30)); hi = float(base.get("rsi_high", 70)); ns = sorted({max(2, min(50, int(n * f))) for f in (0.75, 1.0, 1.25)}); los = sorted({max(5, min(45, lo + d)) for d in (-5, 0, 5)}); his = sorted({max(55, min(95, hi + d)) for d in (-5, 0, 5)})
        return [{"rsi_length": n0, "rsi_low": l, "rsi_high": h} for n0, l, h in itertools.product(ns, los, his) if l < 50 < h][:15]
    if family == "atr_breakout":
        n = int(base.get("atr_length", 14)); m = float(base.get("atr_mult", 1.5)); ns = sorted({max(2, min(50, int(n * f))) for f in (0.75, 1.0, 1.25)}); ms = sorted({round(max(0.25, min(5.0, m * f)), 3) for f in (0.75, 1.0, 1.25)})
        return [{"atr_length": n0, "atr_mult": m0} for n0, m0 in itertools.product(ns, ms)]
    if family == "channel_reversion":
        n = int(base.get("channel_length", 40)); vals = sorted({max(5, min(200, int(n * f))) for f in (0.75, 1.0, 1.25, 1.5)})
        return [{"channel_length": v} for v in vals]
    return [dict(base)]


def _training_objective(metrics: dict) -> float:
    trades = int(metrics.get("trade_count", 0))
    if trades < 4: return -10.0 + trades * 0.25
    ret = float(metrics.get("total_return", 0.0)); pf = min(3.0, float(metrics.get("profit_factor", 0.0))); dd = abs(min(0.0, float(metrics.get("max_drawdown", 0.0)))); sh = float(metrics.get("sharpe", 0.0))
    return 100.0 * ret + 3.0 * pf + 2.0 * sh - 15.0 * dd


def _select_params(train, family, base_params, directions, capital, fee_bps, slippage_bps, market_type="spot", leverage=1.0, funding_rates=None):
    best = None; leaderboard = []
    for candidate in _param_grid(family, base_params):
        result = _run(train, family, candidate, directions, capital, fee_bps, slippage_bps, market_type, leverage, funding_rates); metrics = _metrics(result, result.returns)
        score = _training_objective(metrics); leaderboard.append({"parameters": candidate, "score": float(score), "metrics": metrics})
        if best is None or score > best[0]: best = (score, candidate, metrics)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return (dict(best[1]), {"selected": dict(best[1]), "selected_score": float(best[0]), "candidates": leaderboard[:5]}) if best else (dict(base_params), {"selected": dict(base_params), "candidates": []})


def _walk_forward(df, family, base_params, directions, capital, fee_bps, slippage_bps, windows=4, market_type="spot", leverage=1.0, funding_rates=None):
    n = len(df)
    if n < 240: return {"windows": [], "positive_windows": 0, "median_return": 0.0, "positive_ratio": 0.0, "median_sharpe": 0.0, "min_trade_count": 0, "passed": False, "mode": "tuned"}
    train_size = max(160, int(n * 0.45)); test_size = max(40, int((n - train_size) / windows)); rows = []; start = 0
    while len(rows) < windows and start + train_size + test_size <= n:
        train = df.iloc[start:start + train_size]; test = df.iloc[start + train_size:start + train_size + test_size]
        selected, tuning = _select_params(train, family, base_params, directions, capital, fee_bps, slippage_bps, market_type, leverage, funding_rates)
        result = _run(test, family, selected, directions, capital, fee_bps, slippage_bps, market_type, leverage, funding_rates); m = _metrics(result, result.returns)
        row = {"fold": len(rows) + 1, "train_start": str(train.index[0]), "train_end": str(train.index[-1]), "test_start": str(test.index[0]), "test_end": str(test.index[-1]), "selected_parameters": selected, "tuning": tuning, "total_return": float(m["total_return"]), "max_drawdown": float(m["max_drawdown"]), "profit_factor": float(m["profit_factor"]), "sharpe": float(m["sharpe"]), "trade_count": int(m["trade_count"]), "funding_source": m.get("funding_source", "none"), "liquidation_events": int(m.get("liquidation_events", 0))}
        rows.append(row)
        print(f"      WF Fold {row['fold']}: params={row['selected_parameters']} | return={row['total_return']:.2%} | PF={row['profit_factor']:.2f} | DD={row['max_drawdown']:.2%} | trades={row['trade_count']} | Sharpe={row['sharpe']:.2f}")
        start += test_size
    rets = np.array([r["total_return"] for r in rows]); sharpes = np.array([r["sharpe"] for r in rows]); trades = np.array([r["trade_count"] for r in rows]); positive = int((rets > 0).sum())
    return {"windows": rows, "positive_windows": positive, "positive_ratio": float(positive / len(rows)), "median_return": float(np.median(rets)), "worst_return": float(np.min(rets)), "median_sharpe": float(np.median(sharpes)), "min_trade_count": int(np.min(trades)), "passed": bool(len(rows) >= 3 and positive / len(rows) >= 0.75 and np.median(rets) > 0 and np.min(trades) >= 4), "mode": "tuned"}


def _research_score(oos, walk):
    ret = float(oos.get("total_return", 0)); sh = float(oos.get("sharpe", 0)); dd = abs(min(0.0, float(oos.get("max_drawdown", 0)))); wf = float(walk.get("median_return", 0)); wf_sh = float(walk.get("median_sharpe", 0))
    return float(100 * ret + 5 * sh + 75 * wf + 2 * wf_sh - 20 * dd)


def evaluate(df, family, params, directions, initial_capital, fee_bps, slippage_bps, holdout_ratio=0.30,
             market_type="spot", leverage=1.0, funding_rates=None):
    if len(df) < 240: raise ValueError("Not enough observations for robust evaluation; need at least 240")
    split = int(len(df) * (1 - holdout_ratio)); train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    selected, tuning = _select_params(train, family, params, directions, initial_capital, fee_bps, slippage_bps, market_type, leverage, funding_rates)
    a = _run(train, family, selected, directions, initial_capital, fee_bps, slippage_bps, market_type, leverage, funding_rates); b = _run(test, family, selected, directions, initial_capital, fee_bps, slippage_bps, market_type, leverage, funding_rates)
    variants = []
    for key, base in selected.items():
        try:
            delta = max(1, int(abs(base) * 0.20)) if isinstance(base, int) else max(0.01, abs(base) * 0.20)
            for value in (base - delta, base + delta):
                p = dict(selected); p[key] = max(0.01, int(value) if isinstance(base, int) else value)
                if key == "slow" and "fast" in p: p[key] = max(int(p["fast"]) + 1, int(p[key]))
                if key == "fast" and "slow" in p: p[key] = min(int(p[key]), int(p["slow"]) - 1)
                variants.append(_run(train, family, p, directions, initial_capital, fee_bps, slippage_bps, market_type, leverage, funding_rates).metrics["total_return"])
        except (TypeError, ValueError): pass
    stability = bool(not variants or min(variants) > a.metrics["total_return"] - 0.25)
    stressed = _run(test, family, selected, directions, initial_capital, fee_bps * 2, slippage_bps * 2, market_type, leverage, funding_rates)
    walk = _walk_forward(df, family, selected, directions, initial_capital, fee_bps, slippage_bps, market_type=market_type, leverage=leverage, funding_rates=funding_rates)
    im, om = _metrics(a, a.returns), _metrics(b, b.returns); om["research_score"] = _research_score(om, walk); om["selected_parameters"] = dict(selected)
    robust = {"parameter_stability": stability, "selected_parameters": dict(selected), "main_training_tuning": tuning, "stressed_total_return": float(stressed.metrics["total_return"]), "stressed_max_drawdown": float(stressed.metrics["max_drawdown"]), "stressed_trade_count": _metrics(stressed, stressed.returns)["trade_count"], "walk_forward": walk}
    reasons = []
    if om["trade_count"] < 8: reasons.append("Too few out-of-sample trades")
    if om["total_return"] <= 0: reasons.append("Non-positive out-of-sample return")
    if om["profit_factor"] <= 1: reasons.append("Out-of-sample profit factor <= 1")
    if om["max_drawdown"] < -0.50: reasons.append("Out-of-sample drawdown exceeds 50%")
    if not stability: reasons.append("Parameter stability failed")
    if stressed.metrics["total_return"] <= 0: reasons.append("Fails doubled-cost stress test")
    if not walk["passed"]: reasons.append("Walk-forward consistency failed")
    if market_type == "futures" and om.get("funding_source") != "historical": reasons.append("Missing historical funding rates")
    if market_type == "futures" and om.get("liquidation_events", 0) > 0: reasons.append("Liquidation event detected")
    return Evaluation(im, om, robust, not reasons, reasons)
