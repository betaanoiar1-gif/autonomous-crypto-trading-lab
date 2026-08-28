import math
import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def sharpe(returns: pd.Series, periods_per_year: float = 365.0) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: float = 365.0) -> float:
    r = pd.Series(returns).dropna()
    downside = r[r < 0]
    if len(r) == 0 or len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=1) * math.sqrt(periods_per_year))


def profit_factor(pnl: pd.Series) -> float:
    p = pd.Series(pnl).dropna()
    losses = -p[p < 0].sum()
    if losses <= 0:
        return float("inf") if p[p > 0].sum() > 0 else 0.0
    return float(p[p > 0].sum() / losses)
