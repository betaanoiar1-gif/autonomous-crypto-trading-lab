from dataclasses import dataclass
import pandas as pd
import numpy as np
from .metrics import max_drawdown, sharpe, sortino, profit_factor

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    pnl: pd.Series
    metrics: dict


def run_backtest(df: pd.DataFrame, signal: pd.Series, initial_capital: float = 500.0,
                 commission_bps: float = 10.0, slippage_bps: float = 5.0) -> BacktestResult:
    required = {"close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    close = df["close"].astype(float).copy()
    pos = pd.Series(signal, index=close.index).fillna(0).clip(-1, 1).shift(1).fillna(0)
    asset_ret = close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = turnover * ((commission_bps + slippage_bps) / 10000.0)
    strategy_ret = pos * asset_ret - costs
    equity = initial_capital * (1 + strategy_ret).cumprod()
    pnl = equity.diff().fillna(0)
    metrics = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(strategy_ret),
        "sortino": sortino(strategy_ret),
        "profit_factor": profit_factor(pnl),
        "trades_proxy": int((turnover > 0).sum()),
        "ending_equity": float(equity.iloc[-1]),
    }
    return BacktestResult(equity, strategy_ret, pnl, metrics)
