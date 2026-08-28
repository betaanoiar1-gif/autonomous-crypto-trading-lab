from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_ohlcv(df: pd.DataFrame, signal: pd.Series, initial_capital: float, fee_bps: float = 10.0, slippage_bps: float = 5.0) -> BacktestResult:
    required = {"close"}
    if not required.issubset(df.columns):
        raise ValueError("DataFrame must contain a close column")
    close = df["close"].astype(float).copy()
    pos = signal.astype(float).reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
    asset_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = turnover * ((fee_bps + slippage_bps) / 10_000.0)
    strategy_ret = pos.shift(1).fillna(0.0) * asset_ret - costs
    equity = initial_capital * (1.0 + strategy_ret).cumprod()
    trades = pd.DataFrame({"position": pos, "turnover": turnover, "strategy_return": strategy_ret})
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    pnl = equity.diff().fillna(0.0)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    metrics = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gains / losses) if losses else float("inf"),
        "final_equity": float(equity.iloc[-1]),
        "trade_turnover": float(turnover.sum()),
    }
    return BacktestResult(equity, strategy_ret, trades, metrics)
