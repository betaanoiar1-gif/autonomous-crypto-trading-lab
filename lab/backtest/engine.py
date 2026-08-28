from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict


def _count_position_changes(pos: pd.Series) -> int:
    changes = pos.diff().fillna(pos)
    return int((changes.abs() > 0).sum())


def run_ohlcv(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {sorted(required)} columns")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("fee_bps and slippage_bps must be non-negative")

    data = df.copy().sort_index()
    close = data["close"].astype(float)
    pos = (
        signal.astype(float)
        .reindex(close.index)
        .fillna(0.0)
        .clip(-1.0, 1.0)
    )

    # Execution convention: signal formed on bar t is acted on at bar t+1.
    next_pos = pos.shift(1).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())

    # Fee + slippage are charged on position changes / notional turnover.
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_rate
    strategy_ret = next_pos * asset_ret - costs

    equity = initial_capital * (1.0 + strategy_ret).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    pnl = equity.diff().fillna(0.0)
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())

    metrics = {
        "total_return": float(equity.iloc[-1] / initial_capital - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        "final_equity": float(equity.iloc[-1]),
        "trade_turnover": float(turnover.sum()),
        "trade_count": _count_position_changes(pos),
        "exposure": float(next_pos.abs().mean()),
        "long_exposure": float((next_pos > 0).mean()),
        "short_exposure": float((next_pos < 0).mean()),
        "cost_drag": float(costs.sum()),
        "bars": int(len(data)),
    }

    trades = pd.DataFrame(
        {
            "position": pos,
            "executed_position": next_pos,
            "turnover": turnover,
            "cost": costs,
            "asset_return": asset_ret,
            "strategy_return": strategy_ret,
        },
        index=close.index,
    )
    return BacktestResult(equity, strategy_ret, trades, metrics)
