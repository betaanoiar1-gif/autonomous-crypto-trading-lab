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


def _align_funding_rates(index: pd.DatetimeIndex, funding_rates: pd.Series | None) -> pd.Series:
    if funding_rates is None:
        return pd.Series(np.nan, index=index)
    rates = pd.Series(funding_rates).copy()
    rates.index = pd.to_datetime(rates.index, utc=True)
    rates = pd.to_numeric(rates, errors="coerce").dropna().sort_index()
    if rates.empty:
        return pd.Series(np.nan, index=index)
    return rates.reindex(index, method="ffill")


def run_ohlcv(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    market_type: str = "spot",
    leverage: float = 1.0,
    funding_bps_per_8h: float = 0.0,
    funding_rates: pd.Series | None = None,
) -> BacktestResult:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {sorted(required)} columns")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if fee_bps < 0 or slippage_bps < 0 or funding_bps_per_8h < 0:
        raise ValueError("cost parameters must be non-negative")
    market_type = str(market_type).strip().lower()
    if market_type not in {"spot", "futures"}:
        raise ValueError("market_type must be 'spot' or 'futures'")
    leverage = float(leverage)
    if leverage < 1.0 or leverage > 20.0:
        raise ValueError("leverage must be between 1 and 20")

    data = df.copy().sort_index()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    pos = signal.astype(float).reindex(close.index).fillna(0.0).clip(-1.0, 1.0)

    next_pos = pos.shift(1).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    effective_leverage = leverage if market_type == "futures" else 1.0

    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    trading_costs = turnover * cost_rate
    gross_strategy_ret = next_pos * effective_leverage * asset_ret

    # Prefer actual historical funding-rate observations when supplied. CCXT
    # returns fundingRate as a decimal (e.g. 0.0001 = 1 bp). A funding charge
    # is only applied when a non-zero position is carried through the funding
    # observation. When no history is available, fall back to the configured
    # conservative 8-hour funding assumption.
    actual_funding = _align_funding_rates(data.index, funding_rates)
    if market_type == "futures" and actual_funding.notna().any():
        funding_costs = next_pos * actual_funding.fillna(0.0) * effective_leverage
    elif market_type == "futures" and funding_bps_per_8h > 0 and len(data) > 1:
        median_delta = data.index.to_series().diff().dropna().median()
        hours_per_bar = max(float(median_delta.total_seconds()) / 3600.0, 1e-9)
        funding_per_bar = funding_bps_per_8h / 10_000.0 * (hours_per_bar / 8.0)
        funding_costs = next_pos.abs() * funding_per_bar
    else:
        funding_costs = pd.Series(0.0, index=data.index)

    strategy_ret = gross_strategy_ret - trading_costs - funding_costs

    # Conservative liquidation guard: flag an intrabar adverse move large enough
    # to exhaust the simplified maintenance buffer. This is deliberately a guard,
    # not a claim to reproduce exchange liquidation engines exactly.
    liquidation_events = pd.Series(False, index=data.index)
    if market_type == "futures" and leverage > 1.0:
        prev_close = close.shift(1)
        adverse = pd.Series(0.0, index=data.index)
        long_move = low / prev_close - 1.0
        short_move = -(high / prev_close - 1.0)
        adverse[next_pos > 0] = long_move[next_pos > 0]
        adverse[next_pos < 0] = short_move[next_pos < 0]
        maintenance = max(0.005, min(0.10, 0.005 * leverage))
        threshold = (1.0 - maintenance) / leverage
        liquidation_events = (adverse < -threshold).fillna(False)
        if liquidation_events.any():
            first = liquidation_events[liquidation_events].index[0]
            strategy_ret = strategy_ret.copy()
            strategy_ret.loc[first] = -0.999
            if first != strategy_ret.index[-1]:
                strategy_ret.loc[first < strategy_ret.index] = strategy_ret.loc[first < strategy_ret.index]

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
        "cost_drag": float(trading_costs.sum()),
        "funding_drag": float(funding_costs.sum()),
        "funding_source": "historical" if actual_funding.notna().any() and market_type == "futures" else ("assumption" if market_type == "futures" else "none"),
        "leverage": float(effective_leverage),
        "market_type": market_type,
        "liquidation_events": int(liquidation_events.sum()),
        "bars": int(len(data)),
    }

    trades = pd.DataFrame(
        {
            "position": pos,
            "executed_position": next_pos,
            "turnover": turnover,
            "trading_cost": trading_costs,
            "funding_rate": actual_funding,
            "funding_cost": funding_costs,
            "asset_return": asset_ret,
            "strategy_return": strategy_ret,
            "liquidation_event": liquidation_events,
        },
        index=close.index,
    )
    return BacktestResult(equity, strategy_ret, trades, metrics)
