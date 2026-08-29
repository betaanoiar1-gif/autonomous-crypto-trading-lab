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


def _align_funding_rates(index: pd.DatetimeIndex, funding_rates: pd.Series | None) -> tuple[pd.Series, int]:
    """Map actual funding events to the first OHLCV bar at/after each event."""
    out = pd.Series(0.0, index=index)
    if funding_rates is None or len(index) == 0:
        return out, 0
    rates = pd.Series(funding_rates).copy()
    rates.index = pd.to_datetime(rates.index, utc=True)
    rates = pd.to_numeric(rates, errors="coerce").dropna().sort_index()
    if rates.empty:
        return out, 0
    idx_ns = index.view("int64")
    first_ns = int(idx_ns[0]); last_ns = int(idx_ns[-1])
    event_count = 0
    for ts, rate in rates.groupby(level=0).last().items():
        ts_ns = pd.Timestamp(ts).value
        if ts_ns < first_ns or ts_ns > last_ns:
            continue
        pos = int(np.searchsorted(idx_ns, ts_ns, side="left"))
        if pos < len(out):
            out.iloc[pos] += float(rate)
            event_count += 1
    return out, event_count


def run_ohlcv(df: pd.DataFrame, signal: pd.Series, initial_capital: float, fee_bps: float = 10.0,
              slippage_bps: float = 5.0, market_type: str = "spot", leverage: float = 1.0,
              funding_bps_per_8h: float = 0.0, funding_rates: pd.Series | None = None) -> BacktestResult:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {sorted(required)} columns")
    market_type = str(market_type).strip().lower()
    leverage = float(leverage)
    data = df.copy().sort_index(); close = data["close"].astype(float); high = data["high"].astype(float); low = data["low"].astype(float)
    if market_type == "spot":
        # Spot research is intentionally long/flat. Negative positions are invalid
        # and must never enter the evaluator even if an upstream signal misbehaves.
        pos = signal.astype(float).reindex(close.index).fillna(0.0).clip(0.0, 1.0)
    else:
        pos = signal.astype(float).reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
    next_pos = pos.shift(1).fillna(0.0)
    asset_ret = close.pct_change().fillna(0.0); turnover = pos.diff().abs().fillna(pos.abs()); effective_leverage = leverage if market_type == "futures" else 1.0
    cost_rate = (fee_bps + slippage_bps) / 10000.0; trading_costs = turnover * cost_rate; gross_strategy_ret = next_pos * effective_leverage * asset_ret
    actual_funding, funding_events = _align_funding_rates(data.index, funding_rates)
    has_historical_funding = market_type == "futures" and funding_rates is not None and funding_events > 0
    if has_historical_funding:
        funding_costs = next_pos * actual_funding * effective_leverage
    elif market_type == "futures" and funding_bps_per_8h > 0 and len(data) > 1:
        median_delta = data.index.to_series().diff().dropna().median(); hours_per_bar = max(float(median_delta.total_seconds()) / 3600.0, 1e-9)
        funding_per_bar = funding_bps_per_8h / 10000.0 * (hours_per_bar / 8.0); funding_costs = next_pos.abs() * funding_per_bar
    else:
        funding_costs = pd.Series(0.0, index=data.index)
    strategy_ret = gross_strategy_ret - trading_costs - funding_costs
    liquidation_events = pd.Series(False, index=data.index)
    if market_type == "futures" and leverage > 1.0:
        prev_close = close.shift(1); adverse = pd.Series(0.0, index=data.index)
        adverse[next_pos > 0] = (low / prev_close - 1.0)[next_pos > 0]; adverse[next_pos < 0] = (-(high / prev_close - 1.0))[next_pos < 0]
        maintenance = max(0.005, min(0.10, 0.005 * leverage)); threshold = (1.0 - maintenance) / leverage
        liquidation_events = (adverse < -threshold).fillna(False)
        if liquidation_events.any(): strategy_ret.loc[liquidation_events[liquidation_events].index[0]] = -0.999
    equity = initial_capital * (1.0 + strategy_ret).cumprod(); peak = equity.cummax(); drawdown = (equity / peak.replace(0, np.nan) - 1.0).fillna(-1.0)
    pnl = equity.diff().fillna(0.0); gains = float(pnl[pnl > 0].sum()); losses = float(-pnl[pnl < 0].sum())
    metrics = {"total_return": float(equity.iloc[-1] / initial_capital - 1.0), "max_drawdown": float(drawdown.min()),
               "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
               "final_equity": float(equity.iloc[-1]), "trade_turnover": float(turnover.sum()), "trade_count": _count_position_changes(pos),
               "exposure": float(next_pos.abs().mean()), "long_exposure": float((next_pos > 0).mean()), "short_exposure": float((next_pos < 0).mean()),
               "cost_drag": float(trading_costs.sum()), "funding_drag": float(funding_costs.sum()),
               "funding_source": "historical" if has_historical_funding else ("assumption" if market_type == "futures" else "none"),
               "funding_events": int(funding_events), "leverage": float(effective_leverage), "market_type": market_type,
               "liquidation_events": int(liquidation_events.sum()), "bars": int(len(data))}
    trades = pd.DataFrame({"position": pos, "executed_position": next_pos, "turnover": turnover, "trading_cost": trading_costs,
                           "funding_rate": actual_funding, "funding_cost": funding_costs, "asset_return": asset_ret,
                           "strategy_return": strategy_ret, "liquidation_event": liquidation_events}, index=close.index)
    return BacktestResult(equity, strategy_ret, trades, metrics)
