from dataclasses import dataclass
import pandas as pd

@dataclass
class BacktestConfig:
    initial_capital: float = 500.0
    commission_bps: float = 10.0
    slippage_bps: float = 5.0


def run_equity_backtest(df: pd.DataFrame, signal: pd.Series, config: BacktestConfig) -> pd.DataFrame:
    """Simple transparent baseline engine.

    Signal is -1/0/1 and is applied on the next bar to avoid look-ahead.
    This is a baseline engine; exchange-specific execution belongs in adapters.
    """
    data = df.copy()
    data["signal"] = signal.reindex(data.index).fillna(0).clip(-1, 1)
    data["position"] = data["signal"].shift(1).fillna(0)
    data["asset_return"] = data["close"].pct_change().fillna(0)
    data["gross_return"] = data["position"] * data["asset_return"]
    turnover = data["position"].diff().abs().fillna(data["position"].abs())
    cost_rate = (config.commission_bps + config.slippage_bps) / 10000.0
    data["cost"] = turnover * cost_rate
    data["net_return"] = data["gross_return"] - data["cost"]
    data["equity"] = config.initial_capital * (1 + data["net_return"]).cumprod()
    return data
