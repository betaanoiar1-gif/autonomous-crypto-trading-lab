from dataclasses import dataclass
import pandas as pd

@dataclass
class PortfolioPolicy:
    initial_capital: float = 500.0
    reinvest_profits: bool = True
    max_allocation_per_strategy: float = 0.50
    reserve_fraction: float = 0.10
    max_portfolio_leverage: float = 3.0


def apply_compounding(strategy_returns: pd.DataFrame, policy: PortfolioPolicy) -> pd.Series:
    if strategy_returns.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1.0 / len(strategy_returns.columns), index=strategy_returns.columns)
    weights = weights.clip(upper=policy.max_allocation_per_strategy)
    weights = weights / weights.sum()
    combined = strategy_returns.fillna(0).mul(weights, axis=1).sum(axis=1)
    return (1 + combined).cumprod() * policy.initial_capital
