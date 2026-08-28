from dataclasses import dataclass

@dataclass(frozen=True)
class RiskPolicy:
    risk_per_trade: float = 0.01
    max_position_fraction: float = 1.0
    max_total_exposure: float = 1.0
    reserve_fraction: float = 0.10


def position_notional(equity: float, stop_distance: float, policy: RiskPolicy) -> float:
    if equity <= 0:
        return 0.0
    if stop_distance <= 0:
        raise ValueError("stop_distance must be positive")
    risk_budget = equity * policy.risk_per_trade
    notional = risk_budget / stop_distance
    available = equity * policy.max_position_fraction
    return max(0.0, min(notional, available))
