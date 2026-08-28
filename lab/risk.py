from dataclasses import dataclass

@dataclass
class RiskPolicy:
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.25
    max_daily_loss: float = 0.03
    max_drawdown_stop: float = 0.20


def compounded_equity(initial: float, returns: list[float]) -> list[float]:
    equity = initial
    curve = [equity]
    for r in returns:
        equity *= max(0.0, 1.0 + r)
        curve.append(equity)
    return curve


def position_notional(equity: float, risk_fraction: float, stop_distance: float, max_fraction: float) -> float:
    if equity <= 0 or stop_distance <= 0:
        return 0.0
    by_risk = equity * risk_fraction / stop_distance
    return min(by_risk, equity * max_fraction)
