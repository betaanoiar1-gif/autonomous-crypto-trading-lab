from dataclasses import dataclass

@dataclass
class CapitalState:
    equity: float
    peak_equity: float


def next_equity(current_equity: float, period_return: float) -> float:
    return max(0.0, current_equity * (1.0 + period_return))


def fixed_fraction_notional(equity: float, risk_fraction: float, leverage: float = 1.0) -> float:
    if not 0 <= risk_fraction <= 1:
        raise ValueError("risk_fraction must be between 0 and 1")
    if leverage < 0:
        raise ValueError("leverage must be non-negative")
    return equity * risk_fraction * leverage
