from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class StressCase:
    name: str
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0


def stressed_costs(base_fee_bps: float, base_slippage_bps: float, case: StressCase) -> tuple[float, float]:
    return base_fee_bps * case.fee_multiplier, base_slippage_bps * case.slippage_multiplier


def perturb_returns(returns: pd.Series, scale: float = 1.0) -> pd.Series:
    if scale < 0:
        raise ValueError("scale must be non-negative")
    return returns.astype(float) * scale
