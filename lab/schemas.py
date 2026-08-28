from enum import Enum
from pydantic import BaseModel, Field

class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"

class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    BOTH = "both"

class Hypothesis(BaseModel):
    title: str
    thesis: str
    market_types: list[MarketType]
    directions: list[Direction]
    timeframes: list[str]
    symbols: list[str]
    rules: list[str] = Field(default_factory=list)
    rationale_sources: list[str] = Field(default_factory=list)
    novelty: str = "unknown"
    falsification_plan: list[str] = Field(default_factory=list)
    executable_family: str = ""
    executable_parameters: dict = Field(default_factory=dict)

class StrategyCandidate(BaseModel):
    candidate_id: str
    hypothesis_id: str
    entry_logic: str
    exit_logic: str
    risk_logic: str
    parameters: dict = Field(default_factory=dict)
    code_hash: str

class BacktestSummary(BaseModel):
    total_return: float
    cagr: float | None = None
    max_drawdown: float
    sharpe: float
    sortino: float
    profit_factor: float
    trade_count: int
    win_rate: float
    fees_paid: float
    slippage_paid: float
    funding_paid: float = 0.0
    final_equity: float

class ValidationSummary(BaseModel):
    in_sample: BacktestSummary
    out_of_sample: BacktestSummary
    walk_forward_passed: bool
    parameter_stability_passed: bool
    stress_tests_passed: bool
    falsification_passes: int
    overall_passed: bool
    rejection_reasons: list[str] = Field(default_factory=list)
