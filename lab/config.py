from pathlib import Path
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

class CapitalConfig(BaseModel):
    initial_usd: float = 500.0
    base_currency: str = "USD"

class ResearchConfig(BaseModel):
    market: str = "crypto"
    autonomous: bool = True
    internet_for_hypotheses: bool = True
    max_experiments_per_run: int = 25
    max_autonomous_cycles: int = 10
    require_independent_validation: bool = True

class ExecutionConfig(BaseModel):
    allow_spot: bool = True
    allow_futures: bool = True
    allow_long: bool = True
    allow_short: bool = True
    allow_leverage: bool = True
    max_leverage: float = 3.0
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    include_funding: bool = True

class ValidationConfig(BaseModel):
    holdout_ratio: float = 0.30
    walk_forward: bool = True
    falsification_passes: int = 5
    require_parameter_stability: bool = True

class OutputConfig(BaseModel):
    generate_pine: bool = True
    generate_reports: bool = True
    write_experiment_ledger: bool = True

class Settings(BaseModel):
    project: dict
    capital: CapitalConfig
    research: ResearchConfig
    execution: ExecutionConfig
    validation: ValidationConfig
    output: OutputConfig

def load_settings(path: str | Path = ROOT / "config/default.yaml") -> Settings:
    with open(path, "r", encoding="utf-8") as f:
        return Settings(**yaml.safe_load(f))
