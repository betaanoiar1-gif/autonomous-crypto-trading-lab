from pydantic import BaseModel, Field
from typing import Literal

class Hypothesis(BaseModel):
    title: str
    thesis: str
    market: str = "crypto"
    instruments: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    rationale: str
    expected_edge: str
    falsifiers: list[str] = Field(default_factory=list)
    source_notes: list[str] = Field(default_factory=list)

class Candidate(BaseModel):
    candidate_id: str
    hypothesis_id: str
    mode: Literal["spot", "futures"]
    direction: Literal["long", "short", "long_short"]
    leverage: float = 1.0
    parameters: dict = Field(default_factory=dict)
    status: Literal["PROPOSED", "BACKTESTED", "ATTACKED", "VALIDATED", "REJECTED"] = "PROPOSED"
