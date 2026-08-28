from dataclasses import dataclass, field
from typing import Any

@dataclass
class Hypothesis:
    title: str
    thesis: str
    market: str = "crypto"
    timeframes: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    falsification_plan: list[str] = field(default_factory=list)

@dataclass
class Candidate:
    hypothesis: Hypothesis
    implementation_ref: str
    in_sample: dict[str, Any] = field(default_factory=dict)
    out_of_sample: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    status: str = "PROPOSED"

RESEARCH_PROTOCOL = [
    "Explore multiple mechanisms and timeframes rather than optimizing one idea immediately.",
    "Use external research only to generate hypotheses and implementation clues.",
    "Convert each hypothesis into an explicit, testable rule set before evaluation.",
    "Backtest with fees, slippage, and funding assumptions when applicable.",
    "Separate development data from untouched validation data.",
    "Attack promising candidates with parameter perturbation, regime splits, and falsification tests.",
    "Prefer stable risk-adjusted behavior over spectacular isolated returns.",
    "Keep failed experiments so the research agent can avoid repeating dead ends.",
    "Generate Pine Script only from a reproducible validated specification.",
]
