from dataclasses import dataclass, field
from typing import Protocol
from ..schemas import Hypothesis, StrategyCandidate

class ResearchAgent(Protocol):
    def discover(self, context: dict) -> list[Hypothesis]: ...
    def build(self, hypothesis: Hypothesis, context: dict) -> list[StrategyCandidate]: ...
    def learn(self, result: dict, context: dict) -> None: ...

@dataclass
class ResearchLoop:
    agent: ResearchAgent
    max_hypotheses: int = 10
    history: list[dict] = field(default_factory=list)

    def propose(self, context: dict) -> list[Hypothesis]:
        ideas = self.agent.discover(context)
        return ideas[: self.max_hypotheses]

    def record(self, result: dict, context: dict) -> None:
        self.history.append(result)
        self.agent.learn(result, context)
