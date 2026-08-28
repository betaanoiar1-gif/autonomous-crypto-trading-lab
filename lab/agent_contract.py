from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentTask:
    objective: str
    constraints: dict[str, Any] = field(default_factory=dict)
    prior_failures: list[str] = field(default_factory=list)

@dataclass
class AgentDecision:
    action: str
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)

class ResearchAgent:
    """Provider-neutral contract for the external autonomous research agent."""
    def propose(self, task: AgentTask) -> AgentDecision:
        raise NotImplementedError

    def evaluate(self, task: AgentTask, evidence: dict[str, Any]) -> AgentDecision:
        raise NotImplementedError
