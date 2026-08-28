from dataclasses import dataclass

@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float


def score_candidate(metrics: dict) -> float:
    """Conservative research score; not a promise or direct trading objective."""
    ret = max(-1.0, min(3.0, float(metrics.get("oos_return", 0.0))))
    dd = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
    pf = max(0.0, min(4.0, float(metrics.get("profit_factor", 0.0))))
    stability = 1.0 if metrics.get("parameter_stability", False) else 0.0
    return 0.35 * ret + 0.25 * pf + 0.25 * stability - 0.35 * dd
