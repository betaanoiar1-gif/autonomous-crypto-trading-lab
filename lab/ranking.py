from dataclasses import dataclass

@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def score_candidate(metrics: dict) -> float:
    """Research ranking score; not a direct trading objective or guarantee."""
    ret = _clip(float(metrics.get("oos_return", 0.0)), -1.0, 3.0) / 3.0
    pf = _clip(float(metrics.get("profit_factor", 0.0)), 0.0, 4.0) / 4.0
    dd = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
    dd_quality = 1.0 - _clip(dd, 0.0, 0.75) / 0.75
    sharpe = _clip(float(metrics.get("sharpe", 0.0)), -1.0, 3.0)
    sharpe_quality = (sharpe + 1.0) / 4.0
    stability = 1.0 if metrics.get("parameter_stability") else 0.0
    walk_forward = 1.0 if metrics.get("walk_forward_passed") else 0.0
    stress = 1.0 if metrics.get("stress_tests_passed") else 0.0
    complexity_penalty = _clip(float(metrics.get("complexity", 0.0)), 0.0, 1.0)
    trial_penalty = _clip(float(metrics.get("trial_penalty", 0.0)), 0.0, 1.0)

    score = (
        0.18 * ret
        + 0.14 * pf
        + 0.14 * dd_quality
        + 0.12 * sharpe_quality
        + 0.14 * stability
        + 0.12 * walk_forward
        + 0.10 * stress
        - 0.04 * complexity_penalty
        - 0.02 * trial_penalty
    )
    return float(score)
