from dataclasses import dataclass, field
from math import isfinite

@dataclass
class ValidationResult:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validation_gate(metrics: dict) -> ValidationResult:
    """Research gate. Thresholds reject clearly weak/fragile candidates; it is not a profit guarantee."""
    reasons: list[str] = []
    warnings: list[str] = []
    required = ("oos_return", "max_drawdown", "profit_factor", "trades")
    missing = [k for k in required if k not in metrics]
    if missing:
        return ValidationResult(False, 0.0, warnings=[f"Missing metrics: {missing}"])

    numeric_keys = ("oos_return", "max_drawdown", "profit_factor")
    for key in numeric_keys:
        if not isfinite(float(metrics[key])):
            reasons.append(f"Non-finite metric: {key}")

    if float(metrics["oos_return"]) <= 0:
        reasons.append("Non-positive out-of-sample return")
    if float(metrics["profit_factor"]) <= 1.0:
        reasons.append("Out-of-sample profit factor is not above 1")
    if float(metrics["max_drawdown"]) <= -0.50:
        reasons.append("Drawdown exceeds conservative research limit")
    if int(metrics["trades"]) < 30:
        warnings.append("Low trade count: statistical confidence may be weak")
    if metrics.get("parameter_stability") is False:
        reasons.append("Parameter stability failed")
    if metrics.get("walk_forward_passed") is False:
        reasons.append("Walk-forward validation failed")
    if metrics.get("stress_tests_passed") is False:
        reasons.append("Stress testing failed")
    if metrics.get("lookahead_check_passed") is False:
        reasons.append("Look-ahead/data leakage check failed")

    score_parts = [
        max(0.0, min(1.0, float(metrics.get("oos_return", 0.0)) / 2.0)),
        max(0.0, min(1.0, float(metrics.get("profit_factor", 0.0)) / 2.0)),
        1.0 if metrics.get("parameter_stability", False) else 0.0,
        1.0 if metrics.get("walk_forward_passed", False) else 0.0,
        1.0 if metrics.get("stress_tests_passed", False) else 0.0,
        1.0 if metrics.get("lookahead_check_passed", False) else 0.0,
    ]
    score = sum(score_parts) / len(score_parts)
    return ValidationResult(not reasons, score, reasons, warnings)
