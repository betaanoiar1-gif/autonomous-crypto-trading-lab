from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    passed: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validation_gate(metrics: dict) -> ValidationResult:
    """Conservative gate; real thresholds should be calibrated after data-engine integration."""
    reasons, warnings = [], []
    required = ["oos_return", "max_drawdown", "profit_factor"]
    missing = [k for k in required if k not in metrics]
    if missing:
        return ValidationResult(False, 0.0, warnings=[f"Missing metrics: {missing}"])
    if metrics["oos_return"] <= 0:
        reasons.append("Non-positive out-of-sample return")
    if metrics["profit_factor"] <= 1.0:
        reasons.append("Out-of-sample profit factor is not above 1")
    if metrics["max_drawdown"] <= -0.50:
        reasons.append("Drawdown exceeds conservative research limit")
    if metrics.get("parameter_stability") is False:
        reasons.append("Parameter stability failed")
    passed = not reasons
    score = 1.0 if passed else 0.0
    return ValidationResult(passed, score, reasons, warnings)
