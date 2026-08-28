from lab.validation import validation_gate


def test_validation_rejects_missing_metrics():
    result = validation_gate({})
    assert result.passed is False


def test_validation_rejects_negative_oos_return():
    result = validation_gate({
        "oos_return": -0.1,
        "max_drawdown": -0.1,
        "profit_factor": 1.5,
        "trades": 100,
    })
    assert result.passed is False


def test_validation_accepts_complete_candidate():
    result = validation_gate({
        "oos_return": 0.25,
        "max_drawdown": -0.15,
        "profit_factor": 1.4,
        "trades": 100,
        "parameter_stability": True,
        "walk_forward_passed": True,
        "stress_tests_passed": True,
        "lookahead_check_passed": True,
    })
    assert result.passed is True
