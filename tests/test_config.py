from lab.config import load_settings


def test_default_configuration():
    settings = load_settings()
    assert settings.capital.initial_usd == 500.0
    assert settings.research.autonomous is True
    assert settings.research.require_independent_validation is True
