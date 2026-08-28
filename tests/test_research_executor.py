import pandas as pd

from lab.research.executor import compile_signal


def sample():
    close = pd.Series([100 + i * 0.5 for i in range(100)])
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0})


def test_supported_families_return_bounded_signals():
    df = sample()
    cases = [
        ("momentum", {"lookback": 10}),
        ("mean_reversion", {"lookback": 20, "z_entry": 1.5, "z_exit": 0.25}),
        ("breakout", {"lookback": 10}),
        ("moving_average_cross", {"fast": 5, "slow": 20}),
    ]
    for family, params in cases:
        signal = compile_signal(df, family, params, ["both"])
        assert signal.index.equals(df.index)
        assert signal.dropna().between(-1, 1).all()
