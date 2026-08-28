import pandas as pd
import pytest

from lab.data.validate import validate_ohlcv
from lab.data.split import chronological_holdout


def frame(n=10):
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": range(100, 100+n),
        "high": range(101, 101+n),
        "low": range(99, 99+n),
        "close": range(100, 100+n),
        "volume": [10.0] * n,
    }, index=idx)


def test_validate_ohlcv():
    out = validate_ohlcv(frame())
    assert len(out) == 10


def test_holdout_is_chronological():
    train, test = chronological_holdout(frame(), 0.3)
    assert train.index.max() < test.index.min()
    assert len(test) == 3


def test_invalid_ohlcv_raises():
    bad = frame()
    bad.loc[bad.index[0], "high"] = 1
    with pytest.raises(ValueError):
        validate_ohlcv(bad)
