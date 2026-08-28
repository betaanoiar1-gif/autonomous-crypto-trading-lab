import pandas as pd
from lab.backtest.engine import run_ohlcv


def test_long_signal_grows_equity_without_costs():
    df = pd.DataFrame({'close': [100, 110, 121]})
    signal = pd.Series([1, 1, 1])
    result = run_ohlcv(df, signal, 500, fee_bps=0, slippage_bps=0)
    assert result.metrics['final_equity'] == 605.0


def test_drawdown_is_non_positive():
    df = pd.DataFrame({'close': [100, 120, 90]})
    signal = pd.Series([1, 1, 1])
    result = run_ohlcv(df, signal, 500, fee_bps=0, slippage_bps=0)
    assert result.metrics['max_drawdown'] <= 0
