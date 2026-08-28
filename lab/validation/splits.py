from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def holdout_split(df: pd.DataFrame, test_ratio: float = 0.30) -> TimeSplit:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    cut = int(len(df) * (1 - test_ratio))
    if cut <= 0 or cut >= len(df):
        raise ValueError("Not enough observations for split")
    return TimeSplit(df.iloc[:cut].copy(), df.iloc[cut:].copy())


def walk_forward_splits(df: pd.DataFrame, train_size: int, test_size: int, step: int | None = None):
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    start = 0
    while start + train_size + test_size <= len(df):
        yield TimeSplit(df.iloc[start:start+train_size].copy(), df.iloc[start+train_size:start+train_size+test_size].copy())
        start += step
