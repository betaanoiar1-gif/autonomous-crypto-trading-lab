from __future__ import annotations

import pandas as pd


def chronological_holdout(df: pd.DataFrame, holdout_ratio: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between 0 and 1")
    n = len(df)
    cut = int(n * (1.0 - holdout_ratio))
    if cut <= 0 or cut >= n:
        raise ValueError("Dataset is too small for requested holdout")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def walk_forward_slices(df: pd.DataFrame, train_size: int, test_size: int, step: int | None = None):
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    if step <= 0:
        raise ValueError("step must be positive")
    start = 0
    while start + train_size + test_size <= len(df):
        train = df.iloc[start:start + train_size].copy()
        test = df.iloc[start + train_size:start + train_size + test_size].copy()
        yield train, test
        start += step
