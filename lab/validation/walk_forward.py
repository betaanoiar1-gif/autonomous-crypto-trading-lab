import pandas as pd


def split_walk_forward(df: pd.DataFrame, train_size: int, test_size: int, step: int | None = None):
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    step = step or test_size
    start = 0
    while start + train_size + test_size <= len(df):
        train = df.iloc[start:start + train_size]
        test = df.iloc[start + train_size:start + train_size + test_size]
        yield train, test
        start += step
