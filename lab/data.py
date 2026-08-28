from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class MarketData:
    symbol: str
    timeframe: str
    frame: pd.DataFrame
    source: str


def load_csv(path: str | Path, symbol: str, timeframe: str) -> MarketData:
    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return MarketData(symbol, timeframe, df, f"csv:{Path(path).name}")
