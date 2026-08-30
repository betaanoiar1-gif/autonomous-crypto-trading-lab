from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable
import json
import math

import numpy as np
import pandas as pd

from .backtest.engine import run_ohlcv

SignalFn = Callable[[pd.DataFrame, dict], pd.Series]


@dataclass
class TrialResult:
    name: str
    params: dict
    train_return: float
    test_return: float
    test_drawdown: float
    test_profit_factor: float
    trades: int
    score: float
    accepted: bool
    rejection_reasons: list[str]


class LearningEngine:
    """Research-only optimizer that learns from failed trials without touching holdout data."""

    def __init__(self, initial_capital: float = 500.0, fee_bps: float = 10.0,
                 slippage_bps: float = 5.0, min_trades: int = 20,
                 max_drawdown: float = -0.35):
        self.initial_capital = initial_capital
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.min_trades = min_trades
        self.max_drawdown = max_drawdown
        self.memory: list[dict] = []

    @staticmethod
    def _score(metrics: dict) -> float:
        r = metrics["total_return"]
        dd = abs(min(metrics["max_drawdown"], 0.0))
        pf = min(metrics["profit_factor"], 5.0) if math.isfinite(metrics["profit_factor"]) else 5.0
        return float((r / max(dd, 0.05)) * math.log1p(pf))

    def evaluate(self, name: str, df: pd.DataFrame, signal_fn: SignalFn,
                 params: dict, train_ratio: float = 0.70,
                 market_type: str = "spot", leverage: float = 1.0) -> TrialResult:
        if not 0.5 <= train_ratio < 1.0:
            raise ValueError("train_ratio must be in [0.5, 1.0)")
        cut = int(len(df) * train_ratio)
        train, test = df.iloc[:cut], df.iloc[cut:]
        train_signal = signal_fn(train, params)
        test_signal = signal_fn(test, params)
        tr = run_ohlcv(train, train_signal, self.initial_capital, self.fee_bps,
                       self.slippage_bps, market_type, leverage)
        te = run_ohlcv(test, test_signal, self.initial_capital, self.fee_bps,
                       self.slippage_bps, market_type, leverage)
        reasons: list[str] = []
        if te.metrics["trade_count"] < self.min_trades:
            reasons.append("too_few_test_trades")
        if te.metrics["max_drawdown"] < self.max_drawdown:
            reasons.append("drawdown_limit")
        if te.metrics["total_return"] <= 0:
            reasons.append("negative_test_return")
        score = self._score(te.metrics)
        result = TrialResult(name, dict(params), tr.metrics["total_return"],
                             te.metrics["total_return"], te.metrics["max_drawdown"],
                             te.metrics["profit_factor"], te.metrics["trade_count"],
                             score, not reasons, reasons)
        self.memory.append(asdict(result))
        return result

    def rank(self) -> pd.DataFrame:
        if not self.memory:
            return pd.DataFrame()
        return pd.DataFrame(self.memory).sort_values("score", ascending=False)

    def lessons(self) -> list[str]:
        if not self.memory:
            return []
        df = self.rank()
        lessons = []
        rejected = df[~df.accepted]
        for reason in rejected.rejection_reasons.explode().dropna().value_counts().index:
            lessons.append(f"Avoid recurring failure: {reason}")
        if len(df) >= 3 and df.test_return.std() > abs(df.test_return.mean()):
            lessons.append("Results are unstable across trials; reduce parameter freedom.")
        if len(df) >= 5 and (df.test_return <= 0).mean() >= 0.6:
            lessons.append("Most candidates fail out of sample; change hypothesis family.")
        return lessons

    def export_memory(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"trials": self.memory, "lessons": self.lessons()}, f, indent=2)
