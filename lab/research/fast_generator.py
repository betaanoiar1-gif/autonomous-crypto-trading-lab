from __future__ import annotations

from ..schemas import Hypothesis, MarketType, Direction
from .run import DIVERSITY_SLOTS

POOLS = {
    "momentum": [8, 12, 18, 25, 40, 60, 90, 120, 160, 200],
    "mean_reversion": [(20, 1.25, .25), (30, 1.5, .35), (40, 2.0, .5), (60, 2.5, .75), (90, 3.0, 1.0), (120, 3.5, 1.25)],
    "breakout": [8, 12, 18, 25, 40, 60, 90, 120, 160],
    "moving_average_cross": [(5, 20), (10, 30), (15, 45), (20, 60), (25, 80), (40, 120), (60, 180)],
    "rsi_reversion": [(7, 20, 80), (10, 25, 75), (14, 30, 70), (20, 35, 80), (25, 40, 85)],
    "atr_breakout": [(5, 1.0), (8, 1.25), (11, 1.5), (14, 1.75), (18, 2.0), (25, 2.5)],
    "trend_pullback": [(10, .005), (15, .01), (25, .015), (40, .02), (60, .03), (90, .04)],
    "channel_reversion": [10, 20, 30, 40, 60, 90, 120],
}

def generate(prior_failures: list[dict], target: int) -> list[Hypothesis]:
    offset = len(prior_failures) // max(1, target)
    out: list[Hypothesis] = []
    for i, slot in enumerate(DIVERSITY_SLOTS[:target]):
        family = slot["preferred_family"]
        choice = POOLS[family][(offset + i) % len(POOLS[family])]
        if family in {"momentum", "breakout"}:
            params = {"lookback": choice}
        elif family == "mean_reversion":
            lb, ze, zx = choice; params = {"lookback": lb, "z_entry": ze, "z_exit": zx}
        elif family == "moving_average_cross":
            fast, slow = choice; params = {"fast": fast, "slow": slow}
        elif family == "rsi_reversion":
            n, lo, hi = choice; params = {"rsi_length": n, "rsi_low": lo, "rsi_high": hi}
        elif family == "atr_breakout":
            n, m = choice; params = {"atr_length": n, "atr_mult": m}
        elif family == "trend_pullback":
            lb, th = choice; params = {"lookback": lb, "pullback_threshold": th}
        else:
            params = {"channel_length": choice}
        title = family.replace("_", " ").title() + " | " + " | ".join(f"{k}={v}" for k, v in params.items())
        out.append(Hypothesis(
            title=title,
            thesis=f"Autonomous grid exploration for {family}; validate empirically only.",
            market_types=[MarketType.FUTURES if slot["preferred_market"] == "futures" else MarketType.SPOT],
            directions=[Direction(str(slot["preferred_direction"]))],
            timeframes=[slot["preferred_timeframe"]],
            symbols=[slot["preferred_symbol"]],
            rules=[f"Executable family: {family}"],
            novelty="fast-local-grid",
            falsification_plan=["Reject when OOS/WF/stress/confirmation gates fail."],
            executable_family=family,
            executable_parameters=params,
        ))
    return out
