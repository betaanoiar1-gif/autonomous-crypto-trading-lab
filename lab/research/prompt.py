from __future__ import annotations
import json

SYSTEM = """
You are the autonomous research scientist inside a crypto trading research laboratory.
Generate diverse, falsifiable crypto trading hypotheses. Never promise profits.
Do not invent market data, test results, or sources. External information is only input for hypotheses.
The local model is small, so use this SIMPLE LINE FORMAT and nothing else.
For each hypothesis output one block:
TITLE: ...
THESIS: ...
FAMILY: momentum|mean_reversion|breakout|moving_average_cross
DIRECTION: long|short|both
TIMEFRAME: 1h
SYMBOL: BTC/USDT or ETH/USDT
LOOKBACK: integer
FAST: integer
SLOW: integer
Z_ENTRY: number
Z_EXIT: number
FALSIFY: one sentence
END
Do not output JSON, markdown, code fences, analysis, or commentary.
""".strip()

SUPPORTED_FAMILIES = ["momentum", "mean_reversion", "breakout", "moving_average_cross"]


def build_prompt(market_snapshot: dict, prior_failures: list[dict], max_hypotheses: int = 4) -> str:
    payload = {
        "task": "Generate diverse crypto trading hypotheses for independent backtesting.",
        "market_snapshot": market_snapshot,
        "prior_failures": prior_failures[-10:],
        "max_hypotheses": max_hypotheses,
        "families": SUPPORTED_FAMILIES,
        "parameter_rules": {
            "momentum": "LOOKBACK 2..200",
            "mean_reversion": "LOOKBACK 2..200; Z_ENTRY 0.5..4; Z_EXIT 0..2",
            "breakout": "LOOKBACK 2..200",
            "moving_average_cross": "FAST 2..100; SLOW FAST+1..300",
        },
    }
    return json.dumps(payload, ensure_ascii=False)
