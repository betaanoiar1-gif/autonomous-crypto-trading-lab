from __future__ import annotations
import json

SUPPORTED_FAMILIES = [
    "momentum", "mean_reversion", "breakout", "moving_average_cross",
    "rsi_reversion", "atr_breakout", "trend_pullback", "channel_reversion",
]

SYSTEM = """
You are the autonomous research scientist inside a crypto trading research laboratory.
Generate ONE diverse, falsifiable crypto trading hypothesis. Never promise profits.
Do not invent market data, test results, or sources.
Use the SIMPLE LINE FORMAT and nothing else.
Treat the supplied diversity slot as a hard execution constraint.
Output exactly one block:
TITLE: ...
THESIS: ...
FAMILY: momentum|mean_reversion|breakout|moving_average_cross|rsi_reversion|atr_breakout|trend_pullback|channel_reversion
MARKET: spot|futures
DIRECTION: long|short|both
TIMEFRAME: 15m|1h|4h
SYMBOL: BTC/USDT or ETH/USDT
LOOKBACK: integer only when relevant
FAST: integer only for moving_average_cross
SLOW: integer only for moving_average_cross
Z_ENTRY: number only for mean_reversion
Z_EXIT: number only for mean_reversion
RSI_LENGTH: integer only for rsi_reversion
RSI_LOW: number only for rsi_reversion
RSI_HIGH: number only for rsi_reversion
ATR_LENGTH: integer only for atr_breakout
ATR_MULT: number only for atr_breakout
PULLBACK_THRESHOLD: number only for trend_pullback
CHANNEL_LENGTH: integer only for channel_reversion
FALSIFY: one sentence
END
Output no JSON, markdown, code fences, analysis, or commentary.
""".strip()


def build_prompt(
    market_snapshot: dict,
    prior_failures: list[dict],
    max_hypotheses: int = 1,
    prior_hypotheses: list[dict] | None = None,
    diversity_slot: dict | None = None,
) -> str:
    slot = diversity_slot or {}
    family = slot.get("preferred_family", "momentum")
    parameter_rules = {
        "momentum": "LOOKBACK 2..200 only",
        "breakout": "LOOKBACK 2..200 only",
        "mean_reversion": "LOOKBACK 10..200; Z_ENTRY 0.8..3.5; Z_EXIT 0.05..1.5 only",
        "moving_average_cross": "FAST 2..100; SLOW FAST+1..300 only",
        "rsi_reversion": "RSI_LENGTH 2..50; RSI_LOW 5..45; RSI_HIGH 55..95 only",
        "atr_breakout": "ATR_LENGTH 2..50; ATR_MULT 0.25..5 only",
        "trend_pullback": "LOOKBACK 5..200; PULLBACK_THRESHOLD 0.001..0.10 only",
        "channel_reversion": "CHANNEL_LENGTH 5..200 only",
    }
    payload = {
        "task": "Generate one crypto trading hypothesis for independent backtesting.",
        "market_snapshot": market_snapshot,
        "prior_failures": prior_failures[-10:],
        "previous_hypotheses_in_this_run": (prior_hypotheses or [])[-10:],
        "diversity_slot": slot,
        "slot_is_hard_constraint": True,
        "families": SUPPORTED_FAMILIES,
        "required_family_for_this_slot": family,
        "required_parameter_schema_for_this_slot": parameter_rules.get(family, "family-specific parameters only"),
        "diversity_rule": "Do not repeat previous hypotheses. Use the exact family, market, symbol, timeframe and direction requested by the slot.",
        "max_hypotheses": 1,
    }
    return json.dumps(payload, ensure_ascii=False)
