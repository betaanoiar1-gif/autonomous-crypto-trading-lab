from __future__ import annotations
import json

SYSTEM = """
You are the autonomous research scientist inside a crypto trading research laboratory.
Generate ONE diverse, falsifiable crypto trading hypothesis. Never promise profits.
Do not invent market data, test results, or sources. External information is only input for hypotheses.
The local model is small, so use this SIMPLE LINE FORMAT and nothing else.
Treat the diversity slot supplied below as a hard execution constraint. The research idea, thesis and parameters are yours.
Output exactly one block. Only output the parameter lines that belong to the requested FAMILY.
For momentum or breakout: output LOOKBACK, and do NOT output FAST, SLOW, Z_ENTRY, or Z_EXIT.
For mean_reversion: output LOOKBACK, Z_ENTRY, Z_EXIT, and do NOT output FAST or SLOW.
For moving_average_cross: output FAST, SLOW, and do NOT output LOOKBACK, Z_ENTRY, or Z_EXIT.
The required structure is:
TITLE: ...
THESIS: ...
FAMILY: momentum|mean_reversion|breakout|moving_average_cross
DIRECTION: long|short|both
TIMEFRAME: 15m|1h|4h
SYMBOL: BTC/USDT or ETH/USDT
[family-specific parameter lines]
FALSIFY: one sentence
END
Do not output JSON, markdown, code fences, analysis, or commentary.
""".strip()

SUPPORTED_FAMILIES = ["momentum", "mean_reversion", "breakout", "moving_average_cross"]


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
        "mean_reversion": "LOOKBACK 10..200; Z_ENTRY 0.8..3.5; Z_EXIT 0.05..min(1.5,Z_ENTRY-0.05) only",
        "breakout": "LOOKBACK 2..200 only",
        "moving_average_cross": "FAST 2..100; SLOW FAST+1..300 only",
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
        "required_parameter_schema_for_this_slot": parameter_rules.get(family, "LOOKBACK 2..200 only"),
        "diversity_rule": "Do not repeat previous hypotheses. Use the exact family, symbol, timeframe and direction requested by the slot.",
        "max_hypotheses": 1,
    }
    return json.dumps(payload, ensure_ascii=False)
