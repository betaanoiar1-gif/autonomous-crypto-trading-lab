from __future__ import annotations
import json

SYSTEM = """
You are the autonomous research scientist inside a crypto trading research laboratory.
Propose falsifiable trading hypotheses, not promises of profit.
You may choose assets, timeframes, spot/futures, long/short and research methods.
External claims are hypothesis inputs, never proof. Prefer simple, testable ideas.
Avoid look-ahead bias, leakage and retrospective rule changes.
For executable testing, express each hypothesis with one supported family and numeric parameters.
Return STRICT JSON only. No markdown or prose outside JSON.
""".strip()

SUPPORTED_FAMILIES = ["momentum", "mean_reversion", "breakout", "moving_average_cross"]


def build_prompt(market_snapshot: dict, prior_failures: list[dict], max_hypotheses: int = 5) -> str:
    payload = {
        "task": "Generate diverse crypto trading hypotheses for independent backtesting.",
        "market_snapshot": market_snapshot,
        "prior_failures": prior_failures[-20:],
        "max_hypotheses": max_hypotheses,
        "supported_executable_families": SUPPORTED_FAMILIES,
        "parameter_rules": {
            "momentum": {"lookback": "int 2..200"},
            "mean_reversion": {"lookback": "int 2..200", "z_entry": "float 0.5..4", "z_exit": "float 0..2"},
            "breakout": {"lookback": "int 2..200"},
            "moving_average_cross": {"fast": "int 2..100", "slow": "int 3..300, slow > fast"}
        },
        "required_schema": {"hypotheses": [{
            "title": "string",
            "thesis": "string",
            "market_types": ["spot"],
            "directions": ["long"],
            "timeframes": ["1h"],
            "symbols": ["BTC/USDT"],
            "rules": ["explicit rules"],
            "novelty": "string",
            "falsification_plan": ["specific attacks"],
            "executable_family": "one supported family",
            "executable_parameters": {"family-specific numeric parameters": "value"}
        }]}
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
