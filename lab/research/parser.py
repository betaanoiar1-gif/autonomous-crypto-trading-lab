from __future__ import annotations

import json
from ..schemas import Hypothesis, MarketType, Direction


def parse_hypotheses(text: str) -> list[Hypothesis]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent did not return valid JSON: {exc}") from exc

    items = raw.get("hypotheses")
    if not isinstance(items, list):
        raise ValueError("Missing hypotheses list")

    result: list[Hypothesis] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            result.append(
                Hypothesis(
                    title=str(item.get("title", "Untitled")),
                    thesis=str(item.get("thesis", "")),
                    market_types=[MarketType(x) for x in item.get("market_types", ["spot"])],
                    directions=[Direction(x) for x in item.get("directions", ["long"])],
                    timeframes=[str(x) for x in item.get("timeframes", ["1h"])],
                    symbols=[str(x) for x in item.get("symbols", ["BTC/USDT"])],
                    rules=[str(x) for x in item.get("rules", [])],
                    rationale_sources=[str(x) for x in item.get("rationale_sources", [])],
                    novelty=str(item.get("novelty", "unknown")),
                    falsification_plan=[str(x) for x in item.get("falsification_plan", [])],
                    executable_family=str(item.get("executable_family", "momentum")),
                    executable_parameters=dict(item.get("executable_parameters", {})),
                )
            )
        except Exception as exc:
            raise ValueError(f"Invalid hypothesis payload: {item}") from exc
    return result
