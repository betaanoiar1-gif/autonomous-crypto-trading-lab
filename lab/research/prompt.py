from __future__ import annotations

SYSTEM = """
You are the autonomous research scientist inside a crypto trading research laboratory.
Your job is to propose falsifiable trading hypotheses, not to promise profits.
You may choose assets, timeframes, spot/futures, long/short and research methods.
Use your knowledge to generate ideas; external claims are hypotheses, never proof.
Prefer simple, testable hypotheses. Avoid look-ahead bias and data leakage.
Return STRICT JSON only. No markdown, no prose outside JSON.
""".strip()


def build_prompt(market_snapshot: dict, prior_failures: list[dict], max_hypotheses: int = 5) -> str:
    payload = {
        "task": "Generate diverse crypto trading hypotheses for independent backtesting.",
        "market_snapshot": market_snapshot,
        "prior_failures": prior_failures[-20:],
        "max_hypotheses": max_hypotheses,
        "required_schema": {
            "hypotheses": [
                {
                    "title": "string",
                    "thesis": "string",
                    "market_types": ["spot"],
                    "directions": ["long"],
                    "timeframes": ["1h"],
                    "symbols": ["BTC/USDT"],
                    "rules": ["explicit, testable rules"],
                    "novelty": "what is different",
                    "falsification_plan": ["specific ways to disprove it"]
                }
            ]
        }
    }
    import json
    return json.dumps(payload, ensure_ascii=False, indent=2)
