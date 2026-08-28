from __future__ import annotations

from ..schemas import Hypothesis, MarketType, Direction


def _num(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def parse_hypotheses(text: str) -> list[Hypothesis]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}

    for line in lines:
        if line.upper() == "END":
            if current:
                blocks.append(current)
                current = {}
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in {"TITLE", "THESIS", "FAMILY", "DIRECTION", "TIMEFRAME", "SYMBOL", "LOOKBACK", "FAST", "SLOW", "Z_ENTRY", "Z_EXIT", "FALSIFY"}:
            current[key] = value.strip()

    if current:
        blocks.append(current)

    result: list[Hypothesis] = []
    for i, b in enumerate(blocks, 1):
        family = b.get("FAMILY", "momentum").lower()
        if family not in {"momentum", "mean_reversion", "breakout", "moving_average_cross"}:
            family = "momentum"
        direction = b.get("DIRECTION", "long").lower()
        if direction not in {"long", "short", "both"}:
            direction = "long"
        symbol = b.get("SYMBOL", "BTC/USDT")
        timeframe = b.get("TIMEFRAME", "1h")

        params: dict[str, float | int] = {}
        if family in {"momentum", "breakout", "mean_reversion"}:
            params["lookback"] = max(2, min(200, int(_num(b.get("LOOKBACK", "20"), 20))))
        if family == "mean_reversion":
            params["z_entry"] = max(0.5, min(4.0, _num(b.get("Z_ENTRY", "1.5"), 1.5)))
            params["z_exit"] = max(0.0, min(2.0, _num(b.get("Z_EXIT", "0.25"), 0.25)))
        if family == "moving_average_cross":
            fast = max(2, min(100, int(_num(b.get("FAST", "10"), 10))))
            slow = max(fast + 1, min(300, int(_num(b.get("SLOW", "40"), 40))))
            params["fast"] = fast
            params["slow"] = slow

        result.append(Hypothesis(
            title=b.get("TITLE", f"Hypothesis {i}"),
            thesis=b.get("THESIS", ""),
            market_types=[MarketType.SPOT],
            directions=[Direction(direction)],
            timeframes=[timeframe],
            symbols=[symbol],
            rules=[f"Executable family: {family}"],
            rationale_sources=[],
            novelty="local-agent-generated",
            falsification_plan=[b.get("FALSIFY", "Reject if out-of-sample performance is not robust.")],
            executable_family=family,
            executable_parameters=params,
        ))

    if not result:
        raise ValueError("Agent returned no usable hypotheses")
    return result
