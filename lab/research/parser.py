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
        if key in {"TITLE", "THESIS", "FAMILY", "MARKET", "DIRECTION", "TIMEFRAME", "SYMBOL", "LOOKBACK", "FAST", "SLOW", "Z_ENTRY", "Z_EXIT", "RSI_LENGTH", "RSI_LOW", "RSI_HIGH", "ATR_LENGTH", "ATR_MULT", "CHANNEL_LENGTH", "FALSIFY"}:
            current[key] = value.strip()
    if current:
        blocks.append(current)

    supported = {
        "momentum", "mean_reversion", "breakout", "moving_average_cross",
        "rsi_reversion", "atr_breakout", "trend_pullback", "channel_reversion",
    }
    result: list[Hypothesis] = []
    for i, b in enumerate(blocks, 1):
        family = b.get("FAMILY", "momentum").lower()
        if family not in supported:
            family = "momentum"
        direction = b.get("DIRECTION", "long").lower()
        if direction not in {"long", "short", "both"}:
            direction = "long"
        market = b.get("MARKET", "spot").lower()
        if market not in {"spot", "futures"}:
            market = "spot"
        symbol = b.get("SYMBOL", "BTC/USDT")
        timeframe = b.get("TIMEFRAME", "1h")

        params: dict[str, float | int] = {}
        if family in {"momentum", "breakout", "mean_reversion", "channel_reversion", "trend_pullback"}:
            params["lookback"] = max(2, min(200, int(_num(b.get("LOOKBACK", "20"), 20))))
        if family == "mean_reversion":
            params["z_entry"] = max(0.5, min(4.0, _num(b.get("Z_ENTRY", "1.5"), 1.5)))
            params["z_exit"] = max(0.0, min(2.0, _num(b.get("Z_EXIT", "0.25"), 0.25)))
        if family == "moving_average_cross":
            fast = max(2, min(100, int(_num(b.get("FAST", "10"), 10))))
            slow = max(fast + 1, min(300, int(_num(b.get("SLOW", "40"), 40))))
            params.update(fast=fast, slow=slow)
        if family == "rsi_reversion":
            params.update(
                rsi_length=max(2, min(50, int(_num(b.get("RSI_LENGTH", "14"), 14)))),
                rsi_low=max(5.0, min(45.0, _num(b.get("RSI_LOW", "30"), 30))),
                rsi_high=max(55.0, min(95.0, _num(b.get("RSI_HIGH", "70"), 70))),
            )
        if family == "atr_breakout":
            params.update(
                atr_length=max(2, min(50, int(_num(b.get("ATR_LENGTH", "14"), 14)))),
                atr_mult=max(0.25, min(5.0, _num(b.get("ATR_MULT", "1.5"), 1.5))),
            )
        if family == "channel_reversion":
            params.update(
                channel_length=max(5, min(200, int(_num(b.get("CHANNEL_LENGTH", "40"), 40)))),
            )

        result.append(Hypothesis(
            title=b.get("TITLE", f"Hypothesis {i}"),
            thesis=b.get("THESIS", ""),
            market_types=[MarketType.FUTURES if market == "futures" else MarketType.SPOT],
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
