from __future__ import annotations

from .evaluator import _run, _metrics


def confirm_on_independent_market(
    df,
    family: str,
    params: dict,
    directions: list[str],
    capital: float,
    fee_bps: float,
    slippage_bps: float,
    market_type: str = "spot",
    leverage: float = 1.0,
    funding_rates=None,
) -> dict:
    """Test frozen parameters on an independent market without re-tuning."""
    result = _run(
        df,
        family,
        dict(params),
        directions,
        capital,
        fee_bps,
        slippage_bps,
        market_type=market_type,
        leverage=leverage,
        funding_rates=funding_rates,
    )
    return _metrics(result, result.returns)


def confirmation_passed(metrics: dict) -> bool:
    return bool(
        float(metrics.get("total_return", 0.0)) > 0.0
        and float(metrics.get("profit_factor", 0.0)) > 1.0
        and int(metrics.get("trade_count", 0)) >= 8
        and float(metrics.get("max_drawdown", 0.0)) >= -0.50
    )
