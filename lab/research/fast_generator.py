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


def _key(params: dict) -> tuple:
    return tuple(sorted((str(k), repr(v)) for k, v in params.items()))


def _params_from_record(record: dict) -> tuple[str | None, dict | None]:
    hypothesis = record.get("hypothesis") or {}
    family = hypothesis.get("executable_family")
    params = record.get("out_of_sample", {}).get("selected_parameters") or hypothesis.get("executable_parameters")
    if not isinstance(family, str) or not isinstance(params, dict):
        return None, None
    return family, params


def _local_variants(family: str, base: dict) -> list[dict]:
    """Build a small neighborhood around a previously tested near-miss.

    These candidates are deliberately bounded and only steer discovery. The
    validation gates in the evaluator remain unchanged.
    """
    out: list[dict] = []
    if family in {"momentum", "breakout"} and "lookback" in base:
        b = int(base["lookback"])
        for f in (0.65, 0.8, 0.9, 1.1, 1.25, 1.4):
            out.append({"lookback": max(2, min(200, int(round(b * f))))})
    elif family == "mean_reversion":
        if all(k in base for k in ("lookback", "z_entry", "z_exit")):
            b, ze, zx = int(base["lookback"]), float(base["z_entry"]), float(base["z_exit"])
            for lf, ef, xf in ((0.8, .95, 1.10), (.9, 1.0, .9), (1.0, .9, 1.1), (1.1, 1.05, .95), (1.25, 1.1, .9)):
                entry = max(.8, min(3.5, ze * ef))
                exit_ = max(.05, min(1.5, zx * xf, entry - .05))
                out.append({"lookback": max(10, min(200, int(round(b * lf)))), "z_entry": round(entry, 3), "z_exit": round(exit_, 3)})
    elif family == "moving_average_cross" and "fast" in base and "slow" in base:
        f, s = int(base["fast"]), int(base["slow"])
        ratio = max(1.4, min(4.0, s / max(1, f)))
        for mult in (.75, .9, 1.1, 1.25, 1.4):
            nf = max(2, min(100, int(round(f * mult))))
            ns = max(nf + 1, min(300, int(round(nf * ratio))))
            out.append({"fast": nf, "slow": ns})
        for nf, ns in ((30, 90), (35, 105), (45, 135), (50, 150), (55, 165), (70, 210), (80, 240)):
            out.append({"fast": nf, "slow": ns})
    elif family == "rsi_reversion":
        if all(k in base for k in ("rsi_length", "rsi_low", "rsi_high")):
            n, lo, hi = int(base["rsi_length"]), float(base["rsi_low"]), float(base["rsi_high"])
            for dn in (-4, -2, 2, 4):
                for dlo, dhi in ((-3, 3), (0, 0), (3, -3)):
                    out.append({"rsi_length": max(2, min(50, n + dn)), "rsi_low": max(5, min(45, lo + dlo)), "rsi_high": max(55, min(95, hi + dhi))})
    elif family == "atr_breakout":
        if all(k in base for k in ("atr_length", "atr_mult")):
            n, m = int(base["atr_length"]), float(base["atr_mult"])
            for dn, dm in ((-3, .85), (-2, 1.0), (0, .9), (0, 1.1), (2, 1.0), (3, 1.15)):
                out.append({"atr_length": max(2, min(50, n + dn)), "atr_mult": round(max(.25, min(5.0, m * dm)), 3)})
    elif family == "trend_pullback":
        if all(k in base for k in ("lookback", "pullback_threshold")):
            b, t = int(base["lookback"]), float(base["pullback_threshold"])
            for lf, tf in ((.8, .75), (.9, .9), (1.1, .9), (1.25, 1.0), (1.4, 1.1)):
                out.append({"lookback": max(5, min(200, int(round(b * lf)))), "pullback_threshold": round(max(.001, min(.10, t * tf)), 4)})
    elif family == "channel_reversion" and "channel_length" in base:
        b = int(base["channel_length"])
        for f in (.7, .85, 1.15, 1.3, 1.6):
            out.append({"channel_length": max(5, min(200, int(round(b * f))))})
    return out


def _candidate_params(family: str, prior_failures: list[dict]) -> list[dict]:
    """Return base-grid candidates plus local variants from near-misses.

    The order intentionally prioritizes unexplored local neighborhoods once a
    family has shown useful OOS/WF behavior, preventing the fixed generator
    from cycling forever through the same small grid.
    """
    candidates: list[dict] = []
    seen: set[tuple] = set()

    for choice in POOLS[family]:
        if family in {"momentum", "breakout"}:
            p = {"lookback": choice}
        elif family == "mean_reversion":
            lb, ze, zx = choice; p = {"lookback": lb, "z_entry": ze, "z_exit": zx}
        elif family == "moving_average_cross":
            fast, slow = choice; p = {"fast": fast, "slow": slow}
        elif family == "rsi_reversion":
            n, lo, hi = choice; p = {"rsi_length": n, "rsi_low": lo, "rsi_high": hi}
        elif family == "atr_breakout":
            n, m = choice; p = {"atr_length": n, "atr_mult": m}
        elif family == "trend_pullback":
            lb, th = choice; p = {"lookback": lb, "pullback_threshold": th}
        else:
            p = {"channel_length": choice}
        k = _key(p)
        if k not in seen:
            candidates.append(p)
            seen.add(k)

    # Pull local neighborhoods from prior rejected hypotheses that still had
    # at least one useful signal. This does not weaken validation; it only
    # changes what gets tested next.
    near_misses = []
    for record in prior_failures:
        fam, params = _params_from_record(record)
        if fam != family or params is None:
            continue
        oos = record.get("out_of_sample", {})
        walk = record.get("robustness", {}).get("walk_forward", {})
        useful = (
            float(oos.get("total_return", 0.0)) > 0
            or float(oos.get("profit_factor", 0.0)) > 1.0
            or bool(walk.get("passed", False))
            or int(walk.get("positive_windows", 0)) >= 2
        )
        if useful:
            near_misses.append(params)

    for base in near_misses[-12:]:
        for p in _local_variants(family, base):
            k = _key(p)
            if k not in seen:
                candidates.insert(0, p)
                seen.add(k)

    return candidates


def generate(prior_failures: list[dict], target: int) -> list[Hypothesis]:
    # Extract exact prior failures so the same parameter set is not intentionally
    # replayed forever. The deterministic rotation remains as a fallback when a
    # family has no useful history.
    rejected_keys: set[tuple] = set()
    for record in prior_failures:
        fam, params = _params_from_record(record)
        if fam and isinstance(params, dict):
            rejected_keys.add((fam, _key(params)))

    out: list[Hypothesis] = []
    for i, slot in enumerate(DIVERSITY_SLOTS[:target]):
        family = slot["preferred_family"]
        candidates = _candidate_params(family, prior_failures)
        if not candidates:
            continue

        # Stable rotation across cycles, then skip exact rejected combinations.
        cycle_shift = (len(prior_failures) + i * 7) % len(candidates)
        choice = None
        for j in range(len(candidates)):
            p = candidates[(cycle_shift + j) % len(candidates)]
            if (family, _key(p)) not in rejected_keys:
                choice = p
                break
        if choice is None:
            # All known options have been rejected. Explore the first local
            # variant/base option rather than replaying the entire history.
            choice = candidates[cycle_shift]

        title = family.replace("_", " ").title() + " | " + " | ".join(f"{k}={v}" for k, v in choice.items())
        out.append(Hypothesis(
            title=title,
            thesis=f"Adaptive autonomous grid exploration for {family}; validate empirically only.",
            market_types=[MarketType.FUTURES if slot["preferred_market"] == "futures" else MarketType.SPOT],
            directions=[Direction(str(slot["preferred_direction"]))],
            timeframes=[slot["preferred_timeframe"]],
            symbols=[slot["preferred_symbol"]],
            rules=[f"Executable family: {family}"],
            novelty="adaptive-local-grid",
            falsification_plan=["Reject when OOS/WF/stress/confirmation gates fail."],
            executable_family=family,
            executable_parameters=choice,
        ))
    return out
