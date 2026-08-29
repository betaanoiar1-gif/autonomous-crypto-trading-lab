from __future__ import annotations

"""Phase 1 V3: anti-overfit discovery with nested selection and lockbox."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("DISCOVERY_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase1_discovery_v3_latest.json"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
SCREEN_MARKETS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
    "XRP/USDT", "ADA/USDT",
)


@dataclass(frozen=True)
class Genome:
    family: str
    lookback: int
    fast: int
    slow: int
    threshold: float
    exit_threshold: float
    vol_window: int
    vol_cap: float
    hold_bars: int


@dataclass(frozen=True)
class Eval:
    ret: float
    pf: float
    dd: float
    trades: int
    turnover: float
    stress_ret: float
    stress_pf: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path).sort_index()
    return out


def _common_cut(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    result = {}
    for market, df in data.items():
        result[market] = df.loc[(df.index >= start) & (df.index <= end)].copy()
    return result


def _signal(df: pd.DataFrame, g: Genome) -> pd.Series:
    close = df["close"].astype(float)
    returns = close.pct_change()
    momentum = close.pct_change(g.lookback)
    fast = close.ewm(span=g.fast, adjust=False, min_periods=g.fast).mean()
    slow = close.ewm(span=g.slow, adjust=False, min_periods=g.slow).mean()
    vol = returns.rolling(g.vol_window, min_periods=g.vol_window).std()
    mean = close.rolling(g.lookback, min_periods=g.lookback).mean()
    std = close.rolling(g.lookback, min_periods=g.lookback).std().replace(0.0, np.nan)
    z = (close - mean) / std
    prior_high = close.rolling(g.lookback, min_periods=g.lookback).max().shift(1)

    if g.family == "momentum":
        raw = (momentum > g.threshold) & (fast > slow) & (vol < g.vol_cap)
    elif g.family == "breakout":
        raw = (close > prior_high) & (vol < g.vol_cap)
    elif g.family == "mean_reversion":
        raw = (z < -abs(g.threshold)) & (vol < g.vol_cap)
    elif g.family == "trend_pullback":
        raw = (fast > slow) & (momentum > -abs(g.threshold)) & (momentum < g.exit_threshold)
    elif g.family == "vol_breakout":
        raw = (close > prior_high) & (vol > max(0.001, g.vol_cap * 0.35))
    else:
        raw = (fast > slow) & (momentum > g.threshold) & (vol < g.vol_cap)

    raw = raw.astype(float).fillna(0.0).clip(0.0, 1.0)
    positions = np.zeros(len(raw), dtype=float)
    active = False
    age = g.hold_bars

    for i, value in enumerate(raw.to_numpy()):
        if active:
            age += 1
            if value < 0.5 and age >= g.hold_bars:
                active = False
        elif value > 0.5:
            active = True
            age = 0
        positions[i] = 1.0 if active else 0.0

    return pd.Series(positions, index=df.index)


def _bt(df: pd.DataFrame, g: Genome, settings, cost_mult: float = 1.0) -> Eval:
    result = run_ohlcv(
        df,
        _signal(df, g),
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    m = result.metrics
    return Eval(
        ret=float(m.get("total_return", 0.0)),
        pf=float(m.get("profit_factor", 0.0)),
        dd=float(m.get("max_drawdown", 0.0)),
        trades=int(m.get("trade_count", 0)),
        turnover=float(m.get("trade_turnover", 0.0)),
        stress_ret=float(m.get("total_return", 0.0)),
        stress_pf=float(m.get("profit_factor", 0.0)),
    )


def _evaluate(df: pd.DataFrame, g: Genome, settings) -> dict:
    normal = _bt(df, g, settings, 1.0)
    stress = _bt(df, g, settings, 2.0)
    return {"normal": normal, "stress": stress}


def _wf(df: pd.DataFrame, g: Genome, settings, folds: int = 5) -> dict:
    n = len(df)
    if n < 5000:
        return {"median_return": 0.0, "positive": 0, "dispersion": 0.0, "folds": []}
    fold = n // folds
    values = []
    rows = []
    for k in range(folds):
        lo = k * fold
        hi = (k + 1) * fold if k < folds - 1 else n
        test = df.iloc[lo:hi].copy()
        if len(test) < 500:
            continue
        e = _evaluate(test, g, settings)
        values.append(e["normal"].ret)
        rows.append({
            "fold": k + 1,
            "return": e["normal"].ret,
            "pf": e["normal"].pf,
            "stress_return": e["stress"].ret,
        })
    if not values:
        return {"median_return": 0.0, "positive": 0, "dispersion": 0.0, "folds": []}
    return {
        "median_return": float(np.median(values)),
        "positive": int(sum(v > 0 for v in values)),
        "dispersion": float(np.std(values)),
        "folds": rows,
    }


def _summary(evals: list[dict]) -> dict:
    normals = [x["normal"] for x in evals]
    stresses = [x["stress"] for x in evals]
    return {
        "ret": float(np.median([x.ret for x in normals])),
        "pf": float(np.median([x.pf for x in normals])),
        "dd": float(np.median([x.dd for x in normals])),
        "trades": int(np.median([x.trades for x in normals])),
        "turnover": float(np.median([x.turnover for x in normals])),
        "stress_ret": float(np.median([x.ret for x in stresses])),
        "stress_pf": float(np.median([x.pf for x in stresses])),
        "positive_markets": int(sum(x.ret > 0 and x.pf > 1.0 for x in normals)),
        "stress_positive_markets": int(sum(x.ret > 0 and x.pf > 1.0 for x in stresses)),
    }


def _utility(s: dict, wf_median: float, wf_positive: int, wf_dispersion: float) -> float:
    return (
        55.0 * s["ret"]
        + 18.0 * max(0.0, s["pf"] - 1.0)
        + 30.0 * max(0.0, s["stress_ret"])
        + 12.0 * max(0.0, s["stress_pf"] - 1.0)
        + 45.0 * wf_median
        + 9.0 * wf_positive
        - 35.0 * max(0.0, -s["dd"] - 0.10)
        - 25.0 * max(0.0, -s["stress_ret"])
        - 15.0 * wf_dispersion
        - 0.002 * s["turnover"]
    )


def _pool(seed: int, count: int) -> list[Genome]:
    rng = np.random.default_rng(seed)
    families = ["momentum", "breakout", "mean_reversion", "trend_pullback", "vol_breakout", "ma_cross"]
    lookbacks = [24, 48, 72, 120, 168, 240, 336]
    fasts = [6, 8, 12, 18, 24, 36, 48]
    slows = [40, 60, 90, 120, 180, 240, 360]
    thresholds = [0.0025, 0.005, 0.0075, 0.01, 0.015]
    exits = [0.0015, 0.003, 0.005, 0.0075, 0.01]
    vols = [12, 24, 36, 48, 72]
    caps = [0.012, 0.018, 0.025, 0.035, 0.05]
    holds = [24, 48, 72, 96, 120]

    out: list[Genome] = []
    seen: set[Genome] = set()
    while len(out) < count:
        fast = int(rng.choice(fasts))
        slow = int(rng.choice(slows))
        if fast >= slow:
            continue
        g = Genome(
            family=str(rng.choice(families)),
            lookback=int(rng.choice(lookbacks)),
            fast=fast,
            slow=slow,
            threshold=float(rng.choice(thresholds)),
            exit_threshold=float(rng.choice(exits)),
            vol_window=int(rng.choice(vols)),
            vol_cap=float(rng.choice(caps)),
            hold_bars=int(rng.choice(holds)),
        )
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _mutate(g: Genome, rng: np.random.Generator) -> Genome:
    family = g.family if rng.random() > 0.10 else str(rng.choice(["momentum", "breakout", "mean_reversion", "trend_pullback", "vol_breakout", "ma_cross"]))
    lookback = max(18, min(400, g.lookback + int(rng.choice([-48, -24, 0, 24, 48]))))
    fast = max(6, min(72, g.fast + int(rng.choice([-6, -3, 0, 3, 6]))))
    slow = max(40, min(420, g.slow + int(rng.choice([-30, -15, 0, 15, 30]))))
    if fast >= slow:
        fast = max(6, slow - 6)
    return Genome(
        family=family,
        lookback=lookback,
        fast=fast,
        slow=slow,
        threshold=float(np.clip(g.threshold + rng.choice([-0.0025, 0.0, 0.0025]), 0.0015, 0.025)),
        exit_threshold=float(np.clip(g.exit_threshold + rng.choice([-0.001, 0.0, 0.001]), 0.0005, 0.012)),
        vol_window=max(8, min(96, g.vol_window + int(rng.choice([-12, 0, 12])))),
        vol_cap=float(np.clip(g.vol_cap + rng.choice([-0.005, 0.0, 0.005]), 0.01, 0.06)),
        hold_bars=max(12, min(144, g.hold_bars + int(rng.choice([-24, 0, 24])))),
    )


def _neighborhood(g: Genome, settings, df_map: dict[str, pd.DataFrame]) -> dict:
    variants = [g]
    for dlb in (-24, 24):
        variants.append(Genome(g.family, max(18, min(400, g.lookback + dlb)), g.fast, g.slow, g.threshold, g.exit_threshold, g.vol_window, g.vol_cap, g.hold_bars))
    for df in (-3, 3):
        nf = max(6, min(72, g.fast + df))
        ns = max(40, min(420, g.slow + df))
        if nf < ns:
            variants.append(Genome(g.family, g.lookback, nf, ns, g.threshold, g.exit_threshold, g.vol_window, g.vol_cap, g.hold_bars))

    values = []
    for variant in variants:
        per = [_evaluate(df, variant, settings) for df in df_map.values()]
        sm = _summary(per)
        values.append(sm["ret"])
    return {
        "median_return": float(np.median(values)),
        "min_return": float(np.min(values)),
        "dispersion": float(np.std(values)),
        "variants": len(values),
    }


def run(minutes: float = 180.0, initial_population: int = 64, population: int = 16, generations: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    raw = _load()

    print("=== PHASE 1 DISCOVERY V3 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"Markets loaded: {len(raw)}", flush=True)

    if len(raw) < 8:
        payload = {"decision": "DISCOVERY_BLOCKED_DATA", "markets": list(raw), "version": "v3"}
        _save(payload)
        return payload

    data = _common_cut(raw)
    n = min(len(df) for df in data.values())
    common_end = min(df.index[-1] for df in data.values())

    dev_n = int(n * 0.60)
    val_n = int(n * 0.25)
    lock_n = n - dev_n - val_n

    dev = {m: d.iloc[:dev_n].copy() for m, d in data.items()}
    val = {m: d.iloc[dev_n:dev_n + val_n].copy() for m, d in data.items()}
    lock = {m: d.iloc[dev_n + val_n:].copy() for m, d in data.items()}

    print(f"COMMON CUTOFF: {common_end}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}", flush=True)
    print("LOCKBOX: RESERVED", flush=True)

    rng = np.random.default_rng(seed + 31)
    genes = _pool(seed, initial_population)
    finalist_pool: list[Genome] = []
    evaluations = 0
    best_validation = -float("inf")

    for gen in range(generations):
        if time.monotonic() >= deadline:
            break

        print(f"=== GENERATION {gen + 1}/{generations} ===", flush=True)
        ranked: list[tuple[float, Genome, dict]] = []

        for idx, genome in enumerate(genes[:population], 1):
            if time.monotonic() >= deadline:
                break

            dev_evals = [_evaluate(dev[m].iloc[-12000:].copy(), genome, settings) for m in SCREEN_MARKETS if m in dev]
            dev_summary = _summary(dev_evals)
            wf_rows = [_wf(dev[m], genome, settings) for m in SCREEN_MARKETS[:4] if m in dev]
            wf_med = float(np.median([x["median_return"] for x in wf_rows])) if wf_rows else 0.0
            wf_pos = int(np.median([x["positive"] for x in wf_rows])) if wf_rows else 0
            wf_disp = float(np.median([x["dispersion"] for x in wf_rows])) if wf_rows else 0.0
            dev_util = _utility(dev_summary, wf_med, wf_pos, wf_disp)

            val_candidates = []
            if dev_util > -100.0:
                val_evals = [_evaluate(val[m], genome, settings) for m in SYMBOLS if m in val]
                val_summary = _summary(val_evals)
                val_wf = [_wf(val[m], genome, settings) for m in SCREEN_MARKETS[:4] if m in val]
                val_wf_med = float(np.median([x["median_return"] for x in val_wf])) if val_wf else 0.0
                val_wf_pos = int(np.median([x["positive"] for x in val_wf])) if val_wf else 0
                val_wf_disp = float(np.median([x["dispersion"] for x in val_wf])) if val_wf else 0.0
                val_util = _utility(val_summary, val_wf_med, val_wf_pos, val_wf_disp)
                selection_util = 0.35 * dev_util + 0.65 * val_util
                val_candidates = [{"summary": val_summary, "wf_median": val_wf_med, "wf_positive": val_wf_pos, "wf_dispersion": val_wf_disp, "utility": val_util}]
                best_validation = max(best_validation, val_util)
            else:
                val_util = -999.0
                selection_util = dev_util

            ranked.append((selection_util, genome, {"dev": dev_summary, "dev_utility": dev_util, "validation": val_candidates[0] if val_candidates else None, "validation_utility": val_util}))
            evaluations += 1

            print(
                f"eval {idx}/{min(population, len(genes))} | select={selection_util:.2f} "
                f"dev_ret={dev_summary['ret']:.2%} devPF={dev_summary['pf']:.2f} "
                f"val_util={val_util:.2f}",
                flush=True,
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in ranked[:max(2, population // 4)]]
        finalist_pool.extend(x[1] for x in ranked[:max(4, population // 2)])

        children = []
        for elite in elites:
            for _ in range(3):
                children.append(_mutate(elite, rng))
        genes = list(dict.fromkeys(elites + children))

    finalists = list(dict.fromkeys(finalist_pool))[:12]

    print("=== FINAL VALIDATION + STABILITY ===", flush=True)
    validation_rows = []
    for rank, genome in enumerate(finalists, 1):
        if time.monotonic() >= deadline:
            break
        per = [_evaluate(val[m], genome, settings) for m in SYMBOLS if m in val]
        summary = _summary(per)
        wf_rows = [_wf(val[m], genome, settings) for m in SCREEN_MARKETS[:4] if m in val]
        wf_med = float(np.median([x["median_return"] for x in wf_rows])) if wf_rows else 0.0
        wf_pos = int(np.median([x["positive"] for x in wf_rows])) if wf_rows else 0
        wf_disp = float(np.median([x["dispersion"] for x in wf_rows])) if wf_rows else 0.0
        util = _utility(summary, wf_med, wf_pos, wf_disp)
        stability = _neighborhood(genome, settings, {m: val[m] for m in SCREEN_MARKETS if m in val})
        row = {
            "rank": rank,
            "genome": asdict(genome),
            "utility": util,
            "summary": summary,
            "wf_median": wf_med,
            "wf_positive": wf_pos,
            "wf_dispersion": wf_disp,
            "stability": stability,
        }
        validation_rows.append(row)
        print(
            f"FINALIST {rank}/{len(finalists)} | util={util:.2f} "
            f"ret={summary['ret']:.2%} PF={summary['pf']:.2f} "
            f"stress={summary['stress_ret']:.2%} positive={summary['positive_markets']}/{len(per)} "
            f"stable_med={stability['median_return']:.2%} stable_min={stability['min_return']:.2%}",
            flush=True,
        )

    validation_rows.sort(key=lambda x: (x["stability"]["median_return"], x["utility"]), reverse=True)

    print("=== LOCKBOX TOP 3 ===", flush=True)
    lockbox_rows = []
    for rank, row in enumerate(validation_rows[:3], 1):
        genome = Genome(**row["genome"])
        per = [_evaluate(lock[m], genome, settings) for m in SYMBOLS if m in lock]
        summary = _summary(per)
        lock_row = {
            "rank": rank,
            "genome": row["genome"],
            "median_return": summary["ret"],
            "median_pf": summary["pf"],
            "median_dd": summary["dd"],
            "median_stress_return": summary["stress_ret"],
            "median_stress_pf": summary["stress_pf"],
            "positive_markets": summary["positive_markets"],
            "stress_positive_markets": summary["stress_positive_markets"],
            "markets": [{"market": m, "normal": asdict(_evaluate(lock[m], genome, settings)["normal"]), "stress": asdict(_evaluate(lock[m], genome, settings)["stress"])} for m in SYMBOLS if m in lock],
        }
        lockbox_rows.append(lock_row)
        print(
            f"LOCKBOX {rank}/3 | ret={lock_row['median_return']:.2%} "
            f"PF={lock_row['median_pf']:.2f} stress={lock_row['median_stress_return']:.2%} "
            f"positive={lock_row['positive_markets']}/{len(per)}",
            flush=True,
        )

    eligible = [
        x for x in lockbox_rows
        if x["positive_markets"] >= 6
        and x["stress_positive_markets"] >= 6
        and x["median_return"] > 0.0
        and x["median_stress_return"] > 0.0
        and x["median_pf"] > 1.05
        and x["median_stress_pf"] > 1.0
    ]

    decision = "VALIDATED_STRATEGY_READY" if eligible else "NO_VALIDATED_STRATEGY"
    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (time.monotonic() - (deadline - minutes * 60.0)) / 60.0,
        "version": "v3",
        "decision": decision,
        "evaluations": evaluations,
        "best_validation_utility": best_validation,
        "common_cutoff": str(common_end),
        "split": {"development": dev_n, "validation": val_n, "lockbox": lock_n},
        "finalists": [asdict(g) for g in finalists],
        "validation": validation_rows,
        "lockbox": lockbox_rows,
        "eligible": eligible,
        "protocol": {
            "spot_long_flat": True,
            "ai": False,
            "futures": False,
            "common_cutoff": True,
            "non_overlapping_wf": True,
            "nested_selection": True,
            "neighborhood_stability": True,
            "lockbox_reserved": True,
            "lockbox_candidates_max": 3,
        },
    }
    _save(payload)
    print("=== PHASE 1 V3 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("DISCOVERY_MINUTES", "180")),
        initial_population=int(os.getenv("DISCOVERY_INITIAL_POPULATION", "64")),
        population=int(os.getenv("DISCOVERY_POPULATION", "16")),
        generations=int(os.getenv("DISCOVERY_GENERATIONS", "12")),
        seed=int(os.getenv("DISCOVERY_SEED", "20260829")),
    )
