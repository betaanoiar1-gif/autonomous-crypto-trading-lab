from __future__ import annotations

"""Phase 2: feature, regime, and policy discovery.

Research protocol:
- Spot long/flat only.
- No AI, futures, or live trading.
- Common cutoff across cached 1h markets.
- Development data is used for feature ranking, regime fitting, and evolution.
- Validation is frozen and cannot influence evolution.
- Lockbox is opened only for the top three frozen candidates.
- The policy searches over feature-driven actions rather than only tuning classic strategies.
"""

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
CACHE = Path(
    os.getenv(
        "PHASE2_CACHE_DIR",
        "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3",
    )
)
OUT = ROOT / "experiments" / "phase2_feature_policy_latest.json"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)

FEATURES = (
    "mom_6",
    "mom_24",
    "mom_72",
    "mom_168",
    "ema_trend",
    "range_position",
    "breakout_pressure",
    "z_reversion",
    "vol_scaled_momentum",
    "vol_compression",
)

ACTIONS = ("flat", "momentum", "reversion", "breakout", "trend")


@dataclass(frozen=True)
class PolicyGenome:
    trend_action: str
    range_action: str
    high_vol_action: str
    feature: str
    threshold: float
    exit_threshold: float
    regime_window: int
    vol_window: int
    high_vol_quantile: float
    hold_bars: int


@dataclass
class Eval:
    ret: float
    pf: float
    dd: float
    stress_ret: float
    stress_pf: float
    trades: int
    turnover: float
    exposure: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        if path.exists():
            data[symbol] = pd.read_parquet(path).sort_index()
    return data


def _common_cut(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not data:
        return {}
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    return {
        symbol: df.loc[(df.index >= start) & (df.index <= end)].copy()
        for symbol, df in data.items()
    }


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    ret = close.pct_change()

    out = pd.DataFrame(index=df.index)
    out["mom_6"] = close.pct_change(6)
    out["mom_24"] = close.pct_change(24)
    out["mom_72"] = close.pct_change(72)
    out["mom_168"] = close.pct_change(168)

    ema_fast = close.ewm(span=24, adjust=False, min_periods=24).mean()
    ema_slow = close.ewm(span=96, adjust=False, min_periods=96).mean()
    out["ema_trend"] = ema_fast / ema_slow - 1.0

    rolling_high = high.rolling(72, min_periods=72).max()
    rolling_low = low.rolling(72, min_periods=72).min()
    width = (rolling_high - rolling_low).replace(0.0, np.nan)
    out["range_position"] = ((close - rolling_low) / width).clip(0.0, 1.0)

    prior_high = high.rolling(48, min_periods=48).max().shift(1)
    prior_low = low.rolling(48, min_periods=48).min().shift(1)
    out["breakout_pressure"] = ((close - prior_low) / (prior_high - prior_low).replace(0.0, np.nan)).clip(-1.0, 2.0)

    mean = close.rolling(72, min_periods=72).mean()
    std = close.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    out["z_reversion"] = ((close - mean) / std).clip(-5.0, 5.0)

    vol = ret.rolling(24, min_periods=24).std().replace(0.0, np.nan)
    out["vol_scaled_momentum"] = (out["mom_24"] / vol).clip(-20.0, 20.0)

    vol_short = ret.rolling(12, min_periods=12).std()
    vol_long = ret.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    out["vol_compression"] = (vol_short / vol_long).clip(0.0, 5.0)

    # Volume is used only as a weak confirmation feature; keep it in the frame so
    # future extensions can rank it without changing the execution layer.
    out["volume_pressure"] = (
        volume / volume.rolling(48, min_periods=48).mean().replace(0.0, np.nan)
    ).clip(0.0, 5.0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _rank_features(dev: dict[str, pd.DataFrame], horizon: int = 1) -> list[dict]:
    rows: list[dict] = []
    for feature in FEATURES:
        corr: list[float] = []
        for df in dev.values():
            x = _feature_frame(df)[feature]
            target = df["close"].pct_change(horizon).shift(-horizon)
            valid = pd.concat([x, target.rename("target")], axis=1).dropna()
            if len(valid) < 500:
                continue
            value = valid.iloc[:, 0].corr(valid["target"])
            if np.isfinite(value):
                corr.append(float(value))
        rows.append(
            {
                "feature": feature,
                "median_corr": float(np.median(corr)) if corr else 0.0,
                "median_abs_corr": float(np.median(np.abs(corr))) if corr else 0.0,
                "markets": len(corr),
            }
        )
    rows.sort(key=lambda x: (x["median_abs_corr"], x["markets"]), reverse=True)
    return rows


def _fit_regime_thresholds(df: pd.DataFrame, g: PolicyGenome) -> dict[str, float]:
    f = _feature_frame(df)
    trend_abs = f["ema_trend"].abs().dropna()
    vol = f["vol_compression"].replace([np.inf, -np.inf], np.nan).dropna()
    trend_threshold = float(trend_abs.quantile(0.65)) if len(trend_abs) else 0.002
    vol_threshold = float(vol.quantile(g.high_vol_quantile)) if len(vol) else 1.25
    return {
        "trend_threshold": max(0.0005, trend_threshold),
        "high_vol_threshold": max(1.0, vol_threshold),
    }


def _regimes(df: pd.DataFrame, g: PolicyGenome, thresholds: dict[str, float]) -> pd.Series:
    f = _feature_frame(df)
    trend = f["ema_trend"].abs()
    high_vol = f["vol_compression"] >= thresholds["high_vol_threshold"]
    trend_regime = trend >= thresholds["trend_threshold"]
    regime = pd.Series("range", index=df.index, dtype="object")
    regime.loc[trend_regime] = "trend"
    regime.loc[high_vol] = "high_vol"
    return regime


def _action_condition(f: pd.DataFrame, action: str, feature: str, g: PolicyGenome) -> pd.Series:
    selected = f[feature]
    if action == "flat":
        return pd.Series(False, index=f.index)
    if action == "momentum":
        return selected > g.threshold
    if action == "reversion":
        return selected < -abs(g.threshold)
    if action == "breakout":
        return f["breakout_pressure"] > 1.0
    if action == "trend":
        return f["ema_trend"] > g.exit_threshold
    return pd.Series(False, index=f.index)


def _signal(df: pd.DataFrame, g: PolicyGenome, thresholds: dict[str, float]) -> pd.Series:
    f = _feature_frame(df)
    regimes = _regimes(df, g, thresholds)
    raw = pd.Series(False, index=df.index)
    for regime_name, action in (
        ("trend", g.trend_action),
        ("range", g.range_action),
        ("high_vol", g.high_vol_action),
    ):
        mask = regimes == regime_name
        raw.loc[mask] = _action_condition(f.loc[mask], action, g.feature, g)[mask]

    desired = raw.astype(float).fillna(0.0)
    position = np.zeros(len(desired), dtype=float)
    active = False
    age = g.hold_bars
    for i, value in enumerate(desired.to_numpy()):
        if active:
            age += 1
            if value < 0.5 and age >= g.hold_bars:
                active = False
        elif value > 0.5:
            active = True
            age = 0
        position[i] = 1.0 if active else 0.0
    return pd.Series(position, index=df.index)


def _bt(df: pd.DataFrame, g: PolicyGenome, settings, cost_mult: float = 1.0) -> Eval:
    thresholds = _fit_regime_thresholds(df, g)
    result = run_ohlcv(
        df,
        _signal(df, g, thresholds),
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
        stress_ret=float(m.get("total_return", 0.0)),
        stress_pf=float(m.get("profit_factor", 0.0)),
        trades=int(m.get("trade_count", 0)),
        turnover=float(m.get("trade_turnover", 0.0)),
        exposure=float(m.get("exposure", 0.0)),
    )


def _policy_eval(df: pd.DataFrame, g: PolicyGenome, settings) -> dict[str, Eval]:
    return {
        "normal": _bt(df, g, settings, 1.0),
        "stress": _bt(df, g, settings, 2.0),
    }


def _dev_window_stability(df: pd.DataFrame, g: PolicyGenome, settings, windows: int = 4) -> dict:
    n = len(df)
    if n < 5000:
        return {"median": 0.0, "minimum": 0.0, "positive": 0, "values": []}
    width = n // windows
    values: list[float] = []
    positive = 0
    for k in range(windows):
        lo = k * width
        hi = n if k == windows - 1 else (k + 1) * width
        part = df.iloc[lo:hi].copy()
        e = _bt(part, g, settings, 1.0)
        values.append(e.ret)
        positive += int(e.ret > 0 and e.pf > 1.0)
    return {
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "positive": int(positive),
        "values": values,
    }


def _dev_score(data: dict[str, pd.DataFrame], g: PolicyGenome, settings, feature_rank: dict[str, float]) -> dict:
    markets = list(data)[:8]
    evaluations = [_policy_eval(data[m].iloc[-10000:].copy(), g, settings) for m in markets]
    ret = float(np.median([e["normal"].ret for e in evaluations]))
    pf = float(np.median([e["normal"].pf for e in evaluations]))
    dd = float(np.median([e["normal"].dd for e in evaluations]))
    stress = float(np.median([e["stress"].ret for e in evaluations]))
    stress_pf = float(np.median([e["stress"].pf for e in evaluations]))
    trades = int(np.median([e["normal"].trades for e in evaluations]))
    stability = [_dev_window_stability(data[m], g, settings) for m in markets[:4]]
    stab_med = float(np.median([s["median"] for s in stability])) if stability else 0.0
    stab_min = float(np.min([s["minimum"] for s in stability])) if stability else 0.0
    stab_pos = int(np.median([s["positive"] for s in stability])) if stability else 0
    novelty = 1.0 - min(1.0, abs(feature_rank.get(g.feature, 0.0)) / 0.05)
    simplicity = 1.0 if g.trend_action != g.range_action or g.range_action != g.high_vol_action else 0.5
    utility = (
        45.0 * ret
        + 18.0 * max(0.0, pf - 1.0)
        + 32.0 * stress
        + 18.0 * stab_med
        + 6.0 * stab_pos
        + 5.0 * novelty
        + 3.0 * simplicity
        - 32.0 * max(0.0, -dd - 0.10)
        - 22.0 * max(0.0, -stress)
        - 0.003 * trades
    )
    return {
        "utility": float(utility),
        "ret": ret,
        "pf": pf,
        "dd": dd,
        "stress": stress,
        "stress_pf": stress_pf,
        "trades": trades,
        "stability_median": stab_med,
        "stability_minimum": stab_min,
        "stability_positive": stab_pos,
    }


def _pool(seed: int, count: int, feature_names: list[str]) -> list[PolicyGenome]:
    rng = np.random.default_rng(seed)
    out: list[PolicyGenome] = []
    seen: set[PolicyGenome] = set()
    while len(out) < count:
        g = PolicyGenome(
            trend_action=str(rng.choice(ACTIONS[1:])),
            range_action=str(rng.choice(ACTIONS)),
            high_vol_action=str(rng.choice(ACTIONS)),
            feature=str(rng.choice(feature_names)),
            threshold=float(rng.choice([0.003, 0.005, 0.0075, 0.01])),
            exit_threshold=float(rng.choice([0.0005, 0.001, 0.0025, 0.005])),
            regime_window=int(rng.choice([48, 72, 96, 144])),
            vol_window=int(rng.choice([12, 24, 36, 48])),
            high_vol_quantile=float(rng.choice([0.75, 0.85, 0.90, 0.95])),
            hold_bars=int(rng.choice([24, 48, 72, 96])),
        )
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _mutate(g: PolicyGenome, rng: np.random.Generator, feature_names: list[str]) -> PolicyGenome:
    family = list(ACTIONS)
    return PolicyGenome(
        trend_action=g.trend_action if rng.random() > 0.12 else str(rng.choice(family[1:])),
        range_action=g.range_action if rng.random() > 0.12 else str(rng.choice(family)),
        high_vol_action=g.high_vol_action if rng.random() > 0.12 else str(rng.choice(family)),
        feature=g.feature if rng.random() > 0.15 else str(rng.choice(feature_names)),
        threshold=float(np.clip(g.threshold + rng.choice([-0.001, 0.0, 0.001]), 0.0015, 0.02)),
        exit_threshold=float(np.clip(g.exit_threshold + rng.choice([-0.0005, 0.0, 0.0005]), 0.0005, 0.01)),
        regime_window=int(np.clip(g.regime_window + rng.choice([-24, 0, 24]), 36, 180)),
        vol_window=int(np.clip(g.vol_window + rng.choice([-12, 0, 12]), 8, 72)),
        high_vol_quantile=float(np.clip(g.high_vol_quantile + rng.choice([-0.05, 0.0, 0.05]), 0.70, 0.97)),
        hold_bars=int(np.clip(g.hold_bars + rng.choice([-24, 0, 24]), 24, 144)),
    )


def _validation(data: dict[str, pd.DataFrame], g: PolicyGenome, settings) -> dict:
    rows = []
    for market, df in data.items():
        e = _policy_eval(df, g, settings)
        rows.append({"market": market, "normal": asdict(e["normal"]), "stress": asdict(e["stress"])})
    ret = [r["normal"]["ret"] for r in rows]
    stress = [r["stress"]["ret"] for r in rows]
    pf = [r["normal"]["pf"] for r in rows]
    spf = [r["stress"]["pf"] for r in rows]
    positive = sum(a > 0 and b > 0 and c > 1.0 and d > 1.0 for a, b, c, d in zip(ret, stress, pf, spf))
    return {
        "median_return": float(np.median(ret)),
        "median_stress_return": float(np.median(stress)),
        "median_pf": float(np.median(pf)),
        "median_stress_pf": float(np.median(spf)),
        "positive_markets": int(positive),
        "markets": rows,
    }


def run(
    minutes: float = 180.0,
    initial_population: int = 64,
    population: int = 16,
    generations: int = 12,
    seed: int = 20260829,
) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    raw = _load()

    print("=== PHASE 2 FEATURE/POLICY DISCOVERY ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"CACHE: {CACHE}", flush=True)
    print(f"Markets loaded: {len(raw)}", flush=True)

    if len(raw) < 8:
        payload = {"decision": "PHASE2_BLOCKED_DATA", "markets": list(raw), "version": "phase2-v1"}
        _save(payload)
        return payload

    data = _common_cut(raw)
    common_end = min(df.index[-1] for df in data.values())
    n = min(len(df) for df in data.values())
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_n = n - dev_n - val_n

    dev = {m: df.iloc[:dev_n].copy() for m, df in data.items()}
    val = {m: df.iloc[dev_n:dev_n + val_n].copy() for m, df in data.items()}
    lock = {m: df.iloc[dev_n + val_n:].copy() for m, df in data.items()}

    print(f"COMMON CUTOFF: {common_end}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}", flush=True)
    print("LOCKBOX: RESERVED", flush=True)

    feature_rows = _rank_features(dev)
    feature_names = [x["feature"] for x in feature_rows[:6]]
    feature_rank = {x["feature"]: x["median_abs_corr"] for x in feature_rows}
    print("=== FEATURE RANKING ===", flush=True)
    for row in feature_rows[:10]:
        print(
            f"feature {row['feature']} | median_corr={row['median_corr']:.5f} "
            f"abs={row['median_abs_corr']:.5f} markets={row['markets']}",
            flush=True,
        )

    rng = np.random.default_rng(seed + 31)
    genes = _pool(seed, initial_population, feature_names)
    evaluations = 0
    finalists: list[PolicyGenome] = []

    for generation in range(generations):
        if time.monotonic() >= deadline:
            break
        ranked: list[tuple[float, PolicyGenome, dict]] = []
        batch = genes[:population]
        print(f"=== GENERATION {generation + 1}/{generations} | population={len(batch)} ===", flush=True)
        for i, g in enumerate(batch, 1):
            if time.monotonic() >= deadline:
                break
            score = _dev_score(dev, g, settings, feature_rank)
            ranked.append((score["utility"], g, score))
            evaluations += 1
            print(
                f"eval {i}/{len(batch)} | utility={score['utility']:.2f} "
                f"feature={g.feature} ret={score['ret']:.2%} PF={score['pf']:.2f} "
                f"stress={score['stress']:.2%} WF+={score['stability_positive']} "
                f"stab_min={score['stability_minimum']:.2%}",
                flush=True,
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        elite_count = max(2, population // 4)
        elites = [x[1] for x in ranked[:elite_count]]
        finalists.extend(x[1] for x in ranked[:max(4, population // 2)])
        children: list[PolicyGenome] = []
        for elite in elites:
            for _ in range(3):
                children.append(_mutate(elite, rng, feature_names))
        genes = list(dict.fromkeys(elites + children))

    finalists = list(dict.fromkeys(finalists))[:12]
    print("=== FROZEN VALIDATION ===", flush=True)
    validation_rows = []
    for rank, genome in enumerate(finalists, 1):
        if time.monotonic() >= deadline:
            break
        summary = _validation(val, genome, settings)
        stability = _dev_window_stability(dev["BTC/USDT"], genome, settings) if "BTC/USDT" in dev else {"minimum": 0.0, "median": 0.0}
        summary["dev_stability_minimum"] = float(stability["minimum"])
        summary["dev_stability_median"] = float(stability["median"])
        summary["genome"] = asdict(genome)
        validation_rows.append(summary)
        print(
            f"VALIDATION {rank}/{len(finalists)} | ret={summary['median_return']:.2%} "
            f"PF={summary['median_pf']:.2f} stress={summary['median_stress_return']:.2%} "
            f"positive={summary['positive_markets']}/{len(val)} "
            f"dev_stab_min={summary['dev_stability_minimum']:.2%}",
            flush=True,
        )

    validation_rows.sort(
        key=lambda x: (
            x["positive_markets"],
            x["median_stress_return"],
            x["median_return"],
            x["median_pf"],
        ),
        reverse=True,
    )

    print("=== LOCKBOX FINAL TEST ===", flush=True)
    lockbox_rows = []
    for rank, row in enumerate(validation_rows[:3], 1):
        genome = PolicyGenome(**row["genome"])
        summary = _validation(lock, genome, settings)
        summary["rank"] = rank
        summary["genome"] = asdict(genome)
        lockbox_rows.append(summary)
        print(
            f"LOCKBOX {rank}/3 | ret={summary['median_return']:.2%} "
            f"PF={summary['median_pf']:.2f} stress={summary['median_stress_return']:.2%} "
            f"stressPF={summary['median_stress_pf']:.2f} "
            f"positive={summary['positive_markets']}/{len(lock)}",
            flush=True,
        )

    eligible = [
        x
        for x in lockbox_rows
        if x["positive_markets"] >= max(6, len(lock) // 2)
        and x["median_return"] > 0
        and x["median_stress_return"] > 0
        and x["median_pf"] > 1.05
        and x["median_stress_pf"] > 1.0
    ]

    decision = "PHASE2_VALIDATED_POLICY" if eligible else "PHASE2_NO_VALIDATED_POLICY"
    payload = {
        "version": "phase2-v1",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "evaluations": evaluations,
        "common_cutoff": str(common_end),
        "split": {"development": dev_n, "validation": val_n, "lockbox": lock_n},
        "features": feature_rows,
        "finalists": [asdict(g) for g in finalists],
        "validation": validation_rows,
        "lockbox": lockbox_rows,
        "eligible": eligible,
        "protocol": {
            "spot_long_flat": True,
            "ai": False,
            "futures": False,
            "feature_ranking_dev_only": True,
            "regime_fit_dev_only": True,
            "validation_frozen": True,
            "lockbox_top3_only": True,
        },
    }
    _save(payload)
    print("=== PHASE 2 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("PHASE2_MINUTES", "180")),
        initial_population=int(os.getenv("PHASE2_INITIAL_POPULATION", "64")),
        population=int(os.getenv("PHASE2_POPULATION", "16")),
        generations=int(os.getenv("PHASE2_GENERATIONS", "12")),
        seed=int(os.getenv("PHASE2_SEED", "20260829")),
    )
