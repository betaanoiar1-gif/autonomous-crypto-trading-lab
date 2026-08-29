from __future__ import annotations

"""Phase 2 V2: feature + fixed-regime + policy discovery.

The research split is chronological and explicit. All feature ranking and
regime-threshold fitting are done on development data only; the fitted
thresholds are then reused unchanged on validation and lockbox.
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
CACHE = Path(os.getenv("PHASE2_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase2_feature_policy_v2_latest.json"
SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT")
FEATURES = ("mom_6", "mom_24", "mom_72", "mom_168", "ema_trend", "z_reversion", "breakout_pressure", "range_position", "vol_scaled_momentum", "vol_compression", "volume_pressure")
ACTIONS = ("flat", "momentum", "reversion", "breakout", "trend")

@dataclass(frozen=True)
class PolicyGenome:
    trend_action: str
    range_action: str
    high_vol_action: str
    feature: str
    threshold: float
    exit_threshold: float
    vol_window: int
    high_vol_quantile: float
    hold_bars: int

@dataclass
class Eval:
    ret: float
    pf: float
    dd: float
    trades: int
    turnover: float
    exposure: float


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
    if not data:
        return {}
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    return {k: v.loc[(v.index >= start) & (v.index <= end)].copy() for k, v in data.items()}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    volu = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    x = pd.DataFrame(index=df.index)
    x["mom_6"] = c.pct_change(6)
    x["mom_24"] = c.pct_change(24)
    x["mom_72"] = c.pct_change(72)
    x["mom_168"] = c.pct_change(168)
    fast = c.ewm(span=24, adjust=False, min_periods=24).mean()
    slow = c.ewm(span=96, adjust=False, min_periods=96).mean()
    x["ema_trend"] = fast / slow - 1.0
    mean = c.rolling(72, min_periods=72).mean()
    std = c.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    x["z_reversion"] = ((c - mean) / std).clip(-6.0, 6.0)
    ph = h.rolling(48, min_periods=48).max().shift(1)
    pl = l.rolling(48, min_periods=48).min().shift(1)
    x["breakout_pressure"] = ((c - pl) / (ph - pl).replace(0.0, np.nan)).clip(-1.0, 2.0)
    rh = h.rolling(72, min_periods=72).max()
    rl = l.rolling(72, min_periods=72).min()
    x["range_position"] = ((c - rl) / (rh - rl).replace(0.0, np.nan)).clip(0.0, 1.0)
    rv24 = r.rolling(24, min_periods=24).std().replace(0.0, np.nan)
    x["vol_scaled_momentum"] = (x["mom_24"] / rv24).clip(-20.0, 20.0)
    rv12 = r.rolling(12, min_periods=12).std()
    rv72 = r.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    x["vol_compression"] = (rv12 / rv72).clip(0.0, 5.0)
    x["volume_pressure"] = (volu / volu.rolling(48, min_periods=48).mean().replace(0.0, np.nan)).clip(0.0, 5.0)
    return x.replace([np.inf, -np.inf], np.nan)


def _rank_features(dev: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for name in FEATURES:
        vals: list[float] = []
        for df in dev.values():
            x = _features(df)[name]
            y = df["close"].pct_change().shift(-1)
            z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
            if len(z) >= 1000:
                c = z["x"].corr(z["y"])
                if np.isfinite(c):
                    vals.append(abs(float(c)))
        rows.append({"feature": name, "median_abs_corr": float(np.median(vals)) if vals else 0.0, "markets": len(vals)})
    return sorted(rows, key=lambda r: (r["median_abs_corr"], r["markets"]), reverse=True)


def _fit_regime_model(dev: dict[str, pd.DataFrame], g: PolicyGenome) -> dict[str, float]:
    trends: list[float] = []
    vols: list[float] = []
    for df in dev.values():
        f = _features(df)
        trends.extend(f["ema_trend"].abs().dropna().tail(20000).tolist())
        vols.extend(f["vol_compression"].dropna().tail(20000).tolist())
    trend_cut = float(np.quantile(trends, 0.65)) if trends else 0.002
    vol_cut = float(np.quantile(vols, g.high_vol_quantile)) if vols else 1.25
    return {"trend_cut": max(0.0005, trend_cut), "vol_cut": max(1.0, vol_cut)}


def _regime(df: pd.DataFrame, model: dict[str, float]) -> pd.Series:
    f = _features(df)
    trend = f["ema_trend"].abs()
    high = f["vol_compression"] >= model["vol_cut"]
    out = pd.Series("range", index=df.index, dtype="object")
    out.loc[trend >= model["trend_cut"]] = "trend"
    out.loc[high] = "high_vol"
    return out


def _condition(f: pd.DataFrame, action: str, g: PolicyGenome) -> pd.Series:
    s = f[g.feature]
    if action == "flat":
        return pd.Series(False, index=f.index)
    if action == "momentum":
        return s > g.threshold
    if action == "reversion":
        return s < -abs(g.threshold)
    if action == "breakout":
        return f["breakout_pressure"] > 1.0
    if action == "trend":
        return f["ema_trend"] > g.exit_threshold
    return pd.Series(False, index=f.index)


def _signal(df: pd.DataFrame, g: PolicyGenome, model: dict[str, float]) -> pd.Series:
    f = _features(df)
    regimes = _regime(df, model)
    desired = pd.Series(False, index=df.index)
    mapping = {"trend": g.trend_action, "range": g.range_action, "high_vol": g.high_vol_action}
    for name, action in mapping.items():
        mask = regimes == name
        if mask.any():
            desired.loc[mask] = _condition(f.loc[mask], action, g)[mask]
    pos = np.zeros(len(desired), dtype=float)
    active = False
    age = g.hold_bars
    for i, v in enumerate(desired.fillna(False).to_numpy()):
        if active:
            age += 1
            if not v and age >= g.hold_bars:
                active = False
        elif v:
            active = True
            age = 0
        pos[i] = 1.0 if active else 0.0
    return pd.Series(pos, index=df.index)


def _eval(df: pd.DataFrame, g: PolicyGenome, model: dict[str, float], settings, cost_mult: float = 1.0) -> Eval:
    result = run_ohlcv(df, _signal(df, g, model), settings.capital.initial_usd, settings.execution.commission_bps * cost_mult, settings.execution.slippage_bps * cost_mult, market_type="spot", leverage=1.0, funding_rates=None)
    m = result.metrics
    return Eval(float(m.get("total_return", 0.0)), float(m.get("profit_factor", 0.0)), float(m.get("max_drawdown", 0.0)), int(m.get("trade_count", 0)), float(m.get("trade_turnover", 0.0)), float(m.get("exposure", 0.0)))


def _stable_dev(dev: dict[str, pd.DataFrame], g: PolicyGenome, model_map: dict[str, dict[str, float]], settings) -> dict:
    market_evals = [_eval(dev[m].iloc[-12000:], g, model_map[m], settings, 1.0) for m in dev]
    stress_evals = [_eval(dev[m].iloc[-12000:], g, model_map[m], settings, 2.0) for m in dev]
    med_ret = float(np.median([e.ret for e in market_evals]))
    med_pf = float(np.median([e.pf for e in market_evals]))
    med_dd = float(np.median([e.dd for e in market_evals]))
    med_stress = float(np.median([e.ret for e in stress_evals]))
    med_spf = float(np.median([e.pf for e in stress_evals]))
    breadth = sum(e.ret > 0 and e.pf > 1.0 for e in market_evals)
    stress_breadth = sum(e.ret > 0 and e.pf > 1.0 for e in stress_evals)
    trades = float(np.median([e.trades for e in market_evals]))
    # Temporal stability on four non-overlapping DEV windows.
    folds: list[float] = []
    positives = 0
    for df in list(dev.values())[:4]:
        n = len(df)
        w = n // 4
        for k in range(4):
            part = df.iloc[k * w:(k + 1) * w if k < 3 else n]
            if len(part) < 800:
                continue
            e = _eval(part, g, model_map[next(m for m in dev if dev[m] is df)], settings, 1.0)
            folds.append(e.ret)
            positives += int(e.ret > 0 and e.pf > 1.0)
    wf_med = float(np.median(folds)) if folds else 0.0
    disp = float(np.median([abs(e.ret - med_ret) for e in market_evals]))
    utility = (55 * med_ret + 20 * max(0, med_pf - 1) + 30 * med_stress + 14 * wf_med + 5 * positives + 4 * breadth + 2 * stress_breadth - 25 * max(0, -med_dd - 0.12) - 20 * max(0, -med_stress) - 30 * disp - 0.003 * trades)
    return {"utility": float(utility), "ret": med_ret, "pf": med_pf, "dd": med_dd, "stress": med_stress, "stress_pf": med_spf, "breadth": int(breadth), "stress_breadth": int(stress_breadth), "wf_med": wf_med, "wf_positive": int(positives), "dispersion": disp, "trades": int(trades)}


def _pool(seed: int, count: int, features: list[str]) -> list[PolicyGenome]:
    rng = np.random.default_rng(seed)
    out: list[PolicyGenome] = []
    seen: set[PolicyGenome] = set()
    while len(out) < count:
        g = PolicyGenome(str(rng.choice(ACTIONS[1:])), str(rng.choice(ACTIONS)), str(rng.choice(ACTIONS)), str(rng.choice(features)), float(rng.choice([0.003, 0.005, 0.0075, 0.01])), float(rng.choice([0.0005, 0.001, 0.0025])), int(rng.choice([12, 24, 36, 48])), float(rng.choice([0.75, 0.85, 0.90, 0.95])), int(rng.choice([24, 48, 72, 96])))
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _mutate(g: PolicyGenome, rng: np.random.Generator, features: list[str]) -> PolicyGenome:
    def choose(cur: str, values: tuple[str, ...], p: float) -> str:
        return cur if rng.random() > p else str(rng.choice(values))
    return PolicyGenome(
        choose(g.trend_action, ACTIONS[1:], 0.12),
        choose(g.range_action, ACTIONS, 0.12),
        choose(g.high_vol_action, ACTIONS, 0.12),
        choose(g.feature, tuple(features), 0.15),
        float(np.clip(g.threshold + rng.choice([-0.001, 0.0, 0.001]), 0.0015, 0.02)),
        float(np.clip(g.exit_threshold + rng.choice([-0.0005, 0.0, 0.0005]), 0.0005, 0.01)),
        int(np.clip(g.vol_window + rng.choice([-12, 0, 12]), 8, 72)),
        float(np.clip(g.high_vol_quantile + rng.choice([-0.05, 0.0, 0.05]), 0.70, 0.97)),
        int(np.clip(g.hold_bars + rng.choice([-24, 0, 24]), 24, 144)),
    )


def _frozen_eval(data: dict[str, pd.DataFrame], g: PolicyGenome, model_map: dict[str, dict[str, float]], settings) -> dict:
    normal = [_eval(df, g, model_map[m], settings, 1.0) for m, df in data.items()]
    stress = [_eval(df, g, model_map[m], settings, 2.0) for m, df in data.items()]
    ret = [e.ret for e in normal]
    st = [e.ret for e in stress]
    pf = [e.pf for e in normal]
    spf = [e.pf for e in stress]
    positive = sum(a > 0 and b > 0 and c > 1.0 and d > 1.0 for a, b, c, d in zip(ret, st, pf, spf))
    return {"median_return": float(np.median(ret)), "median_stress_return": float(np.median(st)), "median_pf": float(np.median(pf)), "median_stress_pf": float(np.median(spf)), "positive_markets": int(positive), "markets": [{"market": m, "normal": asdict(n), "stress": asdict(s)} for m, n, s in zip(data, normal, stress)]}


def run(minutes: float = 180.0, initial_population: int = 64, population: int = 16, generations: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    raw = _load()
    print("=== PHASE 2 FEATURE/POLICY DISCOVERY V2 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"Markets loaded: {len(raw)}", flush=True)
    if len(raw) < 8:
        payload = {"version": "phase2-v2", "decision": "PHASE2_BLOCKED_DATA", "markets": list(raw)}
        _save(payload)
        return payload
    data = _common_cut(raw)
    n = min(len(df) for df in data.values())
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_n = n - dev_n - val_n
    dev = {m: df.iloc[:dev_n].copy() for m, df in data.items()}
    val = {m: df.iloc[dev_n:dev_n + val_n].copy() for m, df in data.items()}
    lock = {m: df.iloc[dev_n + val_n:].copy() for m, df in data.items()}
    print(f"COMMON CUTOFF: {min(df.index[-1] for df in data.values())}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}", flush=True)
    feature_rows = _rank_features(dev)
    feature_names = [x["feature"] for x in feature_rows[:6]]
    print("=== FEATURE RANKING (DEV ONLY) ===", flush=True)
    for row in feature_rows[:10]:
        print(f"feature {row['feature']} | abs_corr={row['median_abs_corr']:.5f} markets={row['markets']}", flush=True)
    rng = np.random.default_rng(seed + 77)
    model_anchor = PolicyGenome("trend", "flat", "flat", feature_names[0], 0.005, 0.001, 24, 0.85, 48)
    model_map = {m: _fit_regime_model(dev, model_anchor) for m in dev}
    genes = _pool(seed, initial_population, feature_names)
    finalists: list[PolicyGenome] = []
    evaluations = 0
    for gen in range(generations):
        if time.monotonic() >= deadline:
            break
        ranked: list[tuple[float, PolicyGenome, dict]] = []
        batch = genes[:population]
        print(f"=== GENERATION {gen + 1}/{generations} | population={len(batch)} ===", flush=True)
        for i, g in enumerate(batch, 1):
            if time.monotonic() >= deadline:
                break
            s = _dev_score = _stable_dev(dev, g, model_map, settings)
            ranked.append((s["utility"], g, s))
            evaluations += 1
            print(f"eval {i}/{len(batch)} | utility={s['utility']:.2f} feature={g.feature} ret={s['ret']:.2%} PF={s['pf']:.2f} stress={s['stress']:.2%} WF+={s['wf_positive']} breadth={s['breadth']} stab_min={s['dispersion']:.2%}", flush=True)
        ranked.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in ranked[:max(2, population // 4)]]
        finalists.extend(x[1] for x in ranked[:max(4, population // 2)])
        children: list[PolicyGenome] = []
        for elite in elites:
            for _ in range(3):
                children.append(_mutate(elite, rng, feature_names))
        genes = list(dict.fromkeys(elites + children))
    finalists = list(dict.fromkeys(finalists))[:12]
    print("=== FROZEN VALIDATION (NO EVOLUTION) ===", flush=True)
    validation = []
    for rank, g in enumerate(finalists, 1):
        if time.monotonic() >= deadline:
            break
        s = _frozen_eval(val, g, model_map, settings)
        s["rank"] = rank
        s["genome"] = asdict(g)
        validation.append(s)
        print(f"VALIDATION {rank}/{len(finalists)} | ret={s['median_return']:.2%} PF={s['median_pf']:.2f} stress={s['median_stress_return']:.2%} stressPF={s['median_stress_pf']:.2f} positive={s['positive_markets']}/{len(val)}", flush=True)
    validation.sort(key=lambda x: (x["positive_markets"], x["median_stress_return"], x["median_return"], x["median_pf"]), reverse=True)
    print("=== LOCKBOX FINAL TEST ===", flush=True)
    lockbox = []
    for rank, row in enumerate(validation[:3], 1):
        g = PolicyGenome(**row["genome"])
        s = _frozen_eval(lock, g, model_map, settings)
        s["rank"] = rank
        s["genome"] = asdict(g)
        lockbox.append(s)
        print(f"LOCKBOX {rank}/3 | ret={s['median_return']:.2%} PF={s['median_pf']:.2f} stress={s['median_stress_return']:.2%} stressPF={s['median_stress_pf']:.2f} positive={s['positive_markets']}/{len(lock)}", flush=True)
    eligible = [x for x in lockbox if x["positive_markets"] >= max(6, len(lock) // 2) and x["median_return"] > 0 and x["median_stress_return"] > 0 and x["median_pf"] > 1.05 and x["median_stress_pf"] > 1.0]
    decision = "PHASE2_VALIDATED_POLICY" if eligible else "PHASE2_NO_VALIDATED_POLICY"
    payload = {"version": "phase2-v2", "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "evaluations": evaluations, "common_cutoff": str(min(df.index[-1] for df in data.values())), "split": {"development": dev_n, "validation": val_n, "lockbox": lock_n}, "features": feature_rows, "regime_model": model_map, "finalists": [asdict(g) for g in finalists], "validation": validation, "lockbox": lockbox, "eligible": eligible, "protocol": {"feature_ranking_dev_only": True, "regime_fit_dev_only": True, "validation_frozen": True, "lockbox_top3_only": True, "spot_long_flat": True, "ai": False, "futures": False}}
    _save(payload)
    print("=== PHASE 2 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(minutes=float(os.getenv("PHASE2_MINUTES", "180")), initial_population=int(os.getenv("PHASE2_INITIAL_POPULATION", "64")), population=int(os.getenv("PHASE2_POPULATION", "16")), generations=int(os.getenv("PHASE2_GENERATIONS", "12")), seed=int(os.getenv("PHASE2_SEED", "20260829")))
