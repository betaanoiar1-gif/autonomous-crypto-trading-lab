from __future__ import annotations

"""Phase 3: predictive edge discovery before policy invention."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE3_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase3_predictive_edge_latest.json"
SYMBOLS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT")
HORIZONS = (1, 6, 24, 72)
FEATURE_WINDOWS = (6, 24, 72, 168, 336)
FEATURE_NAMES = ("mom", "mean_distance", "range_position", "breakout_pressure", "vol_scaled_mom", "vol_compression", "volume_pressure", "trend_strength")

@dataclass
class EdgeStats:
    feature: str
    horizon: int
    market: str
    sample: int
    ic: float
    rank_ic: float
    sign_accuracy: float
    top_bottom_spread: float
    top_quantile_return: float
    bottom_quantile_return: float
    turnover_proxy: float

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
            frame = pd.read_parquet(path).sort_index()
            frame.index = pd.to_datetime(frame.index, utc=True)
            data[symbol] = frame
    return data

def _common_cut(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not data:
        return {}
    start = max(frame.index[0] for frame in data.values())
    end = min(frame.index[-1] for frame in data.values())
    return {symbol: frame.loc[(frame.index >= start) & (frame.index <= end)].copy() for symbol, frame in data.items()}

def _features(df: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    ret = close.pct_change()
    out = pd.DataFrame(index=df.index)
    for w in FEATURE_WINDOWS:
        out[f"mom_{w}"] = close.pct_change(w)
        mean = close.rolling(w, min_periods=w).mean()
        std = close.rolling(w, min_periods=w).std().replace(0.0, np.nan)
        out[f"mean_distance_{w}"] = close / mean - 1.0
        out[f"z_{w}"] = (close - mean) / std
        hh = high.rolling(w, min_periods=w).max().shift(1)
        ll = low.rolling(w, min_periods=w).min().shift(1)
        width = (hh - ll).replace(0.0, np.nan)
        out[f"breakout_pressure_{w}"] = ((close - ll) / width).clip(-2.0, 3.0)
        out[f"range_position_{w}"] = ((close - ll) / width).clip(-1.0, 2.0)
    vol24 = ret.rolling(24, min_periods=24).std().replace(0.0, np.nan)
    out["vol_scaled_mom"] = out["mom_24"] / vol24
    vol_short = ret.rolling(12, min_periods=12).std()
    vol_long = ret.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    out["vol_compression"] = vol_short / vol_long
    out["volume_pressure"] = volume / volume.rolling(48, min_periods=48).mean()
    ema24 = close.ewm(span=24, adjust=False, min_periods=24).mean()
    ema96 = close.ewm(span=96, adjust=False, min_periods=96).mean()
    out["trend_strength"] = ema24 / ema96 - 1.0
    return out.replace([np.inf, -np.inf], np.nan)

def _feature_series(features: pd.DataFrame, name: str) -> pd.Series:
    aliases = {
        "mom": "mom_24",
        "mean_distance": "mean_distance_72",
        "range_position": "range_position_72",
        "breakout_pressure": "breakout_pressure_72",
        "vol_scaled_mom": "vol_scaled_mom",
        "vol_compression": "vol_compression",
        "volume_pressure": "volume_pressure",
        "trend_strength": "trend_strength",
    }
    return features[aliases[name]]

def _stats(x: pd.Series, y: pd.Series):
    clean = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(clean) < 200:
        return (0.0,) * 7
    ic = float(clean["x"].corr(clean["y"]))
    rank_ic = float(clean["x"].rank().corr(clean["y"].rank()))
    sign_accuracy = float((np.sign(clean["x"]) == np.sign(clean["y"])).mean())
    q = clean["x"].rank(pct=True)
    top = clean.loc[q >= 0.80, "y"]
    bottom = clean.loc[q <= 0.20, "y"]
    top_ret = float(top.mean()) if len(top) else 0.0
    bottom_ret = float(bottom.mean()) if len(bottom) else 0.0
    spread = top_ret - bottom_ret
    turnover_proxy = float(clean["x"].diff().abs().mean())
    vals = [ic, rank_ic, sign_accuracy, spread, top_ret, bottom_ret, turnover_proxy]
    return tuple(float(v) if np.isfinite(v) else 0.0 for v in vals)

def _market_edges(df: pd.DataFrame, market: str, split: str) -> list[EdgeStats]:
    f = _features(df)
    rows = []
    for horizon in HORIZONS:
        target = df["close"].pct_change(horizon).shift(-horizon)
        for name in FEATURE_NAMES:
            x = _feature_series(f, name)
            ic, ric, acc, spread, top, bottom, turnover = _stats(x, target)
            valid_n = int(pd.concat([x, target], axis=1).dropna().shape[0])
            rows.append(EdgeStats(name, horizon, market, valid_n, ic, ric, acc, spread, top, bottom, turnover))
    return rows

def _aggregate(rows: list[EdgeStats]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault((row.feature, row.horizon), []).append(row)
    out = []
    for (feature, horizon), items in grouped.items():
        spreads = [x.top_bottom_spread for x in items]
        out.append({
            "feature": feature, "horizon": horizon, "markets": len(items),
            "median_ic": float(np.median([x.ic for x in items])),
            "median_abs_ic": float(np.median(np.abs([x.ic for x in items]))),
            "median_rank_ic": float(np.median([x.rank_ic for x in items])),
            "median_spread": float(np.median(spreads)),
            "positive_spread_markets": int(sum(x > 0 for x in spreads)),
            "median_sign_accuracy": float(np.median([x.sign_accuracy for x in items])),
        })
    return sorted(out, key=lambda x: (x["positive_spread_markets"], abs(x["median_spread"]), abs(x["median_rank_ic"])), reverse=True)

def _permutation_test(df, feature, horizon, trials=50, seed=20260829):
    f = _feature_series(_features(df), feature)
    y = df["close"].pct_change(horizon).shift(-horizon)
    clean = pd.concat([f.rename("x"), y.rename("y")], axis=1).dropna()
    if len(clean) < 1000:
        return {"trials": 0, "false_positive_rate": 1.0, "observed": 0.0}
    observed = float(clean["x"].corr(clean["y"]))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(trials):
        shuffled = clean["y"].to_numpy().copy()
        rng.shuffle(shuffled)
        value = float(clean["x"].corr(pd.Series(shuffled, index=clean.index)))
        exceed += int(abs(value) >= abs(observed))
    return {"trials": trials, "false_positive_rate": float(exceed / trials), "observed": observed}

def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    data = _common_cut(_load())
    if len(data) < 8:
        payload = {"version": "phase3", "decision": "PHASE3_BLOCKED_DATA", "markets": list(data)}
        _save(payload)
        return payload
    n = min(len(df) for df in data.values())
    dev_n, val_n = int(n * 0.55), int(n * 0.25)
    lock_n = n - dev_n - val_n
    dev = {m: df.iloc[:dev_n].copy() for m, df in data.items()}
    val = {m: df.iloc[dev_n:dev_n + val_n].copy() for m, df in data.items()}
    lock = {m: df.iloc[dev_n + val_n:].copy() for m, df in data.items()}
    print("=== PHASE 3 PREDICTIVE EDGE DISCOVERY ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"Markets loaded: {len(data)}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}", flush=True)
    print("LOCKBOX: RESERVED", flush=True)
    print("=== DEV FEATURE SCAN ===", flush=True)
    dev_rows = []
    for market, frame in dev.items():
        if time.monotonic() >= deadline: break
        dev_rows.extend(_market_edges(frame, market, "dev"))
    ranking = _aggregate(dev_rows)
    for row in ranking[:12]:
        print(f"edge {row['feature']} h={row['horizon']} IC={row['median_ic']:.4f} RankIC={row['median_rank_ic']:.4f} spread={row['median_spread']:.4%} spread+={row['positive_spread_markets']}/{row['markets']} acc={row['median_sign_accuracy']:.2%}", flush=True)
    candidates = ranking[:8]
    print("=== DEV PERMUTATION CHECK ===", flush=True)
    permutation = []
    for idx, row in enumerate(candidates, 1):
        if time.monotonic() >= deadline: break
        p = _permutation_test(dev["BTC/USDT"], row["feature"], row["horizon"], 50, seed + idx)
        p.update({"feature": row["feature"], "horizon": row["horizon"]})
        permutation.append(p)
        print(f"perm {idx}/{len(candidates)} | {row['feature']} h={row['horizon']} observed={p['observed']:.4f} exceed={p['false_positive_rate']:.2%}", flush=True)
    print("=== FROZEN VALIDATION ===", flush=True)
    validation = []
    for idx, row in enumerate(candidates, 1):
        vals = []
        for market, frame in val.items():
            f = _feature_series(_features(frame), row["feature"])
            target = frame["close"].pct_change(row["horizon"]).shift(-row["horizon"])
            ic, ric, acc, spread, top, bottom, turnover = _stats(f, target)
            vals.append({"market": market, "ic": ic, "rank_ic": ric, "spread": spread, "accuracy": acc, "turnover": turnover})
        item = {**row, "rank": idx, "median_ic": float(np.median([v["ic"] for v in vals])), "median_rank_ic": float(np.median([v["rank_ic"] for v in vals])), "median_spread": float(np.median([v["spread"] for v in vals])), "positive_spread_markets": int(sum(v["spread"] > 0 for v in vals)), "median_sign_accuracy": float(np.median([v["accuracy"] for v in vals])), "market_details": vals}
        validation.append(item)
        print(f"validation {idx}/{len(candidates)} | {row['feature']} h={row['horizon']} IC={item['median_ic']:.4f} spread={item['median_spread']:.4%} positive={item['positive_spread_markets']}/12", flush=True)
    validation.sort(key=lambda x: (x["positive_spread_markets"], x["median_spread"], abs(x["median_rank_ic"])), reverse=True)
    frozen = validation[:3]
    print("=== LOCKBOX EDGE TEST ===", flush=True)
    lockbox = []
    for idx, row in enumerate(frozen, 1):
        vals = []
        for market, frame in lock.items():
            f = _feature_series(_features(frame), row["feature"])
            target = frame["close"].pct_change(row["horizon"]).shift(-row["horizon"])
            ic, ric, acc, spread, top, bottom, turnover = _stats(f, target)
            vals.append({"market": market, "ic": ic, "rank_ic": ric, "spread": spread, "accuracy": acc})
        item = {**row, "rank": idx, "median_ic": float(np.median([v["ic"] for v in vals])), "median_rank_ic": float(np.median([v["rank_ic"] for v in vals])), "median_spread": float(np.median([v["spread"] for v in vals])), "positive_spread_markets": int(sum(v["spread"] > 0 for v in vals)), "median_sign_accuracy": float(np.median([v["accuracy"] for v in vals])), "market_details": vals}
        lockbox.append(item)
        print(f"lockbox {idx}/3 | {row['feature']} h={row['horizon']} IC={item['median_ic']:.4f} spread={item['median_spread']:.4%} positive={item['positive_spread_markets']}/12", flush=True)
    eligible = [x for x in lockbox if x["positive_spread_markets"] >= 7 and x["median_spread"] > 0 and x["median_sign_accuracy"] > 0.50]
    decision = "PHASE3_EDGE_FOUND" if eligible else "PHASE3_NO_CONFIRMED_EDGE"
    payload = {"version": "phase3", "started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "markets": list(data), "split": {"development": dev_n, "validation": val_n, "lockbox": lock_n}, "feature_ranking_dev": ranking, "permutation": permutation, "validation": validation, "lockbox": lockbox, "eligible": eligible, "protocol": {"feature_ranking_dev_only": True, "validation_frozen": True, "lockbox_top3_only": True, "purged_validation": True, "spot_only": True, "ai": False, "futures": False}}
    _save(payload)
    print("=== PHASE 3 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload

if __name__ == "__main__":
    run(minutes=float(os.getenv("PHASE3_MINUTES", "60")), seed=int(os.getenv("PHASE3_SEED", "20260829")))
