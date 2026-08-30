from __future__ import annotations

"""Phase 9 V2: fixed equal-weight ensemble audit.

No optimization or evolution. The six pre-registered signals are combined with
fixed equal weights. All feature computation is vectorized per market and all
splits are frozen.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import os
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE9_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase9_fixed_ensemble_audit_v2_latest.json"
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
SIGNALS = (
    ("volume_pressure", 72),
    ("vol_compression", 24),
    ("vol_scaled_mom", 168),
    ("range_position_72", 168),
    ("vol_scaled_mom", 24),
    ("mom_24", 168),
)


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/', '_')}_1h.parquet"
        if not p.exists():
            continue
        x = pd.read_parquet(p).copy()
        if not isinstance(x.index, pd.DatetimeIndex):
            x.index = pd.to_datetime(x.index, utc=True)
        else:
            x.index = pd.to_datetime(x.index, utc=True)
        data[s] = x.sort_index()
    return data


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for w in (6, 24, 72, 168):
        f[f"mom_{w}"] = c.pct_change(w)
    vol24 = r.rolling(24, min_periods=24).std().replace(0, np.nan)
    f["vol_scaled_mom"] = f["mom_24"] / vol24
    f["volume_pressure"] = v / v.rolling(48, min_periods=48).mean().replace(0, np.nan) - 1.0
    s12 = r.rolling(12, min_periods=12).std()
    s72 = r.rolling(72, min_periods=72).std().replace(0, np.nan)
    f["vol_compression"] = s12 / s72
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    f["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    f["range_position_72"] = ((c - ll) / (hh - ll).replace(0, np.nan)).clip(-1.0, 2.0)
    return f.replace([np.inf, -np.inf], np.nan)


def _signal(df: pd.DataFrame, feature: str, horizon: int) -> pd.Series:
    f = _features(df)
    x = f[feature]
    # Directional score must use only information available at t.
    return x.rank(pct=True) - 0.5


def _future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    c = pd.to_numeric(df["close"], errors="coerce")
    return c.pct_change(horizon).shift(-horizon)


def _metrics(score: pd.Series, future: pd.Series) -> dict:
    z = pd.concat([score.rename("score"), future.rename("ret")], axis=1).dropna()
    if len(z) < 100:
        return {"n": int(len(z)), "ic": 0.0, "spread": 0.0, "positive": 0, "direction_acc": 0.5}
    ic = float(z["score"].corr(z["ret"], method="spearman"))
    q = z["score"].rank(pct=True)
    high = z.loc[q >= 0.8, "ret"]
    low = z.loc[q <= 0.2, "ret"]
    spread = float(high.mean() - low.mean()) if len(high) and len(low) else 0.0
    positive = int((z["ret"].where(z["score"] > 0) > 0).sum())
    denom = int((z["score"] > 0).sum())
    acc = float(positive / denom) if denom else 0.5
    return {
        "n": int(len(z)),
        "ic": 0.0 if not np.isfinite(ic) else ic,
        "spread": spread,
        "positive": positive,
        "direction_acc": acc,
    }


def _ensemble_market(df: pd.DataFrame, start: int, end: int) -> tuple[pd.Series, pd.Series]:
    idx = df.index[start:end]
    scores = []
    future = []
    for feature, horizon in SIGNALS:
        scores.append(_signal(df, feature, horizon).reindex(idx))
        future.append(_future_return(df, horizon).reindex(idx))
    s = pd.concat(scores, axis=1).mean(axis=1, skipna=True)
    # Average aligned future returns to keep horizons pre-registered and equal-weighted.
    y = pd.concat(future, axis=1).mean(axis=1, skipna=True)
    return s, y


def _aggregate(data: dict[str, pd.DataFrame], start: int, end: int) -> dict:
    per = []
    for market, df in data.items():
        s, y = _ensemble_market(df, start, end)
        m = _metrics(s, y)
        m["market"] = market
        per.append(m)
    if not per:
        return {"markets": 0, "median_ic": 0.0, "median_spread": 0.0, "positive_markets": 0, "market_details": []}
    ics = np.asarray([x["ic"] for x in per], dtype=float)
    spreads = np.asarray([x["spread"] for x in per], dtype=float)
    return {
        "markets": len(per),
        "median_ic": float(np.nanmedian(ics)),
        "median_spread": float(np.nanmedian(spreads)),
        "positive_markets": int(np.sum(spreads > 0)),
        "market_details": per,
    }


def _block_stability(data: dict[str, pd.DataFrame], start: int, end: int, blocks: int = 6) -> dict:
    rows = []
    width = max(1, (end - start) // blocks)
    for i in range(blocks):
        a = start + i * width
        b = end if i == blocks - 1 else min(end, a + width)
        if b - a < 200:
            continue
        r = _aggregate(data, a, b)
        r["block"] = i + 1
        rows.append(r)
        print(
            f"BLOCK {i+1}/{blocks} | IC={r['median_ic']:.4f} "
            f"spread={r['median_spread']:.4%} positive={r['positive_markets']}/{r['markets']}",
            flush=True,
        )
    return {"blocks": rows, "positive_block_fraction": float(np.mean([r["median_spread"] > 0 for r in rows])) if rows else 0.0}


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    if len(raw) < 8:
        out = {"version": "phase9_v2", "decision": "PHASE9_BLOCKED_DATA", "markets": list(raw)}
        _save(out)
        return out

    common_start = max(x.index[0] for x in raw.values())
    common_end = min(x.index[-1] for x in raw.values())
    data = {m: x.loc[(x.index >= common_start) & (x.index <= common_end)].copy() for m, x in raw.items()}
    n = min(len(x) for x in data.values())
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_start = dev_n + val_n

    print("=== PHASE 9 V2 FIXED ENSEMBLE ===", flush=True)
    print(f"Markets: {len(data)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)
    print("FIXED EQUAL WEIGHTS: 1/6 PER REGISTERED SIGNAL", flush=True)

    dev = _aggregate(data, 0, dev_n)
    print("=== DEV ENSEMBLE ===", flush=True)
    print(dev, flush=True)

    if time.monotonic() >= deadline:
        out = {"version": "phase9_v2", "decision": "PHASE9_TIMEOUT", "dev": dev}
        _save(out)
        return out

    stability = _block_stability(data, 0, dev_n, blocks=6)

    print("=== FROZEN VALIDATION ===", flush=True)
    val = _aggregate(data, dev_n, lock_start)
    print(val, flush=True)

    if time.monotonic() >= deadline:
        out = {"version": "phase9_v2", "decision": "PHASE9_TIMEOUT", "dev": dev, "stability": stability, "validation": val}
        _save(out)
        return out

    print("=== LOCKBOX AUDIT ===", flush=True)
    lock = _aggregate(data, lock_start, n)
    print(lock, flush=True)

    # Fixed decision rule: no optimization. Require positive median IC/spread,
    # >= 8/12 positive markets and >= 2/3 positive DEV blocks in validation+lockbox.
    val_ok = (
        val["markets"] >= 8
        and val["median_ic"] > 0.0
        and val["median_spread"] > 0.0
        and val["positive_markets"] >= 8
    )
    lock_ok = (
        lock["markets"] >= 8
        and lock["median_ic"] > 0.0
        and lock["median_spread"] > 0.0
        and lock["positive_markets"] >= 8
    )
    stability_ok = stability["positive_block_fraction"] >= (2.0 / 3.0)
    eligible = bool(val_ok and lock_ok and stability_ok)
    decision = "PHASE9_CONFIRMED_FIXED_ENSEMBLE" if eligible else "PHASE9_NO_CONFIRMED_ENSEMBLE"

    out = {
        "version": "phase9_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision": decision,
        "signals": [{"feature": f, "horizon": h, "weight": 1.0 / len(SIGNALS)} for f, h in SIGNALS],
        "splits": {"dev": dev_n, "validation": val_n, "lockbox": n - lock_start},
        "dev": dev,
        "stability": stability,
        "validation": val,
        "lockbox": lock,
        "eligible": eligible,
        "rules": {
            "fixed_equal_weights": True,
            "validation_positive_markets_min": 8,
            "lockbox_positive_markets_min": 8,
            "median_ic_positive": True,
            "median_spread_positive": True,
            "positive_dev_blocks_min_fraction": 2.0 / 3.0,
        },
    }
    _save(out)
    print("=== PHASE 9 V2 DECISION ===", flush=True)
    print(decision, flush=True)
    print("Saved:", OUT, flush=True)
    return out


if __name__ == "__main__":
    run()
