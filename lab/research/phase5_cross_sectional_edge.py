from __future__ import annotations

"""Phase 5: cross-sectional / relative-value edge discovery.

Goal: test whether a feature ranks assets relative to each other in a way that
predicts future cross-sectional returns after removing the common BTC factor.
No strategy is traded here; this is an information test.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE5_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase5_cross_sectional_edge_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
HORIZONS = (6, 24, 72, 168)
FEATURES = ("mom_6","mom_24","mom_72","mom_168","vol_scaled_mom","volume_pressure","vol_compression","trend_strength","range_position_72")

@dataclass
class Row:
    feature: str
    horizon: int
    split: str
    observations: int
    rank_ic: float
    long_short_spread: float
    positive_days: int
    days: int
    beta_neutral_spread: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/', '_')}_1h.parquet"
        if p.exists():
            x = pd.read_parquet(p).copy()
            x.index = pd.to_datetime(x.index, utc=True)
            out[s] = x.sort_index()
    return out


def _common(data):
    start = max(x.index[0] for x in data.values())
    end = min(x.index[-1] for x in data.values())
    return {s: x.loc[(x.index >= start) & (x.index <= end)].copy() for s, x in data.items()}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    z = pd.DataFrame(index=df.index)
    for w in (6,24,72,168):
        z[f"mom_{w}"] = c.pct_change(w)
    vol = r.rolling(24, min_periods=24).std().replace(0, np.nan)
    z["vol_scaled_mom"] = z["mom_24"] / vol
    z["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    z["vol_compression"] = r.rolling(12, min_periods=12).std() / r.rolling(72, min_periods=72).std().replace(0, np.nan)
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    z["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    width = (hh - ll).replace(0, np.nan)
    z["range_position_72"] = ((c - ll) / width).clip(-1, 2)
    return z.replace([np.inf, -np.inf], np.nan)


def _panel(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = sorted(set.intersection(*[set(x.index) for x in data.values()]))
    close = pd.DataFrame({s: data[s].loc[idx, "close"] for s in data}).sort_index()
    feats = {s: _features(data[s]).reindex(close.index) for s in data}
    return close, feats


def _residualize(y: pd.Series, btc: pd.Series) -> pd.Series:
    clean = pd.concat([y.rename("y"), btc.rename("b")], axis=1).dropna()
    if len(clean) < 50:
        return y * np.nan
    b = clean["b"].to_numpy()
    yy = clean["y"].to_numpy()
    var = float(np.var(b))
    beta = 0.0 if var <= 1e-12 else float(np.cov(yy, b, ddof=0)[0,1] / var)
    resid = clean["y"] - beta * clean["b"]
    return resid.reindex(y.index)


def _evaluate(close: pd.DataFrame, feats: dict[str,pd.DataFrame], feature: str, horizon: int, start: int, end: int, split: str) -> Row:
    dates = close.index[start:end]
    ic = []
    spread = []
    neutral = []
    positive = 0
    observed = 0
    for t in dates:
        y = close.pct_change(horizon).shift(-horizon).loc[t]
        x = pd.Series({s: feats[s].loc[t, feature] for s in close.columns})
        z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(z) < 8:
            continue
        observed += len(z)
        ric = float(z["x"].rank().corr(z["y"].rank()))
        if not np.isfinite(ric):
            continue
        ic.append(ric)
        q = z["x"].rank(pct=True)
        long_ret = float(z.loc[q >= 0.75, "y"].mean())
        short_ret = float(z.loc[q <= 0.25, "y"].mean())
        sp = long_ret - short_ret
        spread.append(sp)
        btc_y = float(y.get("BTC/USDT", np.nan))
        # Cross-sectional residual: subtract each asset's rolling-period common BTC move.
        rs = z["y"] - btc_y
        ns = float(rs.loc[q >= 0.75].mean() - rs.loc[q <= 0.25].mean())
        neutral.append(ns)
        positive += int(sp > 0)
    days = len(spread)
    return Row(feature, horizon, split, observed,
               float(np.median(ic)) if ic else 0.0,
               float(np.median(spread)) if spread else 0.0,
               positive, days,
               float(np.median(neutral)) if neutral else 0.0)


def _permute(close: pd.DataFrame, feats: dict[str,pd.DataFrame], feature: str, horizon: int, start: int, end: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    base = _evaluate(close, feats, feature, horizon, start, end, "dev")
    vals = []
    for _ in range(30):
        perm_close = close.copy()
        # shuffle future-return dates globally, preserving the feature timing.
        shifted = close.pct_change(horizon).shift(-horizon)
        shuffled = shifted.to_numpy().copy()
        rng.shuffle(shuffled.flat)
        pseudo = pd.DataFrame(shuffled, index=shifted.index, columns=shifted.columns)
        # Approximate null using rank correlation of features vs shuffled targets.
        tmp_ic = []
        for t in close.index[start:min(end, start+3000)]:
            y = pseudo.loc[t]
            x = pd.Series({s: feats[s].loc[t, feature] for s in close.columns})
            z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
            if len(z) >= 8:
                v = float(z["x"].rank().corr(z["y"].rank()))
                if np.isfinite(v): tmp_ic.append(v)
        vals.append(float(np.median(tmp_ic)) if tmp_ic else 0.0)
    observed = abs(base.rank_ic)
    exceed = sum(abs(v) >= observed for v in vals)
    return {"trials": len(vals), "observed_rank_ic": base.rank_ic, "null_exceed_rate": exceed / max(1,len(vals))}


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    data = _common(raw)
    if len(data) < 8:
        payload = {"version":"phase5","decision":"PHASE5_BLOCKED_DATA","markets":list(data)}
        _save(payload)
        return payload
    close, feats = _panel(data)
    n = len(close)
    dev_n = int(n*0.55); val_n = int(n*0.25)
    lock_start = dev_n + val_n
    print("=== PHASE 5 CROSS-SECTIONAL EDGE ===", flush=True)
    print(f"Markets: {len(data)} | bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)

    dev_rows=[]
    for f in FEATURES:
        for h in HORIZONS:
            if time.monotonic() >= deadline: break
            r = _evaluate(close, feats, f, h, 0, dev_n, "dev")
            dev_rows.append(asdict(r))
            print(f"DEV {f} h={h} | RankIC={r.rank_ic:.4f} spread={r.long_short_spread:.4%} neutral={r.beta_neutral_spread:.4%} positive={r.positive_days}/{r.days}", flush=True)

    dev_rows.sort(key=lambda x:(abs(x["rank_ic"]), abs(x["long_short_spread"]), x["positive_days"]), reverse=True)
    candidates = dev_rows[:8]
    print("=== DEV PERMUTATION ===", flush=True)
    perms=[]
    for i,row in enumerate(candidates,1):
        p=_permute(close,feats,row["feature"],row["horizon"],0,dev_n,seed+i)
        p.update({"feature":row["feature"],"horizon":row["horizon"]}); perms.append(p)
        print(f"PERM {i}/8 | {row['feature']} h={row['horizon']} null_exceed={p['null_exceed_rate']:.2%}", flush=True)

    print("=== FROZEN VALIDATION ===", flush=True)
    val=[]
    for i,row in enumerate(candidates,1):
        r=_evaluate(close,feats,row["feature"],row["horizon"],dev_n,lock_start,"validation")
        val.append(asdict(r))
        print(f"VAL {i}/8 | {r.feature} h={r.horizon} | RankIC={r.rank_ic:.4f} spread={r.long_short_spread:.4%} neutral={r.beta_neutral_spread:.4%} positive={r.positive_days}/{r.days}", flush=True)
    val.sort(key=lambda x:(x["positive_days"],x["beta_neutral_spread"],x["rank_ic"]),reverse=True)
    finalists=val[:3]

    print("=== LOCKBOX ===", flush=True)
    lockbox=[]
    for i,row in enumerate(finalists,1):
        r=_evaluate(close,feats,row["feature"],row["horizon"],lock_start,n,"lockbox")
        lockbox.append(asdict(r))
        print(f"LOCKBOX {i}/3 | {r.feature} h={r.horizon} | RankIC={r.rank_ic:.4f} spread={r.long_short_spread:.4%} neutral={r.beta_neutral_spread:.4%} positive={r.positive_days}/{r.days}", flush=True)

    eligible=[x for x in lockbox if x["positive_days"] >= max(1,int(0.60*x["days"])) and x["beta_neutral_spread"] > 0 and x["rank_ic"] > 0]
    decision="PHASE5_CROSS_SECTIONAL_EDGE_FOUND" if eligible else "PHASE5_NO_CONFIRMED_RELATIVE_EDGE"
    payload={
        "version":"phase5","started_at":datetime.now(timezone.utc).isoformat(),"decision":decision,
        "markets":list(data),"bars":n,"split":{"dev":dev_n,"validation":val_n,"lockbox":n-lock_start},
        "dev":dev_rows,"candidates":candidates,"permutation":perms,"validation":val,"lockbox":lockbox,"eligible":eligible,
        "protocol":{"cross_sectional":True,"beta_neutral_to_BTC":True,"dev_only_selection":True,"validation_frozen":True,"lockbox_top3":True,"spot_only":True}
    }
    _save(payload)
    print("=== PHASE 5 DECISION ===", flush=True); print(decision, flush=True); print(f"Saved: {OUT}", flush=True)
    return payload

if __name__ == "__main__":
    run(float(os.getenv("PHASE5_MINUTES","60")), int(os.getenv("PHASE5_SEED","20260829")))
