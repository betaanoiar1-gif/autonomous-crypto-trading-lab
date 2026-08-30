from __future__ import annotations

"""Phase 7 V2: statistical sanity audit of pre-registered signals only.

No new feature discovery and no strategy optimization. All selection is fixed
before Validation/Lockbox. The audit focuses on time dependence, effective
sample size, block bootstrap, cross-market persistence, and BH correction.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from math import erfc, sqrt
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE7_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase7_statistical_sanity_v2_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
SIGNALS = (("volume_pressure",72),("vol_compression",24),("vol_scaled_mom",168),("range_position_72",168),("vol_scaled_mom",24),("mom_24",168))

@dataclass
class StatRow:
    feature: str
    horizon: int
    market: str
    split: str
    n: int
    ic: float
    hac_se: float
    hac_z: float
    p_hac: float
    lag1_autocorr: float
    effective_n: float
    block_boot_positive: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str,pd.DataFrame]:
    out = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/', '_')}_1h.parquet"
        if not p.exists():
            continue
        x = pd.read_parquet(p).copy()
        x.index = pd.to_datetime(x.index, utc=True)
        out[s] = x.sort_index()
    return out


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    z = pd.DataFrame(index=df.index)
    for w in (6,24,72,168):
        z[f"mom_{w}"] = c.pct_change(w)
    vol = r.rolling(24, min_periods=24).std().replace(0,np.nan)
    z["vol_scaled_mom"] = z["mom_24"] / vol
    z["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    z["vol_compression"] = r.rolling(12, min_periods=12).std() / r.rolling(72, min_periods=72).std().replace(0,np.nan)
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    z["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    z["range_position_72"] = ((c-ll)/(hh-ll).replace(0,np.nan)).clip(-1,2)
    return z.replace([np.inf,-np.inf],np.nan)


def _hac_se(a: np.ndarray, lag: int | None = None) -> float:
    x = np.asarray(a,dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return float("nan")
    if lag is None:
        lag = max(1,min(96,int(n ** 0.25)))
    y = x-x.mean()
    var = float(np.dot(y,y)/n)
    for k in range(1,lag+1):
        cov = float(np.dot(y[k:],y[:-k])/n)
        w = 1.0-k/(lag+1.0)
        var += 2*w*cov
    return float(sqrt(max(var,1e-18)/n))


def _effective_n(a: np.ndarray) -> float:
    x = np.asarray(a,dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(n)
    y = x-x.mean()
    den = float(np.dot(y,y))
    if den <= 1e-18:
        return float(n)
    rho = float(np.dot(y[1:],y[:-1])/den)
    rho = float(np.clip(rho,-0.99,0.99))
    return float(max(1.0,n*(1-rho)/(1+rho)))


def _block_bootstrap_positive(a: np.ndarray, seed: int, block: int = 48, trials: int = 300) -> float:
    x = np.asarray(a,dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block*2:
        return 0.5
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n-block+1,size=(trials,int(np.ceil(n/block))))
    means = np.empty(trials,dtype=float)
    for i,row in enumerate(starts):
        sample = np.concatenate([x[s:s+block] for s in row])[:n]
        means[i] = sample.mean()
    return float(np.mean(means>0))


def _market_stat(df: pd.DataFrame, feature: str, horizon: int, split: str, seed: int) -> StatRow:
    f = _features(df)[feature]
    c = pd.to_numeric(df["close"],errors="coerce")
    y = c.pct_change(horizon).shift(-horizon)
    z = pd.concat([f.rename("x"),y.rename("y")],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    n = len(z)
    if n < 30:
        return StatRow(feature,horizon,"",split,n,0.0,np.nan,0.0,1.0,0.0,float(n),0.5)
    rx = z.x.rank(pct=True).to_numpy(float)
    ry = z.y.rank(pct=True).to_numpy(float)
    signal = (rx-rx.mean())*(ry-ry.mean())
    sx = np.std(rx); sy = np.std(ry)
    ic = float(np.corrcoef(rx,ry)[0,1]) if sx>0 and sy>0 else 0.0
    se = _hac_se(signal)
    q = float(ic/max(se,1e-12)) if np.isfinite(se) else 0.0
    p = float(erfc(abs(q)/sqrt(2.0)))
    ac = float(pd.Series(signal).autocorr(1)) if n>2 else 0.0
    if not np.isfinite(ac): ac = 0.0
    en = _effective_n(signal)
    boot = _block_bootstrap_positive(signal,seed)
    return StatRow(feature,horizon,"",split,n,ic,se,q,p,ac,en,boot)


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic()+minutes*60.0
    raw = _load()
    if len(raw)<8:
        payload={"version":"phase7_v2","decision":"PHASE7_BLOCKED_DATA","markets":list(raw)}
        _save(payload); return payload
    start=max(x.index[0] for x in raw.values()); end=min(x.index[-1] for x in raw.values())
    data={s:x.loc[(x.index>=start)&(x.index<=end)].copy() for s,x in raw.items()}
    n=min(len(x) for x in data.values())
    dev=int(n*0.55); val=int(n*0.25); lock=dev+val
    print("=== PHASE 7 V2 STATISTICAL SANITY ===",flush=True)
    print(f"Markets: {len(data)} | common bars={n}",flush=True)
    print(f"SPLIT: DEV={dev} VALIDATION={val} LOCKBOX={n-lock}",flush=True)
    print("PRE-REGISTERED SIGNALS ONLY",flush=True)

    rows=[]; summaries=[]
    for i,(feat,h) in enumerate(SIGNALS,1):
        if time.monotonic()>=deadline: break
        per=[]
        for m,df in data.items():
            r=_market_stat(df.iloc[:dev],feat,h,"dev",seed+i); r.market=m
            per.append(r); rows.append(asdict(r))
        ic=np.array([r.ic for r in per],float)
        p=np.array([r.p_hac for r in per],float)
        summaries.append({"feature":feat,"horizon":h,"median_ic":float(np.median(ic)),"positive_markets":int(np.sum(ic>0)),"median_p_hac":float(np.median(p)),"median_effective_n":float(np.median([r.effective_n for r in per])),"median_boot_positive":float(np.median([r.block_boot_positive for r in per]))})
        print(f"DEV {i}/{len(SIGNALS)} | {feat} h={h} medianIC={np.median(ic):.4f} pHAC={np.median(p):.3g} positive={np.sum(ic>0)}/{len(ic)}",flush=True)

    print("=== FROZEN VALIDATION ===",flush=True)
    validation=[]
    for i,(feat,h) in enumerate(SIGNALS,1):
        per=[]
        for m,df in data.items():
            r=_market_stat(df.iloc[dev:lock],feat,h,"validation",seed+100+i); r.market=m; per.append(r)
        ic=np.array([r.ic for r in per],float)
        validation.append({"feature":feat,"horizon":h,"median_ic":float(np.median(ic)),"positive_markets":int(np.sum(ic>0)),"median_p_hac":float(np.median([r.p_hac for r in per])),"median_boot_positive":float(np.median([r.block_boot_positive for r in per]))})
        print(f"VAL {i}/{len(SIGNALS)} | {feat} h={h} medianIC={np.median(ic):.4f} pHAC={np.median([r.p_hac for r in per]):.3g} positive={np.sum(ic>0)}/{len(ic)}",flush=True)

    print("=== LOCKBOX ===",flush=True)
    lockbox=[]
    for i,(feat,h) in enumerate(SIGNALS,1):
        per=[]
        for m,df in data.items():
            r=_market_stat(df.iloc[lock:],feat,h,"lockbox",seed+200+i); r.market=m; per.append(r)
        ic=np.array([r.ic for r in per],float)
        lockbox.append({"feature":feat,"horizon":h,"median_ic":float(np.median(ic)),"positive_markets":int(np.sum(ic>0)),"median_p_hac":float(np.median([r.p_hac for r in per])),"median_boot_positive":float(np.median([r.block_boot_positive for r in per]))})
        print(f"LOCKBOX {i}/{len(SIGNALS)} | {feat} h={h} medianIC={np.median(ic):.4f} pHAC={np.median([r.p_hac for r in per]):.3g} positive={np.sum(ic>0)}/{len(ic)}",flush=True)

    ps=np.array([r["p_hac"] for r in rows if np.isfinite(r["p_hac"])],float)
    ps.sort()
    m=len(ps)
    bh=np.minimum(1.0,ps*m/np.arange(1,m+1)) if m else np.array([1.0])
    qmin=float(np.min(bh))
    persistent=[]
    for d,v,l in zip(summaries,validation,lockbox):
        item={"feature":d["feature"],"horizon":d["horizon"],"dev_positive_markets":d["positive_markets"],"validation_positive_markets":v["positive_markets"],"lockbox_positive_markets":l["positive_markets"],"dev_median_ic":d["median_ic"],"validation_median_ic":v["median_ic"],"lockbox_median_ic":l["median_ic"],"lockbox_boot_positive":l["median_boot_positive"]}
        persistent.append(item)

    support=[x for x in persistent if x["dev_positive_markets"]>=8 and x["validation_positive_markets"]>=8 and x["lockbox_positive_markets"]>=8 and x["dev_median_ic"]>0 and x["validation_median_ic"]>0 and x["lockbox_median_ic"]>0 and x["lockbox_boot_positive"]>=0.90]
    decision="PHASE7_PERSISTENT_STATISTICAL_SUPPORT" if support and qmin<0.05 else "PHASE7_NO_PERSISTENT_STATISTICAL_SUPPORT"
    payload={"version":"phase7_v2","started_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"markets":list(data),"bars":n,"split":{"dev":dev,"validation":val,"lockbox":n-lock},"registered_signals":[{"feature":f,"horizon":h} for f,h in SIGNALS],"dev_summary":summaries,"row_diagnostics":rows,"validation":validation,"lockbox":lockbox,"multiple_testing":{"tests":m,"bh_q_min":qmin,"bh_supported":bool(qmin<0.05)},"persistent_candidates":persistent,"supported":support,"protocol":{"no_new_discovery":True,"pre_registered":True,"hac":True,"effective_sample_size":True,"block_bootstrap":True,"bh_correction":True,"frozen_validation":True,"frozen_lockbox":True}}
    _save(payload)
    print("=== PHASE 7 V2 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True)
    return payload


if __name__=="__main__":
    run(float(os.getenv("PHASE7_MINUTES","60")),int(os.getenv("PHASE7_SEED","20260829")))
