from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE4_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase4_conditional_edge_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
HORIZONS = (1, 6, 24, 72, 168)

@dataclass
class EdgeRow:
    feature: str
    horizon: int
    regime: str
    market: str
    n: int
    ic: float
    spread: float
    top_return: float
    bottom_return: float
    accuracy: float


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
            df = pd.read_parquet(p).sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            out[s] = df
    return out


def _common(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not data:
        return {}
    start = max(x.index[0] for x in data.values())
    end = min(x.index[-1] for x in data.values())
    return {k: v.loc[(v.index >= start) & (v.index <= end)].copy() for k,v in data.items()}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    f = pd.DataFrame(index=df.index)
    f["mom_6"] = c.pct_change(6)
    f["mom_24"] = c.pct_change(24)
    f["mom_72"] = c.pct_change(72)
    f["mom_168"] = c.pct_change(168)
    f["mean_distance"] = c / c.rolling(72, min_periods=72).mean() - 1.0
    vol24 = r.rolling(24, min_periods=24).std().replace(0, np.nan)
    f["vol_scaled_mom"] = f["mom_24"] / vol24
    short = r.rolling(12, min_periods=12).std()
    long = r.rolling(72, min_periods=72).std().replace(0, np.nan)
    f["vol_compression"] = short / long
    f["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    ema24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    ema96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    f["trend_strength"] = ema24 / ema96 - 1.0
    hh = h.rolling(48, min_periods=48).max().shift(1)
    ll = l.rolling(48, min_periods=48).min().shift(1)
    width = (hh-ll).replace(0, np.nan)
    f["range_position"] = ((c-ll)/width).clip(-1, 2)
    f["market_ret_24"] = r.rolling(24, min_periods=24).sum()
    f["market_vol_72"] = r.rolling(72, min_periods=72).std()
    return f.replace([np.inf,-np.inf], np.nan)


def _residual_target(df: pd.DataFrame, btc: pd.Series, h: int) -> pd.Series:
    y = df["close"].pct_change(h).shift(-h)
    b = btc.pct_change(h).shift(-h).reindex(df.index)
    z = pd.concat([y.rename("y"), b.rename("b")], axis=1).dropna()
    if len(z) < 200:
        return y * np.nan
    beta = float(z["y"].cov(z["b"]) / z["b"].var()) if z["b"].var() > 1e-12 else 0.0
    return y - beta * b


def _regime(f: pd.DataFrame, trend_q: float = 0.65, vol_q: float = 0.8) -> pd.Series:
    trend_cut = float(f["trend_strength"].abs().quantile(trend_q))
    vol_cut = float(f["market_vol_72"].quantile(vol_q))
    r = pd.Series("range", index=f.index, dtype="object")
    r.loc[f["trend_strength"].abs() >= trend_cut] = "trend"
    r.loc[f["market_vol_72"] >= vol_cut] = "high_vol"
    return r


def _stats(x: pd.Series, y: pd.Series) -> tuple[float,float,float,float,float]:
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(z) < 250:
        return (0.0,0.0,0.0,0.0,0.0)
    ic = float(z.x.corr(z.y))
    q = z.x.rank(pct=True)
    top = z.loc[q >= .8, "y"]
    bot = z.loc[q <= .2, "y"]
    tr = float(top.mean()) if len(top) else 0.0
    br = float(bot.mean()) if len(bot) else 0.0
    acc = float((np.sign(z.x) == np.sign(z.y)).mean())
    vals = [ic,tr-br,tr,br,acc]
    return tuple(float(v) if np.isfinite(v) else 0.0 for v in vals)


def _scan(dev: dict[str,pd.DataFrame], btc: pd.Series) -> list[EdgeRow]:
    rows=[]
    names=("mom_6","mom_24","mom_72","mom_168","mean_distance","vol_scaled_mom","vol_compression","volume_pressure","trend_strength","range_position")
    for m,df in dev.items():
        f=_features(df)
        reg=_regime(f)
        for h in HORIZONS:
            y=_residual_target(df, btc, h)
            for name in names:
                for regime in ("all","trend","range","high_vol"):
                    x=f[name]
                    if regime != "all": x=x.where(reg==regime)
                    ic,spread,tr,br,acc=_stats(x,y)
                    n=int(pd.concat([x,y],axis=1).dropna().shape[0])
                    rows.append(EdgeRow(name,h,regime,m,n,ic,spread,tr,br,acc))
    return rows


def _aggregate(rows: list[EdgeRow]) -> list[dict]:
    g={}
    for x in rows: g.setdefault((x.feature,x.horizon,x.regime),[]).append(x)
    out=[]
    for key,items in g.items():
        spreads=[x.spread for x in items]
        ics=[x.ic for x in items]
        out.append({"feature":key[0],"horizon":key[1],"regime":key[2],"markets":len(items),"median_ic":float(np.median(ics)),"median_abs_ic":float(np.median(np.abs(ics))),"median_spread":float(np.median(spreads)),"positive_spread_markets":int(sum(v>0 for v in spreads)),"median_accuracy":float(np.median([x.accuracy for x in items]))})
    return sorted(out,key=lambda x:(x["positive_spread_markets"],x["median_spread"],abs(x["median_ic"])),reverse=True)


def run(minutes: float=90.0, seed: int=20260829)->dict:
    started=datetime.now(timezone.utc)
    deadline=time.monotonic()+minutes*60
    data=_common(_load())
    print("=== PHASE 4 CONDITIONAL RESIDUAL EDGE ===",flush=True)
    print(f"Markets loaded: {len(data)}",flush=True)
    if len(data)<8:
        payload={"version":"phase4","decision":"PHASE4_BLOCKED_DATA","markets":list(data)}
        _save(payload); return payload
    n=min(len(x) for x in data.values())
    dev_n=int(n*.55); val_n=int(n*.25); lock_n=n-dev_n-val_n
    dev={m:x.iloc[:dev_n].copy() for m,x in data.items()}
    val={m:x.iloc[dev_n:dev_n+val_n].copy() for m,x in data.items()}
    lock={m:x.iloc[dev_n+val_n:].copy() for m,x in data.items()}
    btc_full=data["BTC/USDT"]["close"]
    btc_dev=btc_full.iloc[:dev_n]
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}",flush=True)
    print("LOCKBOX: RESERVED",flush=True)
    print("=== DEV CONDITIONAL SCAN ===",flush=True)
    dev_rows=_scan(dev,btc_dev)
    ranking=_aggregate(dev_rows)
    for i,row in enumerate(ranking[:15],1):
        print(f"edge {i} | {row['feature']} h={row['horizon']} regime={row['regime']} IC={row['median_ic']:.4f} spread={row['median_spread']:.4%} positive={row['positive_spread_markets']}/{row['markets']} acc={row['median_accuracy']:.2%}",flush=True)
    candidates=ranking[:12]
    print("=== FROZEN VALIDATION ===",flush=True)
    validation=[]
    for i,row in enumerate(candidates,1):
        vals=[]
        for m,df in val.items():
            f=_features(df)
            reg=_regime(f)
            x=f[row["feature"]]
            if row["regime"]!="all": x=x.where(reg==row["regime"])
            btc=btc_full.loc[df.index]
            y=_residual_target(df,btc,row["horizon"])
            ic,spread,tr,br,acc=_stats(x,y)
            vals.append((ic,spread,acc))
        item={**row,"median_ic":float(np.median([x[0] for x in vals])),"median_spread":float(np.median([x[1] for x in vals])),"positive_spread_markets":int(sum(x[1]>0 for x in vals)),"median_accuracy":float(np.median([x[2] for x in vals]))}
        validation.append(item)
        print(f"validation {i}/{len(candidates)} | {row['feature']} h={row['horizon']} regime={row['regime']} IC={item['median_ic']:.4f} spread={item['median_spread']:.4%} positive={item['positive_spread_markets']}/12",flush=True)
    validation.sort(key=lambda x:(x["positive_spread_markets"],x["median_spread"],abs(x["median_ic"])),reverse=True)
    top=validation[:3]
    print("=== LOCKBOX ===",flush=True)
    lockbox=[]
    for i,row in enumerate(top,1):
        vals=[]
        for m,df in lock.items():
            f=_features(df); reg=_regime(f)
            x=f[row["feature"]]
            if row["regime"]!="all": x=x.where(reg==row["regime"])
            btc=btc_full.loc[df.index]; y=_residual_target(df,btc,row["horizon"])
            ic,spread,tr,br,acc=_stats(x,y); vals.append((ic,spread,acc))
        item={**row,"median_ic":float(np.median([x[0] for x in vals])),"median_spread":float(np.median([x[1] for x in vals])),"positive_spread_markets":int(sum(x[1]>0 for x in vals)),"median_accuracy":float(np.median([x[2] for x in vals]))}
        lockbox.append(item)
        print(f"lockbox {i}/3 | {row['feature']} h={row['horizon']} regime={row['regime']} IC={item['median_ic']:.4f} spread={item['median_spread']:.4%} positive={item['positive_spread_markets']}/12 acc={item['median_accuracy']:.2%}",flush=True)
    eligible=[x for x in lockbox if x["positive_spread_markets"]>=8 and x["median_spread"]>0 and x["median_accuracy"]>=.50 and abs(x["median_ic"])>=.01]
    decision="PHASE4_EDGE_CONFIRMED" if eligible else "PHASE4_NO_CONFIRMED_CONDITIONAL_EDGE"
    payload={"version":"phase4","started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"split":{"development":dev_n,"validation":val_n,"lockbox":lock_n},"dev_ranking":ranking,"validation":validation,"lockbox":lockbox,"eligible":eligible,"protocol":{"residualized_vs_btc":True,"regime_fit_dev_only":True,"validation_frozen":True,"lockbox_top3_only":True,"spot_long_flat":True,"horizons":list(HORIZONS)}}
    _save(payload)
    print("=== PHASE 4 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True)
    return payload

if __name__=="__main__":
    run(minutes=float(os.getenv("PHASE4_MINUTES","90")),seed=int(os.getenv("PHASE4_SEED","20260829")))
