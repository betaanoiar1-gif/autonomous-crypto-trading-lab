from __future__ import annotations

"""Phase 1 V2: robust cross-market discovery with train/validation/lockbox."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("DISCOVERY_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase1_discovery_v2_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
SCREEN_MARKETS = ("BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","LINK/USDT")

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
    strength: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/','_')}_1h.parquet"
        if p.exists(): out[s] = pd.read_parquet(p).sort_index()
    return out


def _common_cut(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    return {k: v.loc[(v.index >= start) & (v.index <= end)].copy() for k,v in data.items()}


def _signal(df: pd.DataFrame, g: Genome) -> pd.Series:
    c = df["close"].astype(float)
    r = c.pct_change()
    mom = c.pct_change(g.lookback)
    fast = c.ewm(span=g.fast, adjust=False, min_periods=g.fast).mean()
    slow = c.ewm(span=g.slow, adjust=False, min_periods=g.slow).mean()
    vol = r.rolling(g.vol_window, min_periods=g.vol_window).std()
    mean = c.rolling(g.lookback, min_periods=g.lookback).mean()
    std = c.rolling(g.lookback, min_periods=g.lookback).std().replace(0.0, np.nan)
    z = (c-mean)/std
    hh = c.rolling(g.lookback, min_periods=g.lookback).max().shift(1)
    if g.family == "momentum": raw = (mom > g.threshold) & (fast > slow) & (vol < g.vol_cap)
    elif g.family == "breakout": raw = (c > hh) & (vol < g.vol_cap)
    elif g.family == "mean_reversion": raw = (z < -abs(g.threshold)) & (vol < g.vol_cap)
    elif g.family == "trend_pullback": raw = (fast > slow) & (mom > -abs(g.threshold)) & (mom < g.exit_threshold)
    elif g.family == "vol_breakout": raw = (c > hh) & (vol > max(0.001, g.vol_cap*0.35))
    else: raw = (fast > slow) & (mom > g.threshold)
    raw = raw.astype(float).fillna(0.0).clip(0.0,1.0)
    pos = np.zeros(len(raw)); active=False; age=g.hold_bars
    for i,v in enumerate(raw.to_numpy()):
        if active:
            age += 1
            if v < 0.5 and age >= g.hold_bars: active=False
        elif v > 0.5:
            active=True; age=0
        pos[i] = 1.0 if active else 0.0
    return pd.Series(pos,index=df.index)


def _metrics(df, sig, settings, cost_mult=1.0):
    r = run_ohlcv(df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps*cost_mult,
        settings.execution.slippage_bps*cost_mult,
        market_type="spot", leverage=1.0, funding_rates=None)
    return r.metrics


def _evaluate(df,g,settings):
    normal = _metrics(df,_signal(df,g),settings,1.0)
    stress = _metrics(df,_signal(df,g),settings,2.0)
    return {"normal":normal,"stress":stress}


def _wf(df,g,settings,folds=5):
    n=len(df)
    if n < 4000: return {"median_return":0.0,"positive":0,"folds":[]}
    fold=n//folds; vals=[]; rows=[]
    for k in range(folds):
        lo=k*fold; hi=(k+1)*fold if k<folds-1 else n
        test=df.iloc[lo:hi].copy()
        m=_evaluate(test,g,settings)["normal"]
        vals.append(float(m.get("total_return",0.0)))
        rows.append({"fold":k+1,"ret":float(m.get("total_return",0.0)),"pf":float(m.get("profit_factor",0.0))})
    return {"median_return":float(np.median(vals)),"positive":sum(v>0 for v in vals),"folds":rows}


def _utility(ret,pf,dd,stress,stress_pf,wf_med,wf_pos,trades):
    return 90*ret + 18*max(0,pf-1) + 40*wf_med + 8*wf_pos + 18*max(0,stress) + 8*max(0,stress_pf-1) - 30*max(0,-dd-0.10) - 20*max(0,-stress) - 0.003*trades


def _pool(seed,count):
    rng=np.random.default_rng(seed); fam=["momentum","breakout","mean_reversion","trend_pullback","vol_breakout","ma_cross"]
    lbs=[24,48,72,120,168,240,336]; fasts=[6,8,12,18,24,36,48]; slows=[40,60,90,120,180,240,360]
    out=[]; seen=set()
    while len(out)<count:
        fast=int(rng.choice(fasts)); slow=int(rng.choice(slows))
        if fast>=slow: continue
        g=Genome(str(rng.choice(fam)),int(rng.choice(lbs)),fast,slow,float(rng.choice([.0025,.005,.0075,.01,.015])),float(rng.choice([.0015,.003,.005,.0075])),int(rng.choice([12,24,36,48,72])),float(rng.choice([.012,.018,.025,.035,.05])),int(rng.choice([12,24,48,72,96])),float(rng.choice([.75,1,1.25])))
        if g not in seen: seen.add(g); out.append(g)
    return out


def _mutate(g,rng):
    fam=g.family if rng.random()>.12 else str(rng.choice(["momentum","breakout","mean_reversion","trend_pullback","vol_breakout","ma_cross"]))
    nf=g.fast+int(rng.choice([-6,-3,0,3,6]);)
    return g


def run(minutes=180.0,initial_population=64,population=16,generations=20,seed=20260829):
    # Temporary clean evolution; mutation is deliberately conservative.
    started=datetime.now(timezone.utc); deadline=time.monotonic()+minutes*60
    settings=load_settings(); raw=_load()
    print("=== PHASE 1 DISCOVERY V2 ===",flush=True)
    print(f"Markets loaded: {len(raw)}",flush=True)
    if len(raw)<8:
        payload={"decision":"DISCOVERY_BLOCKED_DATA","markets":list(raw),"version":"v2"}; _save(payload); return payload
    data=_common_cut(raw); n=min(map(len,data.values())); dev_n=int(n*.70); val_n=int(n*.15)
    dev={m:d.iloc[:dev_n] for m,d in data.items()}; val={m:d.iloc[dev_n:dev_n+val_n] for m,d in data.items()}; lock={m:d.iloc[dev_n+val_n:] for m,d in data.items()}
    print(f"COMMON CUT: {n} bars | DEV={dev_n} VAL={val_n} LOCKBOX={n-dev_n-val_n}",flush=True)
    rng=np.random.default_rng(seed+17); genes=_pool(seed,initial_population); finalists=[]; evaluations=0
    for gen in range(generations):
        if time.monotonic()>=deadline: break
        ranked=[]; batch=genes[:population]
        print(f"=== GENERATION {gen+1}/{generations} ===",flush=True)
        for i,g in enumerate(batch,1):
            panel=[]
            for m in SCREEN_MARKETS:
                if m in dev:
                    e=_evaluate(dev[m].iloc[-12000:],g,settings); panel.append(e)
            ret=float(np.median([x['normal'].get('total_return',0) for x in panel])); pf=float(np.median([x['normal'].get('profit_factor',0) for x in panel])); dd=float(np.median([x['normal'].get('max_drawdown',0) for x in panel])); stress=float(np.median([x['stress'].get('total_return',0) for x in panel])); spf=float(np.median([x['stress'].get('profit_factor',0) for x in panel])); trades=int(np.median([x['normal'].get('trade_count',0) for x in panel])); wfs=[_wf(dev[m],g,settings) for m in SCREEN_MARKETS[:4] if m in dev]; wf_med=float(np.median([x['median_return'] for x in wfs])) if wfs else 0; wf_pos=int(np.median([x['positive'] for x in wfs])) if wfs else 0
            util=_utility(ret,pf,dd,stress,spf,wf_med,wf_pos,trades); ranked.append((util,g)); evaluations+=1
            print(f"eval {i}/{len(batch)} | util={util:.2f} | ret={ret:.2%} PF={pf:.2f} DD={dd:.2%} stress={stress:.2%} WF={wf_pos}",flush=True)
        ranked.sort(key=lambda x:x[0],reverse=True); elites=[g for _,g in ranked[:max(2,population//4)]]; finalists.extend(g for _,g in ranked[:max(4,population//2)])
        children=[]
        for g in elites:
            for _ in range(3):
                children.append(Genome(g.family,g.lookback,max(12,min(400,g.lookback+int(rng.choice([-48,-24,0,24,48])))),max(40,min(420,g.fast+int(rng.choice([-3,0,3])))),max(60,min(480,g.slow+int(rng.choice([-15,0,15])))),g.threshold,g.exit_threshold,g.vol_window,g.vol_cap,g.hold_bars,g.strength))
        genes=list(dict.fromkeys(elites+children))
    finalists=list(dict.fromkeys(finalists))[:12]
    print("=== VALIDATION ALL MARKETS ===",flush=True); valrows=[]
    for rank,g in enumerate(finalists,1):
        if time.monotonic()>=deadline: break
        es=[_evaluate(val[m],g,settings) for m in data]
        ret=float(np.median([e['normal'].get('total_return',0) for e in es])); stress=float(np.median([e['stress'].get('total_return',0) for e in es])); pf=float(np.median([e['normal'].get('profit_factor',0) for e in es])); spf=float(np.median([e['stress'].get('profit_factor',0) for e in es])); pos=sum(e['normal'].get('total_return',0)>0 and e['stress'].get('total_return',0)>0 and e['normal'].get('profit_factor',0)>1 and e['stress'].get('profit_factor',0)>1 for e in es)
        row={"rank":rank,"genome":asdict(g),"median_return":ret,"median_stress_return":stress,"median_pf":pf,"median_stress_pf":spf,"positive_markets":pos}; valrows.append(row); print(f"VALIDATION {rank}/{len(finalists)} | ret={ret:.2%} PF={pf:.2f} stress={stress:.2%} positive={pos}/{len(es)}",flush=True)
    valrows.sort(key=lambda x:(x['positive_markets'],x['median_stress_return'],x['median_return']),reverse=True)
    lockrows=[]; print("=== LOCKBOX (TOP 3 ONLY) ===",flush=True)
    for rank,row in enumerate(valrows[:3],1):
        g=Genome(**row['genome']); es=[_evaluate(lock[m],g,settings) for m in data]; ret=float(np.median([e['normal'].get('total_return',0) for e in es])); stress=float(np.median([e['stress'].get('total_return',0) for e in es])); pf=float(np.median([e['normal'].get('profit_factor',0) for e in es])); spf=float(np.median([e['stress'].get('profit_factor',0) for e in es])); pos=sum(e['normal'].get('total_return',0)>0 and e['stress'].get('total_return',0)>0 and e['normal'].get('profit_factor',0)>1 and e['stress'].get('profit_factor',0)>1 for e in es); lr={"rank":rank,"genome":asdict(g),"median_return":ret,"median_stress_return":stress,"median_pf":pf,"median_stress_pf":spf,"positive_markets":pos}; lockrows.append(lr); print(f"LOCKBOX {rank}/3 | ret={ret:.2%} PF={pf:.2f} stress={stress:.2%} positive={pos}/{len(es)}",flush=True)
    eligible=[x for x in lockrows if x['positive_markets']>=6 and x['median_return']>0 and x['median_stress_return']>0 and x['median_pf']>1.05 and x['median_stress_pf']>1.0]
    decision="VALIDATED_STRATEGY_READY" if eligible else "NO_VALIDATED_STRATEGY"
    payload={"started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"duration_minutes":(datetime.now(timezone.utc)-started).total_seconds()/60,"version":"v2","decision":decision,"evaluations":evaluations,"common_bars":n,"split":{"development":dev_n,"validation":val_n,"lockbox":n-dev_n-val_n},"finalists":[asdict(g) for g in finalists],"validation":valrows,"lockbox":lockrows,"eligible":eligible,"protocol":{"spot_long_flat":True,"ai":False,"futures":False,"non_overlapping_wf":True,"lockbox_reserved":True}}
    _save(payload); print("=== PHASE 1 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True); return payload

if __name__=="__main__":
    run(minutes=float(os.getenv("DISCOVERY_MINUTES","180")),initial_population=int(os.getenv("DISCOVERY_INITIAL_POPULATION","64")),population=int(os.getenv("DISCOVERY_POPULATION","16")),generations=int(os.getenv("DISCOVERY_GENERATIONS","20")),seed=int(os.getenv("DISCOVERY_SEED","20260829")))
