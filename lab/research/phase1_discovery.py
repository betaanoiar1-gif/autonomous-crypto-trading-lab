from __future__ import annotations
"""Phase 1 discovery engine: typed, fast, multi-objective, cross-market, long/flat only."""
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
OUT = ROOT / "experiments" / "phase1_discovery_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")

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
    strength: float = 1.0

@dataclass
class Score:
    genome: Genome
    market: str
    ret: float
    pf: float
    dd: float
    trades: int
    sharpe: float
    turnover: float
    stress_ret: float
    stress_pf: float
    wf_median: float
    wf_positive: int

def _save(obj: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT)+".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)

def _load() -> dict[str,pd.DataFrame]:
    out={}
    for s in SYMBOLS:
        p=CACHE/f"{s.replace('/','_')}_1h.parquet"
        if p.exists():
            out[s]=pd.read_parquet(p).sort_index()
    return out

def _features(df: pd.DataFrame, g: Genome) -> pd.DataFrame:
    c=df["close"].astype(float)
    r=c.pct_change()
    f=pd.DataFrame(index=df.index)
    f["mom"]=c.pct_change(g.lookback)
    f["fast"]=c.ewm(span=g.fast, adjust=False, min_periods=g.fast).mean()
    f["slow"]=c.ewm(span=g.slow, adjust=False, min_periods=g.slow).mean()
    f["atr"]=r.rolling(g.vol_window, min_periods=g.vol_window).std()
    vol=r.rolling(g.lookback, min_periods=g.lookback).std().replace(0,np.nan)
    f["z"]=(c-c.rolling(g.lookback, min_periods=g.lookback).mean())/vol
    f["hh"]=c.rolling(g.lookback, min_periods=g.lookback).max().shift(1)
    return f

def _signal(df: pd.DataFrame, g: Genome) -> pd.Series:
    x=_features(df,g)
    if g.family=="momentum":
        raw=(x["mom"]>g.threshold)&(x["fast"]>x["slow"])
    elif g.family=="breakout":
        raw=(df["close"]>x["hh"]) & (x["atr"]<g.vol_cap)
    elif g.family=="mean_reversion":
        raw=(x["z"]<-g.threshold) & (x["atr"]<g.vol_cap)
    elif g.family=="trend_pullback":
        raw=(x["fast"]>x["slow"]) & (x["mom"]<g.exit_threshold) & (x["mom"]>-g.threshold)
    else:
        raw=(x["fast"]>x["slow"]) & (x["mom"]>g.threshold)
    raw=raw.astype(float).fillna(0.0).clip(0,1)
    pos=np.zeros(len(raw),dtype=float); active=False; age=g.hold_bars
    for i,v in enumerate(raw.to_numpy()):
        if active:
            age+=1
            if v<0.5 and age>=g.hold_bars:
                active=False
        elif v>0.5:
            active=True; age=0
        pos[i]=1.0 if active else 0.0
    return pd.Series(pos,index=df.index)

def _metrics(df:pd.DataFrame,sig:pd.Series,settings,cost_mult:float=1.0)->dict:
    r=run_ohlcv(df,sig,settings.capital.initial_usd,
        settings.execution.commission_bps*cost_mult,
        settings.execution.slippage_bps*cost_mult,
        market_type="spot",leverage=1.0,funding_rates=None)
    return dict(r.metrics)

def _wf(df:pd.DataFrame,g:Genome,settings)->tuple[float,int]:
    n=len(df)
    if n<2400: return 0.0,0
    fold=n//5; vals=[]
    for k in range(4):
        test=df.iloc[k*fold+fold:(k+2)*fold].copy()
        if len(test)<200: continue
        m=_metrics(test,_signal(test,g),settings)
        vals.append(float(m["total_return"]))
    return (float(np.median(vals)) if vals else 0.0, sum(v>0 for v in vals))

def _score(df:pd.DataFrame,g:Genome,settings,market:str)->Score:
    oos=df.iloc[-8000:].copy() if len(df)>8000 else df.copy()
    sig=_signal(oos,g); m=_metrics(oos,sig,settings); st=_metrics(oos,sig,settings,2.0)
    wfmed,wfpos=_wf(df,g,settings)
    return Score(g,market,float(m["total_return"]),float(m["profit_factor"]),float(m["max_drawdown"]),int(m["trade_count"]),float(m["sharpe"]),float(m["trade_turnover"]),float(st["total_return"]),float(st["profit_factor"]),wfmed,wfpos)

def _utility(s:Score)->float:
    return (80*s.ret + 12*max(0,s.pf-1) + 30*s.wf_median + 10*s.wf_positive + 8*max(0,s.stress_ret) - 25*max(0,-s.dd-0.08) - 12*max(0,-s.stress_ret) - 0.002*s.turnover)

def _pool(seed:int,pop:int)->list[Genome]:
    rng=np.random.default_rng(seed); fam=["momentum","breakout","mean_reversion","trend_pullback","ma_cross"]
    lbs=[24,48,72,120,168,240]; fasts=[8,12,18,24,36,48]; slows=[40,60,90,120,180,240]
    out=[]; seen=set()
    while len(out)<pop:
        f=str(rng.choice(fam)); lb=int(rng.choice(lbs)); fast=int(rng.choice(fasts)); slow=int(rng.choice(slows))
        if fast>=slow: continue
        g=Genome(f,lb,fast,slow,float(rng.choice([0.0025,0.005,0.0075,0.01])),float(rng.choice([0.0015,0.003,0.005])),int(rng.choice([12,24,48])),float(rng.choice([0.018,0.025,0.035])),int(rng.choice([24,48,72])),float(rng.choice([0.75,1.0,1.25])))
        if g in seen: continue
        seen.add(g); out.append(g)
    return out

def _mutate(g:Genome,rng:np.random.Generator)->Genome:
    new=Genome(g.family if rng.random()>0.15 else str(rng.choice(["momentum","breakout","mean_reversion","trend_pullback","ma_cross"])),
        max(12,min(300,g.lookback+int(rng.choice([-24,-12,0,12,24])))),
        max(6,min(60,g.fast+int(rng.choice([-6,-3,0,3,6])))),
        max(30,min(300,g.slow+int(rng.choice([-30,-15,0,15,30])))),
        float(np.clip(g.threshold+rng.choice([-0.0025,0,0.0025]),0.001,0.02)),
        float(np.clip(g.exit_threshold+rng.choice([-0.001,0,0.001]),0,0.01)),
        max(8,min(96,g.vol_window+int(rng.choice([-12,0,12])))),
        float(np.clip(g.vol_cap+rng.choice([-0.005,0,0.005]),0.01,0.06)),
        max(12,min(120,g.hold_bars+int(rng.choice([-12,0,12])))),
        float(np.clip(g.strength+rng.choice([-0.25,0,0.25]),0.5,1.5)))
    if new.fast>=new.slow:
        new=Genome(new.family,new.lookback,min(new.fast,new.slow-6),new.slow,new.threshold,new.exit_threshold,new.vol_window,new.vol_cap,new.hold_bars,new.strength)
    return new

def _elite(scores:list[Score],limit:int)->list[Genome]:
    ranked=sorted(scores,key=_utility,reverse=True); keep=[]; cells=set()
    for s in ranked:
        cell=(s.genome.family,min(4,int(abs(s.dd)*20)),min(4,int(max(0,s.wf_positive))))
        if cell not in cells: cells.add(cell); keep.append(s.genome)
        if len(keep)>=limit: break
    return keep

def run(minutes:float=180.0,initial_population:int=64,population:int=16,generations:int=30,seed:int=20260829)->dict:
    start=time.monotonic(); settings=load_settings(); data=_load()
    print("=== PHASE 1 DISCOVERY ENGINE ===",flush=True); print("AI: DISABLED | Futures: DISABLED | Live: DISABLED",flush=True)
    print(f"Markets cached: {len(data)}",flush=True)
    if len(data)<8:
        p={"decision":"DISCOVERY_BLOCKED_DATA","markets":list(data)}; _save(p); return p
    markets=list(data); rng=np.random.default_rng(seed); genes=_pool(seed,initial_population); finalists=[]; evaluations=0
    for gen in range(generations):
        if time.monotonic()-start>=minutes*60: break
        batch=genes[:population]; scores=[]
        for g in batch:
            per=[_score(data[m],g,settings,m) for m in markets[:6]]
            agg=Score(g,"MULTI",float(np.median([x.ret for x in per])),float(np.median([x.pf for x in per])),float(np.median([x.dd for x in per])),int(np.median([x.trades for x in per])),float(np.median([x.sharpe for x in per])),float(np.median([x.turnover for x in per])),float(np.median([x.stress_ret for x in per])),float(np.median([x.stress_pf for x in per])),float(np.median([x.wf_median for x in per])),int(np.median([x.wf_positive for x in per])))
            scores.append(agg); evaluations+=1
            print(f"GEN {gen+1}/{generations} eval {evaluations} | util={_utility(agg):.2f} | ret={agg.ret:.2%} PF={agg.pf:.2f} DD={agg.dd:.2%} WF={agg.wf_positive}",flush=True)
        finalists.extend(_elite(scores,max(4,population//2))); elites=_elite(scores,population//2)
        genes=list(dict.fromkeys(elites+[_mutate(e,rng) for e in elites for _ in range(2)]))
    finalists=list(dict.fromkeys(finalists))[:12]; deep=[]
    print("=== DEEP FINALIST + CROSS-MARKET ===",flush=True)
    for i,g in enumerate(finalists,1):
        arr=[_score(data[m],g,settings,m) for m in markets]
        med=float(np.median([_utility(x) for x in arr])); positive=sum(x.ret>0 and x.stress_ret>0 and x.pf>1 and x.stress_pf>1 and x.wf_positive>=2 for x in arr)
        deep.append({"rank":i,"genome":asdict(g),"utility":med,"positive_markets":positive,"markets":[asdict(x) for x in arr]})
        print(f"FINALIST {i}/{len(finalists)} | utility={med:.2f} | positive_markets={positive}",flush=True)
    deep.sort(key=lambda x:(x["positive_markets"],x["utility"]),reverse=True); eligible=[x for x in deep if x["positive_markets"]>=max(3,len(markets)//2)]
    decision="DISCOVERY_CANDIDATES_READY" if eligible else "NO_ROBUST_DISCOVERY"
    payload={"started_at":datetime.now(timezone.utc).isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"evaluations":evaluations,"markets":markets,"finalists":deep[:12],"eligible":eligible[:5],"protocol":{"spot_long_flat":True,"ai":False,"futures":False,"lockbox_reserved":True,"screen":"median cross-market utility","final":"all-market median + stress + walk-forward"}}
    _save(payload); print("=== PHASE 1 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True); return payload

if __name__=="__main__":
    run(minutes=float(os.getenv("DISCOVERY_MINUTES","180")),initial_population=int(os.getenv("DISCOVERY_INITIAL_POPULATION","64")),population=int(os.getenv("DISCOVERY_POPULATION","16")),generations=int(os.getenv("DISCOVERY_GENERATIONS","30")),seed=int(os.getenv("DISCOVERY_SEED","20260829")))
