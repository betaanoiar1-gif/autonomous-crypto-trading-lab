from __future__ import annotations
import json, os, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

SYMBOLS = ["BTCUSDT","ETHUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT"]
BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
OUT = Path("experiments/phase10e_flow_reversal")

def _end_day() -> datetime:
    n=datetime.now(timezone.utc); return datetime(n.year,n.month,n.day,tzinfo=timezone.utc)-timedelta(days=1)

def _days(n:int)->list[str]:
    e=_end_day(); return [(e-timedelta(days=i)).date().isoformat() for i in range(n-1,-1,-1)]

def _fetch_one(symbol:str, day:str, out:Path)->pd.DataFrame:
    out.mkdir(parents=True,exist_ok=True); p=out/f"{symbol}_{day}.csv"
    if p.exists(): return pd.read_csv(p)
    url=f"{BASE}/{symbol}/{symbol}-aggTrades-{day}.zip"; t=time.perf_counter()
    r=requests.get(url,timeout=90,headers={"User-Agent":"phase10e-research/1.0"})
    print(f"DOWNLOAD {symbol} {day} | status={r.status_code} | latency={(time.perf_counter()-t)*1000:.1f}ms | bytes={len(r.content)}",flush=True)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        with z.open(names[0]) as fh: df=pd.read_csv(fh,header=None)
    df=df.iloc[:,:7].copy(); df.columns=["agg_id","price","qty","first_id","last_id","timestamp","buyer_maker"]
    for c in ("price","qty","timestamp"): df[c]=pd.to_numeric(df[c],errors="coerce")
    df["buyer_maker"]=df["buyer_maker"].astype(bool)
    df=df.dropna(subset=["agg_id","price","qty","timestamp"]).drop_duplicates("agg_id")
    df.to_csv(p,index=False); return df

def _bars(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy(); unit='us' if float(x.timestamp.median())>10_000_000_000_000 else 'ms'
    x['ts']=pd.to_datetime(x.timestamp,unit=unit,utc=True); x['notional']=x.price*x.qty
    x['flow']=np.where(x.buyer_maker,-x.notional,x.notional); x=x.set_index('ts')
    b=x.resample('1min').agg(close=('price','last'),notional=('notional','sum'),flow=('flow','sum'),trades=('agg_id','count')).dropna(subset=['close'])
    b['imbalance']=b.flow/b.notional.replace(0,np.nan)
    roll=b.flow.rolling(60,min_periods=30); b['flow_z']=(b.flow-roll.mean())/roll.std().replace(0,np.nan)
    b['flow_5m']=b.flow_z.rolling(5,min_periods=5).mean(); b['ret_1m']=b.close.pct_change()
    return b.replace([np.inf,-np.inf],np.nan)

def _corr(a:pd.Series,b:pd.Series,min_n:int=500)->float:
    z=pd.concat([a,b],axis=1).dropna()
    if len(z)<min_n or z.iloc[:,0].std()==0 or z.iloc[:,1].std()==0: return float('nan')
    return float(z.iloc[:,0].corr(z.iloc[:,1]))

def _resid(y:pd.Series, m:pd.Series)->pd.Series:
    z=pd.concat([y,m],axis=1).dropna()
    if len(z)<200: return pd.Series(index=y.index,dtype=float)
    X=np.column_stack([np.ones(len(z)),z.iloc[:,1].to_numpy()]); beta=np.linalg.lstsq(X,z.iloc[:,0].to_numpy(),rcond=None)[0]
    r=pd.Series(index=z.index,data=z.iloc[:,0].to_numpy()-X@beta); return r.reindex(y.index)

def run(minutes:float=15.0,seed:int=20260829,days:int=14)->dict[str,Any]:
    del seed; t0=time.perf_counter(); ds=_days(max(7,min(days,14))); rawdir=OUT/'raw_daily'; barsdir=OUT/'bars_1m'; rawdir.mkdir(parents=True,exist_ok=True); barsdir.mkdir(parents=True,exist_ok=True)
    print('=== PHASE 10E SPOT FLOW REVERSAL ===',flush=True); print('RESEARCH ONLY | NO MODELING | NO TRADING',flush=True); print(f"PERIOD: {ds[0]} → {ds[-1]} | days={len(ds)}",flush=True); print(f"SYMBOLS: {len(SYMBOLS)} | {SYMBOLS}",flush=True)
    frames={}; errors=[]
    def load(s):
        parts=[]
        for d in ds:
            try: parts.append(_bars(_fetch_one(s,d,rawdir)))
            except Exception as e: errors.append({'symbol':s,'day':d,'error':repr(e)})
        return pd.concat(parts).sort_index() if parts else pd.DataFrame()
    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as ex:
        fs={ex.submit(load,s):s for s in SYMBOLS}
        for f in as_completed(fs):
            s=fs[f]; frames[s]=f.result();
            if not frames[s].empty: frames[s].to_parquet(barsdir/f'{s}_1m.parquet')
            print(f"READY {s} | minutes={len(frames[s])}",flush=True)
    if len(frames)<3: raise RuntimeError('Insufficient markets loaded')
    btc=frames.get('BTCUSDT',pd.DataFrame()); market=btc['ret_1m'] if not btc.empty else pd.Series(dtype=float)
    rows=[]
    for s,b in frames.items():
        for h in (60,360):
            fut=b.close.shift(-h)/b.close-1
            resid=_resid(fut,market) if not market.empty else fut
            direct=_corr(b.flow_5m,fut)
            reverse=_corr(-b.flow_5m,fut)
            rres=_corr(-b.flow_5m,resid)
            rows.append({'symbol':s,'horizon':h,'direct_ic':direct,'reverse_ic':reverse,'residual_reverse_ic':rres,'flow_ac1':float(b.imbalance.dropna().autocorr(1)),'flow_ac5':float(b.imbalance.dropna().autocorr(5)),'observations':int(pd.concat([b.flow_5m,fut],axis=1).dropna().shape[0])})
            print(f"SCREEN {s} h={h} | direct={direct:.4f} reverse={reverse:.4f} residual_reverse={rres:.4f} obs={rows[-1]['observations']}",flush=True)
    long=[r for r in rows if np.isfinite(r['residual_reverse_ic'])]
    consistent=sum(r['residual_reverse_ic']>0 for r in long)
    mean_res=float(np.mean([r['residual_reverse_ic'] for r in long])) if long else float('nan')
    med_res=float(np.median([r['residual_reverse_ic'] for r in long])) if long else float('nan')
    decision='PHASE10E_REVERSAL_RESEARCH_GATE' if len(frames)>=3 else 'PHASE10E_INSUFFICIENT_DATA'
    result={'version':'phase10e_spot_flow_reversal','created_at_utc':datetime.now(timezone.utc).isoformat(),'period':{'start':ds[0],'end':ds[-1],'days':len(ds)},'symbols':list(frames),'screens':rows,'summary':{'tests':len(long),'positive_residual_reverse':consistent,'mean_residual_reverse_ic':mean_res,'median_residual_reverse_ic':med_res},'errors':errors,'decision':decision}
    cp=OUT/'phase10e_spot_flow_reversal_latest.json'; cp.write_text(json.dumps(result,indent=2,allow_nan=True),encoding='utf-8')
    print('=== PHASE 10E COMPLETE ===',flush=True); print('DECISION:',decision,flush=True); print('TESTS:',len(rows),flush=True); print('POSITIVE RESIDUAL REVERSE:',consistent,flush=True); print('CHECKPOINT:',cp,flush=True); print('ELAPSED_SEC:',round(time.perf_counter()-t0,2),flush=True); return result

if __name__=='__main__': run()
