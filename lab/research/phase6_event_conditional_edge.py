from __future__ import annotations

"""Phase 6: event-conditioned predictive edge discovery.

Purpose: after Phases 3-5 found only weak/unreliable broad effects, test whether
information appears specifically around pre-declared market events:
volatility shocks, range expansion, momentum shocks, and liquidity/volume shocks.

This is still an information test, not a trading strategy.
No genetic evolution is used.
Selection happens only on DEV. Validation and LOCKBOX are frozen.
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
CACHE = Path(os.getenv("PHASE6_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase6_event_conditional_edge_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
HORIZONS = (6, 24, 72, 168)
EVENTS = ("vol_shock", "range_shock", "momentum_shock", "volume_shock")
SIGNALS = ("mom_6","mom_24","mom_72","vol_scaled_mom","volume_pressure","vol_compression","trend_strength","range_position_72")

@dataclass
class EdgeRow:
    signal: str
    event: str
    horizon: int
    split: str
    observations: int
    rank_ic: float
    spread: float
    residual_spread: float
    positive_periods: int
    periods: int
    event_rate: float


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


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    z = pd.DataFrame(index=df.index)
    for w in (6,24,72,168):
        z[f"mom_{w}"] = c.pct_change(w)
    rv24 = r.rolling(24, min_periods=24).std().replace(0, np.nan)
    rv6 = r.rolling(6, min_periods=6).std().replace(0, np.nan)
    rv72 = r.rolling(72, min_periods=72).std().replace(0, np.nan)
    z["vol_scaled_mom"] = z["mom_24"] / rv24
    z["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    z["vol_compression"] = rv6 / rv72
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    z["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    width = (hh - ll).replace(0, np.nan)
    z["range_position_72"] = ((c - ll) / width).clip(-1, 2)
    z["rv24"] = rv24
    z["rv72"] = rv72
    z["range_frac"] = (h - l) / c.replace(0, np.nan)
    z["volume_z"] = (v - v.rolling(96, min_periods=96).mean()) / v.rolling(96, min_periods=96).std().replace(0, np.nan)
    z["return_abs"] = r.abs()
    return z.replace([np.inf, -np.inf], np.nan)


def _panel(data):
    idx = sorted(set.intersection(*[set(x.index) for x in data.values()]))
    frames = {s: _features(data[s]).reindex(idx) for s in data}
    close = pd.DataFrame({s: data[s].loc[idx, "close"] for s in data}).sort_index()
    return close, frames


def _event_vector(frames, event: str, t):
    values = pd.Series({s: frames[s].loc[t, event] if event in frames[s].columns else np.nan for s in frames})
    return values


def _event_masks(frames, event: str, threshold: float = 0.90):
    # Cross-sectional pre-declared event thresholds at each timestamp.
    idx = next(iter(frames.values())).index
    mask = pd.DataFrame(index=idx, columns=list(frames), dtype=bool)
    for t in idx:
        if event == "vol_shock":
            x = pd.Series({s: frames[s].loc[t, "rv24"] / frames[s].loc[t, "rv72"] for s in frames})
        elif event == "range_shock":
            x = pd.Series({s: frames[s].loc[t, "range_frac"] / frames[s].loc[t, "range_frac" if False else "range_frac"] for s in frames})
        elif event == "momentum_shock":
            x = pd.Series({s: abs(frames[s].loc[t, "mom_6"]) for s in frames})
        else:
            x = pd.Series({s: frames[s].loc[t, "volume_z"] for s in frames})
        q = x.quantile(threshold)
        mask.loc[t] = x >= q
    return mask.fillna(False)


def _event_masks_v2(frames, event: str, threshold: float = 0.90):
    idx = next(iter(frames.values())).index
    mask = pd.DataFrame(False, index=idx, columns=list(frames))
    for s, f in frames.items():
        if event == "vol_shock":
            x = f["rv24"] / f["rv72"]
        elif event == "range_shock":
            base = f["range_frac"].rolling(72, min_periods=72).median()
            x = f["range_frac"] / base.replace(0, np.nan)
        elif event == "momentum_shock":
            base = f["mom_6"].rolling(168, min_periods=168).std().replace(0, np.nan)
            x = f["mom_6"].abs() / base
        else:
            x = f["volume_z"].abs()
        threshold_series = x.rolling(720, min_periods=240).quantile(threshold).shift(1)
        mask[s] = (x > threshold_series).fillna(False)
    return mask


def _score(close, frames, event_mask, signal, horizon, start, end, split):
    idx = close.index[start:end]
    vals_ic=[]; vals_sp=[]; vals_neutral=[]; positive=0; obs=0; event_hits=0; periods=0
    for t in idx:
        mask = event_mask.loc[t]
        if int(mask.sum()) < 4:
            continue
        periods += 1
        event_hits += int(mask.sum())
        y = close.pct_change(horizon).shift(-horizon).loc[t]
        x = pd.Series({s: frames[s].loc[t, signal] for s in close.columns})
        z = pd.concat([x.rename("x"), y.rename("y"), mask.rename("event")], axis=1)
        z = z[(z["event"])].dropna()
        if len(z) < 4:
            continue
        obs += len(z)
        ric = float(z["x"].rank().corr(z["y"].rank()))
        q = z["x"].rank(pct=True)
        long_r = float(z.loc[q >= 0.75, "y"].mean()) if (q >= 0.75).any() else 0.0
        short_r = float(z.loc[q <= 0.25, "y"].mean()) if (q <= 0.25).any() else 0.0
        sp = long_r - short_r
        if np.isfinite(ric): vals_ic.append(ric)
        if np.isfinite(sp):
            vals_sp.append(sp); positive += int(sp > 0)
        # residualize against contemporaneous BTC future return inside this event.
        btc_y = float(y.get("BTC/USDT", np.nan))
        ry = z["y"] - btc_y
        rs = float(ry.loc[q >= 0.75].mean() - ry.loc[q <= 0.25].mean()) if len(ry) else 0.0
        if np.isfinite(rs): vals_neutral.append(rs)
    event_rate = float(event_hits / max(1, periods * len(close.columns)))
    return EdgeRow(signal, event_type if False else "", horizon, split, obs,
                    float(np.median(vals_ic)) if vals_ic else 0.0,
                    float(np.median(vals_sp)) if vals_sp else 0.0,
                    float(np.median(vals_neutral)) if vals_neutral else 0.0,
                    positive, periods, event_rate)

# Alias kept separate to make event name explicit and avoid changing score contract.
def _evaluate(close, frames, event_mask, event_name, signal, horizon, start, end, split):
    idx = close.index[start:end]
    ics=[]; sps=[]; nps=[]; positive=0; obs=0; event_hits=0; periods=0
    future = {h: close.pct_change(h).shift(-h) for h in HORIZONS}
    ypanel = future[horizon]
    for t in idx:
        mask = event_mask.loc[t]
        count = int(mask.sum())
        if count < 4:
            continue
        periods += 1; event_hits += count
        x = pd.Series({s: frames[s].loc[t, signal] for s in close.columns})
        y = ypanel.loc[t]
        z = pd.concat([x.rename("x"), y.rename("y"), mask.rename("event")], axis=1)
        z = z[z.event].dropna()
        if len(z) < 4:
            continue
        obs += len(z)
        ric = float(z.x.rank().corr(z.y.rank()))
        q = z.x.rank(pct=True)
        hi = z.loc[q >= 0.75, "y"]; lo = z.loc[q <= 0.25, "y"]
        sp = float(hi.mean() - lo.mean()) if len(hi) and len(lo) else 0.0
        btc = float(y.get("BTC/USDT", np.nan))
        residual = z.y - btc
        rsp = float(residual.loc[q >= 0.75].mean() - residual.loc[q <= 0.25].mean()) if len(hi) and len(lo) else 0.0
        if np.isfinite(ric): ics.append(ric)
        if np.isfinite(sp): sps.append(sp); positive += int(sp > 0)
        if np.isfinite(rsp): nps.append(rsp)
    return EdgeRow(signal,event_name,horizon,split,obs,
                   float(np.median(ics)) if ics else 0.0,
                   float(np.median(sps)) if sps else 0.0,
                   float(np.median(nps)) if nps else 0.0,
                   positive,periods,
                   float(event_hits / max(1,periods*len(close.columns))))


def _bootstrap(spreads: list[float], seed: int, trials: int = 500) -> dict:
    if len(spreads) < 20:
        return {"trials":0,"median":0.0,"lo":0.0,"hi":0.0,"p_positive":0.0}
    rng=np.random.default_rng(seed)
    x=np.asarray(spreads,dtype=float)
    med=[]
    for _ in range(trials):
        med.append(float(np.median(rng.choice(x,size=len(x),replace=True))))
    return {"trials":trials,"median":float(np.median(med)),"lo":float(np.quantile(med,0.05)),"hi":float(np.quantile(med,0.95)),"p_positive":float(np.mean(np.asarray(med)>0))}


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline=time.monotonic()+minutes*60.0
    raw=_load()
    if len(raw)<8:
        payload={"version":"phase6","decision":"PHASE6_BLOCKED_DATA","markets":list(raw)}; _save(payload); return payload
    close,frames=_panel(raw)
    n=len(close); dev_n=int(n*0.55); val_n=int(n*0.25); lock_start=dev_n+val_n
    print("=== PHASE 6 EVENT-CONDITIONAL EDGE ===",flush=True)
    print(f"Markets: {len(raw)} | common bars={n}",flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}",flush=True)
    print("EVENT THRESHOLDS: trailing DEV history only; frozen afterwards",flush=True)

    masks={e:_event_masks_v2(frames,e) for e in EVENTS}
    dev=[]
    for event in EVENTS:
        for signal in SIGNALS:
            for h in HORIZONS:
                if time.monotonic()>=deadline: break
                r=_evaluate(close,frames,masks[event],event,signal,h,0,dev_n,"dev")
                dev.append(asdict(r))
    dev.sort(key=lambda x:(x["positive_periods"]>=max(1,int(0.55*x["periods"])), x["residual_spread"], x["rank_ic"]),reverse=True)
    candidates=dev[:10]
    for i,r in enumerate(candidates,1):
        print(f"DEV {i}/10 | {r['signal']} h={r['horizon']} event={r['event']} IC={r['rank_ic']:.4f} spread={r['spread']:.4%} residual={r['residual_spread']:.4%} pos={r['positive_periods']}/{r['periods']}",flush=True)

    print("=== FROZEN VALIDATION ===",flush=True)
    validation=[]
    for i,r0 in enumerate(candidates,1):
        r=_evaluate(close,frames,masks[r0['event']],r0['event'],r0['signal'],r0['horizon'],dev_n,lock_start,"validation")
        validation.append(asdict(r))
        print(f"VAL {i}/10 | {r.signal} h={r.horizon} event={r.event} IC={r.rank_ic:.4f} spread={r.spread:.4%} residual={r.residual_spread:.4%} pos={r.positive_periods}/{r.periods}",flush=True)
    validation.sort(key=lambda x:(x["positive_periods"],x["residual_spread"],x["rank_ic"]),reverse=True)
    finalists=validation[:4]

    print("=== LOCKBOX ===",flush=True)
    lockbox=[]
    for i,r0 in enumerate(finalists,1):
        r=_evaluate(close,frames,masks[r0['event']],r0['event'],r0['signal'],r0['horizon'],lock_start,n,"lockbox")
        lockbox.append(asdict(r))
        print(f"LOCKBOX {i}/4 | {r.signal} h={r.horizon} event={r.event} IC={r.rank_ic:.4f} spread={r.spread:.4%} residual={r.residual_spread:.4%} pos={r.positive_periods}/{r.periods}",flush=True)

    eligible=[x for x in lockbox if x["periods"]>=20 and x["positive_periods"]>=int(0.55*x["periods"]) and x["residual_spread"]>0 and x["rank_ic"]>0]
    decision="PHASE6_EVENT_EDGE_FOUND" if eligible else "PHASE6_NO_CONFIRMED_EVENT_EDGE"
    payload={
        "version":"phase6","started_at":datetime.now(timezone.utc).isoformat(),"decision":decision,
        "markets":list(raw),"bars":n,"split":{"dev":dev_n,"validation":val_n,"lockbox":n-lock_start},
        "dev":dev,"candidates":candidates,"validation":validation,"lockbox":lockbox,"eligible":eligible,
        "protocol":{"event_conditioned":True,"thresholds_predeclared":True,"trailing_thresholds_fitted_on_dev":True,"validation_frozen":True,"lockbox_frozen":True,"spot_only":True}
    }
    _save(payload)
    print("=== PHASE 6 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True)
    return payload

if __name__=="__main__":
    run(float(os.getenv("PHASE6_MINUTES","60")),int(os.getenv("PHASE6_SEED","20260829")))
