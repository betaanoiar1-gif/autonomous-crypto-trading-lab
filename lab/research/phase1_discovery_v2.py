from __future__ import annotations

"""Phase 1 V2: robust cross-market discovery with explicit development, validation and lockbox splits."""
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
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol.replace('/','_')}_1h.parquet"
        if path.exists():
            out[symbol] = pd.read_parquet(path).sort_index()
    return out


def _common_cut(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], str]:
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    return ({k: v.loc[(v.index >= start) & (v.index <= end)].copy() for k, v in data.items()}, str(end))


def _signal(df: pd.DataFrame, g: Genome) -> pd.Series:
    c = df["close"].astype(float)
    r = c.pct_change()
    mom = c.pct_change(g.lookback)
    fast = c.ewm(span=g.fast, adjust=False, min_periods=g.fast).mean()
    slow = c.ewm(span=g.slow, adjust=False, min_periods=g.slow).mean()
    vol = r.rolling(g.vol_window, min_periods=g.vol_window).std()
    mean = c.rolling(g.lookback, min_periods=g.lookback).mean()
    std = c.rolling(g.lookback, min_periods=g.lookback).std().replace(0.0, np.nan)
    z = (c - mean) / std
    hh = c.rolling(g.lookback, min_periods=g.lookback).max().shift(1)

    if g.family == "momentum":
        raw = (mom > g.threshold) & (fast > slow) & (vol < g.vol_cap)
    elif g.family == "breakout":
        raw = (c > hh) & (vol < g.vol_cap)
    elif g.family == "mean_reversion":
        raw = (z < -abs(g.threshold)) & (vol < g.vol_cap)
    elif g.family == "trend_pullback":
        raw = (fast > slow) & (mom > -abs(g.threshold)) & (mom < g.exit_threshold)
    elif g.family == "vol_breakout":
        raw = (c > hh) & (vol > max(0.001, g.vol_cap * 0.35))
    else:
        raw = (fast > slow) & (mom > g.threshold)

    raw = raw.astype(float).fillna(0.0).clip(0.0, 1.0)
    pos = np.zeros(len(raw), dtype=float)
    active = False
    age = g.hold_bars
    for i, value in enumerate(raw.to_numpy()):
        if active:
            age += 1
            if value < 0.5 and age >= g.hold_bars:
                active = False
        elif value > 0.5:
            active = True
            age = 0
        pos[i] = 1.0 if active else 0.0
    return pd.Series(pos, index=df.index)


def _evaluate(df: pd.DataFrame, g: Genome, settings) -> dict:
    sig = _signal(df, g)
    normal = run_ohlcv(df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps, settings.execution.slippage_bps,
        market_type="spot", leverage=1.0, funding_rates=None).metrics
    stress = run_ohlcv(df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps * 2.0, settings.execution.slippage_bps * 2.0,
        market_type="spot", leverage=1.0, funding_rates=None).metrics
    return {"normal": normal, "stress": stress}


def _wf(df: pd.DataFrame, g: Genome, settings, folds: int = 5) -> dict:
    n = len(df)
    if n < folds * 500:
        return {"median_return": 0.0, "positive": 0, "folds": []}
    fold = n // folds
    vals, rows = [], []
    for k in range(folds):
        lo, hi = k * fold, (k + 1) * fold if k < folds - 1 else n
        m = _evaluate(df.iloc[lo:hi].copy(), g, settings)["normal"]
        ret = float(m.get("total_return", 0.0)); pf = float(m.get("profit_factor", 0.0))
        vals.append(ret); rows.append({"fold": k + 1, "ret": ret, "pf": pf})
    return {"median_return": float(np.median(vals)), "positive": int(sum(x > 0 for x in vals)), "folds": rows}


def _utility(ret, pf, dd, stress_ret, stress_pf, wf_med, wf_pos, trades) -> float:
    return (90 * ret + 18 * max(0, pf - 1) + 40 * wf_med + 8 * wf_pos
            + 18 * max(0, stress_ret) + 8 * max(0, stress_pf - 1)
            - 30 * max(0, -dd - 0.10) - 20 * max(0, -stress_ret) - 0.003 * trades)


def _pool(seed: int, count: int) -> list[Genome]:
    rng = np.random.default_rng(seed)
    families = ["momentum", "breakout", "mean_reversion", "trend_pullback", "vol_breakout", "ma_cross"]
    out, seen = [], set()
    while len(out) < count:
        fast, slow = int(rng.choice([6,8,12,18,24,36,48])), int(rng.choice([40,60,90,120,180,240,360]))
        if fast >= slow: continue
        g = Genome(str(rng.choice(families)), int(rng.choice([24,48,72,120,168,240,336])), fast, slow,
                   float(rng.choice([.0025,.005,.0075,.01,.015])), float(rng.choice([.0015,.003,.005,.0075,.01])),
                   int(rng.choice([12,24,36,48,72])), float(rng.choice([.012,.018,.025,.035,.05])),
                   int(rng.choice([12,24,48,72,96])), float(rng.choice([.75,1.0,1.25])))
        if g not in seen: seen.add(g); out.append(g)
    return out


def _mutate(g: Genome, rng: np.random.Generator) -> Genome:
    fast = max(6, min(72, g.fast + int(rng.choice([-6,-3,0,3,6]))))
    slow = max(40, min(360, g.slow + int(rng.choice([-30,-15,0,15,30]))))
    if fast >= slow: fast = max(6, slow - 12)
    return Genome(
        family=g.family if rng.random() > 0.12 else str(rng.choice(["momentum","breakout","mean_reversion","trend_pullback","vol_breakout","ma_cross"])),
        lookback=max(18, min(400, g.lookback + int(rng.choice([-48,-24,0,24,48])))),
        fast=fast,
        slow=slow,
        threshold=float(np.clip(g.threshold + rng.choice([-.0025,0,.0025]), .0015, .025)),
        exit_threshold=float(np.clip(g.exit_threshold + rng.choice([-.001,0,.001]), .0005, .012)),
        vol_window=max(8, min(96, g.vol_window + int(rng.choice([-12,0,12])))),
        vol_cap=float(np.clip(g.vol_cap + rng.choice([-.005,0,.005]), .01, .06)),
        hold_bars=max(12, min(144, g.hold_bars + int(rng.choice([-12,0,12])))),
        strength=float(np.clip(g.strength + rng.choice([-.25,0,.25]), .5, 1.5)),
    )


def _screen_one(data, genome, settings):
    panel = [_evaluate(data[m].iloc[-12000:], genome, settings) for m in SCREEN_MARKETS if m in data]
    ret = float(np.median([x["normal"].get("total_return", 0.0) for x in panel]))
    pf = float(np.median([x["normal"].get("profit_factor", 0.0) for x in panel]))
    dd = float(np.median([x["normal"].get("max_drawdown", 0.0) for x in panel]))
    stress_ret = float(np.median([x["stress"].get("total_return", 0.0) for x in panel]))
    stress_pf = float(np.median([x["stress"].get("profit_factor", 0.0) for x in panel]))
    trades = int(np.median([x["normal"].get("trade_count", 0) for x in panel]))
    wfs = [_wf(data[m], genome, settings) for m in SCREEN_MARKETS[:4] if m in data]
    wf_med = float(np.median([x["median_return"] for x in wfs])) if wfs else 0.0
    wf_pos = int(np.median([x["positive"] for x in wfs])) if wfs else 0
    return {"ret": ret, "pf": pf, "dd": dd, "stress_ret": stress_ret, "stress_pf": stress_pf,
            "trades": trades, "wf_med": wf_med, "wf_pos": wf_pos,
            "utility": _utility(ret,pf,dd,stress_ret,stress_pf,wf_med,wf_pos,trades)}


def run(minutes: float = 180.0, initial_population: int = 64, population: int = 16,
        generations: int = 20, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc); deadline = time.monotonic() + minutes * 60
    settings = load_settings(); raw = _load()
    print("=== PHASE 1 DISCOVERY V2 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"Cache: {CACHE}", flush=True)
    print(f"Markets loaded: {len(raw)}", flush=True)
    if len(raw) < 8:
        payload={"decision":"DISCOVERY_BLOCKED_DATA","markets":list(raw),"version":"v2"}; _save(payload); return payload

    data, common_end = _common_cut(raw)
    n = min(map(len, data.values())); dev_n=int(n*.70); val_n=int(n*.15); lock_n=n-dev_n-val_n
    dev={m:d.iloc[:dev_n] for m,d in data.items()}; val={m:d.iloc[dev_n:dev_n+val_n] for m,d in data.items()}; lock={m:d.iloc[dev_n+val_n:] for m,d in data.items()}
    print(f"COMMON CUTOFF: {common_end}", flush=True)
    print(f"SPLIT: DEV={dev_n} | VALIDATION={val_n} | LOCKBOX={lock_n}", flush=True)
    print("LOCKBOX: RESERVED", flush=True)

    rng=np.random.default_rng(seed+17); genes=_pool(seed,initial_population); finalists=[]; evaluations=0; best_utility=-float("inf")
    for gen in range(generations):
        if time.monotonic() >= deadline: break
        batch=genes[:population]; ranked=[]
        print(f"=== GENERATION {gen+1}/{generations} ===", flush=True)
        for i,g in enumerate(batch,1):
            if time.monotonic() >= deadline: break
            s=_screen_one(dev,g,settings); ranked.append((s["utility"],g)); evaluations+=1; best_utility=max(best_utility,s["utility"])
            print(f"eval {i}/{len(batch)} | util={s['utility']:.2f} | ret={s['ret']:.2%} PF={s['pf']:.2f} DD={s['dd']:.2%} stress={s['stress_ret']:.2%} WF={s['wf_pos']}",flush=True)
        ranked.sort(key=lambda x:x[0], reverse=True)
        elites=[g for _,g in ranked[:max(2,population//4)]]
        finalists.extend(g for _,g in ranked[:max(4,population//2)])
        children=[_mutate(g,rng) for g in elites for _ in range(3)]
        genes=list(dict.fromkeys(elites+children))

    finalists=list(dict.fromkeys(finalists))[:12]
    print("=== VALIDATION ALL MARKETS ===", flush=True); validation=[]
    for rank,g in enumerate(finalists,1):
        if time.monotonic() >= deadline: break
        rows=[_evaluate(val[m],g,settings) for m in data]
        ret=float(np.median([x['normal'].get('total_return',0) for x in rows])); stress=float(np.median([x['stress'].get('total_return',0) for x in rows])); pf=float(np.median([x['normal'].get('profit_factor',0) for x in rows])); spf=float(np.median([x['stress'].get('profit_factor',0) for x in rows])); dd=float(np.median([x['normal'].get('max_drawdown',0) for x in rows])); pos=sum(x['normal'].get('total_return',0)>0 and x['stress'].get('total_return',0)>0 and x['normal'].get('profit_factor',0)>1 and x['stress'].get('profit_factor',0)>1 for x in rows)
        r={"rank":rank,"genome":asdict(g),"median_return":ret,"median_stress_return":stress,"median_pf":pf,"median_stress_pf":spf,"median_dd":dd,"positive_markets":pos}; validation.append(r)
        print(f"VALIDATION {rank}/{len(finalists)} | ret={ret:.2%} PF={pf:.2f} stress={stress:.2%} DD={dd:.2%} positive={pos}/{len(rows)}",flush=True)
    validation.sort(key=lambda x:(x['positive_markets'],x['median_stress_return'],x['median_return']),reverse=True)

    lockbox=[]; print("=== LOCKBOX TOP 3 ===",flush=True)
    for rank,item in enumerate(validation[:3],1):
        g=Genome(**item['genome']); rows=[_evaluate(lock[m],g,settings) for m in data]
        ret=float(np.median([x['normal'].get('total_return',0) for x in rows])); stress=float(np.median([x['stress'].get('total_return',0) for x in rows])); pf=float(np.median([x['normal'].get('profit_factor',0) for x in rows])); spf=float(np.median([x['stress'].get('profit_factor',0) for x in rows])); pos=sum(x['normal'].get('total_return',0)>0 and x['stress'].get('total_return',0)>0 and x['normal'].get('profit_factor',0)>1 and x['stress'].get('profit_factor',0)>1 for x in rows)
        r={"rank":rank,"genome":asdict(g),"median_return":ret,"median_stress_return":stress,"median_pf":pf,"median_stress_pf":spf,"positive_markets":pos}; lockbox.append(r)
        print(f"LOCKBOX {rank}/3 | ret={ret:.2%} PF={pf:.2f} stress={stress:.2%} positive={pos}/{len(rows)}",flush=True)

    eligible=[x for x in lockbox if x['positive_markets']>=6 and x['median_return']>0 and x['median_stress_return']>0 and x['median_pf']>1.05 and x['median_stress_pf']>1.0]
    decision="VALIDATED_STRATEGY_READY" if eligible else "NO_VALIDATED_STRATEGY"
    payload={"started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"duration_minutes":(datetime.now(timezone.utc)-started).total_seconds()/60,"version":"v2","decision":decision,"evaluations":evaluations,"best_screen_utility":best_utility,"common_cutoff":common_end,"split":{"development":dev_n,"validation":val_n,"lockbox":lock_n},"finalists":[asdict(g) for g in finalists],"validation":validation,"lockbox":lockbox,"eligible":eligible,"protocol":{"spot_long_flat":True,"ai":False,"futures":False,"non_overlapping_wf":True,"lockbox_reserved":True}}
    _save(payload); print("=== PHASE 1 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True); return payload

if __name__=="__main__":
    run(minutes=float(os.getenv("DISCOVERY_MINUTES","180")),initial_population=int(os.getenv("DISCOVERY_INITIAL_POPULATION","64")),population=int(os.getenv("DISCOVERY_POPULATION","16")),generations=int(os.getenv("DISCOVERY_GENERATIONS","20")),seed=int(os.getenv("DISCOVERY_SEED","20260829")))
