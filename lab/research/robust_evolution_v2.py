from __future__ import annotations

"""AI-free robust evolutionary search with lazy market loading and checkpoints."""

import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics, _run

FAMILIES = ("momentum", "breakout", "trend_pullback", "mean_reversion", "moving_average_cross", "rsi_reversion", "atr_breakout", "channel_reversion")
SEEDS = {
    "momentum": [{"lookback": x} for x in (20, 40, 60, 100, 140, 180, 220)],
    "breakout": [{"lookback": x} for x in (10, 20, 40, 60, 100, 140, 180)],
    "trend_pullback": [{"lookback": x, "pullback_threshold": y} for x in (8, 12, 20, 40, 60) for y in (0.0035, 0.005, 0.0075, 0.01, 0.02)],
    "mean_reversion": [{"lookback": x, "z_entry": z, "z_exit": e} for x in (20, 40, 60, 100, 140) for z in (1.25, 1.75, 2.25, 2.75, 3.25) for e in (0.25, 0.5, 0.75, 1.0) if e < z],
    "moving_average_cross": [{"fast": f, "slow": s} for f in (5, 10, 15, 20, 30, 45, 60) for s in (40, 60, 90, 120, 180, 240) if s > f],
    "rsi_reversion": [{"rsi_length": n, "rsi_low": lo, "rsi_high": hi} for n in (7, 14, 21, 28) for lo in (20, 25, 30, 35, 40) for hi in (60, 65, 70, 75, 80, 85) if lo < 50 < hi],
    "atr_breakout": [{"atr_length": n, "atr_mult": m} for n in (3, 5, 8, 14, 21) for m in (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)],
    "channel_reversion": [{"channel_length": x} for x in (20, 40, 60, 90, 120, 180)],
}
OUT = ROOT / "experiments" / "robust_evolution_v2_latest.json"


def save(state):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def mutate(rng, fam, p):
    p = dict(p)
    mult = rng.choice((0.8, 0.9, 1.0, 1.1, 1.2))
    if fam in ("momentum", "breakout"):
        p["lookback"] = max(5, min(240, int(round(p["lookback"] * mult))))
    elif fam == "channel_reversion":
        p["channel_length"] = max(10, min(240, int(round(p["channel_length"] * mult))))
    elif fam == "trend_pullback":
        p["lookback"] = max(5, min(200, int(round(p["lookback"] * mult))))
        p["pullback_threshold"] = round(max(0.001, min(0.05, p["pullback_threshold"] * rng.choice((0.8, 1.0, 1.2)))), 4)
    elif fam == "mean_reversion":
        p["lookback"] = max(10, min(200, int(round(p["lookback"] * mult))))
        p["z_entry"] = round(max(0.8, min(3.5, p["z_entry"] * rng.choice((0.85, 1.0, 1.15)))), 3)
        p["z_exit"] = round(max(0.1, min(1.5, min(p["z_exit"] * rng.choice((0.8, 1.0, 1.2)), p["z_entry"] * 0.75))), 3)
    elif fam == "moving_average_cross":
        p["fast"] = max(2, min(100, int(round(p["fast"] * mult))))
        p["slow"] = max(p["fast"] + 3, min(300, int(round(p["slow"] * mult))))
    elif fam == "rsi_reversion":
        p["rsi_length"] = max(3, min(50, int(round(p["rsi_length"] * mult))))
        p["rsi_low"] = max(5, min(45, p["rsi_low"] + rng.choice((-5, 0, 5))))
        p["rsi_high"] = max(55, min(95, p["rsi_high"] + rng.choice((-5, 0, 5))))
    elif fam == "atr_breakout":
        p["atr_length"] = max(2, min(50, int(round(p["atr_length"] * mult))))
        p["atr_mult"] = round(max(0.25, min(5.0, p["atr_mult"] * rng.choice((0.8, 1.0, 1.2)))), 3)
    return p


def eval_one(df, fam, p, settings, fee_mult=1.0):
    r = _run(df, fam, dict(p), ["both"], settings.capital.initial_usd,
             settings.execution.commission_bps * fee_mult,
             settings.execution.slippage_bps * fee_mult)
    return _metrics(r, r.returns)


def score(m):
    ret = float(m.get("total_return", 0.0)); pf = float(m.get("profit_factor", 0.0)); dd = abs(min(0.0, float(m.get("max_drawdown", 0.0)))); trades = int(m.get("trade_count", 0)); sh = float(m.get("sharpe", 0.0))
    if trades < 4: return -25.0
    return 100 * ret + 18 * (min(2.5, pf) - 1.0) + 4 * sh - 35 * dd + min(8.0, math.log1p(trades))


def run(hours=3.0, population=12, generations=10):
    settings = load_settings(); adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc); deadline = time.monotonic() + hours * 3600
    state = {"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0, "seen": 0, "ai_generation": False, "futures": False, "live_trading": False}
    save(state)
    print("=== ROBUST EVOLUTION V2 ===", flush=True)
    print("START CHECKPOINT:", OUT, flush=True)

    # Primary markets only at first; fresh markets load after finalists.
    data = {}
    for key, bars in [(('ETH/USDT','1h'), 600), (('ETH/USDT','4h'), 600), (('BTC/USDT','4h'), 600)]:
        if time.monotonic() >= deadline: break
        print(f"LOAD {key[0]} {key[1]}", flush=True)
        data[key] = adapter.fetch_ohlcv_history(key[0], key[1], bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[key])}", flush=True)
        state.update({"decision":"LOADING", "loaded_market":f"{key[0]} {key[1]}", "loaded_bars":len(data[key]), "updated_at":datetime.now(timezone.utc).isoformat()}); save(state)
        gc.collect()

    rng = random.Random(20260829)
    pop=[]; seen=set(); per=max(1,population//len(FAMILIES))
    for fam in FAMILIES:
        for p in rng.sample(SEEDS[fam], min(per, len(SEEDS[fam]))):
            k=(fam,tuple(sorted(p.items())))
            if k not in seen:
                seen.add(k); pop.append((fam,dict(p)))
            if len(pop)>=population: break
        if len(pop)>=population: break

    history=[]; evaluated=0
    for gen in range(1,generations+1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} ===", flush=True)
        genres=[]
        for i,(fam,p) in enumerate(pop,1):
            if time.monotonic() >= deadline: break
            vals=[]; ok=0
            try:
                for key in data:
                    m=eval_one(data[key],fam,p,settings,1.0); vals.append(m)
                    ok += int(m["total_return"]>0 and m["profit_factor"]>1 and m["trade_count"]>=8)
                rs=[float(m["total_return"]) for m in vals]
                v=float(np.mean([score(m) for m in vals])+20*ok-25*(max(rs)-min(rs)))
                rec={"family":fam,"params":dict(p),"title":f"{fam} | {p}","score":v,"ok":ok,"markets":vals}
                genres.append(rec); history.append(rec); evaluated+=1
                state.update({"decision":"RUNNING","generation":gen,"evaluated":evaluated,"seen":len(seen),"latest":rec,"updated_at":datetime.now(timezone.utc).isoformat()}); save(state)
                print(f"eval {i}/{len(pop)} | score={v:.2f} | ok={ok}/3 | {fam} {p}", flush=True)
            except Exception as exc:
                print(f"ERROR {fam} {p}: {type(exc).__name__}: {exc}", flush=True)
            gc.collect()
        if not genres: break
        genres.sort(key=lambda x:(x["ok"],x["score"]), reverse=True)
        best=genres[0]
        print(f"GENERATION RESULT: best={best['score']:.2f} | ok={best['ok']}/3", flush=True)
        elite=genres[:max(2,min(4,len(genres)))]
        nxt=[(x["family"],dict(x["params"])) for x in elite]
        while len(nxt)<population and time.monotonic()<deadline:
            e=rng.choice(elite); fam=e["family"]; child=mutate(rng,fam,e["params"]); k=(fam,tuple(sorted(child.items())))
            if k not in seen: seen.add(k); nxt.append((fam,child))
        pop=nxt

    history.sort(key=lambda x:(x["ok"],x["score"]), reverse=True)
    finals=[]; fam_count={}
    for r in history:
        if fam_count.get(r["family"],0)>=2: continue
        finals.append(r); fam_count[r["family"]]=fam_count.get(r["family"],0)+1
        if len(finals)>=8: break

    # Lazy-load fresh markets only now.
    for key,bars in [(('BTC/USDT','1h'),600),(('ETH/USDT','15m'),600)]:
        if key not in data and time.monotonic()<deadline:
            print(f"LAZY LOAD {key[0]} {key[1]}", flush=True)
            data[key]=adapter.fetch_ohlcv_history(key[0],key[1],bars,page_limit=300,market_type="spot")
            print(f"  bars={len(data[key])}", flush=True)
            state.update({"decision":"FRESH_LOADING","loaded_market":f"{key[0]} {key[1]}","loaded_bars":len(data[key]),"updated_at":datetime.now(timezone.utc).isoformat()}); save(state)

    print("\n=== FINAL FRESH CONFIRMATION ===", flush=True)
    confirmed=[]
    for idx,r in enumerate(finals,1):
        c=(r["family"],r["params"]); fresh=[]
        for key in [("BTC/USDT","1h"),("ETH/USDT","15m")]:
            if key not in data: continue
            n=eval_one(data[key],*c,settings,1.0); s=eval_one(data[key],*c,settings,2.0)
            ok=(n["total_return"]>0 and n["profit_factor"]>1 and n["max_drawdown"]>=-0.50 and n["trade_count"]>=8 and s["total_return"]>0 and s["profit_factor"]>1)
            fresh.append({"market":f"{key[0]} {key[1]}","normal":n,"stress":s,"pass":bool(ok)})
            print(f"  {key[0]} {key[1]}: return={n['total_return']:.2%} PF={n['profit_factor']:.2f} DD={n['max_drawdown']:.2%} trades={n['trade_count']} stress={s['total_return']:.2%} pass={ok}",flush=True)
        validated=bool(r["ok"]==3 and len(fresh)==2 and all(x["pass"] for x in fresh))
        confirmed.append({**r,"fresh":fresh,"validated":validated})
        print(f"FINALIST {idx}: validated={validated}",flush=True)
        if validated: break
        gc.collect()

    good=[x for x in confirmed if x["validated"]]
    decision="VALIDATED_ALGORITHMIC_STRATEGY" if good else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    state={"started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"generation":generations,"evaluated":evaluated,"seen":len(seen),"finalists":len(finals),"validated_count":len(good),"results":confirmed,"generator":"deterministic_evolution_v2"}
    save(state)
    print("\n=== FINAL DECISION ===",flush=True)
    print(decision,flush=True)
    print("Validated:",len(good),flush=True)
    print("Saved:",OUT,flush=True)
    return state

if __name__ == "__main__":
    run()
