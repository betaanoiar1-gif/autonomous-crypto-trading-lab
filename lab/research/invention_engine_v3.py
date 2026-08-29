from __future__ import annotations

"""Architecture-first AI-free strategy invention engine.

The search space is a small deterministic program grammar. An invention controls
signal source, direction transform, confirmation rule, entry persistence,
regime gating, exit rule, and risk filter. Numeric parameters are evolved too.
No LLM, arbitrary code, futures, or live trading.
"""

import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics

OUT = ROOT / "experiments" / "invention_engine_v3_latest.json"

SIGNALS = ("momentum", "breakout", "ma_spread", "roc", "channel_position")
DIRECTIONS = ("normal", "inverse", "long_only", "short_only")
CONFIRMATIONS = ("none", "trend", "momentum", "volatility", "agreement")
REGIMES = ("all", "trend", "range", "low_vol", "normal_vol", "high_vol")
EXITS = ("reversal", "mean_cross", "time_stop", "volatility_exit", "profit_guard")

FAST = (6, 8, 12, 18, 24, 36)
SLOW = (40, 60, 90, 120, 180, 240)
WINDOW = (8, 12, 18, 24, 36, 48, 72, 96, 120)
THRESH = (0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02)
VOL_Q = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
PERSIST = (1, 2, 3, 4, 6)
MAX_HOLD = (8, 12, 18, 24, 36, 48)
Z = (0.8, 1.0, 1.25, 1.5, 1.75, 2.0)

@dataclass(frozen=True)
class Invention:
    signal: str
    direction: str
    confirmation: str
    regime: str
    exit_rule: str
    fast: int
    slow: int
    signal_window: int
    regime_window: int
    vol_window: int
    threshold: float
    high_vol_quantile: float
    persistence: int
    max_hold: int
    z_entry: float
    z_exit: float

    def params(self) -> dict:
        return asdict(self)

    def title(self) -> str:
        return (
            f"I[{self.signal}|{self.direction}|{self.confirmation}|"
            f"{self.regime}|{self.exit_rule}] "
            f"trend={self.fast}/{self.slow} w={self.signal_window}"
        )


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _choice(rng, xs):
    return xs[rng.randrange(len(xs))]


def _valid(p: dict) -> bool:
    return p["slow"] > p["fast"] and p["z_exit"] < p["z_entry"]


def random_invention(rng: random.Random) -> Invention:
    fast = _choice(rng, FAST)
    slow = _choice(rng, [x for x in SLOW if x > fast])
    return Invention(
        signal=_choice(rng, SIGNALS), direction=_choice(rng, DIRECTIONS),
        confirmation=_choice(rng, CONFIRMATIONS), regime=_choice(rng, REGIMES),
        exit_rule=_choice(rng, EXITS), fast=fast, slow=slow,
        signal_window=_choice(rng, WINDOW), regime_window=_choice(rng, WINDOW),
        vol_window=_choice(rng, (12, 18, 24, 36, 48)), threshold=_choice(rng, THRESH),
        high_vol_quantile=_choice(rng, VOL_Q), persistence=_choice(rng, PERSIST),
        max_hold=_choice(rng, MAX_HOLD), z_entry=_choice(rng, Z), z_exit=_choice(rng, (0.1, 0.25, 0.5, 0.75)),
    )


def mutate(x: Invention, rng: random.Random) -> Invention:
    p = x.params()
    choices = {
        "signal": SIGNALS, "direction": DIRECTIONS, "confirmation": CONFIRMATIONS,
        "regime": REGIMES, "exit_rule": EXITS, "fast": FAST, "slow": SLOW,
        "signal_window": WINDOW, "regime_window": WINDOW,
        "vol_window": (12, 18, 24, 36, 48), "threshold": THRESH, "high_vol_quantile": VOL_Q,
        "persistence": PERSIST, "max_hold": MAX_HOLD, "z_entry": Z,
        "z_exit": (0.1, 0.25, 0.5, 0.75),
    }
    probs = {k: 0.18 for k in choices}
    probs.update({"signal": .22, "direction": .22, "confirmation": .22, "regime": .22, "exit_rule": .22})
    for k, xs in choices.items():
        if rng.random() < probs[k]:
            p[k] = _choice(rng, xs)
    if p["slow"] <= p["fast"]:
        p["slow"] = min([v for v in SLOW if v > p["fast"]], default=max(SLOW))
    if p["z_exit"] >= p["z_entry"]:
        p["z_exit"] = 0.25
    return Invention(**p)


def _raw_components(df: pd.DataFrame, p: Invention):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    fast = close.ewm(span=p.fast, adjust=False).mean()
    slow = close.ewm(span=p.slow, adjust=False).mean()
    spread = (fast - slow) / slow.replace(0, np.nan)
    mom = close.pct_change(p.signal_window)
    roc = close.pct_change(p.signal_window)
    prior_hi = high.shift(1).rolling(p.signal_window).max()
    prior_lo = low.shift(1).rolling(p.signal_window).min()
    breakout = pd.Series(0.0, index=df.index)
    breakout[close > prior_hi] = 1.0
    breakout[close < prior_lo] = -1.0
    channel = (close - low.rolling(p.signal_window).min()) / (high.rolling(p.signal_window).max() - low.rolling(p.signal_window).min()).replace(0, np.nan)
    channel_position = (channel - 0.5) * 2.0
    raw = {
        "momentum": mom.clip(-0.2, 0.2),
        "breakout": breakout,
        "ma_spread": spread,
        "roc": roc.clip(-0.2, 0.2),
        "channel_position": channel_position.clip(-1, 1),
    }[p.signal]
    vol = close.pct_change().rolling(p.vol_window).std()
    vol_ref = vol.rolling(p.regime_window).quantile(p.high_vol_quantile)
    trend_regime = spread.abs() >= p.threshold
    high_regime = vol > vol_ref
    range_regime = ~trend_regime & ~high_regime
    low_vol = vol <= vol.rolling(p.regime_window).median()
    return close, fast, slow, raw, vol, trend_regime, range_regime, high_regime, low_vol


def _apply_policy(df: pd.DataFrame, p: Invention) -> pd.Series:
    close, fast, slow, raw, vol, trend, rng, high, low_vol = _raw_components(df, p)
    sig = pd.Series(0.0, index=df.index)

    base = pd.Series(0.0, index=df.index)
    base[raw.shift(1) > p.threshold] = 1.0
    base[raw.shift(1) < -p.threshold] = -1.0

    if p.direction == "inverse":
        base = -base
    elif p.direction == "long_only":
        base = base.clip(lower=0)
    elif p.direction == "short_only":
        base = base.clip(upper=0)

    if p.confirmation == "trend":
        base[(fast.shift(1) - slow.shift(1)) * base <= 0] = 0.0
    elif p.confirmation == "momentum":
        m = close.pct_change(max(2, p.fast // 2)).shift(1)
        base[m * base <= 0] = 0.0
    elif p.confirmation == "volatility":
        base[vol.shift(1).isna()] = 0.0
    elif p.confirmation == "agreement":
        alt = np.sign(close.pct_change(max(2, p.fast // 2)).shift(1).fillna(0))
        base[(alt * base) < 0] = 0.0

    active = pd.Series(True, index=df.index)
    if p.regime == "trend": active = trend
    elif p.regime == "range": active = rng
    elif p.regime == "low_vol": active = low_vol
    elif p.regime == "normal_vol": active = ~high
    elif p.regime == "high_vol": active = high
    sig[active] = base[active]

    if p.persistence > 1:
        pos = sig.copy()
        streak = pos.ne(pos.shift(1)).cumsum()
        counts = pos.groupby(streak).transform("size")
        sig[(pos != 0) & (counts < p.persistence)] = 0.0

    # Exit grammar: convert signals to persistent positions then apply exits.
    pos = pd.Series(0.0, index=df.index)
    current = 0.0
    age = 0
    for i in range(len(sig)):
        s = float(sig.iloc[i])
        if current != 0:
            age += 1
        if s != 0 and s != current:
            current = s
            age = 0
        if current != 0:
            if p.exit_rule == "mean_cross" and ((close.iloc[i] - close.rolling(p.signal_window).mean().iloc[i]) * current <= 0):
                current = 0.0; age = 0
            elif p.exit_rule == "volatility_exit" and float(vol.iloc[i] if pd.notna(vol.iloc[i]) else 0) > 0.05:
                current = 0.0; age = 0
            elif p.exit_rule == "time_stop" and age > p.max_hold:
                current = 0.0; age = 0
        pos.iloc[i] = current
    return pos.fillna(0.0)


def _eval(df, p, settings, fee_mult=1.0):
    sig = _apply_policy(df, p)
    r = run_ohlcv(df, sig, settings.capital.initial_usd,
                  settings.execution.commission_bps * fee_mult,
                  settings.execution.slippage_bps * fee_mult,
                  market_type="spot", leverage=1.0, funding_rates=None)
    return _metrics(r, r.returns)


def _score(m):
    tr = int(m.get("trade_count", 0))
    if tr < 8: return -40 + tr
    ret = float(m.get("total_return", 0))
    pf = min(2.5, float(m.get("profit_factor", 0)))
    dd = abs(min(0, float(m.get("max_drawdown", 0))))
    sh = float(m.get("sharpe", 0))
    return 100*ret + 18*(pf-1) + 4*sh - 30*dd + min(8, math.log1p(tr))


def _fold_eval(df, p, settings):
    max_warm = max(p.slow, p.signal_window, p.regime_window, p.vol_window, p.range_window if hasattr(p,'range_window') else 0)
    # Need enough history for indicators; measure only the following test block.
    n = len(df)
    test_len = max(60, n // 5)
    folds = []
    starts = [int(n*0.40), int(n*0.50), int(n*0.60), int(n*0.70)]
    for start in starts:
        a = max(max_warm + 2, start - test_len)
        b = min(n, start)
        if b-a < 40: continue
        segment = df.iloc[a:b]
        folds.append(_eval(segment, p, settings, 1.0))
    rets = [float(x['total_return']) for x in folds]
    pfs = [float(x['profit_factor']) for x in folds]
    trades = [int(x['trade_count']) for x in folds]
    wf = bool(len(folds)==4 and sum(x>0 for x in rets)>=3 and np.median(rets)>0 and np.median(pfs)>1 and min(trades)>=4)
    return folds, wf, float(np.median(rets)) if rets else 0.0, float(np.median(pfs)) if pfs else 0.0, min(trades) if trades else 0


def run(minutes=180.0, initial_population=36, population=12, generations=20, seed=20260829):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes*60
    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision":"STARTING", "evaluated":0})
    print("=== INVENTION ENGINE V3 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Architecture-first grammar | WF warm-up | independent confirmation", flush=True)
    print("Checkpoint written before data loading", flush=True)

    data = {}
    for symbol, tf, bars in (("ETH/USDT","1h",800),("ETH/USDT","4h",800),("BTC/USDT","4h",800)):
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol,tf)] = adapter.fetch_ohlcv_history(symbol,tf,bars,page_limit=300,market_type="spot")
        print(f"  bars={len(data[(symbol,tf)])}", flush=True)
        gc.collect()

    rng = random.Random(seed)
    population_list=[]; seen=set()
    while len(population_list)<initial_population:
        p=random_invention(rng); k=tuple(sorted(p.params().items()))
        if _valid(p.params()) and k not in seen:
            seen.add(k); population_list.append(p)
    evaluated=0; all_results=[]

    for generation in range(1,generations+1):
        if time.monotonic()>=deadline: break
        print(f"\n=== GENERATION {generation}/{generations} | population={len(population_list)} ===", flush=True)
        gen=[]
        for i,p in enumerate(population_list,1):
            if time.monotonic()>=deadline: break
            try:
                markets={}; scores=[]
                for key,df in data.items():
                    normal=_eval(df.iloc[int(len(df)*0.70):],p,settings,1.0)
                    stress=_eval(df.iloc[int(len(df)*0.70):],p,settings,2.0)
                    folds,wf,medr,medpf,mintr=_fold_eval(df,p,settings)
                    rec={"normal":normal,"stress":stress,"folds":folds,"wf":wf,"wf_median_return":medr,"wf_median_pf":medpf,"wf_min_trades":mintr}
                    rec["primary_pass"] = bool(float(normal['total_return'])>0 and float(normal['profit_factor'])>1 and float(normal['max_drawdown'])>=-0.50 and int(normal['trade_count'])>=8 and float(stress['total_return'])>0 and float(stress['profit_factor'])>1 and wf)
                    markets[f"{key[0]} {key[1]}"]=rec
                    scores.append(_score(normal)+50*medr)
                primary=sum(x['primary_pass'] for x in markets.values())
                rec={"title":p.title(),"parameters":p.params(),"score":float(np.mean(scores)),"primary_passes":primary,"markets":markets}
                gen.append(rec); all_results.append(rec); evaluated+=1
                print(f"eval {i}/{len(population_list)} | score={rec['score']:.2f} | primary={primary}/3 | {p.title()}",flush=True)
            except Exception as exc:
                print(f"eval {i}/{len(population_list)} | ERROR {type(exc).__name__}: {exc}",flush=True)
            _save({"started_at":started.isoformat(),"updated_at":datetime.now(timezone.utc).isoformat(),"decision":"SEARCHING","generation":generation,"evaluated":evaluated,"best":max(all_results,key=lambda x:x['score']) if all_results else None})
            gc.collect()
        if not gen: break
        gen.sort(key=lambda x:(x['primary_passes'],x['score']),reverse=True)
        elite=gen[:max(2,population//3)]
        nxt=[Invention(**x['parameters']) for x in elite]
        while len(nxt)<population:
            child=mutate(rng.choice(nxt),rng)
            k=tuple(sorted(child.params().items()))
            if _valid(child.params()) and k not in seen:
                seen.add(k); nxt.append(child)
            elif rng.random()<0.20:
                child=random_invention(rng); k=tuple(sorted(child.params().items()))
                if _valid(child.params()) and k not in seen:
                    seen.add(k); nxt.append(child)
        population_list=nxt

    ranked=sorted(all_results,key=lambda x:(x['primary_passes'],x['score']),reverse=True)
    finalists=[]; arch=set()
    for x in ranked:
        p=x['parameters']; s=(p['signal'],p['direction'],p['confirmation'],p['regime'],p['exit_rule'])
        if s in arch: continue
        arch.add(s); finalists.append(x)
        if len(finalists)>=population: break

    for symbol,tf in (("BTC/USDT","1h"),("ETH/USDT","15m")):
        if (symbol,tf) not in data and time.monotonic()<deadline:
            print(f"LAZY LOAD {symbol} {tf}",flush=True)
            data[(symbol,tf)] = adapter.fetch_ohlcv_history(symbol,tf,800,page_limit=300,market_type="spot")
            print(f"  bars={len(data[(symbol,tf)])}",flush=True)

    confirmed=[]
    print("\n=== FRESH UNTOUCHED CONFIRMATION ===",flush=True)
    for i,x in enumerate(finalists,1):
        p=Invention(**x['parameters']); checks=[]
        for key in (("BTC/USDT","1h"),("ETH/USDT","15m")):
            n=_eval(data[key],p,settings,1.0); s=_eval(data[key],p,settings,2.0)
            ok=bool(float(n['total_return'])>0 and float(n['profit_factor'])>1 and float(n['max_drawdown'])>=-0.50 and int(n['trade_count'])>=8 and float(s['total_return'])>0 and float(s['profit_factor'])>1)
            checks.append({"market":f"{key[0]} {key[1]}","normal":n,"stress":s,"pass":ok})
            print(f"# {i} {key[0]} {key[1]} | return={n['total_return']:.2%} PF={n['profit_factor']:.2f} DD={n['max_drawdown']:.2%} trades={n['trade_count']} stress={s['total_return']:.2%} pass={ok}",flush=True)
        z=dict(x); z['fresh']=checks; z['fresh_passes']=sum(c['pass'] for c in checks); z['validated']=bool(x['primary_passes']==3 and z['fresh_passes']==2); confirmed.append(z)
        _save({"started_at":started.isoformat(),"updated_at":datetime.now(timezone.utc).isoformat(),"decision":"CONFIRMING","evaluated":evaluated,"finalist":i,"finalists":confirmed})
        if z['validated']:
            print("VALIDATED INVENTION FOUND — STOPPING",flush=True); break
        gc.collect()

    valid=[x for x in confirmed if x['validated']]
    decision='VALIDATED_STRATEGY_INVENTED' if valid else 'NO_VALIDATED_INVENTION'
    payload={"started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"generated":len(all_results),"evaluated":evaluated,"finalists":len(finalists),"validated_count":len(valid),"winner":valid[0] if valid else (confirmed[0] if confirmed else None),"top_confirmed":confirmed}
    _save(payload)
    print("\n=== FINAL DECISION ===",flush=True)
    print(decision,flush=True)
    print("Validated:",len(valid),flush=True)
    print("Saved:",OUT,flush=True)
    return payload

if __name__ == '__main__':
    run()
