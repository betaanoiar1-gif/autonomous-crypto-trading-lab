from __future__ import annotations

"""AI-free strategy invention engine v2.

A candidate is a complete policy: regime classifier + action model per regime.
This version fixes walk-forward validation by using warm-up history before each
validation window and enforces strong population deduplication.
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

OUT = ROOT / "experiments" / "invention_engine_v2_latest.json"

TREND_ACTIONS = ("momentum", "breakout", "ma_cross")
RANGE_ACTIONS = ("mean_reversion", "rsi_reversion", "channel_reversion")
HIGH_ACTIONS = ("flat", "momentum", "breakout")

FAST = (6, 8, 12, 18, 24, 36)
SLOW = (40, 60, 90, 120, 180, 240)
REGIME = (24, 36, 48, 72, 96)
VOLW = (12, 18, 24, 36, 48)
MOMW = (8, 12, 18, 24, 36, 48, 72, 96)
BREAKW = (12, 20, 30, 40, 60, 90)
RANGEW = (20, 30, 40, 60, 90, 120)
TREND_T = (0.0004, 0.0007, 0.001, 0.0015, 0.0025)
HVQ = (0.75, 0.85, 0.90, 0.95)
ZENTRY = (1.0, 1.25, 1.5, 1.75, 2.0)
ZEXIT = (0.1, 0.25, 0.5, 0.75)
RSIL = (7, 14, 21, 28)
RSILO = (20, 25, 30, 35, 40)
RSIHI = (60, 65, 70, 75, 80, 85)
MOMT = (0.003, 0.005, 0.008, 0.01, 0.015, 0.02)
MAXV = (0.02, 0.025, 0.035, 0.05, 0.07)
COOLDOWN = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Invention:
    trend_action: str
    range_action: str
    high_vol_action: str
    trend_fast: int
    trend_slow: int
    regime_window: int
    vol_window: int
    trend_threshold: float
    high_vol_quantile: float
    momentum_window: int
    breakout_window: int
    range_window: int
    z_entry: float
    z_exit: float
    rsi_length: int
    rsi_low: float
    rsi_high: float
    momentum_threshold: float
    max_vol: float
    cooldown: int

    def params(self) -> dict:
        return asdict(self)

    def title(self) -> str:
        return (
            f"Policy[{self.trend_action}/{self.range_action}/{self.high_vol_action}] "
            f"trend={self.trend_fast}/{self.trend_slow} regime={self.regime_window} "
            f"vol={self.vol_window}@{self.high_vol_quantile}"
        )


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _valid(p: dict) -> bool:
    return (
        p["trend_slow"] > p["trend_fast"]
        and p["z_exit"] < p["z_entry"]
        and p["rsi_low"] < p["rsi_high"]
        and p["max_vol"] > 0
    )


def _random(rng: random.Random) -> Invention:
    fast = _pick(rng, FAST)
    slow = _pick(rng, [x for x in SLOW if x > fast])
    return Invention(
        trend_action=_pick(rng, TREND_ACTIONS),
        range_action=_pick(rng, RANGE_ACTIONS),
        high_vol_action=_pick(rng, HIGH_ACTIONS),
        trend_fast=fast,
        trend_slow=slow,
        regime_window=_pick(rng, REGIME),
        vol_window=_pick(rng, VOLW),
        trend_threshold=_pick(rng, TREND_T),
        high_vol_quantile=_pick(rng, HVQ),
        momentum_window=_pick(rng, MOMW),
        breakout_window=_pick(rng, BREAKW),
        range_window=_pick(rng, RANGEW),
        z_entry=_pick(rng, ZENTRY),
        z_exit=_pick(rng, ZEXIT),
        rsi_length=_pick(rng, RSIL),
        rsi_low=_pick(rng, RSILO),
        rsi_high=_pick(rng, RSIHI),
        momentum_threshold=_pick(rng, MOMT),
        max_vol=_pick(rng, MAXV),
        cooldown=_pick(rng, COOLDOWN),
    )


def _mutate(parent: Invention, rng: random.Random) -> Invention:
    p = parent.params()
    choices = {
        "trend_action": TREND_ACTIONS, "range_action": RANGE_ACTIONS, "high_vol_action": HIGH_ACTIONS,
        "trend_fast": FAST, "trend_slow": SLOW, "regime_window": REGIME, "vol_window": VOLW,
        "trend_threshold": TREND_T, "high_vol_quantile": HVQ, "momentum_window": MOMW,
        "breakout_window": BREAKW, "range_window": RANGEW, "z_entry": ZENTRY, "z_exit": ZEXIT,
        "rsi_length": RSIL, "rsi_low": RSILO, "rsi_high": RSIHI, "momentum_threshold": MOMT,
        "max_vol": MAXV, "cooldown": COOLDOWN,
    }
    for key, vals in choices.items():
        prob = 0.30 if "action" in key else 0.20
        if rng.random() < prob:
            p[key] = _pick(rng, vals)
    if p["trend_slow"] <= p["trend_fast"]:
        p["trend_slow"] = min(v for v in SLOW if v > p["trend_fast"])
    if p["z_exit"] >= p["z_entry"]:
        p["z_exit"] = min(ZEXIT)
    if p["rsi_low"] >= p["rsi_high"]:
        p["rsi_low"], p["rsi_high"] = 30, 70
    return Invention(**p)


def _signal(df: pd.DataFrame, p: Invention) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    fast = close.ewm(span=p.trend_fast, adjust=False).mean()
    slow = close.ewm(span=p.trend_slow, adjust=False).mean()
    trend_strength = (fast - slow) / slow.replace(0, np.nan)
    slope = slow.pct_change(p.regime_window) / p.regime_window
    ret = close.pct_change()
    vol = ret.rolling(p.vol_window).std()
    vol_ref = vol.rolling(p.regime_window).quantile(p.high_vol_quantile)
    trend_regime = slope.abs() >= p.trend_threshold
    high_regime = vol > vol_ref

    mom = close.pct_change(p.momentum_window)
    mom_sig = pd.Series(0.0, index=df.index)
    mom_sig[(trend_strength.shift(1) > 0) & (mom.shift(1) > p.momentum_threshold)] = 1.0
    mom_sig[(trend_strength.shift(1) < 0) & (mom.shift(1) < -p.momentum_threshold)] = -1.0

    prev_hi = high.shift(1).rolling(p.breakout_window).max()
    prev_lo = low.shift(1).rolling(p.breakout_window).min()
    br = pd.Series(0.0, index=df.index)
    br[close > prev_hi] = 1.0
    br[close < prev_lo] = -1.0
    br = br.shift(1).fillna(0.0)

    cross = pd.Series(np.where(fast > slow, 1.0, -1.0), index=df.index).shift(1).fillna(0.0)

    w = close.rolling(p.range_window)
    mean = w.mean(); std = w.std(ddof=0).replace(0, np.nan)
    z = (close - mean) / std
    mr = pd.Series(0.0, index=df.index)
    mr[z.shift(1) < -p.z_entry] = 1.0
    mr[z.shift(1) > p.z_entry] = -1.0
    mr[z.shift(1).abs() < p.z_exit] = 0.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/p.rsi_length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/p.rsi_length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100/(1+rs)
    rsi_sig = pd.Series(0.0, index=df.index)
    rsi_sig[rsi.shift(1) < p.rsi_low] = 1.0
    rsi_sig[rsi.shift(1) > p.rsi_high] = -1.0

    ch = close.rolling(p.range_window)
    lo = ch.quantile(0.10); hi = ch.quantile(0.90)
    ch_sig = pd.Series(0.0, index=df.index)
    ch_sig[close.shift(1) < lo.shift(1)] = 1.0
    ch_sig[close.shift(1) > hi.shift(1)] = -1.0

    tmap = {"momentum": mom_sig, "breakout": br, "ma_cross": cross}
    rmap = {"mean_reversion": mr, "rsi_reversion": rsi_sig, "channel_reversion": ch_sig}
    hmap = {"flat": pd.Series(0.0, index=df.index), "momentum": mom_sig, "breakout": br}

    sig = pd.Series(0.0, index=df.index)
    normal = ~high_regime
    sig[trend_regime & normal] = tmap[p.trend_action][trend_regime & normal]
    sig[(~trend_regime) & normal] = rmap[p.range_action][(~trend_regime) & normal]
    sig[high_regime] = hmap[p.high_vol_action][high_regime]
    sig[vol.shift(1) > p.max_vol] = 0.0

    if p.cooldown:
        changes = sig.ne(sig.shift(1)).fillna(False)
        locked = changes.astype(int).rolling(p.cooldown + 1).max().shift(1).fillna(0).astype(bool)
        sig[locked] = sig.shift(1).fillna(0.0)[locked]
    return sig.fillna(0.0)


def _eval(df, p, settings, fee_mult=1.0) -> dict:
    sig = _signal(df, p)
    r = run_ohlcv(df, sig, settings.capital.initial_usd,
                  settings.execution.commission_bps * fee_mult,
                  settings.execution.slippage_bps * fee_mult,
                  market_type="spot", leverage=1.0, funding_rates=None)
    return _metrics(r, r.returns)


def _passes(m, min_trades=8):
    return (float(m["total_return"]) > 0 and float(m["profit_factor"]) > 1
            and float(m["max_drawdown"]) >= -0.50 and int(m["trade_count"]) >= min_trades)


def _score(m):
    ret = float(m["total_return"]); pf = min(2.5, float(m["profit_factor"]))
    dd = abs(min(0.0, float(m["max_drawdown"]))); sh = float(m["sharpe"])
    tr = int(m["trade_count"])
    return 100*ret + 18*(pf-1) + 4*sh - 30*dd + min(8, math.log1p(max(tr,0)))


def _wf(df: pd.DataFrame, p: Invention, settings):
    """Warm-up walk-forward: each test fold gets preceding history for indicators."""
    n = len(df)
    warmup = max(p.trend_slow, p.regime_window, p.vol_window, p.momentum_window, p.breakout_window, p.range_window) + 5
    test = max(80, n // 5)
    usable = n - warmup
    if usable < test * 4:
        return [], False
    folds = []
    for i in range(4):
        start = warmup + i * test
        end = min(start + test, n)
        if end - start < max(40, test // 2):
            return [], False
        context = df.iloc[:end]
        metrics = _eval(context, p, settings, 1.0)
        # Re-run the same signal with the evaluation metrics isolated to the fold window.
        # The warm-up context is retained so indicators are valid, while performance is
        # measured only from the fold start.
        sig = _signal(context, p)
        sig_fold = sig.iloc[start:end]
        r = run_ohlcv(context.iloc[start:end], sig_fold,
                      settings.capital.initial_usd,
                      settings.execution.commission_bps,
                      settings.execution.slippage_bps,
                      market_type="spot", leverage=1.0, funding_rates=None)
        fm = _metrics(r, r.returns)
        folds.append(fm)
    rets = [float(x["total_return"]) for x in folds]
    pfs = [float(x["profit_factor"]) for x in folds]
    trs = [int(x["trade_count"]) for x in folds]
    ok = bool(sum(x > 0 for x in rets) >= 3 and np.median(rets) > 0 and np.median(pfs) > 1 and min(trs) >= 4)
    return folds, ok


def _primary(df, p, settings):
    cut = int(len(df) * 0.70)
    hold = df.iloc[cut:]
    normal = _eval(hold, p, settings, 1.0)
    stress = _eval(hold, p, settings, 2.0)
    folds, wf_ok = _wf(df, p, settings)
    return {
        "normal": normal, "stress": stress, "folds": folds, "wf": wf_ok,
        "score": _score(normal) + (70 * float(np.median([x["total_return"] for x in folds])) if folds else -20),
        "primary_pass": _passes(normal) and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1 and wf_ok,
    }


def run(minutes=180.0, initial_population=24, population=12, generations=20, seed=20260829):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60
    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0})

    print("=== INVENTION ENGINE V2 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("WF: warm-up enabled | Population dedup: enabled", flush=True)
    print("Checkpoint written before data loading", flush=True)

    data = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 800), ("ETH/USDT", "4h", 800), ("BTC/USDT", "4h", 800)):
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        gc.collect()

    rng = random.Random(seed)
    current=[]; seen=set()
    while len(current) < initial_population:
        p = _random(rng); key = tuple(sorted(p.params().items()))
        if _valid(p.params()) and key not in seen:
            seen.add(key); current.append(p)

    all_results=[]; evaluated=0
    for gen in range(1, generations+1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} | population={len(current)} ===", flush=True)
        gen_records=[]; gen_seen=set()
        for i,p in enumerate(current,1):
            if time.monotonic() >= deadline: break
            key = tuple(sorted(p.params().items()))
            if key in gen_seen: continue
            gen_seen.add(key)
            try:
                mr = {f"{k[0]} {k[1]}": _primary(df,p,settings) for k,df in data.items()}
                rec={"title":p.title(),"parameters":p.params(),"score":float(np.mean([x["score"] for x in mr.values()])),"primary_passes":sum(x["primary_pass"] for x in mr.values()),"markets":mr}
                gen_records.append(rec); all_results.append(rec); evaluated+=1
                print(f"eval {i}/{len(current)} | score={rec['score']:.2f} | primary={rec['primary_passes']}/3 | {p.title()}", flush=True)
            except Exception as exc:
                print(f"eval {i}/{len(current)} | ERROR {type(exc).__name__}: {exc}", flush=True)
            best=max(all_results,key=lambda x:x["score"]) if all_results else None
            _save({"started_at":started.isoformat(),"updated_at":datetime.now(timezone.utc).isoformat(),"decision":"SEARCHING","generation":gen,"evaluated":evaluated,"best":best})
            gc.collect()

        if not gen_records: break
        gen_records.sort(key=lambda x:(x["primary_passes"],x["score"]), reverse=True)
        elites=[]; elite_keys=set()
        # Favor diversity of policy architecture, then parameters.
        for rec in gen_records:
            p=Invention(**rec["parameters"])
            arch=(p.trend_action,p.range_action,p.high_vol_action)
            if arch not in elite_keys or len(elites)<max(3,population//3):
                elite_keys.add(arch); elites.append(p)
            if len(elites)>=max(3,population//3): break
        nxt=list(elites); local=set(tuple(sorted(p.params().items())) for p in nxt)
        attempts=0
        while len(nxt)<population and attempts<population*100:
            attempts+=1
            parent=rng.choice(nxt)
            child=_mutate(parent,rng)
            key=tuple(sorted(child.params().items()))
            if _valid(child.params()) and key not in seen and key not in local:
                local.add(key); seen.add(key); nxt.append(child)
            elif rng.random()<0.15:
                child=_random(rng); key=tuple(sorted(child.params().items()))
                if _valid(child.params()) and key not in seen and key not in local:
                    local.add(key); seen.add(key); nxt.append(child)
        current=nxt

    ranked=sorted(all_results,key=lambda x:(x["primary_passes"],x["score"]),reverse=True)
    finalists=[]; sigs=set()
    for rec in ranked:
        p=Invention(**rec["parameters"])
        sig=(p.trend_action,p.range_action,p.high_vol_action,p.trend_fast,p.trend_slow,p.regime_window,p.vol_window,p.momentum_window,p.range_window)
        if sig in sigs: continue
        sigs.add(sig); finalists.append(rec)
        if len(finalists)>=population: break

    print("\n=== FRESH UNTOUCHED CONFIRMATION ===", flush=True)
    for symbol,tf,bars in (("BTC/USDT","1h",800),("ETH/USDT","15m",800)):
        print(f"LAZY LOAD {symbol} {tf}",flush=True)
        data[(symbol,tf)] = adapter.fetch_ohlcv_history(symbol,tf,bars,page_limit=300,market_type="spot")
        print(f"  bars={len(data[(symbol,tf)])}",flush=True)

    confirmed=[]
    for i,rec in enumerate(finalists,1):
        p=Invention(**rec["parameters"]); fresh=[]
        for key in (("BTC/USDT","1h"),("ETH/USDT","15m")):
            normal=_eval(data[key],p,settings,1.0); stress=_eval(data[key],p,settings,2.0)
            ok=_passes(normal) and float(stress["total_return"])>0 and float(stress["profit_factor"])>1
            fresh.append({"market":f"{key[0]} {key[1]}","normal":normal,"stress":stress,"pass":ok})
            print(f"# {i} {key[0]} {key[1]} | return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}",flush=True)
        out=dict(rec); out["fresh"]=fresh; out["fresh_passes"]=sum(x["pass"] for x in fresh); out["validated"]=bool(rec["primary_passes"]==3 and out["fresh_passes"]==2)
        confirmed.append(out)
        if out["validated"]:
            print("VALIDATED INVENTION FOUND",flush=True); break
        gc.collect()

    valid=[x for x in confirmed if x["validated"]]
    decision="VALIDATED_STRATEGY_INVENTED" if valid else "NO_VALIDATED_INVENTION"
    payload={"started_at":started.isoformat(),"finished_at":datetime.now(timezone.utc).isoformat(),"decision":decision,"generated":len(all_results),"evaluated":evaluated,"finalists":len(finalists),"validated_count":len(valid),"winner":valid[0] if valid else (confirmed[0] if confirmed else None),"top_confirmed":confirmed}
    _save(payload)
    print("\n=== FINAL DECISION ===",flush=True)
    print(decision,flush=True)
    print("Validated:",len(valid),flush=True)
    print("Saved:",OUT,flush=True)
    return payload


if __name__ == "__main__":
    run()
