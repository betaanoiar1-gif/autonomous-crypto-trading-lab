from __future__ import annotations

"""AI-free strategy invention engine.

A candidate is a complete policy, not merely a parameter set. The policy
contains a market-regime classifier plus an independently chosen action model
for trend, range, and high-volatility regimes. Candidates are evolved by
mutation and selected only by out-of-sample robustness. No LLM, futures, or
live trading is used.
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

OUT = ROOT / "experiments" / "strategy_invention_latest.json"

TREND_ACTIONS = ("momentum", "breakout", "ma_cross")
RANGE_ACTIONS = ("mean_reversion", "rsi_reversion", "channel_reversion")
HIGH_VOL_ACTIONS = ("flat", "momentum", "breakout")

TREND_FAST = (6, 8, 12, 18, 24, 36)
TREND_SLOW = (40, 60, 90, 120, 180, 240)
REGIME_WINDOW = (24, 36, 48, 72, 96)
VOL_WINDOW = (12, 18, 24, 36, 48)
TREND_THRESH = (0.0004, 0.0007, 0.001, 0.0015, 0.0025)
HIGH_VOL_Q = (0.75, 0.85, 0.90, 0.95)
MOM_WINDOW = (8, 12, 18, 24, 36, 48, 72, 96)
BREAK_WINDOW = (12, 20, 30, 40, 60, 90)
RANGE_WINDOW = (20, 30, 40, 60, 90, 120)
Z_ENTRY = (1.0, 1.25, 1.5, 1.75, 2.0)
Z_EXIT = (0.1, 0.25, 0.5, 0.75)
RSI_LEN = (7, 14, 21, 28)
RSI_LOW = (20, 25, 30, 35, 40)
RSI_HIGH = (60, 65, 70, 75, 80, 85)
MOM_THRESHOLD = (0.003, 0.005, 0.008, 0.01, 0.015, 0.02)
MAX_VOL = (0.02, 0.025, 0.035, 0.05, 0.07)
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

    def title(self) -> str:
        return (
            f"Policy[{self.trend_action}/{self.range_action}/{self.high_vol_action}] "
            f"trend={self.trend_fast}/{self.trend_slow} regime={self.regime_window} "
            f"vol={self.vol_window}@q{self.high_vol_quantile}"
        )

    def params(self) -> dict:
        return asdict(self)


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _choice(rng, seq):
    return seq[rng.randrange(len(seq))]


def _valid(p: dict) -> bool:
    return (
        p["trend_slow"] > p["trend_fast"]
        and p["z_exit"] < p["z_entry"]
        and p["rsi_low"] < p["rsi_high"]
        and p["high_vol_quantile"] > 0.5
        and p["max_vol"] > 0
    )


def random_invention(rng: random.Random) -> Invention:
    fast = _choice(rng, TREND_FAST)
    slow = _choice(rng, [x for x in TREND_SLOW if x > fast])
    return Invention(
        trend_action=_choice(rng, TREND_ACTIONS),
        range_action=_choice(rng, RANGE_ACTIONS),
        high_vol_action=_choice(rng, HIGH_VOL_ACTIONS),
        trend_fast=fast,
        trend_slow=slow,
        regime_window=_choice(rng, REGIME_WINDOW),
        vol_window=_choice(rng, VOL_WINDOW),
        trend_threshold=_choice(rng, TREND_THRESH),
        high_vol_quantile=_choice(rng, HIGH_VOL_Q),
        momentum_window=_choice(rng, MOM_WINDOW),
        breakout_window=_choice(rng, BREAK_WINDOW),
        range_window=_choice(rng, RANGE_WINDOW),
        z_entry=_choice(rng, Z_ENTRY),
        z_exit=_choice(rng, Z_EXIT),
        rsi_length=_choice(rng, RSI_LEN),
        rsi_low=_choice(rng, RSI_LOW),
        rsi_high=_choice(rng, RSI_HIGH),
        momentum_threshold=_choice(rng, MOM_THRESHOLD),
        max_vol=_choice(rng, MAX_VOL),
        cooldown=_choice(rng, COOLDOWN),
    )


def mutate(x: Invention, rng: random.Random) -> Invention:
    p = x.params()
    if rng.random() < 0.25:
        p["trend_action"] = _choice(rng, TREND_ACTIONS)
    if rng.random() < 0.25:
        p["range_action"] = _choice(rng, RANGE_ACTIONS)
    if rng.random() < 0.20:
        p["high_vol_action"] = _choice(rng, HIGH_VOL_ACTIONS)

    for key, values, prob in [
        ("trend_fast", TREND_FAST, 0.30),
        ("trend_slow", TREND_SLOW, 0.30),
        ("regime_window", REGIME_WINDOW, 0.25),
        ("vol_window", VOL_WINDOW, 0.20),
        ("trend_threshold", TREND_THRESH, 0.25),
        ("high_vol_quantile", HIGH_VOL_Q, 0.20),
        ("momentum_window", MOM_WINDOW, 0.25),
        ("breakout_window", BREAK_WINDOW, 0.25),
        ("range_window", RANGE_WINDOW, 0.25),
        ("z_entry", Z_ENTRY, 0.25),
        ("z_exit", Z_EXIT, 0.20),
        ("rsi_length", RSI_LEN, 0.20),
        ("rsi_low", RSI_LOW, 0.20),
        ("rsi_high", RSI_HIGH, 0.20),
        ("momentum_threshold", MOM_THRESHOLD, 0.25),
        ("max_vol", MAX_VOL, 0.20),
        ("cooldown", COOLDOWN, 0.20),
    ]:
        if rng.random() < prob:
            p[key] = _choice(rng, values)

    if p["trend_slow"] <= p["trend_fast"]:
        p["trend_slow"] = min([v for v in TREND_SLOW if v > p["trend_fast"]], default=max(TREND_SLOW))
    if p["z_exit"] >= p["z_entry"]:
        p["z_exit"] = min(Z_EXIT)
    if p["rsi_low"] >= p["rsi_high"]:
        p["rsi_low"], p["rsi_high"] = 30, 70
    return Invention(**p)


def _component_signals(df: pd.DataFrame, p: Invention) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    fast = close.ewm(span=p.trend_fast, adjust=False).mean()
    slow = close.ewm(span=p.trend_slow, adjust=False).mean()
    trend_strength = (fast - slow) / slow.replace(0, np.nan)
    slope = slow.pct_change(p.regime_window) / p.regime_window
    vol = close.pct_change().rolling(p.vol_window).std()
    vol_ref = vol.rolling(p.regime_window).quantile(p.high_vol_quantile)
    trend_regime = slope.abs() >= p.trend_threshold
    high_vol_regime = vol > vol_ref

    mom = close.pct_change(p.momentum_window)
    trend_mom = pd.Series(0.0, index=df.index)
    trend_mom[(trend_strength.shift(1) > 0) & (mom.shift(1) > p.momentum_threshold)] = 1.0
    trend_mom[(trend_strength.shift(1) < 0) & (mom.shift(1) < -p.momentum_threshold)] = -1.0

    prior_high = high.shift(1).rolling(p.breakout_window).max()
    prior_low = low.shift(1).rolling(p.breakout_window).min()
    breakout = pd.Series(0.0, index=df.index)
    breakout[close > prior_high] = 1.0
    breakout[close < prior_low] = -1.0
    breakout = breakout.shift(1).fillna(0.0)

    ma_cross = pd.Series(np.where(fast > slow, 1.0, -1.0), index=df.index).shift(1).fillna(0.0)

    w = close.rolling(p.range_window)
    mean = w.mean()
    std = w.std(ddof=0).replace(0, np.nan)
    z = (close - mean) / std
    mean_rev = pd.Series(0.0, index=df.index)
    mean_rev[z.shift(1) < -p.z_entry] = 1.0
    mean_rev[z.shift(1) > p.z_entry] = -1.0
    mean_rev[z.shift(1).abs() < p.z_exit] = 0.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / p.rsi_length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / p.rsi_length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_sig = pd.Series(0.0, index=df.index)
    rsi_sig[rsi.shift(1) < p.rsi_low] = 1.0
    rsi_sig[rsi.shift(1) > p.rsi_high] = -1.0

    channel = close.rolling(p.range_window)
    ch_lo = channel.quantile(0.10)
    ch_hi = channel.quantile(0.90)
    ch_mid = channel.mean()
    ch_sig = pd.Series(0.0, index=df.index)
    ch_sig[close.shift(1) < ch_lo.shift(1)] = 1.0
    ch_sig[close.shift(1) > ch_hi.shift(1)] = -1.0
    ch_sig[(close.shift(1) - ch_mid.shift(1)).abs() < (ch_mid.shift(1) * 0.005)] = 0.0

    trend_map = {"momentum": trend_mom, "breakout": breakout, "ma_cross": ma_cross}
    range_map = {"mean_reversion": mean_rev, "rsi_reversion": rsi_sig, "channel_reversion": ch_sig}
    high_map = {
        "flat": pd.Series(0.0, index=df.index),
        "momentum": trend_mom,
        "breakout": breakout,
    }

    sig = pd.Series(0.0, index=df.index)
    non_high = ~high_vol_regime
    sig[trend_regime & non_high] = trend_map[p.trend_action][trend_regime & non_high]
    sig[(~trend_regime) & non_high] = range_map[p.range_action][(~trend_regime) & non_high]
    sig[high_vol_regime] = high_map[p.high_vol_action][high_vol_regime]
    sig[vol.shift(1) > p.max_vol] = 0.0

    if p.cooldown:
        changes = sig.ne(sig.shift(1)).fillna(False)
        lock = changes.astype(int).rolling(p.cooldown + 1).max().shift(1).fillna(0).astype(bool)
        sig[lock] = sig.shift(1).fillna(0.0)[lock]

    return sig.fillna(0.0), trend_regime.fillna(False), high_vol_regime.fillna(False)


def _eval(df, p: Invention, settings, fee_mult: float = 1.0) -> dict:
    sig, _, _ = _component_signals(df, p)
    result = run_ohlcv(
        df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
        market_type="spot", leverage=1.0, funding_rates=None,
    )
    return _metrics(result, result.returns)


def _passes(m: dict, min_trades: int = 8) -> bool:
    return bool(
        float(m["total_return"]) > 0
        and float(m["profit_factor"]) > 1
        and float(m["max_drawdown"]) >= -0.50
        and int(m["trade_count"]) >= min_trades
    )


def _score(m: dict) -> float:
    ret = float(m.get("total_return", 0.0))
    pf = min(2.5, float(m.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    tr = int(m.get("trade_count", 0))
    sh = float(m.get("sharpe", 0.0))
    return 100 * ret + 20 * (pf - 1) + 4 * sh - 30 * dd + min(8, math.log1p(max(0, tr)))


def _primary_eval(df: pd.DataFrame, p: Invention, settings) -> dict:
    n = len(df)
    hold = df.iloc[int(n * 0.70):]
    normal = _eval(hold, p, settings, 1.0)
    stress = _eval(hold, p, settings, 2.0)
    block = max(30, len(hold) // 4)
    folds = []
    for i in range(4):
        a = i * block; b = len(hold) if i == 3 else (i + 1) * block
        if b <= a: continue
        folds.append(_eval(hold.iloc[a:b], p, settings, 1.0))
    rets = [float(x["total_return"]) for x in folds]
    pfs = [float(x["profit_factor"]) for x in folds]
    trs = [int(x["trade_count"]) for x in folds]
    wf = bool(len(folds) == 4 and sum(x > 0 for x in rets) >= 3 and np.median(rets) > 0 and np.median(pfs) > 1 and min(trs) >= 4)
    primary = bool(_passes(normal) and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1 and wf)
    return {
        "normal": normal, "stress": stress, "folds": folds,
        "wf": wf, "wf_positive": sum(x > 0 for x in rets),
        "wf_median_return": float(np.median(rets)) if rets else 0.0,
        "wf_median_pf": float(np.median(pfs)) if pfs else 0.0,
        "wf_min_trades": min(trs) if trs else 0,
        "primary_pass": primary,
        "score": _score(normal) + 70 * (float(np.median(rets)) if rets else 0.0),
    }


def run(minutes: float = 180.0, pool_size: int = 240, population: int = 12, generations: int = 20, seed: int = 20260829):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60

    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0})
    print("=== STRATEGY INVENTION ENGINE ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Invention unit: complete regime policy", flush=True)
    print("Checkpoint written before data loading", flush=True)

    data = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 600), ("ETH/USDT", "4h", 600), ("BTC/USDT", "4h", 600)):
        if time.monotonic() >= deadline: break
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "LOADED", "market": f"{symbol} {tf}", "evaluated": 0})
        gc.collect()

    rng = random.Random(seed)
    current = []
    seen = set()
    while len(current) < min(24, population * 2):
        p = random_invention(rng)
        key = tuple(sorted(p.params().items()))
        if _valid(p.params()) and key not in seen:
            seen.add(key); current.append(p)

    all_results = []
    eval_count = 0

    for generation in range(1, generations + 1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {generation}/{generations} | population={len(current)} ===", flush=True)
        gen_results = []
        for idx, p in enumerate(current, 1):
            if time.monotonic() >= deadline: break
            try:
                market_results = {f"{k[0]} {k[1]}": _primary_eval(df, p, settings) for k, df in data.items()}
                score = float(np.mean([x["score"] for x in market_results.values()]))
                passed = sum(x["primary_pass"] for x in market_results.values())
                rec = {"title": p.title(), "parameters": p.params(), "score": score, "primary_passes": passed, "markets": market_results}
                gen_results.append(rec); all_results.append(rec); eval_count += 1
                print(f"eval {idx}/{len(current)} | score={score:.2f} | primary={passed}/3 | {p.title()}", flush=True)
            except Exception as exc:
                print(f"eval {idx}/{len(current)} | ERROR {type(exc).__name__}: {exc}", flush=True)
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "SEARCHING", "generation": generation, "evaluated": eval_count, "best": max(all_results, key=lambda x: x["score"]) if all_results else None})
            gc.collect()

        if not gen_results: break
        gen_results.sort(key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
        elite = gen_results[: max(2, population // 3)]
        next_population = [Invention(**r["parameters"]) for r in elite]
        while len(next_population) < population:
            parent = rng.choice(next_population[:max(1, len(next_population))])
            child = mutate(parent, rng)
            key = tuple(sorted(child.params().items()))
            if _valid(child.params()) and key not in seen:
                seen.add(key); next_population.append(child)
            elif rng.random() < 0.20:
                child = random_invention(rng)
                key = tuple(sorted(child.params().items()))
                if _valid(child.params()) and key not in seen:
                    seen.add(key); next_population.append(child)
        current = next_population

    ranked = sorted(all_results, key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
    finalists = []
    signatures = set()
    for r in ranked:
        p = r["parameters"]
        sig = (p["trend_action"], p["range_action"], p["high_vol_action"], p["trend_fast"], p["trend_slow"], p["regime_window"], p["vol_window"])
        if sig in signatures: continue
        signatures.add(sig); finalists.append(r)
        if len(finalists) >= population: break

    print("\n=== FRESH UNTOUCHED CONFIRMATION ===", flush=True)
    for symbol, tf, bars in (("BTC/USDT", "1h", 800), ("ETH/USDT", "15m", 800)):
        if (symbol, tf) not in data and time.monotonic() < deadline:
            print(f"LAZY LOAD {symbol} {tf}", flush=True)
            data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
            print(f"  bars={len(data[(symbol, tf)])}", flush=True)

    confirmed = []
    for i, r in enumerate(finalists, 1):
        checks = []
        p = Invention(**r["parameters"])
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            if key not in data: continue
            normal = _eval(data[key], p, settings, 1.0)
            stress = _eval(data[key], p, settings, 2.0)
            ok = _passes(normal) and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1
            checks.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"# {i} {key[0]} {key[1]} | return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
        r2 = dict(r); r2["fresh"] = checks; r2["fresh_passes"] = sum(x["pass"] for x in checks); r2["validated"] = bool(r["primary_passes"] == 3 and r2["fresh_passes"] == 2); confirmed.append(r2)
        _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "CONFIRMING", "generation": generation if 'generation' in locals() else 0, "evaluated": eval_count, "finalist": i, "finalists": confirmed})
        if r2["validated"]:
            print("VALIDATED INVENTION FOUND", flush=True)
            break
        gc.collect()

    validated = [x for x in confirmed if x["validated"]]
    decision = "VALIDATED_STRATEGY_INVENTED" if validated else "NO_VALIDATED_INVENTION"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generated": len(all_results), "evaluated": eval_count, "finalists": len(finalists), "validated_count": len(validated), "winner": validated[0] if validated else (confirmed[0] if confirmed else None), "top_confirmed": confirmed}
    _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", OUT, flush=True)
    return payload


if __name__ == "__main__":
    run()