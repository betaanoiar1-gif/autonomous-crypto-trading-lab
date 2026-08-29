from __future__ import annotations

"""AI-free genetic-program strategy discovery v4.

V4 focuses on the failure modes observed in earlier engines:
- normalized primitives so expressions are numerically comparable;
- balanced long/short/normal/inverse directions;
- explicit anti-degeneracy penalties for one-sided and over-trading programs;
- walk-forward folds with warm-up history;
- untouched fresh-market confirmation;
- no LLM, arbitrary code, futures, or live trading.
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

OUT = ROOT / "experiments" / "genetic_program_discovery_v4_latest.json"

FEATURES = (
    "ret_fast", "ret_slow", "ma_spread", "rsi_bias", "vol_ratio",
    "range_position", "candle_pressure", "volume_ratio", "trend_slope",
)
OPS = ("add", "sub", "mean", "mul", "safe_div")
TRANSFORMS = ("identity", "tanh", "sign", "clip")
DIRECTIONS = ("normal", "inverse", "long_only", "short_only")
EXIT_RULES = ("zero_cross", "mean_cross", "time_stop", "volatility_exit")

LOOKBACKS = (8, 12, 18, 24, 36, 48, 72, 96)
SLOW_WINDOWS = (40, 60, 90, 120, 180, 240)
THRESHOLDS = (0.15, 0.25, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00)
MAX_HOLDS = (4, 8, 12, 18, 24, 36)
PERSISTENCE = (0, 1, 2, 3)


@dataclass(frozen=True)
class Node:
    kind: str
    value: str | float
    left: "Node | None" = None
    right: "Node | None" = None


@dataclass(frozen=True)
class Program:
    tree: Node
    transform: str
    direction: str
    threshold: float
    exit_rule: str
    max_hold: int
    persistence: int

    def title(self) -> str:
        return f"GP4[{self.transform}|{self.direction}|{self.exit_rule}] nodes={self.size()} thr={self.threshold} hold={self.max_hold} pers={self.persistence}"

    def params(self) -> dict:
        return {"tree": node_to_dict(self.tree), "transform": self.transform, "direction": self.direction, "threshold": self.threshold, "exit_rule": self.exit_rule, "max_hold": self.max_hold, "persistence": self.persistence}

    def size(self) -> int:
        def rec(n: Node) -> int:
            return 1 + (rec(n.left) if n.left else 0) + (rec(n.right) if n.right else 0)
        return rec(self.tree)


def node_to_dict(n: Node | None) -> dict | None:
    if n is None:
        return None
    return {"kind": n.kind, "value": n.value, "left": node_to_dict(n.left), "right": node_to_dict(n.right)}


def dict_to_node(d: dict) -> Node:
    return Node(d["kind"], d["value"], dict_to_node(d["left"]) if d.get("left") else None, dict_to_node(d["right"]) if d.get("right") else None)


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _leaf(rng: random.Random) -> Node:
    if rng.random() < 0.82:
        return Node("feature", rng.choice(FEATURES))
    return Node("const", rng.choice((-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)))


def _tree(rng: random.Random, depth: int = 0) -> Node:
    if depth >= 3 or rng.random() < 0.32:
        return _leaf(rng)
    return Node("op", rng.choice(OPS), _tree(rng, depth + 1), _tree(rng, depth + 1))


def random_program(rng: random.Random) -> Program:
    return Program(
        tree=_tree(rng),
        transform=rng.choice(TRANSFORMS),
        direction=rng.choice(DIRECTIONS),
        threshold=rng.choice(THRESHOLDS),
        exit_rule=rng.choice(EXIT_RULES),
        max_hold=rng.choice(MAX_HOLDS),
        persistence=rng.choice(PERSISTENCE),
    )


def _mutate_node(n: Node, rng: random.Random, depth: int = 0) -> Node:
    if rng.random() < 0.18:
        return _tree(rng, depth)
    if n.kind == "feature":
        return Node("feature", rng.choice(FEATURES)) if rng.random() < 0.18 else n
    if n.kind == "const":
        return Node("const", rng.choice((-1.0, -0.5, -0.25, 0.25, 0.5, 1.0))) if rng.random() < 0.18 else n
    left = _mutate_node(n.left, rng, depth + 1) if n.left else None
    right = _mutate_node(n.right, rng, depth + 1) if n.right else None
    value = rng.choice(OPS) if rng.random() < 0.12 else n.value
    return Node("op", value, left, right)


def mutate(p: Program, rng: random.Random) -> Program:
    return Program(
        tree=_mutate_node(p.tree, rng),
        transform=rng.choice(TRANSFORMS) if rng.random() < 0.12 else p.transform,
        direction=rng.choice(DIRECTIONS) if rng.random() < 0.18 else p.direction,
        threshold=rng.choice(THRESHOLDS) if rng.random() < 0.18 else p.threshold,
        exit_rule=rng.choice(EXIT_RULES) if rng.random() < 0.14 else p.exit_rule,
        max_hold=rng.choice(MAX_HOLDS) if rng.random() < 0.16 else p.max_hold,
        persistence=rng.choice(PERSISTENCE) if rng.random() < 0.12 else p.persistence,
    )


def _norm_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(1.0, index=df.index)

    ret_fast = close.pct_change(8)
    ret_slow = close.pct_change(32)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=48, adjust=False).mean()
    ma_spread = (fast - slow) / slow.replace(0, np.nan)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_bias = (rsi - 50.0) / 50.0
    rv = close.pct_change().rolling(24).std()
    vol_base = rv.rolling(96).median().replace(0, np.nan)
    vol_ratio = rv / vol_base
    lo = low.rolling(24).min()
    hi = high.rolling(24).max()
    range_position = (close - lo) / (hi - lo).replace(0, np.nan) * 2.0 - 1.0
    candle_pressure = (close - df["open"].astype(float)) / (high - low).replace(0, np.nan)
    vol_base2 = volume.rolling(48).median().replace(0, np.nan)
    volume_ratio = volume / vol_base2
    trend_slope = slow.pct_change(24) / 24.0

    raw = {
        "ret_fast": ret_fast,
        "ret_slow": ret_slow,
        "ma_spread": ma_spread,
        "rsi_bias": rsi_bias,
        "vol_ratio": vol_ratio,
        "range_position": range_position,
        "candle_pressure": candle_pressure,
        "volume_ratio": volume_ratio,
        "trend_slope": trend_slope,
    }
    out = {}
    for k, s in raw.items():
        med = s.rolling(96).median()
        mad = (s - med).abs().rolling(96).median().replace(0, np.nan)
        z = (s - med) / (1.4826 * mad)
        out[k] = z.clip(-4, 4).fillna(0.0)
    return out


def _eval_node(n: Node, f: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    if n.kind == "feature":
        return f[str(n.value)]
    if n.kind == "const":
        return pd.Series(float(n.value), index=index)
    a = _eval_node(n.left, f, index)
    b = _eval_node(n.right, f, index)
    if n.value == "add":
        x = a + b
    elif n.value == "sub":
        x = a - b
    elif n.value == "mean":
        x = (a + b) / 2.0
    elif n.value == "mul":
        x = a * b
    else:
        x = a / b.replace(0, np.nan)
    return x.replace([np.inf, -np.inf], np.nan).clip(-8, 8).fillna(0.0)


def _signal(df: pd.DataFrame, p: Program) -> pd.Series:
    f = _norm_features(df)
    x = _eval_node(p.tree, f, df.index)
    if p.transform == "tanh":
        x = np.tanh(x)
    elif p.transform == "sign":
        x = np.sign(x)
    elif p.transform == "clip":
        x = x.clip(-2, 2) / 2.0

    sig = pd.Series(0.0, index=df.index)
    th = float(p.threshold)
    sig[x.shift(1) > th] = 1.0
    sig[x.shift(1) < -th] = -1.0

    if p.direction == "inverse":
        sig = -sig
    elif p.direction == "long_only":
        sig[sig < 0] = 0.0
    elif p.direction == "short_only":
        sig[sig > 0] = 0.0

    if p.persistence > 0:
        persistent = sig.copy()
        for _ in range(p.persistence):
            persistent = persistent.where(persistent == persistent.shift(1), 0.0)
        sig = sig.where(persistent != 0, 0.0)

    close = df["close"].astype(float)
    vol = close.pct_change().rolling(24).std()
    ma = close.rolling(24).mean()
    if p.exit_rule == "zero_cross":
        cross = x.shift(1) * x.shift(2) < 0
        sig[cross] = 0.0
    elif p.exit_rule == "mean_cross":
        sig[((close.shift(1) - ma.shift(1)) * (close.shift(2) - ma.shift(2)) < 0)] = 0.0
    elif p.exit_rule == "volatility_exit":
        sig[vol.shift(1) > vol.rolling(96).quantile(0.90)] = 0.0

    if p.exit_rule == "time_stop" and p.max_hold > 0:
        pos = sig.ne(0).astype(int)
        age = pos.groupby(pos.eq(0).cumsum()).cumsum()
        sig[age > p.max_hold] = 0.0
    elif p.max_hold > 0:
        # Apply a universal hard max-hold without changing the signal generator.
        nonzero = sig.ne(0)
        groups = nonzero.ne(nonzero.shift()).cumsum()
        age = nonzero.groupby(groups).cumsum()
        sig[age > p.max_hold] = 0.0

    return sig.fillna(0.0)


def _eval(df: pd.DataFrame, p: Program, settings, fee_mult: float = 1.0) -> dict:
    sig = _signal(df, p)
    result = run_ohlcv(
        df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
        market_type="spot", leverage=1.0, funding_rates=None,
    )
    return _metrics(result, result.returns)


def _wf(df: pd.DataFrame, p: Program, settings) -> dict:
    n = len(df)
    hold_start = int(n * 0.55)
    hold = df.iloc[hold_start:]
    fold_n = len(hold) // 4
    folds = []
    max_warmup = 120
    for i in range(4):
        test_a = i * fold_n
        test_b = len(hold) if i == 3 else (i + 1) * fold_n
        global_a = max(0, hold_start + test_a - max_warmup)
        global_b = hold_start + test_b
        chunk = df.iloc[global_a:global_b]
        test_from = hold_start + test_a - global_a
        warm = chunk.iloc[:test_from]
        test = chunk.iloc[test_from:]
        if len(test) < 40:
            continue
        # Warm-up is used solely to initialize rolling features; performance is measured on test.
        combo = pd.concat([warm, test])
        m = _eval(combo, p, settings, 1.0)
        # Re-evaluation on test alone prevents warm-up returns from contaminating reported returns.
        mt = _eval(test, p, settings, 1.0)
        folds.append(mt)
    rets = [float(x["total_return"]) for x in folds]
    pfs = [float(x["profit_factor"]) for x in folds]
    trades = [int(x["trade_count"]) for x in folds]
    return {
        "folds": folds,
        "positive": sum(r > 0 for r in rets),
        "median_return": float(np.median(rets)) if rets else 0.0,
        "median_pf": float(np.median(pfs)) if pfs else 0.0,
        "min_trades": min(trades) if trades else 0,
    }


def _score(normal: dict, stress: dict, wf: dict, p: Program) -> float:
    ret = float(normal.get("total_return", 0.0))
    pf = float(normal.get("profit_factor", 0.0))
    dd = abs(min(0.0, float(normal.get("max_drawdown", 0.0))))
    sharpe = float(normal.get("sharpe", 0.0))
    trades = int(normal.get("trade_count", 0))
    long_exp = float(normal.get("long_exposure", 0.0))
    short_exp = float(normal.get("short_exposure", 0.0))
    balance_penalty = abs(long_exp - short_exp) * 25.0
    one_sided_penalty = 12.0 if (long_exp == 0 or short_exp == 0) else 0.0
    complexity_penalty = max(0, p.size() - 9) * 1.5
    churn_penalty = max(0, trades - 120) * 0.05
    return (
        100 * ret + 18 * (min(2.5, pf) - 1.0) + 3 * sharpe - 28 * dd
        + 55 * wf.get("median_return", 0.0)
        + 8 * max(0, wf.get("positive", 0) - 1)
        - balance_penalty - one_sided_penalty - complexity_penalty - churn_penalty
    )


def _primary_ok(normal: dict, stress: dict, wf: dict) -> bool:
    return bool(
        float(normal["total_return"]) > 0
        and float(normal["profit_factor"]) > 1.05
        and float(normal["max_drawdown"]) >= -0.35
        and int(normal["trade_count"]) >= 8
        and float(stress["total_return"]) > 0
        and float(stress["profit_factor"]) > 1.0
        and wf["positive"] >= 3
        and wf["median_return"] > 0
        and wf["median_pf"] > 1.0
        and wf["min_trades"] >= 4
    )


def run(minutes=180.0, initial_population=64, population=16, generations=30, seed=20260829):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60
    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "evaluated": 0})
    print("=== GENETIC PROGRAM DISCOVERY V4 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Normalized primitives | balanced directions | warm-up WF | complexity penalty", flush=True)
    print("Checkpoint written before data loading", flush=True)

    data = {}
    for symbol, tf, bars in (("ETH/USDT", "1h", 800), ("ETH/USDT", "4h", 800), ("BTC/USDT", "4h", 800)):
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        _save({"decision": "LOADED", "market": f"{symbol} {tf}", "updated_at": datetime.now(timezone.utc).isoformat()})
        gc.collect()

    rng = random.Random(seed)
    all_seen = set()
    current = []
    direction_cycle = list(DIRECTIONS)
    while len(current) < min(initial_population, 64):
        p = random_program(rng)
        sig = json.dumps(p.params(), sort_keys=True)
        if sig in all_seen:
            continue
        all_seen.add(sig)
        current.append(p)

    all_results = []
    eval_count = 0
    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline:
            break
        print(f"\n=== GENERATION {gen}/{generations} | population={len(current)} ===", flush=True)
        gen_results = []
        for i, p in enumerate(current, 1):
            if time.monotonic() >= deadline:
                break
            try:
                market_results = {}
                for key, df in data.items():
                    normal = _eval(df.iloc[int(len(df) * 0.70):], p, settings, 1.0)
                    stress = _eval(df.iloc[int(len(df) * 0.70):], p, settings, 2.0)
                    wf = _wf(df, p, settings)
                    market_results[f"{key[0]} {key[1]}"] = {"normal": normal, "stress": stress, "wf": wf, "pass": _primary_ok(normal, stress, wf)}
                primary = sum(x["pass"] for x in market_results.values())
                score = float(np.mean([_score(x["normal"], x["stress"], x["wf"], p) for x in market_results.values()]))
                rec = {"title": p.title(), "program": p.params(), "score": score, "primary_passes": primary, "markets": market_results}
                gen_results.append(rec); all_results.append(rec); eval_count += 1
                print(f"eval {i}/{len(current)} | score={score:.2f} | primary={primary}/3 | {p.title()}", flush=True)
            except Exception as exc:
                print(f"eval {i}/{len(current)} | ERROR {type(exc).__name__}: {exc}", flush=True)
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "SEARCHING", "generation": gen, "evaluated": eval_count, "best": max(all_results, key=lambda x: x["score"]) if all_results else None})
            gc.collect()

        if not gen_results:
            break
        gen_results.sort(key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
        elite = gen_results[:max(3, population // 3)]
        next_pop = [Program(dict_to_node(e["program"]["tree"]), e["program"]["transform"], e["program"]["direction"], e["program"]["threshold"], e["program"]["exit_rule"], e["program"]["max_hold"], e["program"]["persistence"]) for e in elite]
        while len(next_pop) < population:
            parent = rng.choice(next_pop)
            child = mutate(parent, rng)
            sig = json.dumps(child.params(), sort_keys=True)
            if sig not in all_seen:
                all_seen.add(sig); next_pop.append(child)
            elif rng.random() < 0.15:
                fresh = random_program(rng)
                fsig = json.dumps(fresh.params(), sort_keys=True)
                if fsig not in all_seen:
                    all_seen.add(fsig); next_pop.append(fresh)
        current = next_pop

    ranked = sorted(all_results, key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
    finalists = ranked[: min(population, len(ranked))]

    print("\n=== FRESH UNTOUCHED CONFIRMATION ===", flush=True)
    for symbol, tf, bars in (("BTC/USDT", "1h", 800), ("ETH/USDT", "15m", 800)):
        print(f"LAZY LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)

    confirmed = []
    for idx, r in enumerate(finalists, 1):
        p = Program(dict_to_node(r["program"]["tree"]), r["program"]["transform"], r["program"]["direction"], r["program"]["threshold"], r["program"]["exit_rule"], r["program"]["max_hold"], r["program"]["persistence"])
        checks = []
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            normal = _eval(data[key], p, settings, 1.0)
            stress = _eval(data[key], p, settings, 2.0)
            ok = bool(float(normal["total_return"]) > 0 and float(normal["profit_factor"]) > 1.05 and int(normal["trade_count"]) >= 8 and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1.0)
            checks.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"# {idx} {key[0]} {key[1]} | return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
        out = dict(r); out["fresh"] = checks; out["fresh_passes"] = sum(x["pass"] for x in checks); out["validated"] = bool(r["primary_passes"] == 3 and out["fresh_passes"] == 2); confirmed.append(out)
        _save({"decision": "CONFIRMING", "evaluated": eval_count, "finalist": idx, "finalists": confirmed})
        if out["validated"]:
            print("VALIDATED GENETIC PROGRAM FOUND — stopping.", flush=True)
            break
        gc.collect()

    validated = [x for x in confirmed if x["validated"]]
    decision = "VALIDATED_GENETIC_PROGRAM" if validated else "NO_VALIDATED_GENETIC_PROGRAM"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generated": len(all_seen), "evaluated": eval_count, "finalists": len(finalists), "validated_count": len(validated), "winner": validated[0] if validated else (confirmed[0] if confirmed else None), "top_confirmed": confirmed}
    _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", OUT, flush=True)
    return payload


if __name__ == "__main__":
    run()
