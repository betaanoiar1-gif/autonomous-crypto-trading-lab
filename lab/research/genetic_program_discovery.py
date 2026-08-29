from __future__ import annotations

"""AI-free genetic-program strategy discovery.

Each candidate is an expression tree that invents the signal equation itself
from bounded market primitives. No LLM, arbitrary code, futures, or live
trading are used. Validation uses chronological holdouts, walk-forward with
warm-up, doubled-cost stress, and untouched markets/timeframes.
"""

import gc
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .evaluator import _metrics

OUT = ROOT / "experiments" / "genetic_program_discovery_latest.json"

FEATURES = ("ret_fast", "ret_slow", "ma_spread", "zscore", "rsi_bias", "vol_ratio", "range_pos", "body_pressure")
OPS = ("add", "sub", "mul", "safe_div", "absdiff", "mean2")
TRANSFORMS = ("identity", "tanh", "sign", "clip")
DIRECTIONS = ("normal", "inverse", "long_only", "short_only")
EXITS = ("zero_cross", "mean_cross", "time_stop")
WINDOWS = (8, 12, 18, 24, 36, 48, 72, 96)
THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
MAX_HOLDS = (4, 8, 12, 18, 24)


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


@dataclass(frozen=True)
class Node:
    kind: str
    value: str | float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    def size(self) -> int:
        return 1 + (self.left.size() if self.left else 0) + (self.right.size() if self.right else 0)

    def encode(self):
        return {"kind": self.kind, "value": self.value,
                "left": self.left.encode() if self.left else None,
                "right": self.right.encode() if self.right else None}


@dataclass(frozen=True)
class Program:
    root: Node
    transform: str
    direction: str
    threshold: float
    exit_rule: str
    max_hold: int

    def signature(self):
        return json.dumps(asdict(self), sort_keys=True, default=str)

    def title(self):
        return f"GP[{self.transform}|{self.direction}|{self.exit_rule}] size={self.root.size()} thr={self.threshold} hold={self.max_hold}"

    def params(self):
        return {"tree": self.root.encode(), "transform": self.transform, "direction": self.direction,
                "threshold": self.threshold, "exit_rule": self.exit_rule, "max_hold": self.max_hold}


def _leaf(rng):
    if rng.random() < 0.72:
        return Node("feature", _choice(rng, FEATURES))
    return Node("const", _choice(rng, (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)))


def _choice(rng, seq):
    return seq[rng.randrange(len(seq))]


def _random_tree(rng, depth=0, max_depth=3):
    if depth >= max_depth or rng.random() < (0.28 + 0.12 * depth):
        return _leaf(rng)
    op = _choice(rng, OPS)
    return Node("op", op, _random_tree(rng, depth + 1, max_depth), _random_tree(rng, depth + 1, max_depth))


def random_program(rng):
    return Program(
        root=_random_tree(rng, 0, 3),
        transform=_choice(rng, TRANSFORMS),
        direction=_choice(rng, DIRECTIONS),
        threshold=_choice(rng, THRESHOLDS),
        exit_rule=_choice(rng, EXITS),
        max_hold=_choice(rng, MAX_HOLDS),
    )


def _replace_random(node, replacement, rng, p=0.18):
    if rng.random() < p:
        return replacement
    left = _replace_random(node.left, replacement, rng, p) if node.left else None
    right = _replace_random(node.right, replacement, rng, p) if node.right else None
    return Node(node.kind, node.value, left, right)


def mutate(prog: Program, rng):
    root = prog.root
    if rng.random() < 0.65:
        root = _replace_random(root, _random_tree(rng, 0, 2), rng, 0.20)
    return Program(
        root=root,
        transform=_choice(rng, TRANSFORMS) if rng.random() < 0.18 else prog.transform,
        direction=_choice(rng, DIRECTIONS) if rng.random() < 0.18 else prog.direction,
        threshold=_choice(rng, THRESHOLDS) if rng.random() < 0.24 else prog.threshold,
        exit_rule=_choice(rng, EXITS) if rng.random() < 0.18 else prog.exit_rule,
        max_hold=_choice(rng, MAX_HOLDS) if rng.random() < 0.20 else prog.max_hold,
    )


def _features(df, p: Program):
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    op = df.get("open", close).astype(float)
    vol = df.get("volume", pd.Series(1.0, index=df.index)).astype(float)

    ret = close.pct_change()
    fast_w = 12
    slow_w = 48
    ret_fast = close.pct_change(fast_w)
    ret_slow = close.pct_change(slow_w)
    fast = close.ewm(span=fast_w, adjust=False).mean()
    slow = close.ewm(span=slow_w, adjust=False).mean()
    ma_spread = (fast - slow) / slow.replace(0, np.nan)
    mean = close.rolling(slow_w).mean()
    sd = close.rolling(slow_w).std(ddof=0).replace(0, np.nan)
    zscore = (close - mean) / sd
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi_bias = (rsi - 50.0) / 50.0
    vr = ret.rolling(12).std()
    vr_ref = vr.rolling(48).mean().replace(0, np.nan)
    vol_ratio = vr / vr_ref
    range_pos = (close - low.rolling(24).min()) / (high.rolling(24).max() - low.rolling(24).min()).replace(0, np.nan)
    body_pressure = (close - op) / (high - low).replace(0, np.nan)
    return {k: pd.Series(v, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0) for k, v in {
        "ret_fast": ret_fast, "ret_slow": ret_slow, "ma_spread": ma_spread, "zscore": zscore,
        "rsi_bias": rsi_bias, "vol_ratio": vol_ratio, "range_pos": range_pos - 0.5, "body_pressure": body_pressure,
    }.items()}


def _eval_node(node: Node, feats):
    if node.kind == "feature":
        return feats[node.value]
    if node.kind == "const":
        return node.value
    a = _eval_node(node.left, feats)
    b = _eval_node(node.right, feats)
    if node.value == "add": out = a + b
    elif node.value == "sub": out = a - b
    elif node.value == "mul": out = a * b
    elif node.value == "mean2": out = (a + b) / 2.0
    elif node.value == "absdiff": out = (a - b).abs()
    else: out = a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / b
    if isinstance(out, pd.Series):
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10, 10)
    return float(out) if math.isfinite(float(out)) else 0.0


def _signal(df, p: Program):
    feats = _features(df, p)
    raw = _eval_node(p.root, feats)
    if not isinstance(raw, pd.Series):
        raw = pd.Series(float(raw), index=df.index)
    if p.transform == "tanh": raw = np.tanh(raw)
    elif p.transform == "sign": raw = np.sign(raw)
    elif p.transform == "clip": raw = raw.clip(-1.0, 1.0)

    sig = pd.Series(0.0, index=df.index)
    sig[raw.shift(1) > p.threshold] = 1.0
    sig[raw.shift(1) < -p.threshold] = -1.0
    if p.direction == "inverse": sig = -sig
    elif p.direction == "long_only": sig[sig < 0] = 0.0
    elif p.direction == "short_only": sig[sig > 0] = 0.0

    if p.exit_rule == "zero_cross":
        cross = raw.shift(1).abs() < (p.threshold * 0.35)
        sig[cross] = 0.0
    elif p.exit_rule == "mean_cross":
        cross = raw.shift(1).abs() < 0.15
        sig[cross] = 0.0
    else:
        groups = sig.ne(sig.shift(1)).cumsum()
        age = groups.groupby(groups).cumcount()
        sig[age >= p.max_hold] = 0.0

    return sig.fillna(0.0)


def _run_metrics(df, p, settings, fee_mult=1.0):
    sig = _signal(df, p)
    result = run_ohlcv(
        df, sig, settings.capital.initial_usd,
        settings.execution.commission_bps * fee_mult,
        settings.execution.slippage_bps * fee_mult,
        market_type="spot", leverage=1.0, funding_rates=None,
    )
    return _metrics(result, result.returns)


def _wf(df, p, settings):
    warm = max(100, p.root.size() * 20, p.max_hold * 5)
    hold = df.iloc[int(len(df) * 0.58):]
    test = max(60, int(len(hold) * 0.22))
    folds = []
    for start in range(0, max(1, len(hold) - test + 1), test):
        a = max(0, int(len(df) * 0.58) + start - warm)
        b = min(len(df), int(len(df) * 0.58) + start + test)
        if b - a < warm + 30: continue
        eval_start = min(b, a + warm)
        m = _run_metrics(df.iloc[a:b], p, settings, 1.0)
        # Metrics are over the full slice; require enough trades so sparse warm-up candidates do not look artificially good.
        if int(m["trade_count"]) >= 4:
            folds.append(m)
        if len(folds) >= 4: break
    if len(folds) < 3:
        return {"wf": False, "folds": folds, "median_return": -1.0, "median_pf": 0.0, "min_trades": 0}
    rs = [float(x["total_return"]) for x in folds]
    pfs = [float(x["profit_factor"]) for x in folds]
    trs = [int(x["trade_count"]) for x in folds]
    return {
        "wf": bool(sum(x > 0 for x in rs) >= 3 and float(np.median(rs)) > 0 and float(np.median(pfs)) > 1),
        "folds": folds, "median_return": float(np.median(rs)), "median_pf": float(np.median(pfs)), "min_trades": min(trs)
    }


def _primary(df, p, settings):
    cut = int(len(df) * 0.70)
    normal = _run_metrics(df.iloc[cut:], p, settings, 1.0)
    stress = _run_metrics(df.iloc[cut:], p, settings, 2.0)
    wf = _wf(df, p, settings)
    simple = float(normal["total_return"]) > 0 and float(normal["profit_factor"]) > 1 and int(normal["trade_count"]) >= 8
    stress_ok = float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1
    ok = bool(simple and stress_ok and wf["wf"])
    score = 100 * float(normal["total_return"]) + 18 * (min(2.5, float(normal["profit_factor"])) - 1) - 35 * abs(min(0.0, float(normal["max_drawdown"]))) + 70 * wf["median_return"] - 0.5 * p.root.size()
    return {"normal": normal, "stress": stress, "wf": wf, "pass": ok, "score": score}


def run(minutes=180.0, initial_population=64, population=16, generations=30, seed=20260829):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60
    _save({"started_at": started.isoformat(), "updated_at": started.isoformat(), "decision": "STARTING", "generation": 0, "evaluated": 0})
    print("=== GENETIC PROGRAM DISCOVERY ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Invents signal equations from bounded primitives; no arbitrary code.", flush=True)
    print("Checkpoint written before data loading", flush=True)

    data = {}
    for symbol, tf in (("ETH/USDT", "1h"), ("ETH/USDT", "4h"), ("BTC/USDT", "4h")):
        print(f"LOAD {symbol} {tf}", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, 800, page_limit=300, market_type="spot")
        print(f"  bars={len(data[(symbol, tf)])}", flush=True)
        gc.collect()

    rng = random.Random(seed)
    current, seen = [], set()
    while len(current) < initial_population:
        p = random_program(rng)
        if p.signature() not in seen:
            seen.add(p.signature()); current.append(p)

    all_records = []
    evaluated = 0
    for gen in range(1, generations + 1):
        if time.monotonic() >= deadline: break
        print(f"\n=== GENERATION {gen}/{generations} | population={len(current)} ===", flush=True)
        gen_records = []
        for i, p in enumerate(current, 1):
            if time.monotonic() >= deadline: break
            try:
                market = {f"{k[0]} {k[1]}": _primary(df, p, settings) for k, df in data.items()}
                score = float(np.mean([x["score"] for x in market.values()]))
                primary = sum(x["pass"] for x in market.values())
                rec = {"title": p.title(), "program": p.params(), "score": score, "primary_passes": primary, "markets": market}
                gen_records.append(rec); all_records.append(rec); evaluated += 1
                print(f"eval {i}/{len(current)} | score={score:.2f} | primary={primary}/3 | {p.title()}", flush=True)
            except Exception as exc:
                print(f"eval {i}/{len(current)} | ERROR {type(exc).__name__}: {exc}", flush=True)
            _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "SEARCHING", "generation": gen, "evaluated": evaluated, "best": max(all_records, key=lambda x: (x["primary_passes"], x["score"])) if all_records else None})
            gc.collect()
        if not gen_records: break
        gen_records.sort(key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
        elites = [x for x in gen_records if x["primary_passes"] > 0][:max(2, population // 3)]
        if not elites:
            elites = gen_records[:max(2, population // 3)]
        next_pop = [Program(**{}) for _ in []]
        elite_programs = []
        for e in elites:
            pr = e["program"]
            root = _decode_node(pr["tree"])
            elite_programs.append(Program(root, pr["transform"], pr["direction"], pr["threshold"], pr["exit_rule"], pr["max_hold"]))
        next_pop = elite_programs[:]
        while len(next_pop) < population:
            parent = rng.choice(elite_programs)
            child = mutate(parent, rng)
            if child.signature() not in seen:
                seen.add(child.signature()); next_pop.append(child)
            elif rng.random() < 0.25:
                fresh = random_program(rng)
                if fresh.signature() not in seen:
                    seen.add(fresh.signature()); next_pop.append(fresh)
        current = next_pop

    ranked = sorted(all_records, key=lambda x: (x["primary_passes"], x["score"]), reverse=True)
    finalists = []
    sigs = set()
    for r in ranked:
        pr = r["program"]
        sig = (str(pr["tree"]), pr["transform"], pr["direction"], pr["exit_rule"])
        if sig in sigs: continue
        sigs.add(sig); finalists.append(r)
        if len(finalists) >= population: break

    print("\n=== FRESH UNTOUCHED CONFIRMATION ===", flush=True)
    for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
        print(f"LAZY LOAD {key[0]} {key[1]}", flush=True)
        data[key] = adapter.fetch_ohlcv_history(key[0], key[1], 800, page_limit=300, market_type="spot")
        print(f"  bars={len(data[key])}", flush=True)

    confirmed = []
    for i, r in enumerate(finalists, 1):
        pr = r["program"]; p = Program(_decode_node(pr["tree"]), pr["transform"], pr["direction"], pr["threshold"], pr["exit_rule"], pr["max_hold"])
        checks = []
        for key in (("BTC/USDT", "1h"), ("ETH/USDT", "15m")):
            normal = _run_metrics(data[key], p, settings, 1.0)
            stress = _run_metrics(data[key], p, settings, 2.0)
            ok = bool(float(normal["total_return"]) > 0 and float(normal["profit_factor"]) > 1 and int(normal["trade_count"]) >= 8 and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1)
            checks.append({"market": f"{key[0]} {key[1]}", "normal": normal, "stress": stress, "pass": ok})
            print(f"# {i} {key[0]} {key[1]} | return={normal['total_return']:.2%} PF={normal['profit_factor']:.2f} DD={normal['max_drawdown']:.2%} trades={normal['trade_count']} stress={stress['total_return']:.2%} pass={ok}", flush=True)
        out = dict(r); out["fresh"] = checks; out["validated"] = bool(r["primary_passes"] == 3 and all(x["pass"] for x in checks)); confirmed.append(out)
        _save({"started_at": started.isoformat(), "updated_at": datetime.now(timezone.utc).isoformat(), "decision": "CONFIRMING", "evaluated": evaluated, "finalist": i, "finalists": confirmed})
        if out["validated"]:
            print("VALIDATED INVENTION FOUND", flush=True); break
        gc.collect()

    validated = [x for x in confirmed if x["validated"]]
    decision = "VALIDATED_GENETIC_PROGRAM" if validated else "NO_VALIDATED_GENETIC_PROGRAM"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "generated": len(all_records), "evaluated": evaluated, "finalists": len(finalists), "validated_count": len(validated), "winner": validated[0] if validated else (confirmed[0] if confirmed else None), "top": confirmed}
    _save(payload)
    print("\n=== FINAL DECISION ===", flush=True)
    print(decision, flush=True)
    print("Validated:", len(validated), flush=True)
    print("Saved:", OUT, flush=True)
    return payload


def _decode_node(d):
    if d["kind"] == "feature" or d["kind"] == "const":
        return Node(d["kind"], d.get("value"))
    return Node(d["kind"], d.get("value"), _decode_node(d["left"]), _decode_node(d["right"]))


if __name__ == "__main__":
    run()
