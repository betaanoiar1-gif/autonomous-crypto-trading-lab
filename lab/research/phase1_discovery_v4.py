from __future__ import annotations

"""Phase 1 V4: strict anti-overfit strategy discovery.

Discovery only uses DEVELOPMENT data. Validation is used only after a
candidate has been frozen for selection. Lockbox is opened only for the
final three candidates and never influences evolution.
"""

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
OUT = ROOT / "experiments" / "phase1_discovery_v4_latest.json"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)

SCREEN_MARKETS = ("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT")


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


@dataclass
class Eval:
    ret: float
    pf: float
    dd: float
    stress_ret: float
    stress_pf: float
    trades: int
    turnover: float
    exposure: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/', '_')}_1h.parquet"
        if p.exists():
            out[s] = pd.read_parquet(p).sort_index()
    return out


def _common_cut(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    start = max(df.index[0] for df in data.values())
    end = min(df.index[-1] for df in data.values())
    return {k: v.loc[(v.index >= start) & (v.index <= end)].copy() for k, v in data.items()}


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


def _bt(df: pd.DataFrame, g: Genome, settings, cost_mult: float = 1.0) -> Eval:
    result = run_ohlcv(
        df,
        _signal(df, g),
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    m = result.metrics
    return Eval(
        ret=float(m.get("total_return", 0.0)),
        pf=float(m.get("profit_factor", 0.0)),
        dd=float(m.get("max_drawdown", 0.0)),
        stress_ret=float(m.get("total_return", 0.0)),
        stress_pf=float(m.get("profit_factor", 0.0)),
        trades=int(m.get("trade_count", 0)),
        turnover=float(m.get("trade_turnover", 0.0)),
        exposure=float(m.get("exposure", 0.0)),
    )


def _window_scores(df: pd.DataFrame, g: Genome, settings, windows: int = 5) -> list[Eval]:
    n = len(df)
    if n < 5000:
        return []
    width = n // windows
    out: list[Eval] = []
    for i in range(windows):
        lo = i * width
        hi = n if i == windows - 1 else (i + 1) * width
        part = df.iloc[lo:hi].copy()
        if len(part) < 700:
            continue
        out.append(_bt(part, g, settings, 1.0))
    return out


def _robustness(dev: dict[str, pd.DataFrame], g: Genome, settings) -> dict:
    xs: list[Eval] = []
    for m in SCREEN_MARKETS:
        if m not in dev:
            continue
        xs.append(_bt(dev[m].iloc[-16000:].copy(), g, settings, 1.0))

    if not xs:
        return {"utility": -1e9, "ret": 0.0, "pf": 0.0, "dd": 0.0, "stress": 0.0, "wf_pos": 0, "dispersion": 1.0, "trades": 0}

    # Stability is measured only on DEVELOPMENT data.
    wf_returns: list[float] = []
    wf_pfs: list[float] = []
    wf_positive = 0
    for m in SCREEN_MARKETS[:4]:
        if m in dev:
            for e in _window_scores(dev[m], g, settings, windows=5):
                wf_returns.append(e.ret)
                wf_pfs.append(e.pf)
                wf_positive += int(e.ret > 0 and e.pf > 1.0)

    ret = float(np.median([x.ret for x in xs]))
    pf = float(np.median([x.pf for x in xs]))
    dd = float(np.median([x.dd for x in xs]))
    stress = float(np.median([_bt(dev[m].iloc[-16000:].copy(), g, settings, 2.0).ret for m in SCREEN_MARKETS if m in dev]))
    dispersion = float(np.median([abs(x.ret - ret) for x in xs]))
    trades = int(np.median([x.trades for x in xs]))

    utility = (
        55.0 * ret
        + 20.0 * max(0.0, pf - 1.0)
        + 7.0 * wf_positive
        + 22.0 * max(0.0, stress)
        - 35.0 * max(0.0, -dd - 0.12)
        - 18.0 * max(0.0, -stress)
        - 35.0 * dispersion
        - 0.004 * trades
    )
    return {
        "utility": float(utility),
        "ret": ret,
        "pf": pf,
        "dd": dd,
        "stress": stress,
        "wf_pos": int(wf_positive),
        "dispersion": dispersion,
        "trades": trades,
    }


def _pool(seed: int, count: int) -> list[Genome]:
    rng = np.random.default_rng(seed)
    families = ["momentum", "breakout", "mean_reversion", "trend_pullback", "vol_breakout", "ma_cross"]
    lbs = [24, 48, 72, 120, 168, 240, 336]
    fasts = [6, 8, 12, 18, 24, 36, 48]
    slows = [40, 60, 90, 120, 180, 240, 360]
    thresholds = [0.0025, 0.005, 0.0075, 0.01, 0.015]
    exits = [0.0015, 0.003, 0.005, 0.0075, 0.01]
    vols = [12, 24, 36, 48, 72]
    caps = [0.012, 0.018, 0.025, 0.035, 0.05]
    holds = [24, 48, 72, 96, 120]
    out: list[Genome] = []
    seen: set[Genome] = set()
    while len(out) < count:
        fast = int(rng.choice(fasts))
        slow = int(rng.choice(slows))
        if fast >= slow:
            continue
        g = Genome(
            str(rng.choice(families)),
            int(rng.choice(lbs)),
            fast,
            slow,
            float(rng.choice(thresholds)),
            float(rng.choice(exits)),
            int(rng.choice(vols)),
            float(rng.choice(caps)),
            int(rng.choice(holds)),
        )
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def _mutate(g: Genome, rng: np.random.Generator) -> Genome:
    family = g.family if rng.random() > 0.10 else str(rng.choice(["momentum", "breakout", "mean_reversion", "trend_pullback", "vol_breakout", "ma_cross"]))
    lookback = max(18, min(400, g.lookback + int(rng.choice([-48, -24, 0, 24, 48]))))
    fast = max(6, min(72, g.fast + int(rng.choice([-6, -3, 0, 3, 6]))))
    slow = max(40, min(420, g.slow + int(rng.choice([-30, -15, 0, 15, 30]))))
    if fast >= slow:
        fast = max(6, slow - 6)
    return Genome(
        family,
        lookback,
        fast,
        slow,
        float(np.clip(g.threshold + rng.choice([-0.0025, 0.0, 0.0025]), 0.0015, 0.025)),
        float(np.clip(g.exit_threshold + rng.choice([-0.001, 0.0, 0.001]), 0.0005, 0.012)),
        max(8, min(96, g.vol_window + int(rng.choice([-12, 0, 12])))),
        float(np.clip(g.vol_cap + rng.choice([-0.005, 0.0, 0.005]), 0.01, 0.06)),
        max(24, min(144, g.hold_bars + int(rng.choice([-24, 0, 24])))),
    )


def _parameter_stability(dev: dict[str, pd.DataFrame], g: Genome, settings) -> dict:
    rng = np.random.default_rng(hash(g) & 0xffffffff)
    neighbors: list[float] = []
    for _ in range(5):
        ng = _mutate(g, rng)
        xs = [_bt(dev[m].iloc[-12000:].copy(), ng, settings, 1.0).ret for m in SCREEN_MARKETS if m in dev]
        if xs:
            neighbors.append(float(np.median(xs)))
    if not neighbors:
        return {"median": 0.0, "minimum": 0.0, "count": 0}
    return {"median": float(np.median(neighbors)), "minimum": float(np.min(neighbors)), "count": len(neighbors)}


def run(minutes: float = 180.0, initial_population: int = 64, population: int = 16, generations: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    raw = _load()

    print("=== PHASE 1 DISCOVERY V4 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print(f"Markets loaded: {len(raw)}", flush=True)
    if len(raw) < 8:
        payload = {"decision": "DISCOVERY_BLOCKED_DATA", "markets": list(raw), "version": "v4"}
        _save(payload)
        return payload

    data = _common_cut(raw)
    n = min(len(df) for df in data.values())
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_n = n - dev_n - val_n
    dev = {m: df.iloc[:dev_n].copy() for m, df in data.items()}
    val = {m: df.iloc[dev_n:dev_n + val_n].copy() for m, df in data.items()}
    lock = {m: df.iloc[dev_n + val_n:].copy() for m, df in data.items()}

    print(f"COMMON CUTOFF: {min(df.index[-1] for df in data.values())}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={lock_n}", flush=True)
    print("LOCKBOX: RESERVED", flush=True)

    rng = np.random.default_rng(seed + 101)
    genes = _pool(seed, initial_population)
    evaluations = 0
    finalists: list[Genome] = []
    selected_validation: list[dict] = []

    for gen in range(generations):
        if time.monotonic() >= deadline:
            break
        ranked: list[tuple[float, Genome, dict]] = []
        print(f"=== GENERATION {gen + 1}/{generations} ===", flush=True)
        for i, g in enumerate(genes[:population], 1):
            if time.monotonic() >= deadline:
                break
            r = _robustness(dev, g, settings)
            stab = _parameter_stability(dev, g, settings)
            utility = r["utility"] + 12.0 * max(0.0, stab["minimum"])
            ranked.append((utility, g, {**r, "stability": stab}))
            evaluations += 1
            print(
                f"eval {i}/{min(population, len(genes))} | util={utility:.2f} "
                f"ret={r['ret']:.2%} PF={r['pf']:.2f} DD={r['dd']:.2%} "
                f"stress={r['stress']:.2%} WF={r['wf_pos']} "
                f"stab_min={stab['minimum']:.2%}",
                flush=True,
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        # Evolution sees DEV only. No validation values enter ranking.
        elites = [x[1] for x in ranked[:max(2, population // 4)]]
        finalists.extend(x[1] for x in ranked[:max(6, population // 2)])
        children = []
        for elite in elites:
            for _ in range(4):
                children.append(_mutate(elite, rng))
        genes = list(dict.fromkeys(elites + children))

    finalists = list(dict.fromkeys(finalists))[:12]

    # First gate: validation is evaluated only after evolution is frozen.
    print("=== FROZEN VALIDATION ===", flush=True)
    for rank, g in enumerate(finalists, 1):
        if time.monotonic() >= deadline:
            break
        rows = []
        for m, df in val.items():
            normal = _bt(df, g, settings, 1.0)
            stress = _bt(df, g, settings, 2.0)
            rows.append({"market": m, "normal": asdict(normal), "stress": asdict(stress)})
        rets = [x["normal"]["ret"] for x in rows]
        srets = [x["stress"]["ret"] for x in rows]
        pfs = [x["normal"]["pf"] for x in rows]
        spfs = [x["stress"]["pf"] for x in rows]
        positive = sum(r > 0 and sr > 0 and pf > 1.0 and spf > 1.0 for r, sr, pf, spf in zip(rets, srets, pfs, spfs))
        stab = _parameter_stability(dev, g, settings)
        summary = {
            "rank": rank,
            "genome": asdict(g),
            "median_return": float(np.median(rets)),
            "median_stress_return": float(np.median(srets)),
            "median_pf": float(np.median(pfs)),
            "median_stress_pf": float(np.median(spfs)),
            "positive_markets": int(positive),
            "stability_min": float(stab["minimum"]),
            "markets": rows,
        }
        selected_validation.append(summary)
        print(
            f"VALIDATION {rank}/{len(finalists)} | ret={summary['median_return']:.2%} "
            f"PF={summary['median_pf']:.2f} stress={summary['median_stress_return']:.2%} "
            f"positive={positive}/{len(rows)} stab_min={stab['minimum']:.2%}",
            flush=True,
        )

    # Validation itself now selects only the strongest stable candidates.
    selected_validation.sort(
        key=lambda x: (
            x["positive_markets"],
            x["stability_min"],
            x["median_stress_return"],
            x["median_return"],
        ),
        reverse=True,
    )

    lock_candidates = selected_validation[:3]
    print("=== LOCKBOX FINAL TEST ===", flush=True)
    lock_results = []
    for rank, item in enumerate(lock_candidates, 1):
        g = Genome(**item["genome"])
        rows = []
        for m, df in lock.items():
            normal = _bt(df, g, settings, 1.0)
            stress = _bt(df, g, settings, 2.0)
            rows.append({"market": m, "normal": asdict(normal), "stress": asdict(stress)})
        rets = [x["normal"]["ret"] for x in rows]
        srets = [x["stress"]["ret"] for x in rows]
        pfs = [x["normal"]["pf"] for x in rows]
        spfs = [x["stress"]["pf"] for x in rows]
        positive = sum(r > 0 and sr > 0 and pf > 1.0 and spf > 1.0 for r, sr, pf, spf in zip(rets, srets, pfs, spfs))
        summary = {
            "rank": rank,
            "genome": asdict(g),
            "median_return": float(np.median(rets)),
            "median_stress_return": float(np.median(srets)),
            "median_pf": float(np.median(pfs)),
            "median_stress_pf": float(np.median(spfs)),
            "positive_markets": int(positive),
            "markets": rows,
        }
        lock_results.append(summary)
        print(
            f"LOCKBOX {rank}/{len(lock_candidates)} | ret={summary['median_return']:.2%} "
            f"PF={summary['median_pf']:.2f} stress={summary['median_stress_return']:.2%} "
            f"positive={positive}/{len(rows)}",
            flush=True,
        )

    eligible = [
        x for x in lock_results
        if x["positive_markets"] >= max(6, len(lock) // 2)
        and x["median_return"] > 0
        and x["median_stress_return"] > 0
        and x["median_pf"] > 1.05
        and x["median_stress_pf"] > 1.0
    ]
    decision = "VALIDATED_STRATEGY_READY" if eligible else "NO_VALIDATED_STRATEGY"

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "version": "v4",
        "decision": decision,
        "evaluations": evaluations,
        "common_cutoff": str(min(df.index[-1] for df in data.values())),
        "split": {"development": dev_n, "validation": val_n, "lockbox": lock_n},
        "finalists": [asdict(g) for g in finalists],
        "validation": selected_validation,
        "lockbox": lock_results,
        "eligible": eligible,
        "protocol": {
            "spot_long_flat": True,
            "ai": False,
            "futures": False,
            "evolution_uses_validation": False,
            "lockbox_reserved_until_final": True,
            "parameter_stability_source": "development_only",
            "selection": "development robustness only",
        },
    }
    _save(payload)
    print("=== PHASE 1 V4 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("DISCOVERY_MINUTES", "180")),
        initial_population=int(os.getenv("DISCOVERY_INITIAL_POPULATION", "64")),
        population=int(os.getenv("DISCOVERY_POPULATION", "16")),
        generations=int(os.getenv("DISCOVERY_GENERATIONS", "12")),
        seed=int(os.getenv("DISCOVERY_SEED", "20260829")),
    )
