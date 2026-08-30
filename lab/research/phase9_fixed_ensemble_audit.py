from __future__ import annotations

"""Phase 9: fixed ensemble audit.

This phase does not discover or optimize new weights. It combines six
pre-registered signals with equal fixed weights and audits whether the
ensemble is more stable than its components across DEV, VALIDATION and
LOCKBOX. No futures, leverage, shorting, or live trading.
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
CACHE = Path(os.getenv("PHASE9_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase9_fixed_ensemble_audit_latest.json"
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)

# Pre-registered signals from Phases 3-6. Equal fixed weights only.
SIGNALS = (
    ("volume_pressure", 72),
    ("vol_compression", 24),
    ("vol_scaled_mom", 168),
    ("range_position_72", 168),
    ("vol_scaled_mom", 24),
    ("mom_24", 168),
)

@dataclass
class EvalRow:
    split: str
    feature: str
    horizon: int
    median_ic: float
    median_spread: float
    positive_times: int
    times: int
    markets_positive: int
    markets: int


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        p = CACHE / f"{s.replace('/', '_')}_1h.parquet"
        if not p.exists():
            continue
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
    for w in (6, 24):
        z[f"mom_{w}"] = c.pct_change(w)
    vol = r.rolling(24, min_periods=24).std().replace(0, np.nan)
    z["vol_scaled_mom"] = z["mom_24"] / vol
    z["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    z["vol_compression"] = r.rolling(12, min_periods=12).std() / r.rolling(72, min_periods=72).std().replace(0, np.nan)
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    z["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    z["range_position_72"] = ((c - ll) / (hh - ll).replace(0, np.nan)).clip(-1, 2)
    return z.replace([np.inf, -np.inf], np.nan)


def _panel(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    common = sorted(set.intersection(*[set(x.index) for x in data.values()]))
    close = pd.DataFrame({s: data[s].loc[common, "close"] for s in data}).sort_index()
    feats = {s: _features(data[s]).reindex(close.index) for s in data}
    return close, feats


def _signal_panel(close: pd.DataFrame, feats: dict[str, pd.DataFrame], feature: str, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.DataFrame({s: feats[s][feature] for s in close.columns}, index=close.index)
    y = close.pct_change(horizon).shift(-horizon)
    return x, y


def _cross_sectional_z(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mean, axis=0).div(std, axis=0)


def _ensemble(close: pd.DataFrame, feats: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = []
    returns = []
    for feature, horizon in SIGNALS:
        x, y = _signal_panel(close, feats, feature, horizon)
        # Equal fixed weight; cross-sectional z-score prevents one feature from
        # dominating because of scale. The corresponding future horizon is
        # aligned to the signal and averaged across the registered horizons.
        parts.append(_cross_sectional_z(x))
        returns.append(y)
    signal = sum(parts) / float(len(parts))
    # Average the six future-return horizons with fixed equal weights.
    target = sum(returns) / float(len(returns))
    return signal, target


def _evaluate(signal: pd.DataFrame, target: pd.DataFrame, lo: int, hi: int, split: str) -> EvalRow:
    idx = signal.index[lo:hi]
    ics = []
    spreads = []
    positive = 0
    market_pos = 0
    market_total = 0

    for t in idx:
        x = signal.loc[t]
        y = target.loc[t]
        z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(z) < 8:
            continue
        r = float(z["x"].corr(z["y"], method="spearman"))
        if not np.isfinite(r):
            continue
        q = z["x"].rank(pct=True)
        top = float(z.loc[q >= 0.75, "y"].mean())
        bottom = float(z.loc[q <= 0.25, "y"].mean())
        sp = top - bottom
        ics.append(r)
        spreads.append(sp)
        positive += int(sp > 0)

    # Market-level audit over the same frozen ensemble.
    for s in signal.columns:
        sx = signal[s].iloc[lo:hi]
        sy = target[s].iloc[lo:hi]
        z = pd.concat([sx.rename("x"), sy.rename("y")], axis=1).dropna()
        if len(z) >= 30:
            rr = float(z["x"].corr(z["y"], method="spearman"))
            if np.isfinite(rr):
                market_total += 1
                market_pos += int(rr > 0)

    return EvalRow(
        split=split,
        feature="fixed_equal_weight_ensemble",
        horizon=0,
        median_ic=float(np.median(ics)) if ics else 0.0,
        median_spread=float(np.median(spreads)) if spreads else 0.0,
        positive_times=positive,
        times=len(spreads),
        markets_positive=market_pos,
        markets=market_total,
    )


def _component_rows(close: pd.DataFrame, feats: dict[str, pd.DataFrame], lo: int, hi: int, split: str) -> list[dict]:
    rows = []
    for feature, horizon in SIGNALS:
        x, y = _signal_panel(close, feats, feature, horizon)
        x = _cross_sectional_z(x)
        r = _evaluate(x, y, lo, hi, split)
        r.feature = feature
        r.horizon = horizon
        rows.append(asdict(r))
    return rows


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    if len(raw) < 8:
        payload = {"version": "phase9", "decision": "PHASE9_BLOCKED_DATA", "markets": list(raw)}
        _save(payload)
        return payload

    close, feats = _panel(raw)
    n = len(close)
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_start = dev_n + val_n

    print("=== PHASE 9 FIXED ENSEMBLE AUDIT ===", flush=True)
    print(f"Markets: {len(close.columns)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)
    print("FIXED EQUAL WEIGHTS: 1/6 PER REGISTERED SIGNAL", flush=True)

    signal, target = _ensemble(close, feats)

    component_dev = _component_rows(close, feats, 0, dev_n, "dev")
    component_val = _component_rows(close, feats, dev_n, lock_start, "validation")
    component_lock = _component_rows(close, feats, lock_start, n, "lockbox")

    if time.monotonic() >= deadline:
        payload = {"version": "phase9", "decision": "PHASE9_TIMEOUT", "component_dev": component_dev}
        _save(payload)
        return payload

    print("=== DEV COMPONENTS ===", flush=True)
    for r in component_dev:
        print(f"DEV {r['feature']} h={r['horizon']} IC={r['median_ic']:.4f} spread={r['median_spread']:.4%}", flush=True)

    dev = _evaluate(signal, target, 0, dev_n, "dev")
    val = _evaluate(signal, target, dev_n, lock_start, "validation")
    lock = _evaluate(signal, target, lock_start, n, "lockbox")

    print("=== ENSEMBLE ===", flush=True)
    print(f"DEV | IC={dev.median_ic:.4f} spread={dev.median_spread:.4%} positive={dev.positive_times}/{dev.times} markets+={dev.markets_positive}/{dev.markets}", flush=True)
    print(f"VAL | IC={val.median_ic:.4f} spread={val.median_spread:.4%} positive={val.positive_times}/{val.times} markets+={val.markets_positive}/{val.markets}", flush=True)
    print(f"LOCK | IC={lock.median_ic:.4f} spread={lock.median_spread:.4%} positive={lock.positive_times}/{lock.times} markets+={lock.markets_positive}/{lock.markets}", flush=True)

    # Fixed, conservative eligibility: positive median IC/spread and at least
    # 60% positive timestamps in validation and lockbox. No fitting here.
    def eligible_row(r: EvalRow) -> bool:
        return (
            r.median_ic > 0.0
            and r.median_spread > 0.0
            and r.times >= 100
            and r.positive_times / max(1, r.times) >= 0.60
        )

    eligible = []
    if eligible_row(val) and eligible_row(lock):
        eligible.append(asdict(lock))

    decision = "PHASE9_FIXED_ENSEMBLE_SUPPORTED" if eligible else "PHASE9_NO_ENSEMBLE_SUPPORT"

    payload = {
        "version": "phase9",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision": decision,
        "signals": list(SIGNALS),
        "weights": {f"{f}:{h}": 1.0 / len(SIGNALS) for f, h in SIGNALS},
        "bars": n,
        "split": {"dev": dev_n, "validation": val_n, "lockbox": n - lock_start},
        "ensemble": {
            "dev": asdict(dev),
            "validation": asdict(val),
            "lockbox": asdict(lock),
        },
        "components": {
            "dev": component_dev,
            "validation": component_val,
            "lockbox": component_lock,
        },
        "eligible": eligible,
        "protocol": {
            "new_strategy_search": False,
            "weight_optimization": False,
            "equal_fixed_weights": True,
            "dev_selection": False,
            "validation_frozen": True,
            "lockbox_audit_only": True,
            "spot_only": True,
        },
    }
    _save(payload)
    print("=== PHASE 9 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        float(os.getenv("PHASE9_MINUTES", "60")),
        int(os.getenv("PHASE9_SEED", "20260829")),
    )
