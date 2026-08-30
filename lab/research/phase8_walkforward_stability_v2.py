from __future__ import annotations

"""Phase 8: pre-registered walk-forward stability audit.

Tests only previously observed signals. No new feature discovery and no
parameter fitting on validation/lockbox. Primary criterion is temporal sign
consistency across contiguous blocks, supplemented by median IC/spread and a
block bootstrap probability of a positive mean.
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
CACHE = Path(os.getenv("PHASE8_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase8_walkforward_stability_v2_latest.json"
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
SIGNALS = (
    ("volume_pressure", 72),
    ("vol_compression", 24),
    ("vol_scaled_mom", 168),
    ("range_position_72", 168),
    ("vol_scaled_mom", 24),
    ("mom_24", 168),
)

@dataclass
class BlockRow:
    feature: str
    horizon: int
    block: int
    start: str
    end: str
    markets: int
    median_ic: float
    median_spread: float
    positive_markets: int
    positive_share: float


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
    for w in (6, 24, 72, 168):
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
    idx = sorted(set.intersection(*[set(x.index) for x in data.values()]))
    close = pd.DataFrame({s: data[s].loc[idx, "close"] for s in data}).sort_index()
    feats = {s: _features(data[s]).reindex(close.index) for s in data}
    return close, feats


def _evaluate_block(close: pd.DataFrame, feats: dict[str, pd.DataFrame], feature: str, horizon: int, a: int, b: int) -> tuple[float, float, int, int]:
    # Compute time-series average of cross-sectional rank IC and long-short spread
    # across timestamps in one contiguous block. All thresholds are fixed quartiles.
    ymat = close.pct_change(horizon).shift(-horizon)
    ics: list[float] = []
    spreads: list[float] = []
    by_market_ic: dict[str, list[float]] = {s: [] for s in close.columns}
    positive = 0
    total = 0
    idx = close.index[a:b]
    for t in idx:
        y = ymat.loc[t]
        x = pd.Series({s: feats[s].loc[t, feature] for s in close.columns})
        z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
        if len(z) < 8:
            continue
        rx = z["x"].rank(pct=True)
        ry = z["y"].rank(pct=True)
        ric = float(rx.corr(ry)) if len(z) > 2 else np.nan
        if np.isfinite(ric):
            ics.append(ric)
        q = rx
        lo = z.loc[q <= 0.25, "y"]
        hi = z.loc[q >= 0.75, "y"]
        if len(lo) and len(hi):
            spreads.append(float(hi.mean() - lo.mean()))
        total += 1
        positive += int((float(hi.mean()) - float(lo.mean())) > 0) if len(lo) and len(hi) else 0
    return (
        float(np.median(ics)) if ics else 0.0,
        float(np.median(spreads)) if spreads else 0.0,
        positive,
        total,
    )


def _market_block(close: pd.DataFrame, feats: dict[str, pd.DataFrame], feature: str, horizon: int, a: int, b: int) -> tuple[float, int]:
    ymat = close.pct_change(horizon).shift(-horizon)
    vals = []
    for m in close.columns:
        z = pd.concat([feats[m][feature].rename("x"), ymat[m].rename("y")], axis=1).iloc[a:b].dropna()
        if len(z) < 30:
            continue
        rx = z.x.rank(pct=True)
        ry = z.y.rank(pct=True)
        ic = float(rx.corr(ry))
        if np.isfinite(ic):
            vals.append(ic)
    if not vals:
        return 0.0, 0
    return float(np.median(vals)), int(sum(v > 0 for v in vals))


def _bootstrap_block_sign(values: np.ndarray, seed: int, trials: int = 500) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return 0.5
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(trials):
        sample = rng.choice(x, size=n, replace=True)
        out.append(float(np.mean(sample) > 0.0))
    return float(np.mean(out))


def _bootstrap_blocks(block_values: list[float], seed: int, trials: int = 2000) -> float:
    x = np.asarray(block_values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.5
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(trials, len(x)), replace=True).mean(axis=1)
    return float(np.mean(means > 0.0))


def _signal_summary(rows: list[dict], feature: str, horizon: int) -> dict:
    r = [x for x in rows if x["feature"] == feature and x["horizon"] == horizon]
    ics = np.asarray([x["median_ic"] for x in r], dtype=float)
    spreads = np.asarray([x["median_spread"] for x in r], dtype=float)
    signs_ic = int(np.sum(ics > 0))
    signs_sp = int(np.sum(spreads > 0))
    return {
        "feature": feature,
        "horizon": horizon,
        "blocks": len(r),
        "median_block_ic": float(np.median(ics)) if len(ics) else 0.0,
        "median_block_spread": float(np.median(spreads)) if len(spreads) else 0.0,
        "positive_ic_blocks": signs_ic,
        "positive_spread_blocks": signs_sp,
        "ic_sign_share": signs_ic / max(1, len(r)),
        "spread_sign_share": signs_sp / max(1, len(r)),
        "block_bootstrap_positive_mean": _bootstrap_blocks([float(v) for v in spreads], 91000 + horizon + len(feature)),
        "persistent_candidate": signs_sp >= max(4, int(0.75 * len(r))) and float(np.median(spreads)) > 0.0,
    }


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    if len(raw) < 8:
        payload = {"version": "phase8", "decision": "PHASE8_BLOCKED_DATA", "markets": list(raw)}
        _save(payload)
        return payload

    start = max(x.index[0] for x in raw.values())
    end = min(x.index[-1] for x in raw.values())
    data = {s: x.loc[(x.index >= start) & (x.index <= end)].copy() for s, x in raw.items()}
    close, feats = _panel(data)
    n = len(close)
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_start = dev_n + val_n

    print("=== PHASE 8 V2 WALK-FORWARD STABILITY ===", flush=True)
    print(f"Markets: {len(data)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)
    print("PRE-REGISTERED SIGNALS ONLY", flush=True)

    # Eight contiguous blocks across DEV+VALIDATION. Selection is not performed.
    audit_end = lock_start
    audit_start = 0
    block_edges = np.linspace(audit_start, audit_end, 9, dtype=int)
    block_rows: list[dict] = []

    for si, (feature, horizon) in enumerate(SIGNALS, 1):
        if time.monotonic() >= deadline:
            break
        for bi in range(8):
            a, b = int(block_edges[bi]), int(block_edges[bi + 1])
            if b - a < max(200, horizon * 3):
                continue
            ic, spread, pos, days = _evaluate_block(close, feats, feature, horizon, a, b)
            row = BlockRow(
                feature, horizon, bi + 1,
                close.index[a].isoformat(),
                close.index[b - 1].isoformat(),
                len(data), ic, spread, pos, pos / max(1, days),
            )
            block_rows.append(asdict(row))
            print(
                f"BLOCK {si}/{len(SIGNALS)} {bi+1}/8 | {feature} h={horizon} "
                f"IC={ic:.4f} spread={spread:.4%} positive={pos}/{days}",
                flush=True,
            )

    summaries = [_signal_summary(block_rows, f, h) for f, h in SIGNALS]

    print("=== LOCKBOX (AUDIT ONLY) ===", flush=True)
    lockbox = []
    # Lockbox is audited only for the pre-registered candidate signals; it does not select models.
    for i, (feature, horizon) in enumerate(SIGNALS, 1):
        if time.monotonic() >= deadline:
            break
        a, b = lock_start, n
        ic, spread, pos, days = _evaluate_block(close, feats, feature, horizon, a, b)
        item = {
            "feature": feature,
            "horizon": horizon,
            "median_ic": ic,
            "median_spread": spread,
            "positive_observations": pos,
            "observations": days,
            "positive_share": pos / max(1, days),
        }
        lockbox.append(item)
        print(
            f"LOCKBOX {i}/{len(SIGNALS)} | {feature} h={horizon} "
            f"IC={ic:.4f} spread={spread:.4%} positive={pos}/{days}",
            flush=True,
        )

    persistent = [x for x in summaries if x["persistent_candidate"] and x["block_bootstrap_positive_mean"] >= 0.80]
    lockbox_confirmed = [x for x in lockbox if x["median_spread"] > 0.0 and x["positive_share"] >= 0.50]
    eligible = [x for x in persistent if any(y["feature"] == x["feature"] and y["horizon"] == x["horizon"] for y in lockbox_confirmed)]

    decision = "PHASE8_PERSISTENT_EDGE_FOUND" if eligible else "PHASE8_NO_STABLE_EDGE"
    payload = {
        "version": "phase8_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "markets": list(data),
        "bars": n,
        "split": {"dev": dev_n, "validation": val_n, "lockbox": n - lock_start},
        "signals": list(SIGNALS),
        "blocks": block_rows,
        "summaries": summaries,
        "lockbox": lockbox,
        "eligible": eligible,
        "protocol": {
            "new_discovery": False,
            "contiguous_blocks": 8,
            "selection_on_lockbox": False,
            "primary_metric": "positive spread block share",
            "bootstrap_metric": "probability mean block spread > 0",
            "spot_only": True,
        },
    }
    _save(payload)
    print("=== PHASE 8 V2 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(float(os.getenv("PHASE8_MINUTES", "60")), int(os.getenv("PHASE8_SEED", "20260829")))
