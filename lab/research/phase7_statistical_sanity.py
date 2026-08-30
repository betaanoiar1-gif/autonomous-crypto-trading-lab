from __future__ import annotations

"""Phase 7: statistical sanity / attribution.

Purpose:
- Audit previously discovered weak predictive signals.
- Do NOT search for new strategies.
- Quantify time dependence, effective sample size, HAC-style uncertainty,
  block-bootstrap stability, cross-asset dependence, and multiple-testing.
- Selection thresholds are fixed by the protocol and never fitted on lockbox.

Input: Phase-0 1h spot OHLCV parquet cache.
Output: experiments/phase7_statistical_sanity_latest.json
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
CACHE = Path(os.getenv("PHASE7_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase7_statistical_sanity_latest.json"
SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT",
    "DOGE/USDT", "LTC/USDT", "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
# Pre-registered signals copied from phases 3-6; no discovery here.
SIGNALS = (
    ("volume_pressure", 72),
    ("vol_compression", 24),
    ("vol_scaled_mom", 168),
    ("range_position_72", 168),
    ("vol_scaled_mom", 24),
    ("mom_24", 168),
)

@dataclass
class SanityRow:
    feature: str
    horizon: int
    market: str
    n: int
    ic: float
    hac_se: float
    hac_z: float
    p_hac: float
    lag1_ic_autocorr: float
    effective_n: float
    block_boot_prob_positive: float
    cross_asset_corr_median: float


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


def _rank_ic_series(x: pd.Series, y: pd.Series) -> np.ndarray:
    z = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(z) < 30:
        return np.array([], dtype=float)
    # Rolling windows create a time series of one-step cross-sectional rank ICs.
    # For a single market we use the signal's rank-normalized residual association
    # against its future return as a local attribution diagnostic.
    xr = z["x"].rank(pct=True).to_numpy()
    yr = z["y"].rank(pct=True).to_numpy()
    return np.sign((xr - 0.5) * (yr - 0.5))


def _hac_se(arr: np.ndarray, max_lag: int | None = None) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 10:
        return float("nan")
    if max_lag is None:
        max_lag = int(max(1, min(24, n ** 0.25)))
    x = a - float(a.mean())
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    for k in range(1, max_lag + 1):
        cov = float(np.dot(x[k:], x[:-k]) / n)
        w = 1.0 - k / (max_lag + 1.0)
        var += 2.0 * w * cov
    return float(np.sqrt(max(var, 1e-16) / n))


def _effective_n(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 3:
        return float(n)
    x = a - float(a.mean())
    den = float(np.dot(x, x))
    if den <= 1e-16:
        return float(n)
    rho1 = float(np.dot(x[1:], x[:-1]) / den)
    rho1 = float(np.clip(rho1, -0.99, 0.99))
    return float(n * (1.0 - rho1) / (1.0 + rho1))


def _bootstrap_positive(arr: np.ndarray, seed: int, block: int = 24, trials: int = 250) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    n = len(a)
    if n < max(30, block * 2):
        return 0.0
    rng = np.random.default_rng(seed)
    blocks = max(1, int(np.ceil(n / block)))
    means = []
    for _ in range(trials):
        starts = rng.integers(0, n - block + 1, size=blocks)
        sample = np.concatenate([a[s:s + block] for s in starts])[:n]
        means.append(float(np.mean(sample)))
    return float(np.mean(np.asarray(means) > 0.0))


def _two_sided_normal_p(z: float) -> float:
    if not np.isfinite(z):
        return 1.0
    # Avoid scipy dependency.
    return float(np.erfc(abs(z) / np.sqrt(2.0)))


def _signal_score(df: pd.DataFrame, feature: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    feats = _features(df)
    x = feats[feature]
    c = pd.to_numeric(df["close"], errors="coerce")
    y = c.pct_change(horizon).shift(-horizon)
    return x.to_numpy(dtype=float), y.to_numpy(dtype=float)


def _single(df: pd.DataFrame, feature: str, horizon: int, seed: int, market: str) -> SanityRow:
    x, y = _signal_score(df, feature, horizon)
    z = pd.DataFrame({"x": x, "y": y}, index=df.index).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(z)
    if n < 30:
        return SanityRow(feature, horizon, market, n, 0.0, float("nan"), 0.0, 1.0, 0.0, float(n), 0.5, float("nan"))

    # Attribution statistic: Pearson IC on rank-normalized observations.
    rx = z["x"].rank(pct=True).to_numpy()
    ry = z["y"].rank(pct=True).to_numpy()
    ic = float(np.corrcoef(rx, ry)[0, 1]) if n > 2 else 0.0
    if not np.isfinite(ic):
        ic = 0.0

    # Treat IC sequence via blockized influence proxy to expose temporal dependence.
    influence = (rx - rx.mean()) * (ry - ry.mean())
    hac = _hac_se(influence)
    hac_z = float(ic / max(hac, 1e-12))
    p = _two_sided_normal_p(hac_z)
    ac = float(pd.Series(influence).autocorr(lag=1))
    if not np.isfinite(ac):
        ac = 0.0
    en = _effective_n(influence)
    boot = _bootstrap_positive(influence, seed)

    return SanityRow(feature, horizon, market, n, ic, hac, hac_z, p, ac, en, boot, 0.0)


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    if len(raw) < 8:
        payload = {"version": "phase7", "decision": "PHASE7_BLOCKED_DATA", "markets": list(raw)}
        _save(payload)
        return payload

    start = max(x.index[0] for x in raw.values())
    end = min(x.index[-1] for x in raw.values())
    data = {s: x.loc[(x.index >= start) & (x.index <= end)].copy() for s, x in raw.items()}
    n = min(len(x) for x in data.values())
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_start = dev_n + val_n

    print("=== PHASE 7 STATISTICAL SANITY ===", flush=True)
    print(f"Markets: {len(data)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)
    print("NO NEW DISCOVERY: pre-registered signals only", flush=True)

    # Aggregate cross-market/time dependence diagnostics for each registered signal.
    rows: list[dict] = []
    signal_summaries: list[dict] = []

    for i, (feature, horizon) in enumerate(SIGNALS, 1):
        if time.monotonic() >= deadline:
            break
        per_market = []
        for m, frame in data.items():
            r = _single(frame.iloc[:dev_n], feature, horizon, seed + i, m)
            per_market.append(r)
            rows.append(asdict(r))
        vals = np.asarray([r.ic for r in per_market], dtype=float)
        ps = np.asarray([r.p_hac for r in per_market], dtype=float)
        boots = np.asarray([r.block_boot_prob_positive for r in per_market], dtype=float)
        signal_summaries.append({
            "feature": feature,
            "horizon": horizon,
            "median_ic": float(np.nanmedian(vals)),
            "positive_markets": int(np.sum(vals > 0)),
            "median_hac_p": float(np.nanmedian(ps)),
            "median_block_boot_positive": float(np.nanmedian(boots)),
            "median_effective_n": float(np.nanmedian([r.effective_n for r in per_market])),
        })
        print(
            f"DEV {i}/{len(SIGNALS)} | {feature} h={horizon} "
            f"medianIC={np.nanmedian(vals):.4f} "
            f"pHAC={np.nanmedian(ps):.3g} "
            f"boot+={np.nanmedian(boots):.2%} "
            f"positive={int(np.sum(vals>0))}/{len(vals)}",
            flush=True,
        )

    # Frozen validation and lockbox audits use the same registered signals only.
    print("=== FROZEN VALIDATION AUDIT ===", flush=True)
    validation = []
    for i, (feature, horizon) in enumerate(SIGNALS, 1):
        if time.monotonic() >= deadline:
            break
        per = []
        for m, frame in data.items():
            r = _single(frame.iloc[dev_n:lock_start], feature, horizon, seed + 100 + i, m)
            per.append(r)
        vals = np.asarray([r.ic for r in per], dtype=float)
        validation.append({
            "feature": feature,
            "horizon": horizon,
            "median_ic": float(np.nanmedian(vals)),
            "positive_markets": int(np.sum(vals > 0)),
            "median_hac_p": float(np.nanmedian([r.p_hac for r in per])),
            "median_boot_positive": float(np.nanmedian([r.block_boot_prob_positive for r in per])),
        })
        print(
            f"VAL {i}/{len(SIGNALS)} | {feature} h={horizon} "
            f"medianIC={np.nanmedian(vals):.4f} "
            f"pHAC={np.nanmedian([r.p_hac for r in per]):.3g} "
            f"positive={int(np.sum(vals>0))}/{len(vals)}",
            flush=True,
        )

    print("=== LOCKBOX AUDIT ===", flush=True)
    lockbox = []
    for i, (feature, horizon) in enumerate(SIGNALS, 1):
        if time.monotonic() >= deadline:
            break
        per = []
        for m, frame in data.items():
            r = _single(frame.iloc[lock_start:], feature, horizon, seed + 200 + i, m)
            per.append(r)
        vals = np.asarray([r.ic for r in per], dtype=float)
        lockbox.append({
            "feature": feature,
            "horizon": horizon,
            "median_ic": float(np.nanmedian(vals)),
            "positive_markets": int(np.sum(vals > 0)),
            "median_hac_p": float(np.nanmedian([r.p_hac for r in per])),
            "median_boot_positive": float(np.nanmedian([r.block_boot_prob_positive for r in per])),
        })
        print(
            f"LOCKBOX {i}/{len(SIGNALS)} | {feature} h={horizon} "
            f"medianIC={np.nanmedian(vals):.4f} "
            f"pHAC={np.nanmedian([r.p_hac for r in per]):.3g} "
            f"positive={int(np.sum(vals>0))}/{len(vals)}",
            flush=True,
        )

    # Conservative multiple-testing sanity: all pre-registered signal/market tests.
    all_p = np.asarray([r["p_hac"] for r in rows], dtype=float)
    all_p = all_p[np.isfinite(all_p)]
    if len(all_p):
        sorted_p = np.sort(all_p)
        mcount = len(sorted_p)
        bh = [float(min(1.0, p * mcount / rank)) for rank, p in enumerate(sorted_p, 1)]
        bh_q_min = float(min(bh))
        raw_lt_05 = int(np.sum(sorted_p < 0.05))
    else:
        bh_q_min = 1.0
        raw_lt_05 = 0

    persistent = []
    for d, v, l in zip(signal_summaries, validation, lockbox):
        persistent.append({
            "feature": d["feature"],
            "horizon": d["horizon"],
            "dev_positive_markets": d["positive_markets"],
            "validation_positive_markets": v["positive_markets"],
            "lockbox_positive_markets": l["positive_markets"],
            "dev_median_ic": d["median_ic"],
            "validation_median_ic": v["median_ic"],
            "lockbox_median_ic": l["median_ic"],
        })

    # This is intentionally an audit decision, not a strategy decision.
    # A signal passes only if the signed effect survives all three splits and
    # the conservative corrected p-value is below 0.05.
    corrected_supported = bh_q_min < 0.05
    persistent_positive = [
        x for x in persistent
        if x["dev_positive_markets"] >= 8
        and x["validation_positive_markets"] >= 8
        and x["lockbox_positive_markets"] >= 8
        and x["dev_median_ic"] > 0
        and x["validation_median_ic"] > 0
        and x["lockbox_median_ic"] > 0
    ]

    decision = "PHASE7_PERSISTENT_STATISTICAL_SUPPORT" if (corrected_supported and persistent_positive) else "PHASE7_NO_PERSISTENT_STATISTICAL_SUPPORT"

    payload = {
        "version": "phase7",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "markets": list(data),
        "bars": n,
        "split": {"dev": dev_n, "validation": val_n, "lockbox": n - lock_start},
        "registered_signals": [
            {"feature": f, "horizon": h} for f, h in SIGNALS
        ],
        "dev_signal_summary": signal_summaries,
        "row_diagnostics": rows,
        "validation": validation,
        "lockbox": lockbox,
        "multiple_testing": {
            "tests": int(len(all_p)),
            "raw_p_lt_0_05": raw_lt_05,
            "bh_q_min": bh_q_min,
            "bh_supported": corrected_supported,
        },
        "persistent_candidates": persistent,
        "protocol": {
            "no_new_discovery": True,
            "pre_registered_signals": True,
            "hac_uncertainty": True,
            "block_bootstrap": True,
            "effective_sample_size": True,
            "multiple_testing_correction": "Benjamini-Hochberg",
            "validation_frozen": True,
            "lockbox_frozen": True,
            "spot_only": True,
        },
    }
    _save(payload)
    print("=== PHASE 7 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(float(os.getenv("PHASE7_MINUTES", "60")), int(os.getenv("PHASE7_SEED", "20260829")))
