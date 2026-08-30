from __future__ import annotations

"""Phase 5 V2: vectorized cross-sectional / relative-value edge discovery.

The first Phase 5 implementation evaluated every timestamp inside nested Python
loops. This version precomputes feature panels and future-return panels once,
then evaluates each feature/horizon with NumPy/Pandas vectorized operations.

Research protocol:
- spot 1h cached markets only
- common timestamp intersection across markets
- DEV ranking only
- frozen validation
- top-3 lockbox
- BTC beta is estimated on DEV only and frozen for all later splits
- no strategy optimization here; information test only
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.getenv("PHASE5_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase5_cross_sectional_edge_v2_latest.json"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
HORIZONS = (6, 24, 72, 168)
FEATURES = (
    "mom_6", "mom_24", "mom_72", "mom_168",
    "vol_scaled_mom", "volume_pressure", "vol_compression",
    "trend_strength", "range_position_72",
)


@dataclass
class Row:
    feature: str
    horizon: int
    split: str
    observations: int
    rank_ic: float
    long_short_spread: float
    positive_days: int
    days: int
    beta_neutral_spread: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        if path.exists():
            frame = pd.read_parquet(path).copy()
            frame.index = pd.to_datetime(frame.index, utc=True)
            out[symbol] = frame.sort_index()
    return out


def _common(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not data:
        return {}
    idx = None
    for frame in data.values():
        current = pd.Index(frame.index)
        idx = current if idx is None else idx.intersection(current)
    if idx is None or len(idx) == 0:
        return {}
    idx = idx.sort_values()
    return {s: frame.loc[idx].copy() for s, frame in data.items()}


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")
    r = c.pct_change()
    out = pd.DataFrame(index=df.index)
    for w in (6, 24, 72, 168):
        out[f"mom_{w}"] = c.pct_change(w)
    vol24 = r.rolling(24, min_periods=24).std().replace(0.0, np.nan)
    out["vol_scaled_mom"] = out["mom_24"] / vol24
    out["volume_pressure"] = v / v.rolling(48, min_periods=48).mean()
    out["vol_compression"] = (
        r.rolling(12, min_periods=12).std()
        / r.rolling(72, min_periods=72).std().replace(0.0, np.nan)
    )
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    out["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    width = (hh - ll).replace(0.0, np.nan)
    out["range_position_72"] = ((c - ll) / width).clip(-1.0, 2.0)
    return out.replace([np.inf, -np.inf], np.nan)


def _panels(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    idx = next(iter(data.values())).index
    close = pd.DataFrame({s: data[s]["close"] for s in data}, index=idx).sort_index()
    feats = {s: _features(data[s]).reindex(close.index) for s in data}
    return close, feats


def _frozen_betas(close: pd.DataFrame, end: int) -> pd.Series:
    """Estimate per-asset beta to BTC using DEV returns only."""
    rets = close.pct_change().iloc[:end]
    btc = rets["BTC/USDT"]
    betas: dict[str, float] = {}
    btc_var = float(np.nanvar(btc.to_numpy()))
    for symbol in close.columns:
        y = rets[symbol]
        clean = pd.concat([y.rename("y"), btc.rename("b")], axis=1).dropna()
        if len(clean) < 1000 or btc_var <= 1e-12:
            betas[symbol] = 0.0 if symbol == "BTC/USDT" else 1.0
            continue
        cov = float(np.cov(clean["y"].to_numpy(), clean["b"].to_numpy(), ddof=0)[0, 1])
        var = float(np.var(clean["b"].to_numpy()))
        betas[symbol] = cov / var if var > 1e-12 else 0.0
    return pd.Series(betas)


def _rank_ic_by_row(x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
    """Row-wise Spearman IC using vectorized rank transforms."""
    xr = x.rank(axis=1, method="average", pct=False)
    yr = y.rank(axis=1, method="average", pct=False)
    xm = xr.mean(axis=1)
    ym = yr.mean(axis=1)
    xa = xr.sub(xm, axis=0).to_numpy(dtype=float)
    ya = yr.sub(ym, axis=0).to_numpy(dtype=float)
    num = np.nansum(xa * ya, axis=1)
    den = np.sqrt(np.nansum(xa * xa, axis=1) * np.nansum(ya * ya, axis=1))
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return out


def _evaluate(
    close: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    feature: str,
    horizon: int,
    start: int,
    end: int,
    split: str,
    betas: pd.Series,
) -> Row:
    """Vectorized cross-sectional evaluation for one feature/horizon/split."""
    x = pd.DataFrame({s: feats[s][feature] for s in close.columns}, index=close.index)
    y = close.pct_change(horizon).shift(-horizon)

    x = x.iloc[start:end]
    y = y.iloc[start:end]

    valid_counts = x.notna().sum(axis=1).to_numpy()
    enough = valid_counts >= 8
    if not enough.any():
        return Row(feature, horizon, split, 0, 0.0, 0.0, 0, 0, 0.0)

    x = x.loc[enough]
    y = y.loc[enough]
    pair_ok = x.notna() & y.notna()
    observed = int(pair_ok.sum(axis=1).sum())

    rank_ic = _rank_ic_by_row(x.where(pair_ok), y.where(pair_ok))
    ic_mask = np.isfinite(rank_ic)

    ranks = x.rank(axis=1, pct=True)
    long_mask = ranks >= 0.75
    short_mask = ranks <= 0.25

    long_ret = y.where(long_mask).mean(axis=1)
    short_ret = y.where(short_mask).mean(axis=1)
    spread = (long_ret - short_ret).to_numpy(dtype=float)

    # True beta-neutral return: y_i - beta_i * BTC_return.
    btc_ret = y["BTC/USDT"]
    beta_frame = pd.DataFrame({s: betas.get(s, 0.0) * btc_ret for s in close.columns}, index=y.index)
    residual = y - beta_frame
    neutral_long = residual.where(long_mask).mean(axis=1)
    neutral_short = residual.where(short_mask).mean(axis=1)
    neutral = (neutral_long - neutral_short).to_numpy(dtype=float)

    valid = ic_mask & np.isfinite(spread) & np.isfinite(neutral)
    rank_ic = rank_ic[valid]
    spread = spread[valid]
    neutral = neutral[valid]

    days = int(len(spread))
    positive_days = int(np.sum(spread > 0.0))
    return Row(
        feature=feature,
        horizon=horizon,
        split=split,
        observations=observed,
        rank_ic=float(np.median(rank_ic)) if days else 0.0,
        long_short_spread=float(np.median(spread)) if days else 0.0,
        positive_days=positive_days,
        days=days,
        beta_neutral_spread=float(np.median(neutral)) if days else 0.0,
    )


def _permutation(
    close: pd.DataFrame,
    feats: dict[str, pd.DataFrame],
    feature: str,
    horizon: int,
    start: int,
    end: int,
    betas: pd.Series,
    seed: int,
    trials: int = 20,
) -> dict:
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({s: feats[s][feature] for s in close.columns}, index=close.index).iloc[start:end]
    y = close.pct_change(horizon).shift(-horizon).iloc[start:end]
    observed = _evaluate(close, feats, feature, horizon, start, end, "dev", betas).rank_ic

    vals = []
    arr = y.to_numpy(dtype=float)
    for _ in range(trials):
        shuffled = arr.copy()
        # Shuffle within each asset, preserving the cross-sectional structure.
        for j in range(shuffled.shape[1]):
            rng.shuffle(shuffled[:, j])
        py = pd.DataFrame(shuffled, index=y.index, columns=y.columns)
        ric = _rank_ic_by_row(x, py)
        finite = ric[np.isfinite(ric)]
        vals.append(float(np.median(finite)) if finite.size else 0.0)

    exceed = int(sum(abs(v) >= abs(observed) for v in vals))
    return {
        "trials": trials,
        "observed_rank_ic": observed,
        "null_exceed_rate": exceed / max(1, trials),
    }


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()

    if len(raw) < 8:
        payload = {"version": "phase5_v2", "decision": "PHASE5_BLOCKED_DATA", "markets": list(raw)}
        _save(payload)
        return payload

    data = _common(raw)
    if len(data) < 8:
        payload = {"version": "phase5_v2", "decision": "PHASE5_BLOCKED_COMMON_INDEX", "markets": list(data)}
        _save(payload)
        return payload

    close, feats = _panels(data)
    n = len(close)
    dev_n = int(n * 0.55)
    val_n = int(n * 0.25)
    lock_start = dev_n + val_n
    betas = _frozen_betas(close, dev_n)

    print("=== PHASE 5 V2 CROSS-SECTIONAL EDGE ===", flush=True)
    print(f"Markets: {len(data)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_n} VALIDATION={val_n} LOCKBOX={n-lock_start}", flush=True)
    print("BTC BETAS: FROZEN FROM DEV", flush=True)

    dev_rows: list[dict] = []
    for feature in FEATURES:
        for horizon in HORIZONS:
            if time.monotonic() >= deadline:
                break
            row = _evaluate(close, feats, feature, horizon, 0, dev_n, "dev", betas)
            dev_rows.append(asdict(row))
            print(
                f"DEV {feature} h={horizon} | RankIC={row.rank_ic:.4f} "
                f"spread={row.long_short_spread:.4%} "
                f"neutral={row.beta_neutral_spread:.4%} "
                f"positive={row.positive_days}/{row.days}",
                flush=True,
            )

    dev_rows.sort(
        key=lambda r: (
            r["positive_days"] / max(1, r["days"]),
            r["beta_neutral_spread"],
            r["rank_ic"],
        ),
        reverse=True,
    )
    candidates = dev_rows[:8]

    print("=== DEV PERMUTATION ===", flush=True)
    permutations = []
    for i, row in enumerate(candidates, 1):
        result = _permutation(
            close,
            feats,
            row["feature"],
            row["horizon"],
            0,
            dev_n,
            betas,
            seed + i,
        )
        result.update({"feature": row["feature"], "horizon": row["horizon"]})
        permutations.append(result)
        print(
            f"PERM {i}/8 | {row['feature']} h={row['horizon']} "
            f"null_exceed={result['null_exceed_rate']:.2%}",
            flush=True,
        )

    print("=== FROZEN VALIDATION ===", flush=True)
    validation = []
    for i, row in enumerate(candidates, 1):
        r = _evaluate(
            close,
            feats,
            row["feature"],
            row["horizon"],
            dev_n,
            lock_start,
            "validation",
            betas,
        )
        validation.append(asdict(r))
        print(
            f"VAL {i}/8 | {r.feature} h={r.horizon} | RankIC={r.rank_ic:.4f} "
            f"spread={r.long_short_spread:.4%} neutral={r.beta_neutral_spread:.4%} "
            f"positive={r.positive_days}/{r.days}",
            flush=True,
        )

    validation.sort(
        key=lambda r: (
            r["positive_days"] / max(1, r["days"]),
            r["beta_neutral_spread"],
            r["rank_ic"],
        ),
        reverse=True,
    )
    finalists = validation[:3]

    print("=== LOCKBOX FINAL TEST ===", flush=True)
    lockbox = []
    for i, row in enumerate(finalists, 1):
        r = _evaluate(
            close,
            feats,
            row["feature"],
            row["horizon"],
            lock_start,
            n,
            "lockbox",
            betas,
        )
        lockbox.append(asdict(r))
        print(
            f"LOCKBOX {i}/3 | {r.feature} h={r.horizon} | RankIC={r.rank_ic:.4f} "
            f"spread={r.long_short_spread:.4%} neutral={r.beta_neutral_spread:.4%} "
            f"positive={r.positive_days}/{r.days}",
            flush=True,
        )

    eligible = [
        r for r in lockbox
        if r["days"] >= 1000
        and r["positive_days"] / max(1, r["days"]) >= 0.60
        and r["beta_neutral_spread"] > 0.0
        and r["rank_ic"] > 0.0
    ]
    decision = "PHASE5_CROSS_SECTIONAL_EDGE_FOUND" if eligible else "PHASE5_NO_CONFIRMED_RELATIVE_EDGE"

    payload = {
        "version": "phase5_v2",
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "markets": list(data),
        "bars": n,
        "split": {"dev": dev_n, "validation": val_n, "lockbox": n - lock_start},
        "btc_betas_dev_frozen": betas.to_dict(),
        "dev_edges": dev_rows,
        "candidates": candidates,
        "permutation": permutations,
        "validation": validation,
        "lockbox": lockbox,
        "eligible": eligible,
        "protocol": {
            "vectorized": True,
            "cross_sectional": True,
            "beta_neutral_to_BTC": True,
            "btc_betas_fit_on_dev_only": True,
            "dev_only_selection": True,
            "validation_frozen": True,
            "lockbox_top3": True,
            "spot_only": True,
        },
    }
    _save(payload)
    print("=== PHASE 5 V2 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        float(os.getenv("PHASE5_MINUTES", "60")),
        int(os.getenv("PHASE5_SEED", "20260829")),
    )
