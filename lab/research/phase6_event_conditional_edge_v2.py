from __future__ import annotations

"""Phase 6 V2: leakage-safe event-conditioned edge discovery.

Research-only information test. No trading strategy is optimized here.
Event thresholds are fitted on DEV only and frozen for VALIDATION/LOCKBOX.
BTC betas are also fitted on DEV only and frozen.
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
CACHE = Path(os.getenv("PHASE6_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
OUT = ROOT / "experiments" / "phase6_event_conditional_edge_v2_latest.json"
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")
HORIZONS = (6, 24, 72, 168)
SIGNALS = ("mom_6","mom_24","mom_72","mom_168","vol_scaled_mom","volume_pressure","vol_compression","trend_strength","range_position_72")
EVENTS = ("vol_shock","range_shock","momentum_shock","volume_shock")

@dataclass
class EdgeRow:
    signal: str
    event: str
    horizon: int
    split: str
    observations: int
    rank_ic: float
    spread: float
    residual_spread: float
    positive_periods: int
    periods: int
    event_rate: float


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _load() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        out[symbol] = frame.sort_index()
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
    rv6 = r.rolling(6, min_periods=6).std()
    rv24 = r.rolling(24, min_periods=24).std()
    rv72 = r.rolling(72, min_periods=72).std()
    z["vol_scaled_mom"] = z["mom_24"] / rv24.replace(0, np.nan)
    vol_mean = v.rolling(48, min_periods=48).mean()
    z["volume_pressure"] = v / vol_mean.replace(0, np.nan)
    z["vol_compression"] = rv6 / rv72.replace(0, np.nan)
    e24 = c.ewm(span=24, adjust=False, min_periods=24).mean()
    e96 = c.ewm(span=96, adjust=False, min_periods=96).mean()
    z["trend_strength"] = e24 / e96 - 1.0
    hh = h.rolling(72, min_periods=72).max().shift(1)
    ll = l.rolling(72, min_periods=72).min().shift(1)
    width = (hh - ll).replace(0, np.nan)
    z["range_position_72"] = ((c - ll) / width).clip(-1, 2)
    z["rv_shock_metric"] = rv24 / rv72.replace(0, np.nan)
    rng = (h - l) / c.replace(0, np.nan)
    z["range_shock_metric"] = rng / rng.rolling(72, min_periods=72).median().replace(0, np.nan)
    mom_std = z["mom_6"].rolling(168, min_periods=168).std().replace(0, np.nan)
    z["momentum_shock_metric"] = z["mom_6"].abs() / mom_std
    vz = (v - v.rolling(96, min_periods=96).mean()) / v.rolling(96, min_periods=96).std().replace(0, np.nan)
    z["volume_shock_metric"] = vz.abs()
    return z.replace([np.inf, -np.inf], np.nan)


def _panel(data: dict[str, pd.DataFrame]):
    common = sorted(set.intersection(*[set(x.index) for x in data.values()]))
    close = pd.DataFrame({s: data[s].loc[common, "close"] for s in data}, index=common).sort_index()
    feats = {s: _features(data[s]).reindex(close.index) for s in data}
    return close, feats


def _signal_panel(feats: dict[str, pd.DataFrame], signal: str) -> pd.DataFrame:
    return pd.DataFrame({s: f[signal] for s, f in feats.items()})


def _event_metric_panel(feats: dict[str, pd.DataFrame], event: str) -> pd.DataFrame:
    mapping = {
        "vol_shock": "rv_shock_metric",
        "range_shock": "range_shock_metric",
        "momentum_shock": "momentum_shock_metric",
        "volume_shock": "volume_shock_metric",
    }
    col = mapping[event]
    return pd.DataFrame({s: f[col] for s, f in feats.items()})


def _frozen_event_mask(metric: pd.DataFrame, dev_end: int, q: float = 0.90) -> tuple[pd.DataFrame, dict[str, float]]:
    dev = metric.iloc[:dev_end]
    thresholds = dev.quantile(q, axis=0, interpolation="linear").to_dict()
    mask = metric.gt(pd.Series(thresholds), axis="columns")
    return mask.fillna(False), {k: float(v) if np.isfinite(v) else 0.0 for k, v in thresholds.items()}


def _frozen_betas(close: pd.DataFrame, dev_end: int) -> dict[str, float]:
    ret = close.pct_change().iloc[:dev_end]
    btc = ret["BTC/USDT"]
    betas: dict[str, float] = {}
    for symbol in close.columns:
        if symbol == "BTC/USDT":
            betas[symbol] = 1.0
            continue
        z = pd.concat([ret[symbol], btc], axis=1).dropna()
        if len(z) < 1000:
            betas[symbol] = 0.0
            continue
        x = z.iloc[:, 1].to_numpy()
        y = z.iloc[:, 0].to_numpy()
        var = float(np.var(x))
        betas[symbol] = 0.0 if var <= 1e-12 else float(np.cov(y, x, ddof=0)[0, 1] / var)
    return betas


def _rowwise_rank_corr(x: pd.DataFrame, y: pd.DataFrame, mask: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xx = x.where(mask)
    yy = y.where(mask)
    rx = xx.rank(axis=1, method="average", pct=False)
    ry = yy.rank(axis=1, method="average", pct=False)
    xmean = rx.mean(axis=1)
    ymean = ry.mean(axis=1)
    xc = rx.sub(xmean, axis=0)
    yc = ry.sub(ymean, axis=0)
    num = (xc * yc).sum(axis=1, min_count=4)
    den = np.sqrt((xc * xc).sum(axis=1, min_count=4) * (yc * yc).sum(axis=1, min_count=4))
    ric = num / den.replace(0, np.nan)
    counts = mask.sum(axis=1).to_numpy(dtype=float)
    return ric.to_numpy(dtype=float), counts


def _evaluate(x: pd.DataFrame, future: pd.DataFrame, residual: pd.DataFrame, mask: pd.DataFrame, start: int, end: int, signal: str, event: str, horizon: int, split: str) -> tuple[EdgeRow, np.ndarray]:
    xs = x.iloc[start:end]
    ys = future.iloc[start:end]
    rs = residual.iloc[start:end]
    ms = mask.iloc[start:end]
    ric, counts = _rowwise_rank_corr(xs, ys, ms)
    valid = np.isfinite(ric) & (counts >= 4)
    if not valid.any():
        row = EdgeRow(signal,event,horizon,split,0,0.0,0.0,0.0,0,0,0.0)
        return row, np.empty(0)

    ranks = xs.where(ms).rank(axis=1, pct=True)
    top = (ranks >= 0.75) & ms
    bot = (ranks <= 0.25) & ms
    long_mean = ys.where(top).mean(axis=1)
    short_mean = ys.where(bot).mean(axis=1)
    spread = (long_mean - short_mean).to_numpy(dtype=float)
    residual_spread = (rs.where(top).mean(axis=1) - rs.where(bot).mean(axis=1)).to_numpy(dtype=float)
    usable_spread = spread[np.isfinite(spread) & valid]
    usable_resid = residual_spread[np.isfinite(residual_spread) & valid]
    ric_use = ric[valid]
    positive = int(np.sum(usable_spread > 0))
    periods = int(valid.sum())
    event_periods = int(np.sum(ms.any(axis=1).to_numpy()[valid]))
    event_rate = float(ms.to_numpy(dtype=bool).sum() / max(1, periods * ms.shape[1]))
    obs = int(np.sum(counts[valid]))
    row = EdgeRow(signal,event,horizon,split,obs,
                  float(np.median(ric_use)) if ric_use.size else 0.0,
                  float(np.median(usable_spread)) if usable_spread.size else 0.0,
                  float(np.median(usable_resid)) if usable_resid.size else 0.0,
                  positive,periods,event_rate)
    return row, usable_spread


def _permutation(x: pd.DataFrame, future: pd.DataFrame, mask: pd.DataFrame, start: int, end: int, observed: float, seed: int, trials: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    y = future.iloc[start:end].to_numpy(copy=True)
    m = mask.iloc[start:end].to_numpy(copy=False)
    xx = x.iloc[start:end]
    null = []
    for _ in range(trials):
        order = rng.permutation(len(y))
        pseudo = pd.DataFrame(y[order], index=xx.index, columns=xx.columns)
        ric, counts = _rowwise_rank_corr(xx, pseudo, mask.iloc[start:end])
        use = ric[np.isfinite(ric) & (counts >= 4)]
        null.append(float(np.median(use)) if use.size else 0.0)
    null_arr = np.asarray(null)
    return {
        "trials": trials,
        "observed_rank_ic": float(observed),
        "null_median": float(np.median(null_arr)) if null_arr.size else 0.0,
        "null_exceed_rate": float(np.mean(np.abs(null_arr) >= abs(observed))) if null_arr.size else 1.0,
    }


def run(minutes: float = 60.0, seed: int = 20260829) -> dict:
    deadline = time.monotonic() + minutes * 60.0
    raw = _load()
    if len(raw) < 8:
        payload = {"version":"phase6_v2","decision":"PHASE6_BLOCKED_DATA","markets":list(raw)}
        _save(payload)
        return payload

    close, feats = _panel(raw)
    n = len(close)
    dev_end = int(n * 0.55)
    val_end = dev_end + int(n * 0.25)
    print("=== PHASE 6 V2 EVENT EDGE ===", flush=True)
    print(f"Markets: {len(raw)} | common bars={n}", flush=True)
    print(f"SPLIT: DEV={dev_end} VALIDATION={val_end-dev_end} LOCKBOX={n-val_end}", flush=True)
    print("EVENT/BETA THRESHOLDS: FIT ON DEV ONLY", flush=True)

    betas = _frozen_betas(close, dev_end)
    btc_future = {h: close["BTC/USDT"].pct_change(h).shift(-h) for h in HORIZONS}
    residual_future: dict[int, pd.DataFrame] = {}
    for h in HORIZONS:
        fy = close.pct_change(h).shift(-h)
        residual_future[h] = pd.DataFrame({s: fy[s] - betas.get(s, 0.0) * btc_future[h] for s in close.columns}, index=close.index)

    masks: dict[str, pd.DataFrame] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for event in EVENTS:
        metric = _event_metric_panel(feats, event)
        mask, th = _frozen_event_mask(metric, dev_end, 0.90)
        masks[event] = mask
        thresholds[event] = th

    dev=[]
    for event in EVENTS:
        for signal in SIGNALS:
            x = _signal_panel(feats, signal)
            for h in HORIZONS:
                if time.monotonic() >= deadline:
                    break
                row, _ = _evaluate(x, {hh: close.pct_change(hh).shift(-hh) for hh in [h]}[h], residual_future[h], masks[event], 0, dev_end, signal,event,h,"dev")
                dev.append(asdict(row))
    dev.sort(key=lambda r: (r["periods"] >= 50, r["residual_spread"], r["rank_ic"], r["positive_periods"]), reverse=True)
    candidates = dev[:10]
    for i,r in enumerate(candidates,1):
        print(f"DEV {i}/10 | {r['signal']} h={r['horizon']} event={r['event']} IC={r['rank_ic']:.4f} spread={r['spread']:.4%} residual={r['residual_spread']:.4%} pos={r['positive_periods']}/{r['periods']}", flush=True)

    print("=== DEV PERMUTATION ===", flush=True)
    perm=[]
    for i,r in enumerate(candidates,1):
        x=_signal_panel(feats,r['signal']); future=close.pct_change(r['horizon']).shift(-r['horizon'])
        perm.append({**_permutation(x,future,masks[r['event']],0,dev_end,r['rank_ic'],seed+i),"signal":r['signal'],"event":r['event'],"horizon":r['horizon']})
        print(f"PERM {i}/10 | {r['signal']} h={r['horizon']} event={r['event']} null_exceed={perm[-1]['null_exceed_rate']:.2%}",flush=True)

    print("=== FROZEN VALIDATION ===", flush=True)
    validation=[]
    for i,r in enumerate(candidates,1):
        x=_signal_panel(feats,r['signal']); future=close.pct_change(r['horizon']).shift(-r['horizon'])
        row,_=_evaluate(x,future,residual_future[r['horizon']],masks[r['event']],dev_end,val_end,r['signal'],r['event'],r['horizon'],"validation")
        validation.append(asdict(row))
        print(f"VAL {i}/10 | {row.signal} h={row.horizon} event={row.event} IC={row.rank_ic:.4f} spread={row.spread:.4%} residual={row.residual_spread:.4%} pos={row.positive_periods}/{row.periods}",flush=True)
    validation.sort(key=lambda r:(r['residual_spread'],r['positive_periods'],r['rank_ic']),reverse=True)
    finalists=validation[:4]

    print("=== LOCKBOX ===", flush=True)
    lockbox=[]
    for i,r in enumerate(finalists,1):
        x=_signal_panel(feats,r['signal']); future=close.pct_change(r['horizon']).shift(-r['horizon'])
        row,_=_evaluate(x,future,residual_future[r['horizon']],masks[r['event']],val_end,n,r['signal'],r['event'],r['horizon'],"lockbox")
        lockbox.append(asdict(row))
        print(f"LOCKBOX {i}/4 | {row.signal} h={row.horizon} event={row.event} IC={row.rank_ic:.4f} spread={row.spread:.4%} residual={row.residual_spread:.4%} pos={row.positive_periods}/{row.periods}",flush=True)

    eligible=[r for r in lockbox if r['periods']>=50 and r['positive_periods']>=int(0.55*r['periods']) and r['residual_spread']>0 and r['rank_ic']>0]
    decision="PHASE6_EVENT_EDGE_FOUND" if eligible else "PHASE6_NO_CONFIRMED_EVENT_EDGE"
    payload={
        "version":"phase6_v2","started_at":datetime.now(timezone.utc).isoformat(),"decision":decision,
        "markets":list(raw),"bars":n,"split":{"dev":dev_end,"validation":val_end-dev_end,"lockbox":n-val_end},
        "frozen_btc_betas":betas,"event_thresholds_dev":thresholds,
        "dev":dev,"candidates":candidates,"permutation":perm,"validation":validation,"lockbox":lockbox,"eligible":eligible,
        "protocol":{"event_conditioned":True,"thresholds_frozen_from_dev":True,"btc_betas_frozen_from_dev":True,"validation_frozen":True,"lockbox_frozen":True,"spot_only":True}
    }
    _save(payload)
    print("=== PHASE 6 V2 DECISION ===",flush=True); print(decision,flush=True); print(f"Saved: {OUT}",flush=True)
    return payload

if __name__ == "__main__":
    run(float(os.getenv("PHASE6_MINUTES","60")), int(os.getenv("PHASE6_SEED","20260829")))
