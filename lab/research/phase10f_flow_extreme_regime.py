from __future__ import annotations

import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
OUT = Path("experiments/phase10f_flow_extreme_regime")

@dataclass(frozen=True)
class ScreenRow:
    symbol: str
    horizon_min: int
    condition: str
    observations: int
    median_reverse_ic: float
    positive_days: int
    total_days: int
    median_abs_flow_z: float
    median_beta_residual_reverse_ic: float


def _last_complete_day() -> datetime.date:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date()


def _date_list(days: int) -> list[str]:
    end = _last_complete_day()
    return [(end - timedelta(days=i)).isoformat() for i in range(max(1, days))[::-1]]


def _download_one(symbol: str, day: str) -> tuple[str, str, bytes, float]:
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day}.zip"
    started = time.perf_counter()
    req = Request(url, headers={"User-Agent": "phase10f-research/1.0"})
    with urlopen(req, timeout=120.0) as response:
        payload = response.read()
        status = int(getattr(response, "status", 200))
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {url}")
    return symbol, day, payload, (time.perf_counter() - started) * 1000.0


def _read_agg(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csvs = [x for x in archive.namelist() if x.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError("No CSV in archive")
        with archive.open(csvs[0]) as fh:
            raw = pd.read_csv(fh, header=None)
    if raw.shape[1] < 7:
        raise RuntimeError(f"Unexpected columns={raw.shape[1]}")
    raw = raw.iloc[:, :7].copy()
    raw.columns = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "buyer_maker"]
    for c in ("agg_id", "price", "qty", "timestamp"):
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["buyer_maker"] = raw["buyer_maker"].astype(bool)
    raw = raw.dropna(subset=["agg_id", "price", "qty", "timestamp"])
    unit = "us" if float(raw["timestamp"].median()) > 10_000_000_000_000 else "ms"
    raw["ts"] = pd.to_datetime(raw["timestamp"], unit=unit, utc=True)
    return raw.drop_duplicates(subset=["agg_id"]).sort_values("ts")


def _bars(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.set_index("ts").copy()
    x["notional"] = x["price"] * x["qty"]
    x["signed"] = np.where(x["buyer_maker"], -x["notional"], x["notional"])
    b = x.resample("1min").agg(close=("price", "last"), notional=("notional", "sum"), signed=("signed", "sum"), trades=("agg_id", "count")).dropna(subset=["close"])
    b["ret_1m"] = b["close"].pct_change()
    b["imbalance"] = b["signed"] / b["notional"].replace(0.0, np.nan)
    mu = b["signed"].rolling(60, min_periods=30).mean()
    sd = b["signed"].rolling(60, min_periods=30).std()
    b["flow_z"] = (b["signed"] - mu) / sd.replace(0.0, np.nan)
    b["flow_5m"] = b["flow_z"].rolling(5, min_periods=5).mean()
    for h in (60, 360):
        b[f"fwd_{h}"] = b["close"].shift(-h) / b["close"] - 1.0
    return b.replace([np.inf, -np.inf], np.nan)


def _pearson(a: pd.Series, b: pd.Series) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < 200 or z.iloc[:, 0].std() == 0 or z.iloc[:, 1].std() == 0:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def _residual_reverse(signal: pd.Series, target: pd.Series, market_ret: pd.Series) -> float:
    z = pd.concat([-signal, target, market_ret], axis=1).dropna()
    if len(z) < 300 or z.iloc[:, 0].std() == 0:
        return float("nan")
    x = z.iloc[:, 2].to_numpy(dtype=float)
    y = z.iloc[:, 1].to_numpy(dtype=float)
    s = z.iloc[:, 0].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    if np.std(resid) == 0 or np.std(s) == 0:
        return float("nan")
    return float(np.corrcoef(s, resid)[0, 1])


def _daily_screen(frames: dict[str, pd.DataFrame], day: str) -> list[dict[str, Any]]:
    ret = pd.concat({s: f["ret_1m"] for s, f in frames.items()}, axis=1)
    market_ret = ret.mean(axis=1)
    flow = pd.concat({s: f["flow_z"] for s, f in frames.items()}, axis=1)
    rows: list[dict[str, Any]] = []
    for s, frame in frames.items():
        fz = frame["flow_z"]
        conditions = {
            "all": fz.notna(),
            "extreme_high": fz > 1.5,
            "extreme_low": fz < -1.5,
            "extreme_abs": fz.abs() > 1.5,
            "extreme_cross": (fz.abs() > 1.5) & (flow.drop(columns=[s]).abs().median(axis=1) > 0.5),
        }
        for condition, mask in conditions.items():
            for h in (60, 360):
                sub = frame.loc[mask]
                if len(sub) < 200:
                    continue
                reverse_ic = _pearson(-sub["flow_5m"], sub[f"fwd_{h}"])
                residual = _residual_reverse(sub["flow_5m"], sub[f"fwd_{h}"], market_ret.reindex(sub.index))
                rows.append({"symbol": s, "date": day, "condition": condition, "horizon_min": h, "observations": int(len(sub)), "reverse_ic": reverse_ic, "residual_reverse_ic": residual, "median_abs_flow_z": float(sub["flow_z"].abs().median())})
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> list[ScreenRow]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: list[ScreenRow] = []
    for (s, h, c), g in df.groupby(["symbol", "horizon_min", "condition"]):
        v = g.dropna(subset=["residual_reverse_ic"])
        out.append(ScreenRow(
            symbol=str(s), horizon_min=int(h), condition=str(c), observations=int(g["observations"].sum()),
            median_reverse_ic=float(g["reverse_ic"].median()), positive_days=int((v["residual_reverse_ic"] > 0).sum()),
            total_days=int(len(v)), median_abs_flow_z=float(g["median_abs_flow_z"].median()),
            median_beta_residual_reverse_ic=float(v["residual_reverse_ic"].median()) if len(v) else float("nan"),
        ))
    return sorted(out, key=lambda x: (-x.positive_days / max(1, x.total_days), -abs(x.median_beta_residual_reverse_ic if np.isfinite(x.median_beta_residual_reverse_ic) else 0.0)))


def run(minutes: float = 15.0, seed: int = 20260829, days: int = 14) -> dict[str, Any]:
    del seed
    started = time.perf_counter()
    dates = _date_list(days)
    OUT.mkdir(parents=True, exist_ok=True)
    bars_dir = OUT / "1m"
    bars_dir.mkdir(exist_ok=True)
    print("=== PHASE 10F EXTREME FLOW REGIME ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"PERIOD: {dates[0]} → {dates[-1]} | days={len(dates)}", flush=True)
    print(f"SYMBOLS: {len(SYMBOLS)} | {SYMBOLS}", flush=True)
    print("CONDITIONS: all | extreme_high | extreme_low | extreme_abs | extreme_cross", flush=True)
    print("BETA: cross-sectional mean 1m return", flush=True)

    frames_by_day: dict[str, dict[str, pd.DataFrame]] = {d: {} for d in dates}
    errors: list[dict[str, str]] = []
    jobs = [(s, d) for d in dates for s in SYMBOLS]
    with ThreadPoolExecutor(max_workers=min(12, len(jobs))) as ex:
        futs = {ex.submit(_download_one, s, d): (s, d) for s, d in jobs}
        for fut in as_completed(futs):
            s, d = futs[fut]
            try:
                _, _, payload, ms = fut.result()
                print(f"DOWNLOAD {s} {d} | ok=True | ms={ms:.1f} | bytes={len(payload)}", flush=True)
                b = _bars(_read_agg(payload))
                frames_by_day[d][s] = b
                b.to_parquet(bars_dir / f"{s}_{d}_1m.parquet")
            except Exception as exc:
                errors.append({"symbol": s, "date": d, "error": repr(exc)})
                print(f"ERROR {s} {d}: {exc!r}", flush=True)

    rows: list[dict[str, Any]] = []
    for d in dates:
        if len(frames_by_day[d]) < 4:
            continue
        day_rows = _daily_screen(frames_by_day[d], d)
        rows.extend(day_rows)
        print(f"SCREEN DAY {d} | symbols={len(frames_by_day[d])} | rows={len(day_rows)}", flush=True)

    top = _aggregate(rows)
    eligible = [x for x in top if x.total_days >= 7 and x.positive_days >= int(np.ceil(0.65 * x.total_days)) and np.isfinite(x.median_beta_residual_reverse_ic) and x.median_beta_residual_reverse_ic >= 0.02]
    decision = "PHASE10F_PERSISTENT_EXTREME_FLOW_CANDIDATES" if eligible else "PHASE10F_NO_PERSISTENT_EXTREME_FLOW_EDGE"
    result: dict[str, Any] = {"version": "phase10f_flow_extreme_regime", "created_at_utc": datetime.now(timezone.utc).isoformat(), "period": {"start": dates[0], "end": dates[-1], "days": len(dates)}, "symbols": SYMBOLS, "conditions": ["all", "extreme_high", "extreme_low", "extreme_abs", "extreme_cross"], "rows": len(rows), "top": [asdict(x) for x in top[:20]], "eligible": [asdict(x) for x in eligible[:10]], "errors": errors, "decision": decision, "elapsed_sec": time.perf_counter() - started, "next": "Only candidates surviving this gate should enter purged walk-forward model testing."}
    cp = OUT / "phase10f_flow_extreme_regime_latest.json"
    cp.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print("=== PHASE 10F COMPLETE ===", flush=True)
    print("DECISION:", decision, flush=True)
    print("ROWS:", len(rows), flush=True)
    print("TOP_CANDIDATES:", len(top[:20]), flush=True)
    print("ELIGIBLE:", len(eligible), flush=True)
    print("ERRORS:", len(errors), flush=True)
    print("CHECKPOINT:", cp, flush=True)
    print("ELAPSED_SEC:", f"{time.perf_counter() - started:.2f}", flush=True)
    return result
