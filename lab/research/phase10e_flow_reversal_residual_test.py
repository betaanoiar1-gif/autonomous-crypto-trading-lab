from __future__ import annotations

import json
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
BASE = "https://data.binance.vision/data/spot/daily/aggTrades"
OUT = Path("experiments/phase10e_flow")

@dataclass(frozen=True)
class TestRow:
    symbol: str
    days: int
    horizon_h: int
    direct_ic: float
    residual_ic: float
    reversed_ic: float
    nonoverlap_ic: float
    sign_days: int
    positive_days: int
    observations: int


def _fetch(url: str, timeout: float = 90.0) -> bytes:
    req = Request(url, headers={"User-Agent": "phase10e/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _last_complete_days(days: int) -> list[str]:
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    return [(end - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]


def _download(symbol: str, day: str) -> pd.DataFrame:
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{day}.zip"
    payload = _fetch(url)
    with zipfile.ZipFile(BytesIO(payload)) as z:
        members = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not members:
            raise RuntimeError(f"no csv in {url}")
        with z.open(members[0]) as fh:
            raw = pd.read_csv(fh, header=None)
    raw = raw.iloc[:, :7].copy()
    raw.columns = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "buyer_maker"]
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["buyer_maker"] = raw["buyer_maker"].astype(bool)
    raw = raw.dropna(subset=["price", "qty", "timestamp"]).drop_duplicates("agg_id")
    unit = "us" if float(raw["timestamp"].median()) > 10_000_000_000_000 else "ms"
    raw["ts"] = pd.to_datetime(raw["timestamp"], unit=unit, utc=True)
    x = raw.set_index("ts")
    x["notional"] = x["price"] * x["qty"]
    # buyer_maker=True means seller initiated the trade, therefore negative signed flow.
    x["signed"] = np.where(x["buyer_maker"], -x["notional"], x["notional"])
    bars = x.resample("1min").agg(
        close=("price", "last"),
        notional=("notional", "sum"),
        signed=("signed", "sum"),
    ).dropna(subset=["close"])
    bars["imbalance"] = bars["signed"] / bars["notional"].replace(0.0, np.nan)
    bars["flow60"] = bars["imbalance"].rolling(60, min_periods=30).mean()
    bars["ret1m"] = bars["close"].pct_change()
    return bars.replace([np.inf, -np.inf], np.nan)


def _corr(a: pd.Series, b: pd.Series) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < 200 or z.iloc[:, 0].std() == 0 or z.iloc[:, 1].std() == 0:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def _residualize(target: pd.Series, market: pd.Series) -> pd.Series:
    z = pd.concat([target, market], axis=1).dropna()
    if len(z) < 200 or z.iloc[:, 1].std() == 0:
        return target.copy()
    x = z.iloc[:, 1].to_numpy(dtype=float)
    y = z.iloc[:, 0].to_numpy(dtype=float)
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    resid = y - beta * x
    out = pd.Series(index=z.index, data=resid)
    return out.reindex(target.index)


def _nonoverlap_ic(signal: pd.Series, future: pd.Series, step: int) -> float:
    idx = np.arange(0, len(signal), max(1, step))
    a = signal.iloc[idx].reset_index(drop=True)
    b = future.iloc[idx].reset_index(drop=True)
    return _corr(a, b)


def run(minutes: float = 15.0, seed: int = 20260829, days: int = 14) -> dict[str, Any]:
    del seed
    started = time.perf_counter()
    days = max(7, min(days, 30))
    day_list = _last_complete_days(days)
    print("=== PHASE 10E FLOW REVERSAL / RESIDUAL TEST ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"PERIOD: {day_list[0]} -> {day_list[-1]} | days={len(day_list)}", flush=True)

    series: dict[str, list[pd.DataFrame]] = {s: [] for s in SYMBOLS}
    errors: list[dict[str, str]] = []
    for d in day_list:
        for s in SYMBOLS:
            if time.perf_counter() - started > minutes * 60:
                break
            try:
                series[s].append(_download(s, d))
            except Exception as exc:
                errors.append({"symbol": s, "day": d, "error": repr(exc)})
        if time.perf_counter() - started > minutes * 60:
            break

    usable = {s: pd.concat(v).sort_index() for s, v in series.items() if v}
    market = pd.concat([usable[s]["ret1m"].rename(s) for s in usable], axis=1).median(axis=1)
    rows: list[TestRow] = []
    for s, bars in usable.items():
        if len(bars) < 5000:
            continue
        for horizon_h in (1, 6, 24):
            h = horizon_h * 60
            future = bars["close"].shift(-h) / bars["close"] - 1.0
            sig = bars["flow60"]
            direct = _corr(sig, future)
            rev = _corr(-sig, future)
            resid_future = _residualize(future, market.reindex(future.index).rolling(h).sum())
            residual = _corr(sig, resid_future)
            nonoverlap = _nonoverlap_ic(sig, future, h)
            byday = pd.DataFrame({"sig": sig, "f": future}).groupby(sig.index.date).apply(
                lambda g: _corr(g["sig"], g["f"])
            )
            vals = byday.dropna().to_numpy(dtype=float)
            sign_days = int(np.sum(vals < 0))
            positive_days = int(np.sum(vals > 0))
            row = TestRow(
                symbol=s,
                days=len(vals),
                horizon_h=horizon_h,
                direct_ic=float(direct),
                residual_ic=float(residual),
                reversed_ic=float(rev),
                nonoverlap_ic=float(nonoverlap),
                sign_days=sign_days,
                positive_days=positive_days,
                observations=int(pd.concat([sig, future], axis=1).dropna().shape[0]),
            )
            rows.append(row)
            print(
                f"TEST {s} h={horizon_h} | directIC={direct:.4f} reverseIC={rev:.4f} "
                f"residIC={residual:.4f} nonoverlapIC={nonoverlap:.4f} "
                f"sign_days={sign_days}/{len(vals)}",
                flush=True,
            )

    stable = [
        asdict(r) for r in rows
        if r.days >= 7
        and r.sign_days >= max(5, r.days - 2)
        and abs(r.nonoverlap_ic) >= 0.02
        and abs(r.residual_ic) >= 0.02
    ]
    decision = "PHASE10E_PERSISTENT_FLOW_REVERSAL" if stable else "PHASE10E_NO_PERSISTENT_FLOW_EDGE"
    result: dict[str, Any] = {
        "version": "phase10e_flow_reversal_residual_test",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start": day_list[0], "end": day_list[-1], "days": len(day_list)},
        "symbols": SYMBOLS,
        "rows": [asdict(r) for r in rows],
        "stable_candidates": stable,
        "errors": errors,
        "decision": decision,
        "next": "Only persistent non-overlapping residual edges advance to modeling; otherwise change information source.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    cp = OUT / "phase10e_flow_reversal_residual_latest.json"
    cp.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print("=== PHASE 10E COMPLETE ===", flush=True)
    print("DECISION:", decision, flush=True)
    print("TEST ROWS:", len(rows), flush=True)
    print("STABLE CANDIDATES:", len(stable), flush=True)
    print("ERRORS:", len(errors), flush=True)
    print("CHECKPOINT:", cp, flush=True)
    return result
