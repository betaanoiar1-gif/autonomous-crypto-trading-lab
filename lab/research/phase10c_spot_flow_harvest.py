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

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
DEFAULT_OUT = Path("experiments/phase10c_flow")


@dataclass(frozen=True)
class FlowStats:
    symbol: str
    date: str
    rows: int
    minutes: int
    total_notional: float
    signed_notional_ratio: float
    flow_autocorr_1: float
    flow_autocorr_5: float
    imbalance_mean: float
    imbalance_std: float
    price_impact_median: float
    return_1h_ic: float
    return_6h_ic: float
    return_24h_ic: float


def _last_complete_day() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()


def _fetch(url: str, timeout: float = 90.0) -> tuple[int, bytes, float]:
    started = time.perf_counter()
    req = Request(url, headers={"User-Agent": "phase10c-research/1.0"})
    with urlopen(req, timeout=timeout) as response:
        payload = response.read()
        status = int(getattr(response, "status", 200))
    return status, payload, (time.perf_counter() - started) * 1000.0


def _download_one(symbol: str, day: str) -> tuple[pd.DataFrame, float]:
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day}.zip"
    status, payload, latency = _fetch(url)
    print(
        f"DOWNLOAD {symbol} {day} | status={status} | latency={latency:.1f}ms | bytes={len(payload)}",
        flush=True,
    )
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {url}")
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        csvs = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError("No CSV member in archive")
        with archive.open(csvs[0]) as fh:
            raw = pd.read_csv(fh, header=None)
    if raw.shape[1] < 7:
        raise RuntimeError(f"Unexpected aggTrades column count: {raw.shape[1]}")
    raw = raw.iloc[:, :7].copy()
    raw.columns = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "buyer_maker"]
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["qty"] = pd.to_numeric(raw["qty"], errors="coerce")
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["buyer_maker"] = raw["buyer_maker"].astype(bool)
    raw = raw.dropna(subset=["price", "qty", "timestamp"])
    raw = raw.drop_duplicates(subset=["agg_id"]).sort_values("timestamp")
    unit = "us" if float(raw["timestamp"].median()) > 10_000_000_000_000 else "ms"
    raw["ts"] = pd.to_datetime(raw["timestamp"], unit=unit, utc=True)
    return raw, latency


def _minute_features(raw: pd.DataFrame) -> pd.DataFrame:
    x = raw.set_index("ts").copy()
    x["notional"] = x["price"] * x["qty"]
    x["signed_notional"] = np.where(x["buyer_maker"], -x["notional"], x["notional"])
    bars = x.resample("1min").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("qty", "sum"),
        notional=("notional", "sum"),
        signed_notional=("signed_notional", "sum"),
        trades=("agg_id", "count"),
    ).dropna(subset=["close"])

    bars["return"] = bars["close"].pct_change()
    bars["imbalance"] = bars["signed_notional"] / bars["notional"].replace(0.0, np.nan)
    mu = bars["signed_notional"].rolling(60, min_periods=30).mean()
    sd = bars["signed_notional"].rolling(60, min_periods=30).std()
    bars["flow_z"] = (bars["signed_notional"] - mu) / sd.replace(0.0, np.nan)
    bars["impact"] = bars["return"] / (bars["notional"] / 1_000_000.0).replace(0.0, np.nan)
    bars["flow_5m"] = bars["flow_z"].rolling(5, min_periods=5).mean()
    bars["future_1h"] = bars["close"].shift(-60) / bars["close"] - 1.0
    bars["future_6h"] = bars["close"].shift(-360) / bars["close"] - 1.0
    bars["future_24h"] = bars["close"].shift(-1440) / bars["close"] - 1.0
    return bars.replace([np.inf, -np.inf], np.nan)


def _corr(a: pd.Series, b: pd.Series) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < 100 or z.iloc[:, 0].std() == 0 or z.iloc[:, 1].std() == 0:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def _stats(symbol: str, day: str, raw: pd.DataFrame, bars: pd.DataFrame) -> FlowStats:
    flow = bars["imbalance"].dropna()
    ac1 = flow.autocorr(1) if len(flow) > 3 else np.nan
    ac5 = flow.autocorr(5) if len(flow) > 7 else np.nan
    signal = bars["flow_5m"]
    return FlowStats(
        symbol=symbol,
        date=day,
        rows=int(len(raw)),
        minutes=int(len(bars)),
        total_notional=float(bars["notional"].sum()),
        signed_notional_ratio=float(bars["signed_notional"].sum() / max(bars["notional"].sum(), 1e-12)),
        flow_autocorr_1=float(ac1),
        flow_autocorr_5=float(ac5),
        imbalance_mean=float(bars["imbalance"].mean()),
        imbalance_std=float(bars["imbalance"].std()),
        price_impact_median=float(bars["impact"].median()),
        return_1h_ic=_corr(signal, bars["future_1h"]),
        return_6h_ic=_corr(signal, bars["future_6h"]),
        return_24h_ic=_corr(signal, bars["future_24h"]),
    )


def run(minutes: float = 15.0, seed: int = 20260829, sample_symbols: int = 6) -> dict[str, Any]:
    del seed
    started = time.perf_counter()
    day = _last_complete_day()
    symbols = DEFAULT_SYMBOLS[: max(1, min(sample_symbols, len(DEFAULT_SYMBOLS)))]
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 10C SPOT FLOW HARVEST ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"LAST COMPLETE DAY: {day}", flush=True)
    print(f"SYMBOLS: {len(symbols)} | {symbols}", flush=True)

    completed: list[FlowStats] = []
    errors: list[dict[str, str]] = []

    for symbol in symbols:
        if time.perf_counter() - started >= minutes * 60.0:
            errors.append({"symbol": symbol, "error": "time_budget_exceeded"})
            break
        try:
            raw, _ = _download_one(symbol, day)
            bars = _minute_features(raw)
            stats = _stats(symbol, day, raw, bars)
            completed.append(stats)
            bars.to_parquet(out / f"{symbol}_1m_{day}.parquet", index=True)
            print(
                f"FLOW {symbol} | rows={stats.rows} minutes={stats.minutes} "
                f"AC1={stats.flow_autocorr_1:.3f} AC5={stats.flow_autocorr_5:.3f} "
                f"IC1h={stats.return_1h_ic:.4f} IC6h={stats.return_6h_ic:.4f} IC24h={stats.return_24h_ic:.4f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "error": repr(exc)})
            print(f"ERROR {symbol}: {exc!r}", flush=True)

    decision = (
        "PHASE10C_READY_FOR_FLOW_MODELING"
        if len(completed) >= max(2, len(symbols) // 2)
        else "PHASE10C_INSUFFICIENT_FLOW_DATA"
    )

    result: dict[str, Any] = {
        "version": "phase10c_spot_flow_harvest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "date": day,
        "symbols": symbols,
        "completed": [asdict(x) for x in completed],
        "errors": errors,
        "decision": decision,
        "next": "Run a fast multi-market flow screen before expanding the raw-data universe.",
    }

    checkpoint = out / "phase10c_spot_flow_harvest_latest.json"
    checkpoint.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== PHASE 10C COMPLETE ===", flush=True)
    print("DECISION:", decision, flush=True)
    print("COMPLETED:", len(completed), flush=True)
    print("ERRORS:", len(errors), flush=True)
    print("CHECKPOINT:", checkpoint, flush=True)

    return result
