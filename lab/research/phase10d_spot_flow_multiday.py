from __future__ import annotations

import json
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT"]
BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
DEFAULT_OUT = Path("experiments/phase10d_flow")
DEFAULT_DAYS = 7


@dataclass(frozen=True)
class FlowDay:
    symbol: str
    day: str
    trades: int
    minutes: int
    notional: float
    mean_imbalance: float
    flow_ac1: float
    flow_ac5: float
    ic_1h: float
    ic_6h: float
    ic_24h: float
    reverse_ic_1h: float
    reverse_ic_6h: float
    reverse_ic_24h: float


@dataclass(frozen=True)
class PairEdge:
    source: str
    target: str
    horizon: int
    ic: float
    reverse_ic: float
    observations: int


def _last_complete_day() -> datetime:
    now = datetime.now(timezone.utc)
    return datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)


def _days(n: int, end: datetime) -> list[str]:
    return [(end - timedelta(days=i)).date().isoformat() for i in range(n - 1, -1, -1)]


def _url(symbol: str, day: str) -> str:
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{day}.zip"


def _get(url: str, timeout: float = 90.0) -> tuple[int, bytes, float]:
    t0 = time.perf_counter()
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "autonomous-crypto-trading-lab/phase10d"})
    return r.status_code, r.content, (time.perf_counter() - t0) * 1000.0


def _download(symbol: str, day: str, out: Path) -> pd.DataFrame:
    raw_dir = out / "raw_daily"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}_{day}.csv"
    if path.exists():
        return pd.read_csv(path)
    status, payload, latency = _get(_url(symbol, day))
    print(f"DOWNLOAD {symbol} {day} | status={status} | latency={latency:.1f}ms | bytes={len(payload)}", flush=True)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {_url(symbol, day)}")
    with zipfile.ZipFile(BytesIO(payload)) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise RuntimeError("CSV not found in archive")
        with z.open(csvs[0]) as fh:
            df = pd.read_csv(fh, header=None)
    if df.shape[1] < 7:
        raise RuntimeError(f"Expected >=7 columns, got {df.shape[1]}")
    df = df.iloc[:, :7].copy()
    df.columns = ["agg_id", "price", "qty", "first_id", "last_id", "timestamp", "buyer_maker"]
    for c in ("price", "qty", "timestamp"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["buyer_maker"] = df["buyer_maker"].astype(bool)
    df = df.dropna(subset=["agg_id", "price", "qty", "timestamp"]).drop_duplicates("agg_id")
    df.to_csv(path, index=False)
    return df


def _to_1m(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    unit = "us" if float(x["timestamp"].median()) > 10_000_000_000_000 else "ms"
    x["ts"] = pd.to_datetime(x["timestamp"], unit=unit, utc=True)
    x["notional"] = x["price"] * x["qty"]
    # Binance: buyer_maker=True means the buyer was the maker, so the taker/aggressor was a seller.
    x["signed_notional"] = np.where(x["buyer_maker"], -x["notional"], x["notional"])
    x = x.set_index("ts")
    bars = x.resample("1min").agg(
        close=("price", "last"),
        notional=("notional", "sum"),
        signed_notional=("signed_notional", "sum"),
        trades=("agg_id", "count"),
    ).dropna(subset=["close"])
    bars["imbalance"] = bars["signed_notional"] / bars["notional"].replace(0.0, np.nan)
    roll = bars["signed_notional"].rolling(60, min_periods=30)
    bars["flow_z"] = (bars["signed_notional"] - roll.mean()) / roll.std().replace(0.0, np.nan)
    bars["flow_5m"] = bars["flow_z"].rolling(5, min_periods=5).mean()
    bars["future_1h"] = bars["close"].shift(-60) / bars["close"] - 1.0
    bars["future_6h"] = bars["close"].shift(-360) / bars["close"] - 1.0
    bars["future_24h"] = bars["close"].shift(-1440) / bars["close"] - 1.0
    return bars.replace([np.inf, -np.inf], np.nan)


def _corr(a: pd.Series, b: pd.Series, min_n: int = 200) -> float:
    z = pd.concat([a, b], axis=1).dropna()
    if len(z) < min_n:
        return float("nan")
    if z.iloc[:, 0].std() == 0.0 or z.iloc[:, 1].std() == 0.0:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1]))


def _daily_stats(symbol: str, day: str, bars: pd.DataFrame) -> FlowDay:
    f = bars["imbalance"].dropna()
    ac1 = f.autocorr(1) if len(f) > 3 else np.nan
    ac5 = f.autocorr(5) if len(f) > 7 else np.nan
    signal = bars["flow_5m"]
    return FlowDay(
        symbol=symbol,
        day=day,
        trades=int(bars["trades"].sum()),
        minutes=int(len(bars)),
        notional=float(bars["notional"].sum()),
        mean_imbalance=float(bars["imbalance"].mean()),
        flow_ac1=float(ac1),
        flow_ac5=float(ac5),
        ic_1h=_corr(signal, bars["future_1h"]),
        ic_6h=_corr(signal, bars["future_6h"]),
        ic_24h=_corr(signal, bars["future_24h"]),
        reverse_ic_1h=_corr(-signal, bars["future_1h"]),
        reverse_ic_6h=_corr(-signal, bars["future_6h"]),
        reverse_ic_24h=_corr(-signal, bars["future_24h"]),
    )


def _load_symbol(symbol: str, days: list[str], out: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for day in days:
        parts.append(_to_1m(_download(symbol, day, out)))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts).sort_index()[~pd.concat(parts).sort_index().index.duplicated(keep="first")]


def _pair_edge(source: str, target: str, frames: dict[str, pd.DataFrame], horizon: int) -> PairEdge:
    src = frames[source]["flow_5m"].copy()
    dst = frames[target]["close"].copy()
    joined = pd.concat([src.rename("src"), dst.rename("close")], axis=1).sort_index().ffill(limit=2)
    future = joined["close"].shift(-horizon) / joined["close"] - 1.0
    z = pd.concat([joined["src"], future.rename("future")], axis=1).dropna()
    ic = _corr(z["src"], z["future"], min_n=500)
    ric = _corr(-z["src"], z["future"], min_n=500)
    return PairEdge(source, target, horizon, ic, ric, int(len(z)))


def run(minutes: float = 15.0, seed: int = 20260829, days: int = DEFAULT_DAYS, sample_symbols: int = 6) -> dict[str, Any]:
    del seed
    t0 = time.perf_counter()
    symbols = DEFAULT_SYMBOLS[: max(2, min(sample_symbols, len(DEFAULT_SYMBOLS)))]
    end = _last_complete_day()
    day_list = _days(max(3, min(days, 14)), end)
    out = Path(os.environ.get("PHASE10D_OUT", str(DEFAULT_OUT)))
    out.mkdir(parents=True, exist_ok=True)

    print("=== PHASE 10D MULTI-DAY SPOT FLOW ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"PERIOD: {day_list[0]} → {day_list[-1]} | days={len(day_list)}", flush=True)
    print(f"SYMBOLS: {len(symbols)} | {symbols}", flush=True)
    print("SIGN CONVENTION: buyer_maker=True => aggressive seller => negative flow", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    daily: list[FlowDay] = []
    errors: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=len(symbols)) as ex:
        futs = {ex.submit(_load_symbol, s, day_list, out): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                frame = fut.result()
                frames[s] = frame
                for day in day_list:
                    d0 = frame.loc[frame.index.date == datetime.fromisoformat(day).date()]
                    if not d0.empty:
                        daily.append(_daily_stats(s, day, d0))
                frame.to_parquet(out / f"{s}_1m_multiday.parquet")
                print(f"READY {s} | minutes={len(frame):d}", flush=True)
            except Exception as exc:
                errors.append({"symbol": s, "error": repr(exc)})
                print(f"ERROR {s}: {exc!r}", flush=True)

    # Fast multi-day direct flow screen using pooled minute observations.
    direct: list[dict[str, Any]] = []
    for s, frame in frames.items():
        signal = frame["flow_5m"]
        row = {
            "symbol": s,
            "days": len(day_list),
            "ic_1h": _corr(signal, frame["future_1h"], 500),
            "ic_6h": _corr(signal, frame["future_6h"], 500),
            "ic_24h": _corr(signal, frame["future_24h"], 500),
            "reverse_ic_1h": _corr(-signal, frame["future_1h"], 500),
            "reverse_ic_6h": _corr(-signal, frame["future_6h"], 500),
            "reverse_ic_24h": _corr(-signal, frame["future_24h"], 500),
            "ac1": float(frame["imbalance"].dropna().autocorr(1)),
            "ac5": float(frame["imbalance"].dropna().autocorr(5)),
        }
        direct.append(row)

    # Cross-crypto leadership screen: source flow(t) -> target return(t+h).
    pair_edges: list[PairEdge] = []
    horizons = (60, 360, 1440)
    for source in frames:
        for target in frames:
            if source == target:
                continue
            for h in horizons:
                if time.perf_counter() - t0 > minutes * 60.0:
                    break
                pair_edges.append(_pair_edge(source, target, frames, h))

    pair_edges_sorted = sorted(
        [e for e in pair_edges if np.isfinite(e.ic)],
        key=lambda e: abs(e.ic),
        reverse=True,
    )

    result: dict[str, Any] = {
        "version": "phase10d_spot_flow_multiday",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start": day_list[0], "end": day_list[-1], "days": len(day_list)},
        "symbols": symbols,
        "daily": [asdict(x) for x in sorted(daily, key=lambda x: (x.symbol, x.day))],
        "direct": direct,
        "pair_edges": [asdict(x) for x in pair_edges_sorted[:50]],
        "errors": errors,
        "decision": "PHASE10D_FLOW_SCREEN_COMPLETE" if len(frames) >= 2 else "PHASE10D_INSUFFICIENT_DATA",
        "next": "Use only persistent multi-day direction-consistent edges for the next modeling gate.",
    }
    cp = out / "phase10d_spot_flow_multiday_latest.json"
    cp.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")

    print("=== PHASE 10D COMPLETE ===", flush=True)
    print("DECISION:", result["decision"], flush=True)
    print("LOADED:", len(frames), flush=True)
    print("PAIR EDGES:", len(pair_edges_sorted), flush=True)
    print("ERRORS:", len(errors), flush=True)
    print("CHECKPOINT:", cp, flush=True)
    print("ELAPSED_SEC:", round(time.perf_counter() - t0, 2), flush=True)
    return result


if __name__ == "__main__":
    run()
