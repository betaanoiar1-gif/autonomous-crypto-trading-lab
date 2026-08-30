from __future__ import annotations

"""Phase 10 — Information Acquisition Lab.

Research only. No trading, optimization, or lockbox use.
Build a current availability matrix for richer market information before
investing compute in predictive modelling.

Sources checked:
- Binance Spot public market-data API
- Binance USD-M Futures public market-data API
- Binance public historical-data host (data.binance.vision)

The module is intentionally lightweight: it probes metadata/endpoints and
records availability, latency, schema, and historical-file reachability.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import requests
except Exception as exc:  # pragma: no cover
    requests = None
    _REQUESTS_ERROR = repr(exc)
else:
    _REQUESTS_ERROR = None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase10_information_acquisition_latest.json"

SPOT_API = "https://data-api.binance.vision"
FUTURES_API = "https://fapi.binance.com"
DATA_HOST = "https://data.binance.vision"

TARGETS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT",
)

@dataclass
class Probe:
    source: str
    endpoint: str
    status: int | None
    ok: bool
    latency_ms: float | None
    bytes: int | None
    note: str


def _save(payload: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _request(url: str, params: dict[str, Any] | None = None, timeout: float = 8.0) -> tuple[int | None, float | None, int | None, Any, str]:
    if requests is None:
        return None, None, None, None, f"requests unavailable: {_REQUESTS_ERROR}"
    started = time.perf_counter()
    try:
        r = requests.get(url, params=params, timeout=timeout)
        latency = (time.perf_counter() - started) * 1000.0
        raw = r.content
        try:
            body = r.json()
        except Exception:
            body = None
        return r.status_code, latency, len(raw), body, ""
    except Exception as exc:
        latency = (time.perf_counter() - started) * 1000.0
        return None, latency, None, None, repr(exc)


def _probe(url: str, source: str, note: str, params: dict[str, Any] | None = None) -> Probe:
    status, latency, size, body, err = _request(url, params=params)
    return Probe(
        source=source,
        endpoint=url,
        status=status,
        ok=(status is not None and 200 <= status < 300),
        latency_ms=latency,
        bytes=size,
        note=err or note,
    )


def _historical_file_candidates(symbol: str) -> list[str]:
    today = datetime.now(timezone.utc)
    year = today.year
    month = today.month
    # We only probe one recent monthly and one recent daily file. The
    # official downloader documents the same naming scheme and daily/monthly
    # organization; this is an availability check, not a bulk download.
    monthly = f"{DATA_HOST}/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip"
    daily = f"{DATA_HOST}/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{today:%Y-%m-%d}.zip"
    return [monthly, daily]


def run(minutes: float = 20.0, seed: int = 20260829) -> dict[str, Any]:
    del seed
    deadline = time.monotonic() + minutes * 60.0
    print("=== PHASE 10 INFORMATION ACQUISITION ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)

    if requests is None:
        payload = {"decision": "PHASE10_BLOCKED_REQUESTS", "error": _REQUESTS_ERROR}
        _save(payload)
        return payload

    probes: list[Probe] = []

    # Current symbol universe metadata.
    status, latency, size, spot_info, err = _request(f"{SPOT_API}/api/v3/exchangeInfo")
    spot_symbols = []
    if isinstance(spot_info, dict):
        for row in spot_info.get("symbols", []):
            if row.get("quoteAsset") == "USDT" and row.get("status") == "TRADING":
                spot_symbols.append(row.get("symbol"))
    probes.append(Probe("spot", "/api/v3/exchangeInfo", status,
                        bool(status and 200 <= status < 300), latency, size,
                        err or f"trading USDT symbols={len(spot_symbols)}"))
    print(f"SPOT EXCHANGE INFO | status={status} | symbols={len(spot_symbols)}", flush=True)

    # Current futures metadata.
    status, latency, size, fut_info, err = _request(f"{FUTURES_API}/fapi/v1/exchangeInfo")
    futures_symbols = []
    if isinstance(fut_info, dict):
        for row in fut_info.get("symbols", []):
            if row.get("quoteAsset") == "USDT" and row.get("status") == "TRADING":
                futures_symbols.append(row.get("symbol"))
    probes.append(Probe("usd_m_futures", "/fapi/v1/exchangeInfo", status,
                        bool(status and 200 <= status < 300), latency, size,
                        err or f"trading USDT symbols={len(futures_symbols)}"))
    print(f"FUTURES EXCHANGE INFO | status={status} | symbols={len(futures_symbols)}", flush=True)

    # Cheap endpoint probes for the research variables we care about.
    endpoint_probes = [
        (f"{SPOT_API}/api/v3/aggTrades", "spot", "public aggTrades"),
        (f"{SPOT_API}/api/v3/depth", "spot", "public order book"),
        (f"{SPOT_API}/api/v3/trades", "spot", "public recent trades"),
        (f"{FUTURES_API}/fapi/v1/aggTrades", "usd_m_futures", "public futures aggTrades"),
        (f"{FUTURES_API}/fapi/v1/fundingRate", "usd_m_futures", "public funding history"),
        (f"{FUTURES_API}/fapi/v1/openInterest", "usd_m_futures", "public current open interest"),
        (f"{FUTURES_API}/futures/data/openInterestHist", "usd_m_futures", "public historical open interest"),
        (f"{FUTURES_API}/futures/data/globalLongShortAccountRatio", "usd_m_futures", "public long/short ratio"),
    ]

    for url, source, note in endpoint_probes:
        params = None
        if "aggTrades" in url or "depth" in url or url.endswith("/trades") or "openInterest" in url:
            params = {"symbol": TARGETS[0]}
        if "fundingRate" in url:
            params = {"symbol": TARGETS[0], "limit": 1}
        if "globalLongShort" in url:
            params = {"symbol": TARGETS[0], "period": "1h", "limit": 1}
        p = _probe(url, source, note, params)
        probes.append(p)
        print(f"PROBE {source} | {note} | status={p.status} | ok={p.ok} | latency={p.latency_ms:.1f}ms" if p.latency_ms is not None else f"PROBE {source} | {note} | status={p.status} | ok={p.ok}", flush=True)
        if time.monotonic() >= deadline:
            break

    # Historical file reachability for the official public-data host.
    hist = []
    for symbol in TARGETS:
        for url in _historical_file_candidates(symbol):
            if time.monotonic() >= deadline:
                break
            started = time.perf_counter()
            try:
                r = requests.head(url, timeout=8.0, allow_redirects=True)
                ms = (time.perf_counter() - started) * 1000.0
                hist.append({
                    "symbol": symbol,
                    "url": url,
                    "status": r.status_code,
                    "ok": 200 <= r.status_code < 300,
                    "latency_ms": ms,
                    "content_length": r.headers.get("Content-Length"),
                })
            except Exception as exc:
                ms = (time.perf_counter() - started) * 1000.0
                hist.append({"symbol": symbol, "url": url, "status": None, "ok": False, "latency_ms": ms, "error": repr(exc)})
        if time.monotonic() >= deadline:
            break

    available_hist = sum(1 for x in hist if x["ok"])
    print(f"HISTORICAL FILE PROBES | ok={available_hist}/{len(hist)}", flush=True)

    categories = {
        "spot_trades": any(p.ok and p.note == "public aggTrades" for p in probes),
        "spot_orderbook": any(p.ok and p.note == "public order book" for p in probes),
        "futures_trades": any(p.ok and p.note == "public futures aggTrades" for p in probes),
        "funding": any(p.ok and p.note == "public funding history" for p in probes),
        "open_interest_current": any(p.ok and p.note == "public current open interest" for p in probes),
        "open_interest_history": any(p.ok and p.note == "public historical open interest" for p in probes),
        "long_short_ratio": any(p.ok and p.note == "public long/short ratio" for p in probes),
        "historical_public_files": available_hist > 0,
    }

    payload = {
        "version": "phase10_information_acquisition",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PHASE10_DATA_SOURCES_MAPPED",
        "markets": {
            "target_symbols": list(TARGETS),
            "spot_trading_usdt_count": len(spot_symbols),
            "futures_trading_usdt_count": len(futures_symbols),
        },
        "categories": categories,
        "probes": [asdict(x) for x in probes],
        "historical_file_probes": hist,
        "next": "Build permanent enriched cache only for categories marked available.",
    }
    _save(payload)
    print("=== PHASE 10 COMPLETE ===", flush=True)
    print("DECISION:", payload["decision"], flush=True)
    print("CHECKPOINT:", OUT, flush=True)
    return payload
