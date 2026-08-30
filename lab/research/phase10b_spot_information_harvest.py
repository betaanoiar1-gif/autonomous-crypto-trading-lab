from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
    "DOGEUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT",
]
DEFAULT_DATA_API = "https://data-api.binance.vision"
DEFAULT_PUBLIC_DATA = "https://data.binance.vision"


@dataclass
class Probe:
    category: str
    symbol: str
    url: str
    status: int | None
    ok: bool
    latency_ms: float | None
    bytes: int | None
    note: str = ""


def _get(session: requests.Session, url: str, timeout: float = 15.0) -> tuple[int | None, bytes, float, str]:
    started = time.perf_counter()
    try:
        r = session.get(url, timeout=timeout)
        return r.status_code, r.content, (time.perf_counter() - started) * 1000.0, r.headers.get("content-type", "")
    except Exception as exc:
        return None, b"", (time.perf_counter() - started) * 1000.0, repr(exc)


def _last_completed_month(ref: datetime | None = None) -> tuple[int, int]:
    now = ref or datetime.now(timezone.utc)
    first_this = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    last = first_this - timedelta(days=1)
    return last.year, last.month


def _last_completed_day(ref: datetime | None = None) -> datetime:
    now = ref or datetime.now(timezone.utc)
    today = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    return today - timedelta(days=1)


def _monthly_url(symbol: str, year: int, month: int) -> str:
    return (
        f"{DEFAULT_PUBLIC_DATA}/data/spot/monthly/aggTrades/{symbol}/"
        f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
    )


def _daily_url(symbol: str, day: datetime) -> str:
    return (
        f"{DEFAULT_PUBLIC_DATA}/data/spot/daily/aggTrades/{symbol}/"
        f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    )


def _save(payload: dict[str, Any], repo: Path) -> Path:
    exp = repo / "experiments"
    exp.mkdir(parents=True, exist_ok=True)
    out = exp / "phase10b_spot_information_harvest_latest.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def run(minutes: float = 15.0, seed: int = 20260829) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    cache = Path(
        os.environ.get(
            "PHASE0_CACHE_DIR",
            "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3",
        )
    ).resolve()
    symbols = [s.strip().upper() for s in os.environ.get("PHASE10B_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
    sample_symbols = symbols[: int(os.environ.get("PHASE10B_SAMPLE_SYMBOLS", "6"))]

    print("=== PHASE 10B SPOT INFORMATION HARVEST ===", flush=True)
    print("RESEARCH ONLY | NO MODELING | NO TRADING", flush=True)
    print(f"Cache: {cache}", flush=True)
    print(f"Target symbols: {len(symbols)} | sample: {len(sample_symbols)}", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "autonomous-crypto-trading-lab/phase10b"})

    probes: list[Probe] = []

    # Spot API reachability.
    status, body, lat, ctype = _get(session, f"{DEFAULT_DATA_API}/api/v3/exchangeInfo")
    spot_ok = status == 200
    print(f"SPOT EXCHANGE INFO | status={status} | ok={spot_ok}", flush=True)

    exchange_symbols = 0
    if spot_ok:
        try:
            payload = json.loads(body.decode("utf-8"))
            exchange_symbols = sum(
                1 for x in payload.get("symbols", [])
                if x.get("status") == "TRADING" and x.get("quoteAsset") == "USDT"
            )
        except Exception:
            pass

    print(f"SPOT USDT TRADING SYMBOLS: {exchange_symbols}", flush=True)

    # Use the most recent completed month automatically. If that fails, try the most recent completed day.
    year, month = _last_completed_month()
    day = _last_completed_day()
    print(f"AUTO PERIOD | month={year:04d}-{month:02d} | day={day:%Y-%m-%d}", flush=True)

    deadline = time.time() + max(10.0, minutes * 60.0)

    monthly_ok = 0
    daily_ok = 0
    sample_results: list[dict[str, Any]] = []

    for symbol in sample_symbols:
        if time.time() >= deadline:
            print("TIME BUDGET REACHED", flush=True)
            break

        murl = _monthly_url(symbol, year, month)
        status, body, lat, ctype = _get(session, murl)
        ok = status == 200 and len(body) > 0 and body[:2] == b"PK"
        probes.append(Probe("monthly_aggTrades", symbol, murl, status, ok, lat, len(body)))
        print(
            f"MONTHLY {symbol} | status={status} | ok={ok} | "
            f"latency={lat:.1f}ms | bytes={len(body)}",
            flush=True,
        )
        if ok:
            monthly_ok += 1
            sample_results.append({"symbol": symbol, "monthly": True, "monthly_bytes": len(body)})
            continue

        durl = _daily_url(symbol, day)
        status, body, lat, ctype = _get(session, durl)
        ok = status == 200 and len(body) > 0 and body[:2] == b"PK"
        probes.append(Probe("daily_aggTrades", symbol, durl, status, ok, lat, len(body)))
        print(
            f"DAILY {symbol} | status={status} | ok={ok} | "
            f"latency={lat:.1f}ms | bytes={len(body)}",
            flush=True,
        )
        if ok:
            daily_ok += 1
        sample_results.append({"symbol": symbol, "monthly": False, "daily": ok, "daily_bytes": len(body)})

    # Small live endpoints: these are cheap proxies for order-flow enrichment availability.
    for symbol in sample_symbols[: min(3, len(sample_symbols))]:
        if time.time() >= deadline:
            break
        for name, path in (
            ("aggTrades", "/api/v3/aggTrades"),
            ("depth", "/api/v3/depth"),
            ("trades", "/api/v3/trades"),
        ):
            url = f"{DEFAULT_DATA_API}{path}?symbol={symbol}&limit=1000" if name != "depth" else f"{DEFAULT_DATA_API}{path}?symbol={symbol}&limit=100"
            status, body, lat, ctype = _get(session, url)
            ok = status == 200 and len(body) > 0
            probes.append(Probe(f"spot_{name}", symbol, url, status, ok, lat, len(body)))
            print(
                f"LIVE {name} {symbol} | status={status} | ok={ok} | latency={lat:.1f}ms | bytes={len(body)}",
                flush=True,
            )

    decision = "PHASE10B_READY_FOR_SPOT_FLOW_HARVEST" if monthly_ok + daily_ok >= max(3, len(sample_symbols) // 2) else "PHASE10B_SPOT_HISTORICAL_ACCESS_WEAK"

    result = {
        "version": "phase10b_spot_information_harvest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "decision": decision,
        "cache_exists": cache.exists(),
        "symbols": symbols,
        "sample_symbols": sample_symbols,
        "spot_usdt_trading_symbols": exchange_symbols,
        "auto_period": {"completed_month": f"{year:04d}-{month:02d}", "completed_day": day.strftime("%Y-%m-%d")},
        "monthly_success": monthly_ok,
        "daily_success": daily_ok,
        "probes": [asdict(p) for p in probes],
        "sample_results": sample_results,
        "next": "Build 1m/5m order-flow feature cache from accessible historical aggTrades before any modeling.",
    }
    cp = _save(result, repo)
    print("=== PHASE 10B COMPLETE ===", flush=True)
    print("DECISION:", decision, flush=True)
    print("CHECKPOINT:", cp, flush=True)
    return result


if __name__ == "__main__":
    run()
