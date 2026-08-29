from __future__ import annotations

"""Phase 0 v4: aligned research-foundation audit.

This gate validates the research plumbing before discovery. It is intentionally
spot-only, long/flat, unlevered, and AI-free.

Key design points:
- reuses cached parquet when available;
- aligns all research markets to one common timestamp cutoff;
- reports gaps instead of treating every missing hour as corrupt data;
- uses a causal synthetic edge aligned with the backtest execution convention;
- estimates a random-noise false-positive floor;
- includes a deliberate lookahead canary;
- writes a durable checkpoint throughout the run.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData

OUT = ROOT / "experiments" / "phase0_foundation_v4_latest.json"
DATA_DIR = ROOT / "experiments" / "phase0_data_v4"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)
DEFAULT_CACHE = Path("/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3")


@dataclass
class Audit:
    name: str
    passed: bool
    details: dict


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(OUT)


def _hash_df(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(raw).hexdigest()


def _metrics(df: pd.DataFrame, signal: pd.Series, settings, cost_mult: float = 1.0) -> dict:
    # Phase 0 is spot-only. Any accidental negative signal is clipped away.
    signal = pd.Series(signal, index=df.index).astype(float).clip(0.0, 1.0)
    result = run_ohlcv(
        df,
        signal,
        settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot",
        leverage=1.0,
        funding_rates=None,
    )
    return dict(result.metrics)


def _flat(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def _buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    """Use a deliberately impossible signal and require it to be detectable."""
    flat = _metrics(df, _flat(df), settings, 1.0)
    future_return = df["close"].pct_change().shift(-1).fillna(0.0)
    leaked = _metrics(df, (future_return > 0).astype(float), settings, 1.0)
    distinguishable = bool(
        leaked["total_return"] > flat["total_return"] + 0.05
        or leaked["profit_factor"] > max(2.0, flat["profit_factor"] + 1.0)
    )
    return Audit(
        "lookahead_canary",
        distinguishable,
        {
            "flat_return": flat["total_return"],
            "leaked_return": leaked["total_return"],
            "leaked_pf": leaked["profit_factor"],
            "leaked_trades": leaked["trade_count"],
        },
    )


def _noise_floor(data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns: list[float] = []
    for i in range(trials):
        df = data[keys[i % len(keys)]]
        signal = pd.Series((rng.random(len(df)) > 0.65).astype(float), index=df.index)
        metrics = _metrics(df, signal, settings, 2.0)
        ok = bool(
            metrics["total_return"] > 0
            and metrics["profit_factor"] > 1.0
            and metrics["max_drawdown"] > -0.20
        )
        accepted += int(ok)
        returns.append(float(metrics["total_return"]))
    return {
        "trials": trials,
        "accepted": accepted,
        "false_positive_rate": accepted / max(trials, 1),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
    }


def _synthetic_edge_frame(seed: int, n: int = 10000, sigma: float = 0.0015, alpha: float = 0.00025) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a realistic bounded causal AR(1)-style edge.

    At close(t), r[t] is observable. The edge contributes alpha*sign(r[t])
    to r[t+1], which matches run_ohlcv's signal[t] -> return[t+1].
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        r[i] = float(np.clip(noise[i] + alpha * np.sign(r[i - 1]), -0.02, 0.02))

    close = 100.0 * np.cumprod(1.0 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(index=idx)
    frame["close"] = close
    frame["open"] = np.r_[close[0], close[:-1]]
    frame["high"] = np.maximum(frame["open"].to_numpy(), frame["close"].to_numpy())
    frame["low"] = np.minimum(frame["open"].to_numpy(), frame["close"].to_numpy())
    frame["volume"] = 1_000_000.0
    signal = pd.Series((r > 0).astype(float), index=idx)
    return frame, signal


def _synthetic_detection(settings, seed: int) -> dict:
    frame, signal = _synthetic_edge_frame(seed)
    normal = _metrics(frame, signal, settings, 1.0)
    stress = _metrics(frame, signal, settings, 2.0)
    detected = bool(
        normal["total_return"] > 0
        and stress["total_return"] > 0
        and normal["profit_factor"] > 1.0
    )
    return {
        "normal": normal,
        "stress": stress,
        "alpha_design": 0.00025,
        "noise_sigma": 0.0015,
        "detected": detected,
        "alignment": "signal(t) observes close(t) return and is applied to return(t+1)",
    }


def _integrity(df: pd.DataFrame, symbol: str) -> Audit:
    required = ["open", "high", "low", "close", "volume"]
    values = df[required].to_numpy(dtype=float)
    monotonic = bool(df.index.is_monotonic_increasing)
    unique = bool(df.index.is_unique)
    finite = bool(np.isfinite(values).all())
    valid_ohlc = bool(
        (
            (df["high"] >= df[["open", "close"]].max(axis=1))
            & (df["low"] <= df[["open", "close"]].min(axis=1))
            & (df["high"] >= df["low"])
        ).all()
    deltas = df.index.to_series().diff().dropna()
    bad_intervals = int((deltas != pd.Timedelta(hours=1)).sum()) if len(deltas) else 0
    max_gap = deltas.max() if len(deltas) else pd.Timedelta(0)
    # Gaps are reported, not automatically treated as corruption.
    passed = monotonic and unique and finite and valid_ohlc
    return Audit(
        f"integrity:{symbol}",
        passed,
        {
            "bars": len(df),
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "monotonic": monotonic,
            "unique": unique,
            "finite": finite,
            "valid_ohlc": valid_ohlc,
            "bad_1h_intervals": bad_intervals,
            "max_gap": str(max_gap),
            "sha256": _hash_df(df),
        },
    )


def _load_data(minutes: float, bars: int, max_symbols: int, cache_dir: Path, seed: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    deadline = time.monotonic() + minutes * 60.0
    adapter = CCXTMarketData()
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []
    symbols = list(SYMBOLS)[:max_symbols]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        if time.monotonic() >= deadline:
            errors.append({"symbol": symbol, "error": "deadline_before_load"})
            break
        filename = f"{symbol.replace('/', '_')}_1h.parquet"
        cache_path = cache_dir / filename
        local_path = DATA_DIR / filename
        try:
            if cache_path.exists() and cache_path.stat().st_size > 0:
                print(f"CACHE {symbol} 1h ...", flush=True)
                df = pd.read_parquet(cache_path)
            elif local_path.exists() and local_path.stat().st_size > 0:
                print(f"LOCAL {symbol} 1h ...", flush=True)
                df = pd.read_parquet(local_path)
            else:
                print(f"FETCH {symbol} 1h ...", flush=True)
                df = adapter.fetch_ohlcv_history(symbol, "1h", bars, page_limit=1500, market_type="spot")
            df = df.sort_index().copy()
            if len(df) < min(bars, 5000):
                raise ValueError(f"only {len(df)} bars available")
            data[symbol] = df.iloc[-min(len(df), bars):].copy()
            data[symbol].to_parquet(local_path, index=True)
            print(f"OK {symbol}: {len(data[symbol])} bars | {data[symbol].index[0]} -> {data[symbol].index[-1]}", flush=True)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)
    return data, errors


def _align_common_cutoff(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    common_end = min(df.index[-1] for df in data.values())
    aligned = {symbol: df.loc[df.index <= common_end].copy() for symbol, df in data.items()}
    common_start = max(df.index[0] for df in aligned.values())
    aligned = {symbol: df.loc[df.index >= common_start].copy() for symbol, df in aligned.items()}
    return aligned, {
        "common_start": str(common_start),
        "common_end": str(common_end),
        "bars": {symbol: len(df) for symbol, df in aligned.items()},
    }


def run(minutes: float = 30.0, bars: int = 50000, max_symbols: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    settings = load_settings()
    cache_dir = Path(os.getenv("PHASE0_CACHE_DIR", str(DEFAULT_CACHE)))

    print("=== PHASE 0 FOUNDATION V4 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Target: {bars} 1h bars × {max_symbols} spot markets", flush=True)
    print(f"Cache: {cache_dir}", flush=True)
    print(f"Checkpoint: {OUT}", flush=True)
    _save({"started_at": started.isoformat(), "decision": "LOADING", "cache": str(cache_dir)})

    raw_data, errors = _load_data(minutes, bars, max_symbols, cache_dir, seed)
    if len(raw_data) < min(max_symbols, 8):
        payload = {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": "PHASE0_DATA_INSUFFICIENT",
            "markets_loaded": list(raw_data),
            "errors": errors,
        }
        _save(payload)
        return payload

    data, alignment = _align_common_cutoff(raw_data)
    audits = [_integrity(df, symbol) for symbol, df in data.items()]
    print("=== DATA INTEGRITY ===", flush=True)
    for audit in audits:
        print(f"{audit.name}: {'PASS' if audit.passed else 'FAIL'}", flush=True)

    # Require sufficient aligned history, but do not reject ordinary historical gaps.
    aligned_min_bars = min(len(df) for df in data.values())
    data_ok = len(data) >= min(max_symbols, 8) and aligned_min_bars >= min(bars, 5000)
    integrity_ok = all(a.passed for a in audits)

    probe_df = next(iter(data.values())).iloc[:120]
    alternating = pd.Series(np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0), index=probe_df.index)
    probe = _metrics(probe_df, alternating, settings)
    spot_safe = bool(probe["short_exposure"] == 0.0 and probe["long_exposure"] > 0.0)
    print(f"SPOT POSITION PROBE: {'PASS' if spot_safe else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    common_reference = next(iter(data.values()))
    canary = _lookahead_canary(common_reference, settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | leaked_return={canary.details['leaked_return']:.2%} | PF={canary.details['leaked_pf']:.2f}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_detection(settings, seed + 1)
    synthetic_ok = bool(synthetic["detected"])
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic_ok else 'FAIL'}", flush=True)

    gates = {
        "data": data_ok,
        "integrity": integrity_ok,
        "spot_long_flat": spot_safe,
        "lookahead_canary": canary.passed,
        "noise_fp_le_5pct": noise_ok,
        "synthetic_edge_detected": synthetic_ok,
    }
    decision = "PHASE0_READY_FOR_DISCOVERY" if all(gates.values()) else "PHASE0_BLOCKED"

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (datetime.now(timezone.utc) - started).total_seconds() / 60.0,
        "decision": decision,
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "alignment": alignment,
        "markets_loaded": list(data),
        "bars": {symbol: len(df) for symbol, df in data.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "gap_policy": "gaps reported; ordinary missing hourly candles do not fail integrity",
        "spot_position_probe": probe,
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "gates": gates,
        "data_dir": str(DATA_DIR),
        "data_files": [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))],
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run(
        minutes=float(os.getenv("PHASE0_MINUTES", "30")),
        bars=int(os.getenv("PHASE0_BARS", "50000")),
        max_symbols=int(os.getenv("PHASE0_MAX_SYMBOLS", "12")),
        seed=int(os.getenv("PHASE0_SEED", "20260829")),
    )
