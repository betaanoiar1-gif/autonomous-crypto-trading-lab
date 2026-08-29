from __future__ import annotations

"""Phase 0 v3: research-microscope validation.

No strategy discovery happens here. The goal is to prove that the data,
execution timing, cost model, noise controls, and known-edge detector are
sane before any evolutionary search is allowed to run.
"""

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData

OUT = ROOT / "experiments" / "phase0_foundation_v3_latest.json"
DATA_DIR = ROOT / "experiments" / "phase0_data_v3"

SYMBOLS = (
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
    "SOL/USDT", "ADA/USDT", "DOGE/USDT", "LTC/USDT",
    "LINK/USDT", "DOT/USDT", "AVAX/USDT", "TRX/USDT",
)


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
    # Phase 0 is spot-only and long/flat only.
    pos = pd.Series(signal, index=df.index).astype(float).clip(0.0, 1.0)
    result = run_ohlcv(
        df,
        pos,
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


def _vol_matched(df: pd.DataFrame, target_vol: float = 0.25) -> pd.Series:
    ret = df["close"].astype(float).pct_change().fillna(0.0)
    vol = ret.rolling(168).std() * math.sqrt(8760.0)
    return (target_vol / vol.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)


def _lookahead_canary(df: pd.DataFrame, settings) -> Audit:
    flat = _metrics(df, _flat(df), settings)
    future_return = df["close"].astype(float).pct_change().shift(-1).fillna(0.0)
    leaked = _metrics(df, (future_return > 0.0).astype(float), settings)
    distinguishable = bool(
        leaked["profit_factor"] > 2.0
        or leaked["total_return"] > flat["total_return"] + 0.05
    )
    return Audit(
        "lookahead_canary",
        distinguishable,
        {
            "flat_return": float(flat["total_return"]),
            "leaked_return": float(leaked["total_return"]),
            "leaked_pf": float(leaked["profit_factor"]),
            "leaked_trades": int(leaked["trade_count"]),
        },
    )


def _noise_floor(data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns: list[float] = []
    for i in range(trials):
        df = data[keys[i % len(keys)]]
        signal = pd.Series(
            (rng.random(len(df)) > 0.65).astype(float),
            index=df.index,
        )
        m = _metrics(df, signal, settings, cost_mult=2.0)
        ok = bool(
            m["total_return"] > 0.0
            and m["profit_factor"] > 1.0
            and m["max_drawdown"] > -0.20
        )
        accepted += int(ok)
        returns.append(float(m["total_return"]))
    return {
        "trials": trials,
        "accepted": accepted,
        "false_positive_rate": accepted / max(trials, 1),
        "median_return": float(np.median(returns)) if returns else 0.0,
        "max_return": float(np.max(returns)) if returns else 0.0,
    }


def _synthetic_edge(seed: int, n: int = 8000, alpha: float = 0.0025, sigma: float = 0.0015) -> tuple[pd.DataFrame, pd.Series]:
    """Known causal edge aligned to run_ohlcv timing.

    run_ohlcv uses signal[t] as the position during return[t+1]. Therefore
    synthetic r[t+1] is constructed from the information available at close t:
    sign(r[t]). This is deliberately causal and should remain profitable after
    realistic costs.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, size=n)
    for t in range(n - 1):
        state = 1.0 if r[t] >= 0.0 else -1.0
        r[t + 1] = noise[t + 1] + alpha * state

    close = 100.0 * np.cumprod(1.0 + r)
    index = pd.date_range("2019-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["open"] = np.r_[close[0], close[:-1]]
    frame["close"] = close
    frame["high"] = np.maximum(frame["open"].to_numpy(), close)
    frame["low"] = np.minimum(frame["open"].to_numpy(), close)
    frame["volume"] = 1_000_000.0

    # Current r[t] is known at close(t), so it may causally choose position
    # for r[t+1]. The first point is neutral.
    signal = pd.Series((r >= 0.0).astype(float), index=index)
    signal.iloc[0] = 0.0
    return frame, signal


def _synthetic_detection(settings, seed: int) -> dict:
    frame, signal = _synthetic_edge(seed)
    normal = _metrics(frame, signal, settings, cost_mult=1.0)
    stress = _metrics(frame, signal, settings, cost_mult=2.0)
    detected = bool(
        normal["total_return"] > 0.0
        and normal["profit_factor"] > 1.0
        and stress["total_return"] > 0.0
        and stress["profit_factor"] > 1.0
    )
    return {
        "normal": normal,
        "stress": stress,
        "alpha_design": 0.0025,
        "noise_sigma": 0.0015,
        "detected": detected,
        "alignment": "signal(t) observes close(t) return and is applied to return(t+1)",
    }


def _integrity(data: dict[str, pd.DataFrame]) -> list[Audit]:
    audits: list[Audit] = []
    for symbol, df in data.items():
        values = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        monotonic = bool(df.index.is_monotonic_increasing)
        unique = bool(df.index.is_unique)
        finite = bool(np.isfinite(values).all())
        valid_ohlc = bool(
            ((df["high"] >= df[["open", "close"]].max(axis=1))
             & (df["low"] <= df[["open", "close"]].min(axis=1))).all()
        )
        interval_ok = bool(
            np.all(np.diff(df.index.view("int64")) == 3_600_000_000_000)
        ) if len(df) > 1 else False
        closed = bool(
            (datetime.now(timezone.utc) - df.index[-1].to_pydatetime()).total_seconds() >= 3600
        )
        passed = monotonic and unique and finite and valid_ohlc and interval_ok and closed
        audits.append(Audit(
            f"integrity:{symbol}",
            passed,
            {
                "bars": len(df),
                "monotonic": monotonic,
                "unique": unique,
                "finite": finite,
                "valid_ohlc": valid_ohlc,
                "hourly_spacing": interval_ok,
                "closed_candle": closed,
                "sha256": _hash_df(df),
                "start": str(df.index[0]),
                "end": str(df.index[-1]),
            },
        ))
    return audits


def run(minutes: float = 180.0, bars: int = 50000, max_symbols: int = 12, seed: int = 20260829) -> dict:
    started = datetime.now(timezone.utc)
    deadline = time.monotonic() + minutes * 60.0
    settings = load_settings()
    adapter = CCXTMarketData()
    symbols = list(SYMBOLS)[:max_symbols]
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []

    print("=== PHASE 0 FOUNDATION V3 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Target: {bars} 1h bars × {len(symbols)} spot markets", flush=True)
    print(f"Checkpoint: {OUT}", flush=True)
    _save({"started_at": started.isoformat(), "decision": "LOADING", "target_bars": bars, "symbols": symbols})

    for symbol in symbols:
        if time.monotonic() >= deadline:
            errors.append({"symbol": symbol, "error": "deadline_before_load"})
            break
        print(f"LOAD {symbol} 1h ...", flush=True)
        try:
            df = adapter.fetch_ohlcv_history(symbol, "1h", bars, page_limit=1500, market_type="spot")
            data[symbol] = df
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = DATA_DIR / f"{symbol.replace('/', '_')}_1h.parquet"
            df.to_parquet(path, index=True)
            print(f"OK {symbol}: {len(df)} bars | {df.index[0]} -> {df.index[-1]}", flush=True)
            _save({
                "started_at": started.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "decision": "LOADING",
                "loaded": list(data),
                "bars": {k: len(v) for k, v in data.items()},
                "errors": errors,
            })
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {type(exc).__name__}: {exc}", flush=True)

    # Full Phase 0 needs at least eight markets. For a deliberately smaller
    # smoke run, the caller should set max_symbols accordingly and the minimum
    # becomes min(8, requested symbols).
    required_markets = min(8, len(symbols))
    if len(data) < required_markets or not data or min(map(len, data.values())) < min(bars, 5000):
        payload = {
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "decision": "PHASE0_DATA_INSUFFICIENT",
            "loaded_markets": len(data),
            "required_markets": required_markets,
            "required_bars": min(bars, 5000),
            "errors": errors,
        }
        _save(payload)
        print("=== PHASE 0 DECISION ===", flush=True)
        print(payload["decision"], flush=True)
        return payload

    audits = _integrity(data)
    print("=== DATA INTEGRITY ===", flush=True)
    for a in audits:
        print(f"{a.name}: {'PASS' if a.passed else 'FAIL'}", flush=True)

    probe_df = next(iter(data.values())).iloc[:120]
    probe_signal = pd.Series(np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0), index=probe_df.index)
    probe = _metrics(probe_df, probe_signal, settings)
    spot_safe = bool(probe["short_exposure"] == 0.0 and probe["long_exposure"] > 0.0)
    print(f"SPOT POSITION PROBE: {'PASS' if spot_safe else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)

    canary = _lookahead_canary(next(iter(data.values())), settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | leaked_return={canary.details['leaked_return']:.2%} | PF={canary.details['leaked_pf']:.2f}", flush=True)

    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise_floor(data, settings, seed, trials=int(os.getenv("PHASE0_NOISE_TRIALS", "100")))
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)

    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic_detection(settings, seed + 1)
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic['detected'] else 'FAIL'}", flush=True)

    benchmark = {}
    for symbol, df in list(data.items())[:6]:
        benchmark[symbol] = {
            "buy_hold": _metrics(df, _buy_hold(df), settings, 1.0),
            "buy_hold_stress": _metrics(df, _buy_hold(df), settings, 2.0),
            "vol_matched": _metrics(df, _vol_matched(df), settings, 1.0),
        }

    integrity_ok = all(a.passed for a in audits)
    data_ok = len(data) >= required_markets and min(map(len, data.values())) >= min(bars, 5000)
    noise_ok = noise["false_positive_rate"] <= 0.05
    decision = "PHASE0_READY_FOR_DISCOVERY" if all((integrity_ok, data_ok, spot_safe, canary.passed, noise_ok, synthetic["detected"])) else "PHASE0_BLOCKED"

    payload = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_minutes": (datetime.now(timezone.utc) - started).total_seconds() / 60.0,
        "decision": decision,
        "ai_generation": False,
        "spot_only": True,
        "long_flat_only": True,
        "leverage": 1.0,
        "markets_loaded": list(data),
        "bars": {k: len(v) for k, v in data.items()},
        "errors": errors,
        "integrity": [asdict(a) for a in audits],
        "spot_position_probe": {"passed": spot_safe, "details": probe},
        "lookahead_canary": asdict(canary),
        "noise_floor": noise,
        "synthetic_edge": synthetic,
        "benchmarks": benchmark,
        "gates": {
            "data": data_ok,
            "integrity": integrity_ok,
            "spot_long_flat": spot_safe,
            "lookahead_canary": canary.passed,
            "noise_fp_le_5pct": noise_ok,
            "synthetic_edge_detected": synthetic["detected"],
        },
        "data_dir": str(DATA_DIR),
        "data_files": [str(p) for p in sorted(DATA_DIR.glob("*.parquet"))],
    }
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload


if __name__ == "__main__":
    run()
