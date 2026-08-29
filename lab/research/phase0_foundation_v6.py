from __future__ import annotations
"""Phase 0 v6: calibrated synthetic-edge validation."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

from ..backtest.engine import run_ohlcv
from ..config import load_settings

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "phase0_foundation_v6_latest.json"
CACHE = Path(os.getenv("PHASE0_CACHE_DIR", "/tmp/autonomous_crypto_trading_lab_phase0/experiments/phase0_data_v3"))
SYMBOLS = ("BTC/USDT","ETH/USDT","BNB/USDT","XRP/USDT","SOL/USDT","ADA/USDT","DOGE/USDT","LTC/USDT","LINK/USDT","DOT/USDT","AVAX/USDT","TRX/USDT")

@dataclass
class Audit:
    name: str
    passed: bool
    details: dict


def _save(payload: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(OUT) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(OUT)


def _metrics(df: pd.DataFrame, signal: pd.Series, settings, cost_mult: float = 1.0) -> dict:
    pos = pd.Series(signal, index=df.index, dtype=float).clip(0.0, 1.0)
    result = run_ohlcv(
        df, pos, settings.capital.initial_usd,
        settings.execution.commission_bps * cost_mult,
        settings.execution.slippage_bps * cost_mult,
        market_type="spot", leverage=1.0, funding_rates=None,
    )
    return dict(result.metrics)


def _load_cache(bars: int, max_symbols: int) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    data: dict[str, pd.DataFrame] = {}
    errors: list[dict] = []
    for symbol in SYMBOLS[:max_symbols]:
        path = CACHE / f"{symbol.replace('/', '_')}_1h.parquet"
        print(f"CACHE {symbol} 1h ...", flush=True)
        try:
            df = pd.read_parquet(path).sort_index()
            if len(df) < min(bars, 5000):
                raise ValueError(f"only {len(df)} bars")
            data[symbol] = df.iloc[-min(len(df), bars):].copy()
            print(f"OK {symbol}: {len(data[symbol])} bars", flush=True)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})
            print(f"ERROR {symbol}: {exc}", flush=True)
    return data, errors


def _align(data: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    common_end = min(df.index[-1] for df in data.values())
    common_start = max(df.index[0] for df in data.values())
    out = {s: df.loc[(df.index >= common_start) & (df.index <= common_end)].copy() for s, df in data.items()}
    return out, {"common_start": str(common_start), "common_end": str(common_end), "bars": {s: len(df) for s, df in out.items()}}


def _integrity(symbol: str, df: pd.DataFrame) -> Audit:
    req = ["open", "high", "low", "close", "volume"]
    x = df[req].to_numpy(float)
    finite = bool(np.isfinite(x).all())
    structural = bool(df.index.is_monotonic_increasing and df.index.is_unique)
    ohlc = bool((
        (df.high >= df[["open", "close"]].max(axis=1))
        & (df.low <= df[["open", "close"]].min(axis=1))
        & (df.high >= df.low)
        & (df.volume >= 0)
    ).all())
    gaps = df.index.to_series().diff().dropna()
    bad = int((gaps != pd.Timedelta(hours=1)).sum()) if len(gaps) else 0
    return Audit(f"integrity:{symbol}", structural and finite and ohlc, {"bars": len(df), "finite": finite, "structural": structural, "valid_ohlc": ohlc, "bad_1h_intervals": bad, "max_gap": str(gaps.max()) if len(gaps) else "0s", "sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()})


def _lookahead(df: pd.DataFrame, settings) -> Audit:
    future = df.close.pct_change().shift(-1).fillna(0.0)
    metrics = _metrics(df, (future > 0).astype(float), settings)
    ok = bool(metrics["total_return"] > 0.20 or metrics["profit_factor"] > 1.5)
    return Audit("lookahead_canary", ok, metrics)


def _noise(data: dict[str, pd.DataFrame], settings, seed: int, trials: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(data)
    accepted = 0
    returns: list[float] = []
    for i in range(trials):
        df = data[keys[i % len(keys)]]
        signal = pd.Series((rng.random(len(df)) > 0.70).astype(float), index=df.index)
        m = _metrics(df, signal, settings, 2.0)
        accepted += int(m["total_return"] > 0 and m["profit_factor"] > 1.0 and m["max_drawdown"] > -0.20)
        returns.append(float(m["total_return"]))
    return {"trials": trials, "accepted": accepted, "false_positive_rate": accepted / trials, "median_return": float(np.median(returns)), "max_return": float(np.max(returns))}


def _synthetic(settings, seed: int, n: int = 12000, sigma: float = 0.0015, alpha: float = 0.0006, hold: int = 12) -> dict:
    # Known causal edge with low turnover: sign(r[t]) predicts r[t+1]. Signal is held for a block.
    rng = np.random.default_rng(seed)
    r = np.zeros(n, dtype=float)
    noise = rng.normal(0.0, sigma, n)
    for t in range(n - 1):
        r[t + 1] = float(np.clip(noise[t + 1] + alpha * np.sign(r[t]), -0.02, 0.02))
    close = 100.0 * np.cumprod(1.0 + r)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    prev_close = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": prev_close, "high": np.maximum(prev_close, close), "low": np.minimum(prev_close, close), "close": close, "volume": 1_000_000.0}, index=idx)
    raw = pd.Series(np.where(r > 0, 1.0, 0.0), index=idx)
    block = raw.groupby(np.arange(n) // hold).transform("first")
    normal = _metrics(df, block, settings, 1.0)
    stress = _metrics(df, block, settings, 2.0)
    detected = bool(normal["total_return"] > 0 and stress["total_return"] > 0 and normal["profit_factor"] > 1.0)
    return {"normal": normal, "stress": stress, "sigma": sigma, "alpha": alpha, "hold_bars": hold, "detected": detected, "alignment": "signal(t)=sign(r[t]) held for blocks and applied to return(t+1) onward"}


def run(minutes: float = 30.0, bars: int = 50000, max_symbols: int = 12, seed: int = 20260829) -> dict:
    start = datetime.now(timezone.utc)
    settings = load_settings()
    print("=== PHASE 0 FOUNDATION V6 ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Short: DISABLED | Leverage: DISABLED", flush=True)
    print(f"Cache: {CACHE}", flush=True)
    _save({"started_at": start.isoformat(), "decision": "LOADING", "bars": bars, "max_symbols": max_symbols})
    raw, errors = _load_cache(bars, max_symbols)
    required = min(max_symbols, 8)
    if len(raw) < required:
        payload = {"decision": "PHASE0_DATA_INSUFFICIENT", "markets_loaded": list(raw), "errors": errors}
        _save(payload)
        return payload
    data, alignment = _align(raw)
    audits = [_integrity(s, df) for s, df in data.items()]
    print("=== DATA INTEGRITY ===", flush=True)
    for a in audits:
        print(f"{a.name}: {'PASS' if a.passed else 'FAIL'} | gaps={a.details['bad_1h_intervals']}", flush=True)
    probe_df = next(iter(data.values())).iloc[:120]
    probe = _metrics(probe_df, pd.Series(np.where(np.arange(len(probe_df)) % 3 == 0, -1.0, 1.0), index=probe_df.index), settings)
    spot_ok = probe["short_exposure"] == 0.0
    print(f"SPOT POSITION PROBE: {'PASS' if spot_ok else 'FAIL'} | long={probe['long_exposure']:.2%} short={probe['short_exposure']:.2%}", flush=True)
    canary = _lookahead(probe_df, settings)
    print(f"LOOKAHEAD CANARY: {'PASS' if canary.passed else 'FAIL'} | return={canary.details['total_return']:.2%} PF={canary.details['profit_factor']:.2f}", flush=True)
    print("=== NOISE FLOOR ===", flush=True)
    noise = _noise(data, settings, seed)
    noise_ok = noise["false_positive_rate"] <= 0.05
    print(f"Noise false-positive rate: {noise['false_positive_rate']:.2%} ({noise['accepted']}/{noise['trials']})", flush=True)
    print("=== SYNTHETIC EDGE ===", flush=True)
    synthetic = _synthetic(settings, seed + 1)
    print(f"Synthetic normal return={synthetic['normal']['total_return']:.2%} PF={synthetic['normal']['profit_factor']:.2f} trades={synthetic['normal']['trade_count']}", flush=True)
    print(f"Synthetic stress return={synthetic['stress']['total_return']:.2%} PF={synthetic['stress']['profit_factor']:.2f} trades={synthetic['stress']['trade_count']}", flush=True)
    print(f"Synthetic detection: {'PASS' if synthetic['detected'] else 'FAIL'}", flush=True)
    gates = {"data": len(data) >= required and min(map(len, data.values())) >= min(bars, 5000), "integrity": all(a.passed for a in audits), "spot_long_flat": spot_ok, "lookahead_canary": canary.passed, "noise_fp_le_5pct": noise_ok, "synthetic_edge_detected": synthetic["detected"]}
    decision = "PHASE0_READY_FOR_DISCOVERY" if all(gates.values()) else "PHASE0_BLOCKED"
    payload = {"started_at": start.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(), "decision": decision, "alignment": alignment, "markets_loaded": list(data), "bars": {s: len(df) for s,df in data.items()}, "errors": errors, "integrity": [asdict(a) for a in audits], "lookahead_canary": asdict(canary), "noise_floor": noise, "synthetic_edge": synthetic, "gates": gates}
    _save(payload)
    print("=== PHASE 0 DECISION ===", flush=True)
    print(decision, flush=True)
    print(f"Saved: {OUT}", flush=True)
    return payload

if __name__ == "__main__":
    run(minutes=float(os.getenv("PHASE0_MINUTES", "30")), bars=int(os.getenv("PHASE0_BARS", "50000")), max_symbols=int(os.getenv("PHASE0_MAX_SYMBOLS", "12")), seed=int(os.getenv("PHASE0_SEED", "20260829")))
