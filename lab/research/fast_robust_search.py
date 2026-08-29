from __future__ import annotations

"""Fast AI-free robust strategy search.

Stage 1: cheap cross-market screening on bounded recent samples.
Stage 2: full-data stress and fresh confirmation for a small diverse finalist set.
No LLM generation, no futures, no live trading.
"""

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import ROOT, load_settings
from ..data.ccxt_adapter import CCXTMarketData
from .ai_free_autosearch import _make_candidates
from .evaluator import _metrics, _run

SCREEN = (("ETH/USDT", "1h", 1200), ("ETH/USDT", "4h", 900), ("BTC/USDT", "4h", 900))
FRESH = (("BTC/USDT", "1h", 1200), ("ETH/USDT", "15m", 1200))


def metrics(df, c, settings, fee_mult=1.0):
    r = _run(df, c.family, dict(c.params), ["both"], settings.capital.initial_usd,
             settings.execution.commission_bps * fee_mult,
             settings.execution.slippage_bps * fee_mult)
    return _metrics(r, r.returns)


def score(m):
    ret = float(m.get("total_return", 0.0))
    pf = min(2.5, float(m.get("profit_factor", 0.0)))
    dd = abs(min(0.0, float(m.get("max_drawdown", 0.0))))
    trades = int(m.get("trade_count", 0))
    sharpe = float(m.get("sharpe", 0.0))
    return 100 * ret + 18 * (pf - 1) + 4 * sharpe - 30 * dd + min(7, math.log1p(max(0, trades)))


def passes(m, min_trades=8):
    return bool(float(m["total_return"]) > 0 and float(m["profit_factor"]) > 1
                and float(m["max_drawdown"]) >= -0.50 and int(m["trade_count"]) >= min_trades)


def run(candidate_limit=240, finalists=12):
    settings = load_settings()
    adapter = CCXTMarketData(exchange_id="binance")
    started = datetime.now(timezone.utc)
    print("=== FAST AI-FREE ROBUST SEARCH ===", flush=True)
    print("AI: DISABLED | Futures: DISABLED | Live: DISABLED", flush=True)
    print("Stage 1: cheap 3-market screen", flush=True)
    print("Stage 2: full stress + fresh confirmation", flush=True)

    data = {}
    for symbol, tf, bars in SCREEN + FRESH:
        print(f"LOADING {symbol} {tf} ...", flush=True)
        data[(symbol, tf)] = adapter.fetch_ohlcv_history(symbol, tf, bars, page_limit=1000, market_type="spot")
        print(f"LOADED {symbol} {tf}: {len(data[(symbol, tf)])} bars", flush=True)

    candidates = _make_candidates(20260829, candidate_limit)
    print(f"GENERATED {len(candidates)} UNIQUE CANDIDATES", flush=True)
    screened = []

    for i, c in enumerate(candidates, 1):
        try:
            mm = []
            ok = 0
            for symbol, tf, sample_bars in SCREEN:
                df = data[(symbol, tf)].iloc[-sample_bars:]
                m = metrics(df, c, settings, 1.0)
                mm.append(m)
                ok += int(passes(m))
            rets = [float(m["total_return"]) for m in mm]
            value = float(np.mean([score(m) for m in mm]) + 18 * ok - 18 * (max(rets) - min(rets)))
            screened.append({"candidate": c, "score": value, "ok": ok, "screen": mm})
        except Exception as exc:
            print(f"CANDIDATE {i} ERROR: {type(exc).__name__}: {exc}", flush=True)
        if i == 1 or i % 5 == 0:
            best = max(screened, key=lambda x: x["score"]) if screened else None
            print(f"PROGRESS {i}/{len(candidates)} | best_score={best['score']:.2f} | best_ok={best['ok']}/3" if best else f"PROGRESS {i}", flush=True)
        gc.collect()

    screened.sort(key=lambda x: (x["ok"], x["score"]), reverse=True)
    selected, seen, family_counts = [], set(), {}
    for item in screened:
        c = item["candidate"]
        sig = (c.family, tuple(sorted(c.params.items())))
        if sig in seen or family_counts.get(c.family, 0) >= 4:
            continue
        seen.add(sig); selected.append(item); family_counts[c.family] = family_counts.get(c.family, 0) + 1
        if len(selected) >= finalists:
            break

    print("\n=== FINALIST STAGE ===", flush=True)
    final = []
    for i, item in enumerate(selected, 1):
        c = item["candidate"]
        print(f"FINALIST {i}/{len(selected)}: {c.title}", flush=True)
        full = []
        for symbol, tf, _ in SCREEN:
            df = data[(symbol, tf)]
            normal = metrics(df, c, settings, 1.0)
            stress = metrics(df, c, settings, 2.0)
            full.append({"market": f"{symbol} {tf}", "normal": normal, "stress": stress,
                         "pass": passes(normal) and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1})
        fresh = []
        for symbol, tf, _ in FRESH:
            df = data[(symbol, tf)]
            normal = metrics(df, c, settings, 1.0)
            stress = metrics(df, c, settings, 2.0)
            fresh.append({"market": f"{symbol} {tf}", "normal": normal, "stress": stress,
                          "pass": passes(normal) and float(stress["total_return"]) > 0 and float(stress["profit_factor"]) > 1})
        fp = sum(x["pass"] for x in full); fr = sum(x["pass"] for x in fresh)
        validated = fp == 3 and fr == 2
        rec = {"title": c.title, "family": c.family, "parameters": dict(c.params),
               "screen_score": item["score"], "full": full, "fresh": fresh,
               "full_pass": fp, "fresh_pass": fr, "validated": validated}
        final.append(rec)
        print(f"  FULL={fp}/3 | FRESH={fr}/2 | VALIDATED={validated}", flush=True)
        if validated:
            break
        gc.collect()

    decision = "VALIDATED_ALGORITHMIC_STRATEGY" if any(x["validated"] for x in final) else "NO_VALIDATED_ALGORITHMIC_STRATEGY"
    payload = {"started_at": started.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
               "decision": decision, "generated": len(candidates), "screened": len(screened),
               "finalists": len(selected), "validated_count": sum(x["validated"] for x in final), "results": final}
    out = ROOT / "experiments" / "fast_robust_search_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(out)
    print("\n=== SEARCH FINISHED ===", flush=True)
    print("Decision:", decision, flush=True)
    print("Validated:", payload["validated_count"], flush=True)
    print("Saved:", out, flush=True)
    return payload


if __name__ == "__main__":
    run()
