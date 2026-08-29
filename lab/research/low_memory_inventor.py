from __future__ import annotations

"""Process-isolated strategy invention for small-RAM Colab sessions.

The LLM runs in a short-lived child process. It writes only compact strategy
specifications to disk and exits before numerical backtesting begins. The
backtest process therefore never holds the LLM model in RAM.
"""

import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config import ROOT, load_settings
from .self_inventor import _ask_agent, _fallback_specs, _normalize, _gate, _slim_record
from .evaluator import evaluate, frozen_confirmation
from .run import _fetch_markets_cached


def _generate_child(out_path: Path, count: int) -> int:
    from ..local_agent import LocalAgent

    agent = None
    try:
        agent = LocalAgent()
        health = agent.healthcheck()
        print(
            f"Inventor child: model={health.get('model')} "
            f"device={health.get('device')} healthy={health.get('ok')}",
            flush=True,
        )
        specs = _ask_agent(agent, count)
        if not specs:
            specs = _fallback_specs()
        normalized = []
        seen = set()
        for i, raw in enumerate(specs, 1):
            if not isinstance(raw, dict):
                continue
            item = _normalize(raw, i)
            if item["candidate_id"] in seen:
                continue
            seen.add(item["candidate_id"])
            normalized.append(item)
            if len(normalized) >= count:
                break
        out_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Invented specs written: {out_path}", flush=True)
        return 0
    except Exception as exc:
        print(f"Inventor child failed: {type(exc).__name__}: {exc}", flush=True)
        return 2
    finally:
        try:
            del agent
        except Exception:
            pass
        gc.collect()


def _evaluate_parent(spec_path: Path) -> int:
    settings = load_settings()
    spot, _, _, _, snapshot = _fetch_markets_cached()
    primary = spot[("ETH/USDT", "1h")]
    independent = spot[("BTC/USDT", "4h")]
    specs = json.loads(spec_path.read_text(encoding="utf-8"))

    run_id = datetime.now(timezone.utc).strftime("INVENT-LM-%Y%m%dT%H%M%SZ")
    ledger_dir = ROOT / "experiments" / run_id
    ledger_dir.mkdir(parents=True, exist_ok=False)
    rows = []

    print("=== LOW-MEMORY AUTONOMOUS STRATEGY INVENTOR ===", flush=True)
    print(f"Candidates: {len(specs)}", flush=True)
    print("LLM process: FINISHED BEFORE BACKTESTS", flush=True)
    print("Arbitrary code execution: DISABLED", flush=True)
    print("Futures: DISABLED | Live trading: DISABLED", flush=True)

    for i, spec in enumerate(specs, 1):
        try:
            ev = evaluate(
                primary,
                "invented_composite",
                spec["parameters"],
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                settings.validation.holdout_ratio,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )
            provisional = {"holdout": ev.out_of_sample, "robustness": ev.robustness}
            primary_pass, reasons = _gate(provisional)

            confirmation = frozen_confirmation(
                independent,
                "invented_composite",
                spec["parameters"],
                ["both"],
                settings.capital.initial_usd,
                settings.execution.commission_bps,
                settings.execution.slippage_bps,
                market_type="spot",
                leverage=1.0,
                funding_rates=None,
            )

            row = _slim_record(spec, ev, primary_pass, reasons, confirmation)
            rows.append(row)
            (ledger_dir / f"candidate_{i:02d}.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(
                f"[{i:02d}] {spec['title']} | "
                f"OOS={row['holdout']['total_return']:.2%} | "
                f"PF={row['holdout']['profit_factor']:.2f} | "
                f"DD={row['holdout']['max_drawdown']:.2%} | "
                f"WF={row['robustness']['walk_forward']['passed']} | "
                f"CONF={row['independent_confirmation']['passed']} | "
                f"{row['status']}",
                flush=True,
            )
            del ev, provisional, confirmation, row
        except Exception as exc:
            err = {
                "candidate_id": spec.get("candidate_id"),
                "title": spec.get("title"),
                "parameters": spec.get("parameters", {}),
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(err)
            print(f"[{i:02d}] ERROR={err['error']}", flush=True)
        finally:
            gc.collect()

    rows.sort(
        key=lambda r: float(r.get("holdout", {}).get("research_score", -1e9)),
        reverse=True,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "primary": {"symbol": "ETH/USDT", "timeframe": "1h", "market_type": "spot"},
        "independent": {"symbol": "BTC/USDT", "timeframe": "4h", "market_type": "spot"},
        "generator": "local_llm_safe_dsl_process_isolated",
        "memory_mode": "process_isolated_low_memory",
        "arbitrary_code_execution": False,
        "candidate_count": len(rows),
        "validated": [r for r in rows if r.get("status") == "VALIDATED"],
        "leaderboard": rows,
        "market_snapshot": snapshot,
    }
    latest = ROOT / "experiments" / "low_memory_inventor_latest.json"
    latest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (ledger_dir / "result.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("=== FINAL DECISION ===", flush=True)
    if manifest["validated"]:
        print("Decision: STOP_VALIDATED", flush=True)
        for item in manifest["validated"]:
            print(item["title"], item["parameters"], flush=True)
    else:
        print("Decision: NO_VALIDATED_INVENTION", flush=True)
    print("Saved:", latest, flush=True)
    return 0


def run(count: int = 4) -> int:
    count = max(2, min(6, int(count)))
    tmp_dir = ROOT / "experiments" / "_inventor_staging"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_dir / "generated_specs.json"

    if spec_path.exists():
        spec_path.unlink()

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(ROOT)
    child_env["TOKENIZERS_PARALLELISM"] = "false"
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"

    print("=== PROCESS-ISOLATED INVENTION ===", flush=True)
    print(f"Generation batch: {count}", flush=True)

    result = subprocess.run(
        [sys.executable, "-m", "lab.research.low_memory_inventor", "--generate", str(spec_path), str(count)],
        cwd=str(ROOT),
        env=child_env,
        check=False,
    )
    if result.returncode != 0 or not spec_path.exists():
        print("Generation process failed; stopping safely.", flush=True)
        return result.returncode or 2

    return _evaluate_parent(spec_path)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--generate":
        path = Path(sys.argv[2])
        n = int(sys.argv[3])
        raise SystemExit(_generate_child(path, n))
    raise SystemExit(run())
