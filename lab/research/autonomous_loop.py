from __future__ import annotations

"""Long-running autonomous research loop.

Runs the fast research engine repeatedly in one process so the user can start
one command instead of pasting and launching each experiment manually.

The loop is deliberately conservative:
- research/paper-only; no orders are created
- every candidate keeps the existing validation gates
- failed hypotheses are remembered by the existing memory.jsonl mechanism
- a cycle stops immediately when a genuinely VALIDATED candidate appears
- all cycle outcomes are checkpointed to a single JSON ledger
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from ..config import ROOT, load_settings
from .fast_run import run as run_fast


def _validated_records(manifest: dict) -> list[dict]:
    return [
        r for r in manifest.get("records", [])
        if r.get("status") == "VALIDATED"
    ]


def run(cycles: int | None = None, pause_seconds: float = 1.0) -> dict:
    settings = load_settings()
    max_cycles = int(cycles if cycles is not None else settings.research.max_autonomous_cycles)
    max_cycles = max(1, max_cycles)
    pause_seconds = max(0.0, float(pause_seconds))

    ledger_path = ROOT / "experiments" / "autonomous_loop_latest.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    ledger = {
        "started_at": started.isoformat(),
        "max_cycles": max_cycles,
        "pause_seconds": pause_seconds,
        "live_trading": False,
        "cycles": [],
        "validated": [],
        "decision": "NO_VALIDATED_STRATEGY",
    }

    print("=== AUTONOMOUS RESEARCH LOOP ===", flush=True)
    print(f"Cycles allowed: {max_cycles}", flush=True)
    print("Mode: research/paper-only | live trading: disabled", flush=True)
    print("Stop condition: first candidate that passes primary + independent validation", flush=True)
    print("The loop owns iteration; no manual copy/paste between experiments.", flush=True)
    print()

    for cycle in range(1, max_cycles + 1):
        cycle_started = datetime.now(timezone.utc)
        print(f"=== CYCLE {cycle}/{max_cycles} ===", flush=True)
        try:
            manifest = run_fast(max_hypotheses=min(12, int(settings.research.max_experiments_per_run)))
            validated = _validated_records(manifest)
            summary = {
                "cycle": cycle,
                "started_at": cycle_started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "run_id": manifest.get("run_id"),
                "hypothesis_count": manifest.get("hypothesis_count", 0),
                "validated_count": len(validated),
                "leaderboard": manifest.get("leaderboard", []),
            }
            ledger["cycles"].append(summary)

            if validated:
                ledger["validated"] = validated
                ledger["decision"] = "STOP_VALIDATED"
                print()
                print("=== VALIDATED STRATEGY FOUND ===", flush=True)
                for candidate in validated:
                    h = candidate.get("hypothesis", {})
                    print(
                        f"{h.get('title', 'candidate')} | "
                        f"{candidate.get('symbol')} | {candidate.get('timeframe')} | "
                        f"family={h.get('executable_family')} | "
                        f"params={h.get('executable_parameters')}",
                        flush=True,
                    )
                break

        except Exception as exc:
            error = {
                "cycle": cycle,
                "started_at": cycle_started.isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
            ledger["cycles"].append(error)
            print(f"Cycle {cycle} failed: {type(exc).__name__}: {exc}", flush=True)

        ledger_path.write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        if cycle < max_cycles and pause_seconds > 0:
            time.sleep(pause_seconds)

    ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
    ledger["cycles_completed"] = len(ledger["cycles"])
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print()
    print("=== LOOP FINISHED ===", flush=True)
    print("Decision:", ledger["decision"], flush=True)
    print("Cycles completed:", ledger["cycles_completed"], flush=True)
    print("Checkpoint:", ledger_path, flush=True)
    return ledger


if __name__ == "__main__":
    run()
