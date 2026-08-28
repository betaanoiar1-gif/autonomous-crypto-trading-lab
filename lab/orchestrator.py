from pathlib import Path
from datetime import datetime, timezone
import json
from .config import ROOT, load_settings


def run() -> None:
    settings = load_settings()
    run_id = datetime.now(timezone.utc).strftime("RUN-%Y%m%dT%H%M%SZ")
    exp_dir = ROOT / "experiments" / run_id
    exp_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "status": "BOOTSTRAPPED",
        "capital": settings.capital.model_dump(),
        "research": settings.research.model_dump(),
        "execution": settings.execution.model_dump(),
        "validation": settings.validation.model_dump(),
        "note": "Research adapters and autonomous-agent runtime are intentionally separated from the controller.",
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"LAB READY: {run_id}")
    print(f"Initial capital: ${settings.capital.initial_usd:,.2f}")
    print("Next layer: data + research agent + backtest + validation adapters.")


if __name__ == "__main__":
    run()
