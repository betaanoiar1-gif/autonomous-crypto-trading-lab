from datetime import datetime, timezone
import json
from pathlib import Path


def write_report(path: str | Path, experiment_id: str, hypothesis: str,
                 metrics: dict, validation: dict, provenance: dict) -> None:
    report = {
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis,
        "metrics": metrics,
        "validation": validation,
        "provenance": provenance,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
