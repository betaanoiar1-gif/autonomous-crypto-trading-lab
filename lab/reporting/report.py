from pathlib import Path
import json


def write_report(path: str | Path, experiment: dict, results: dict, validation: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "results": results,
        "validation": validation,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
