from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json


def stable_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def new_experiment(root: Path, hypothesis: str, config: dict) -> Path:
    now = datetime.now(timezone.utc)
    experiment_id = now.strftime("EXP-%Y%m%dT%H%M%SZ")
    path = root / "experiments" / experiment_id
    path.mkdir(parents=True, exist_ok=False)
    record = {
        "experiment_id": experiment_id,
        "created_at": now.isoformat(),
        "hypothesis": hypothesis,
        "config_hash": stable_hash(config),
        "status": "PROPOSED",
    }
    (path / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
