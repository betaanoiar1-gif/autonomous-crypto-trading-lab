from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = os.getenv("LAB_REPO_URL", "https://github.com/betaanoiar1-gif/autonomous-crypto-trading-lab.git")
REPO_DIR = Path("/content/autonomous_crypto_trading_lab")


def main() -> None:
    if REPO_DIR.exists():
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
    else:
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO_DIR / "requirements.txt")], check=True)
    subprocess.run([sys.executable, "-m", "lab.cli", "doctor"], cwd=REPO_DIR, check=True)

if __name__ == "__main__":
    main()
