from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from .config import ROOT, load_settings


def doctor() -> int:
    settings = load_settings()
    print("AUTONOMOUS CRYPTO TRADING LAB — SYSTEM CHECK")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Repository root: {ROOT}")
    print(f"Initial capital: ${settings.capital.initial_usd:,.2f}")
    print(f"Agent endpoint configured: {bool(os.getenv('AGENT_ENDPOINT'))}")
    print(f"GitHub token configured: {bool(os.getenv('GITHUB_TOKEN'))}")
    print("Core imports: OK")
    return 0


def config() -> int:
    print(json.dumps(load_settings().model_dump(), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="crypto-lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("config")
    args = parser.parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "config":
        return config()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
