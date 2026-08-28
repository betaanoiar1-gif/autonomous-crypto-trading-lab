from __future__ import annotations

import os


def build_agent():
    provider = os.getenv("AGENT_PROVIDER", "local").lower()

    if provider == "kimi":
        from .kimi_agent import KimiAgent
        return KimiAgent()

    if provider == "local":
        from .local_agent import LocalAgent
        return LocalAgent()

    if provider == "auto":
        if os.getenv("MOONSHOT_API_KEY"):
            try:
                from .kimi_agent import KimiAgent
                return KimiAgent()
            except Exception as exc:
                print(f"Kimi unavailable; falling back to local agent: {type(exc).__name__}")
        from .local_agent import LocalAgent
        return LocalAgent()

    raise ValueError("AGENT_PROVIDER must be one of: local, kimi, auto")
