from __future__ import annotations

import os
import requests


class AgentGateway:
    """Small provider-agnostic HTTP bridge for an external research agent.

    The agent remains outside the lab. The lab sends structured context and expects
    a JSON response. No provider-specific SDK is required.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None, timeout: int = 300):
        self.endpoint = endpoint or os.getenv("AGENT_ENDPOINT")
        self.api_key = api_key or os.getenv("AGENT_API_KEY")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.endpoint)

    def run(self, task: dict) -> dict:
        if not self.endpoint:
            raise RuntimeError("AGENT_ENDPOINT is not configured")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(self.endpoint, json=task, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Agent response must be a JSON object")
        return payload
