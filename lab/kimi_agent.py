import os
from openai import OpenAI

DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k2.6"

SYSTEM_PROMPT = """
You are the autonomous research agent for the Autonomous Crypto Trading Lab.
You are a quantitative research agent, not a strategy oracle.
Discover hypotheses independently, use external information only as hypothesis input,
and require reproducible empirical evidence before accepting claims.
Never expose or log secrets. Never claim guaranteed profitability.
""".strip()

class KimiAgent:
    def __init__(self, api_key: str | None = None, model: str | None = None, base_url: str | None = None):
        key = api_key or os.getenv("MOONSHOT_API_KEY")
        if not key:
            raise ValueError("MOONSHOT_API_KEY is not configured.")
        self.model = model or os.getenv("KIMI_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("KIMI_BASE_URL", DEFAULT_BASE_URL),
        )

    def chat(self, prompt: str, system: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def healthcheck(self) -> dict:
        text = self.chat("Reply with exactly: KIMI_OK")
        return {"ok": text.strip() == "KIMI_OK", "model": self.model}
