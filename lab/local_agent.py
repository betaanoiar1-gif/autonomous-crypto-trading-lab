from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen3-0.6B")

@dataclass
class LocalAgent:
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 512
    temperature: float = 0.2

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load local model {self.model_name}: {exc}"
            ) from exc

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        output = self.pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        generated = output[0]["generated_text"]
        if isinstance(generated, list):
            return str(generated[-1].get("content", ""))
        return str(generated)

    def healthcheck(self) -> dict:
        text = self.chat("Reply with exactly: LOCAL_OK")
        return {"ok": "LOCAL_OK" in text, "model": self.model_name}
