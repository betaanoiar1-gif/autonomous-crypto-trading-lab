from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SMALL_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LARGER_MODEL = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = """
You are an autonomous quantitative crypto research scientist.
Generate falsifiable hypotheses and useful reasoning, but never claim guaranteed profitability.
Use information to form hypotheses; empirical validation must come from the research lab.
Do not invent data, results or tests. Return requested structured output exactly when a schema is provided.
""".strip()


def choose_default_model() -> str:
    explicit = os.getenv("LOCAL_MODEL")
    if explicit:
        return explicit
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            if props.total_memory >= 8 * 1024**3:
                return DEFAULT_LARGER_MODEL
    except Exception:
        pass
    return DEFAULT_SMALL_MODEL


@dataclass
class LocalAgent:
    model_name: str | None = None
    max_new_tokens: int = 768
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.model_name = self.model_name or choose_default_model()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=dtype,
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load local model {self.model_name}: {exc}") from exc

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
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
