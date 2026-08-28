from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SMALL_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_LARGER_MODEL = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = """
You are an autonomous quantitative crypto research scientist.
Generate falsifiable hypotheses and useful reasoning, but never claim guaranteed profitability.
Use information to form hypotheses; empirical validation must come from the research lab.
Do not invent data, test results or sources.
When structured output is requested, follow the requested format exactly.
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
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self.model_name = self.model_name or choose_default_model()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                dtype=dtype,
            )
        except Exception as exc:
            raise RuntimeError(f"Unable to load local model {self.model_name}: {exc}") from exc

    def chat(self, prompt: str, system: str | None = None) -> str:
        messages = [
            {"role": "system", "content": system or SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )

        input_ids = input_ids.to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_id if pad_id is not None else eos_id)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=eos_id,
        )
        generated = output_ids[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def healthcheck(self) -> dict:
        """Check that model/tokenizer loaded and the runtime device is available.

        Do not use an LLM generation request as a health check: small local models
        may legally emit empty or unexpected text even when fully operational.
        """
        try:
            device = str(self.model.device)
            vocab = len(self.tokenizer)
            ok = bool(self.model and self.tokenizer and vocab > 0)
            return {"ok": ok, "model": self.model_name, "device": device}
        except Exception:
            return {"ok": False, "model": self.model_name, "device": "unknown"}
