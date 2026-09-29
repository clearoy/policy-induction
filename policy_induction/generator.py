"""Rule-generation LLM: text prompt in, list of rule strings out."""

from __future__ import annotations

import logging
import os
from typing import List, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Per-request timeout. Generation prompts are large and reasoning models can be
# slow, but a call that exceeds this is treated as failed and retried.
REQUEST_TIMEOUT_S = 180


class Rules(BaseModel):
    rules: List[str] = Field(..., description="The proposed rules.")


class RuleGenerator(Protocol):
    """Anything that turns a (system, prompt) pair into a list of rules."""

    model: str

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        ...


class OpenAIGenerator:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT_S
        )

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        response = await self._client.responses.parse(
            model=self.model,
            instructions=system,
            input=prompt,
            text_format=Rules,
            temperature=temperature,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"{self.model} returned no parseable rules")
        return parsed.rules


class GoogleGenerator:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        self._client = genai.Client(
            api_key=api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY"),
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),  # ms
        )

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        from google.genai import types

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=Rules,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, Rules):
            return parsed.rules
        if response.text:
            return Rules.model_validate_json(response.text).rules
        raise RuntimeError(f"{self.model} returned no parseable rules")


def make_generator(model: str) -> RuleGenerator:
    """Pick a backend from the model name."""
    name = model.lower()
    if name.startswith("gemini"):
        return GoogleGenerator(model)
    if name.startswith(("gpt", "o1", "o3", "o4", "o5")):
        return OpenAIGenerator(model)
    raise ValueError(
        f"Unknown generation model {model!r}. Use a gemini-* or gpt-* model, "
        "or pass a RuleGenerator instance."
    )
