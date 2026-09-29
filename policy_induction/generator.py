"""Rule-generation LLM: text prompt in, list of rule strings out."""

from __future__ import annotations

import logging
import os
from typing import List, Protocol

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Per-request timeout. Generation prompts are large and reasoning models can be
# slow, but a call that exceeds this is treated as failed and retried.
REQUEST_TIMEOUT_S = 180


class Rules(BaseModel):
    rules: List[str] = Field(..., description="The proposed rules.")

    @field_validator("rules", mode="before")
    @classmethod
    def _unwrap_objects(cls, value):
        """Accept [{"rule": "..."}] as well as ["..."].

        JSON-mode models (DeepSeek) sometimes wrap each rule in an object,
        especially after seeing rules listed with metadata in the prompt.
        """
        if not isinstance(value, list):
            return value
        out = []
        for item in value:
            if isinstance(item, dict):
                texts = [v for v in item.values() if isinstance(v, str) and v.strip()]
                item = item.get("rule") or item.get("text") or (texts[0] if texts else item)
            out.append(item)
        return out


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
        # Reasoning models reject `temperature`; learned from the first 400.
        self._supports_temperature = True

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        from openai import BadRequestError

        kwargs = dict(model=self.model, instructions=system, input=prompt, text_format=Rules)
        try:
            if self._supports_temperature:
                response = await self._client.responses.parse(**kwargs, temperature=temperature)
            else:
                response = await self._client.responses.parse(**kwargs)
        except BadRequestError as e:
            if not self._supports_temperature or "temperature" not in str(e).lower():
                raise
            logger.info("%s does not accept temperature; using its default.", self.model)
            self._supports_temperature = False
            response = await self._client.responses.parse(**kwargs)
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


class DeepSeekGenerator:
    """DeepSeek via its OpenAI-compatible chat API.

    DeepSeek has no schema-constrained parsing, only a JSON mode, so the reply
    is validated against ``Rules`` here. Models that reject JSON mode (e.g.
    older ``deepseek-reasoner``) fall back to extracting JSON from plain text.
    """

    BASE_URL = "https://api.deepseek.com"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY is not set (add it to .env).")
        self.model = model
        self._client = AsyncOpenAI(api_key=key, base_url=self.BASE_URL, timeout=REQUEST_TIMEOUT_S)
        self._json_mode = True

    async def generate(self, system: str, prompt: str, temperature: float) -> List[str]:
        from openai import BadRequestError

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        try:
            if self._json_mode:
                response = await self._client.chat.completions.create(
                    **kwargs, response_format={"type": "json_object"}
                )
            else:
                response = await self._client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if not self._json_mode or "response_format" not in str(e).lower():
                raise
            logger.info("%s does not support JSON mode; parsing plain text.", self.model)
            self._json_mode = False
            response = await self._client.chat.completions.create(**kwargs)
        return parse_rules(response.choices[0].message.content or "", self.model)


def parse_rules(text: str, model: str) -> List[str]:
    """Rules from a reply that should be {"rules": [...]}, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"{model} returned no JSON object: {text[:200]!r}")
    return Rules.model_validate_json(text[start : end + 1]).rules


def make_generator(model: str) -> RuleGenerator:
    """Pick a backend from the model name."""
    name = model.lower()
    if name.startswith("deepseek"):
        return DeepSeekGenerator(model)
    if name.startswith("gemini"):
        return GoogleGenerator(model)
    if name.startswith(("gpt", "o1", "o3", "o4", "o5")):
        return OpenAIGenerator(model)
    raise ValueError(
        f"Unknown generation model {model!r}. Use a deepseek-*, gemini-* or gpt-* "
        "model, or pass a RuleGenerator instance."
    )
