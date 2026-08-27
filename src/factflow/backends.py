"""Model backends.

Two are supported.  Anthropic has native schema-constrained output, so the
Pydantic model goes straight to the API and comes back validated.  OpenAI-
compatible endpoints (DeepSeek, vLLM, together, ...) vary: most offer only a
"return some JSON" mode with no schema enforcement, so the schema is injected
into the prompt and validation happens client-side with a bounded repair loop.

The repair loop is not optional at scale - a few percent of malformed responses
across tens of thousands of adjudication calls is a lot of silently dropped
pairs, and a dropped pair looks exactly like "these facts are unrelated".
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class Backend(Protocol):
    name: str

    def generate(self, *, system: str, user: str, output_format: type[T], model: str, max_tokens: int) -> T: ...
    def chat(self, *, system: str, user: str, model: str, max_tokens: int, temperature: float) -> str: ...


class AnthropicBackend:
    """Native structured output - the schema is enforced server-side."""

    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(self, client: Any | None = None, effort: str | None = None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.effort = effort

    def generate(self, *, system, user, output_format, model, max_tokens):
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        response = self.client.messages.parse(**kwargs)
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(f"model refused: {getattr(response, 'stop_details', None)}")
        return response.parsed_output

    def chat(self, *, system, user, model, max_tokens, temperature):
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


SCHEMA_INSTRUCTION = """\

You must reply with a single JSON object and nothing else - no prose, no \
markdown fence, no explanation before or after.

The object must validate against this JSON Schema:

{schema}
"""

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class OpenAICompatBackend:
    """Any OpenAI-compatible chat endpoint. Schema is prompted, not enforced."""

    name = "openai-compat"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "deepseek-chat",
        max_repairs: int = 2,
        client: Any | None = None,
    ):
        from openai import OpenAI

        self.default_model = default_model
        self.max_repairs = max_repairs
        self.client = client or OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    @staticmethod
    def _strip(text: str) -> str:
        return _FENCE.sub("", text or "").strip()

    def generate(self, *, system, user, output_format, model, max_tokens):
        schema = json.dumps(output_format.model_json_schema(), ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system + SCHEMA_INSTRUCTION.format(schema=schema)},
            {"role": "user", "content": user},
        ]

        last_error = ""
        for attempt in range(self.max_repairs + 1):
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = self._strip(response.choices[0].message.content)
            try:
                return output_format.model_validate_json(raw)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)[:1500]
                if attempt == self.max_repairs:
                    break
                # Show the model its own output and the validator's complaint.
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "That output failed schema validation:\n"
                            f"{last_error}\n\n"
                            "Return the corrected JSON object only."
                        ),
                    },
                ]
        raise ValueError(f"schema validation failed after {self.max_repairs + 1} attempts: {last_error}")

    def chat(self, *, system, user, model, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


def deepseek(model: str = "deepseek-chat", **kw) -> OpenAICompatBackend:
    """DeepSeek via its OpenAI-compatible endpoint. Reads DEEPSEEK_API_KEY."""
    return OpenAICompatBackend(base_url="https://api.deepseek.com", default_model=model, **kw)
