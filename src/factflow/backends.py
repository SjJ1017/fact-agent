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
from pathlib import Path
import re
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Optional, Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


PRICES_PATH = Path(__file__).resolve().parents[2] / "experiments" / "out" / "prices.json"


@dataclass
class Usage:
    """Token accounting, so cost comparisons rest on measured numbers.

    Estimating this instead is a reliable way to be wrong by a multiple. A cost
    model built from one model's reasoning on one task was off by 3.5x when
    carried to a different model on a different task: the reasoning budget is a
    property of the pair, not of either one.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cached: int = 0

    def add(self, i: int, o: int) -> None:
        self.input_tokens += i
        self.output_tokens += o
        self.calls += 1

    def cost(self, model: str) -> Optional[float]:
        """USD, from experiments/out/prices.json. None when the model is absent -
        an unpriced model must not silently read as free."""
        try:
            prices = json.loads(PRICES_PATH.read_text())
        except Exception:
            return None
        p = prices.get(model)
        if not p:
            return None
        return self.input_tokens / 1e6 * p[0] + self.output_tokens / 1e6 * p[1]

    def report(self, model: str) -> str:
        c = self.cost(model)
        money = f"${c:.4f}" if c is not None else "cost unknown (model not in prices.json)"
        return (f"{self.calls} calls, {self.input_tokens:,} in / "
                f"{self.output_tokens:,} out, {money}")


class Backend(Protocol):
    name: str
    usage: "Usage"

    def generate(self, *, system: str, user: str, output_format: type[T], model: str, max_tokens: int) -> T: ...
    def chat(self, *, system: str, user: str, model: str, max_tokens: int,
             temperature: float,
             history: Sequence[tuple[str, str]] = ()) -> str: ...


class AnthropicBackend:
    """Native structured output - the schema is enforced server-side."""

    name = "anthropic"
    default_model = "claude-opus-5"

    def __init__(self, client: Any | None = None, effort: str | None = None):
        import anthropic

        self.client = client or anthropic.Anthropic()
        self.effort = effort
        self.usage = Usage()

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
        u = getattr(response, "usage", None)
        if u:
            self.usage.add(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        if getattr(response, "stop_reason", None) == "refusal":
            raise RuntimeError(f"model refused: {getattr(response, 'stop_details', None)}")
        return response.parsed_output

    def chat(self, *, system, user, model, max_tokens, temperature, history=()):
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[*({"role": r, "content": c} for r, c in history),
                      {"role": "user", "content": user}],
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
        timeout: float = 120.0,
        max_retries: int = 2,
    ):
        from openai import OpenAI

        self.default_model = default_model
        self.max_repairs = max_repairs
        self.usage = Usage()
        self._token_param = "max_tokens"
        self._unsupported: set[str] = set()
        # A per-request timeout is not optional on a multi-provider gateway:
        # one wedged upstream otherwise hangs a whole batch indefinitely, and
        # the concurrency pool has no way to notice.
        self.client = client or OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _strip(text: str) -> str:
        return _FENCE.sub("", text or "").strip()

    def _create(self, **kwargs):
        """Call the endpoint, adapting to per-model parameter differences.

        Endpoints that are "OpenAI-compatible" disagree about the spelling of
        the output cap (`max_tokens` vs `max_completion_tokens`), and reasoning
        models reject `temperature` outright. Both surface as a 400 naming the
        offending parameter, so the fix is read off the error and remembered for
        the rest of the process rather than hard-coded per model name.
        """
        for _ in range(4):
            call = dict(kwargs)
            for drop in self._unsupported:
                call.pop(drop, None)
            if self._token_param != "max_tokens" and "max_tokens" in call:
                call[self._token_param] = call.pop("max_tokens")
            try:
                return self.client.chat.completions.create(**call)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "max_tokens" in msg and "max_completion_tokens" in msg:
                    self._token_param = "max_completion_tokens"
                    continue
                m = re.search(r"[Uu]nsupported (?:parameter|value): '([^']+)'", msg)
                if m and m.group(1).split(".")[0] not in self._unsupported:
                    self._unsupported.add(m.group(1).split(".")[0])
                    continue
                raise
        raise RuntimeError("could not find a working parameter set for this endpoint")

    def generate(self, *, system, user, output_format, model, max_tokens):
        schema = json.dumps(output_format.model_json_schema(), ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system + SCHEMA_INSTRUCTION.format(schema=schema)},
            {"role": "user", "content": user},
        ]

        last_error = ""
        for attempt in range(self.max_repairs + 1):
            response = self._create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                response_format={"type": "json_object"},
            )
            u = getattr(response, "usage", None)
            if u:
                self.usage.add(u.prompt_tokens or 0, u.completion_tokens or 0)
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

    def chat(self, *, system, user, model, max_tokens, temperature, history=()):
        response = self._create(
            model=model,
            messages=[{"role": "system", "content": system},
                      *({"role": r, "content": c} for r, c in history),
                      {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        u = getattr(response, "usage", None)
        if u:
            self.usage.add(u.prompt_tokens or 0, u.completion_tokens or 0)
        return response.choices[0].message.content or ""


def deepseek(model: str = "deepseek-chat", **kw) -> OpenAICompatBackend:
    """DeepSeek via its OpenAI-compatible endpoint. Reads DEEPSEEK_API_KEY."""
    return OpenAICompatBackend(base_url="https://api.deepseek.com", default_model=model, **kw)


OPENCODE_ENDPOINTS = {
    "go": "https://opencode.ai/zen/go/v1",   # OpenCode Go subscription
    "zen": "https://opencode.ai/zen/v1",     # pay-as-you-go Zen
}


def opencode(model: str = "kimi-k2.6", tier: str = "go", **kw) -> OpenAICompatBackend:
    """OpenCode gateway - one OpenAI-compatible endpoint over many providers.

    Valuable here because cross-provider comparison becomes a model-string
    change rather than a client change, so a benchmark measures the model and
    not the differences between SDKs.

    `tier` picks the endpoint: "go" for an OpenCode Go subscription, "zen" for
    pay-as-you-go. The two expose different model catalogues. Reads
    OPENCODE_API_KEY.
    """
    import os as _os

    if tier not in OPENCODE_ENDPOINTS:
        raise ValueError(f"tier must be one of {sorted(OPENCODE_ENDPOINTS)}")
    return OpenAICompatBackend(
        api_key=_os.environ.get("OPENCODE_API_KEY"),
        base_url=OPENCODE_ENDPOINTS[tier],
        default_model=model,
        **kw,
    )


def openai(model: str = "gpt-5.4-mini", **kw) -> OpenAICompatBackend:
    """OpenAI. Reads OPENAI_API_KEY."""
    import os as _os

    return OpenAICompatBackend(
        api_key=_os.environ.get("OPENAI_API_KEY"), default_model=model, **kw
    )
