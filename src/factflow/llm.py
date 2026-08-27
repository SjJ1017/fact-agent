"""Thin Anthropic wrapper: structured output, disk cache, bounded concurrency.

Every call in this package goes through `LLM.parse`, which returns a validated
Pydantic object.  Structured output is not a nicety here - extraction and
pairwise adjudication both need machine-readable results at volume, and free-text
parsing failures would silently corrupt the fact store.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, TypeVar

import anthropic
from pydantic import BaseModel

from .cache import DiskCache

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5"


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    effort: Optional[str] = None  # None -> server default (high); "low"|"medium"|"high"|"xhigh"|"max"
    max_concurrency: int = 8
    cache_dir: str = ".factflow_cache"
    cache_enabled: bool = True


class LLM:
    def __init__(self, config: LLMConfig | None = None, client: anthropic.Anthropic | None = None):
        self.config = config or LLMConfig()
        # Zero-arg client resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
        # `ant auth login` profile - do not hardcode a key.
        self.client = client or anthropic.Anthropic()
        self.cache = DiskCache(self.config.cache_dir, self.config.cache_enabled)
        self._effort_supported = self.config.effort is not None

    def parse(
        self,
        *,
        system: str,
        user: str,
        output_format: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        model = model or self.config.model
        max_tokens = max_tokens or self.config.max_tokens

        key = self.cache.key(
            kind="parse",
            model=model,
            system=system,
            user=user,
            schema=output_format.model_json_schema(),
            effort=self.config.effort,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return output_format.model_validate(cached)

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
        if self._effort_supported:
            kwargs["output_config"] = {"effort": self.config.effort}

        try:
            response = self.client.messages.parse(**kwargs)
        except (TypeError, anthropic.BadRequestError):
            # Older SDK builds reject `output_config` alongside `output_format`.
            # Drop it once and stop trying for the rest of the process.
            if not self._effort_supported:
                raise
            self._effort_supported = False
            kwargs.pop("output_config", None)
            response = self.client.messages.parse(**kwargs)

        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError(f"Model refused the request: {details}")

        parsed = response.parsed_output
        self.cache.put(key, parsed.model_dump(mode="json"))
        return parsed

    def map(self, fn: Callable[..., T], items: Sequence, desc: str = "") -> list[T]:
        """Run `fn` over `items` with bounded concurrency, preserving order."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
            return list(pool.map(fn, items))
