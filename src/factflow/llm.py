"""Backend-agnostic LLM wrapper: structured output, disk cache, bounded concurrency.

Every call in this package goes through `LLM.parse`, which returns a validated
Pydantic object.  Structured output is not a nicety here - extraction and
pairwise adjudication both need machine-readable results at volume, and parsing
failures would silently corrupt the fact store.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, TypeVar

from pydantic import BaseModel

from .backends import (
    AnthropicBackend,
    Backend,
    deepseek,
    openai as openai_backend,
    opencode as opencode_backend,
)
from .cache import DiskCache

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMConfig:
    model: Optional[str] = None  # None -> the backend's default
    max_tokens: int = 8000
    max_concurrency: int = 8
    cache_dir: str = ".factflow_cache"
    cache_enabled: bool = True


class LLM:
    def __init__(self, config: LLMConfig | None = None, backend: Backend | None = None):
        self.config = config or LLMConfig()
        self.backend = backend or AnthropicBackend()
        self.model = self.config.model or getattr(self.backend, "default_model", None)
        if not self.model:
            raise ValueError("no model: set LLMConfig.model or use a backend with a default")
        self.cache = DiskCache(self.config.cache_dir, self.config.cache_enabled)
        self.failures: list[str] = []

    @classmethod
    def deepseek(cls, model: str = "deepseek-chat", **cfg) -> "LLM":
        return cls(LLMConfig(model=model, **cfg), backend=deepseek(model))

    @classmethod
    def openai(cls, model: str = "gpt-5.4-mini", **cfg) -> "LLM":
        return cls(LLMConfig(model=model, **cfg), backend=openai_backend(model))

    @classmethod
    def opencode(cls, model: str = "kimi-k2.6", tier: str = "go", **cfg) -> "LLM":
        return cls(LLMConfig(model=model, **cfg), backend=opencode_backend(model, tier=tier))

    @property
    def usage(self):
        return self.backend.usage

    def parse(
        self,
        *,
        system: str,
        user: str,
        output_format: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        model = model or self.model
        max_tokens = max_tokens or self.config.max_tokens

        key = self.cache.key(
            kind="parse",
            backend=self.backend.name,
            model=model,
            system=system,
            user=user,
            schema=output_format.model_json_schema(),
        )
        cached = self.cache.get(key)
        if cached is not None:
            return output_format.model_validate(cached)

        parsed = self.backend.generate(
            system=system, user=user, output_format=output_format, model=model, max_tokens=max_tokens
        )
        self.cache.put(key, parsed.model_dump(mode="json"))
        return parsed

    def chat(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Free-text generation, used to *produce* traces rather than analyse them."""
        model = model or self.model
        max_tokens = max_tokens or self.config.max_tokens
        key = self.cache.key(
            kind="chat",
            backend=self.backend.name,
            model=model,
            system=system,
            user=user,
            temperature=temperature,
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        text = self.backend.chat(
            system=system, user=user, model=model, max_tokens=max_tokens, temperature=temperature
        )
        self.cache.put(key, text)
        return text

    def map(self, fn: Callable[..., T], items: Sequence, tolerate_failures: bool = True) -> list[T]:
        """Run `fn` over `items` with bounded concurrency, preserving order.

        With `tolerate_failures`, a call that fails after its repair attempts is
        skipped and recorded rather than killing a long run; check `.failures`.
        """
        if not items:
            return []

        def _guarded(item):
            try:
                return fn(item)
            except Exception as exc:  # noqa: BLE001
                if not tolerate_failures:
                    raise
                self.failures.append(f"{type(exc).__name__}: {exc}")
                return None

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as pool:
            results = list(pool.map(_guarded, items))
        return [r for r in results if r is not None]
