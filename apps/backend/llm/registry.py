"""
Registry: which providers are configured, which models can we offer right now?

This is the single source of truth for both the `/models` HTTP endpoint
(Phase 4) and the eval matrix runner (Phase 7). Both need the same
question answered: "given the current env, what can I actually call?"

Construction is cheap (no network). The expensive ping happens lazily on
`describe()`, and we cache it with a short TTL so repeated UI polls don't
hammer the providers.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from apps.backend.core.settings import settings
from apps.backend.llm.anthropic import AnthropicAdapter
from apps.backend.llm.base import AdapterError, LLMAdapter, ModelInfo
from apps.backend.llm.ollama import OllamaAdapter
from apps.backend.llm.openai import OpenAIAdapter


@dataclass(slots=True)
class ProviderStatus:
    provider: str
    available: bool
    default_model: str
    models: list[ModelInfo] = field(default_factory=list)
    error: str | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "available": self.available,
            "default_model": self.default_model,
            "models": [{"id": m.id, "label": m.label or m.id} for m in self.models],
            "error": self.error,
        }


class ModelRegistry:
    """Construct once; call describe() to get a fresh snapshot."""

    CACHE_TTL_SECONDS = 30.0

    def __init__(self) -> None:
        self._adapters: dict[str, LLMAdapter] = {}
        self._init_errors: dict[str, str] = {}
        self._cache: list[ProviderStatus] | None = None
        self._cache_at: float = 0.0
        self._try_init(OpenAIAdapter, "openai")
        self._try_init(AnthropicAdapter, "anthropic")
        # Ollama doesn't need a key; we always try to construct it (cheap
        # — just storing the base URL). Real availability is decided by
        # the health ping below.
        self._try_init(OllamaAdapter, "ollama")

    def _try_init(self, cls: type, name: str) -> None:
        try:
            self._adapters[name] = cls()
        except AdapterError as e:
            self._init_errors[name] = str(e)
        except Exception as e:  # pragma: no cover - defensive
            self._init_errors[name] = f"unexpected: {e!r}"

    def get(self, provider: str) -> LLMAdapter | None:
        return self._adapters.get(provider)

    def all_adapters(self) -> dict[str, LLMAdapter]:
        return dict(self._adapters)

    # --- snapshot --------------------------------------------------------

    async def describe(self, force_refresh: bool = False) -> list[ProviderStatus]:
        if (
            not force_refresh
            and self._cache is not None
            and time.monotonic() - self._cache_at < self.CACHE_TTL_SECONDS
        ):
            return self._cache

        statuses: list[ProviderStatus] = []

        # Providers that failed to even initialise (missing key, bad config)
        for name, err in self._init_errors.items():
            statuses.append(
                ProviderStatus(
                    provider=name,
                    available=False,
                    default_model=_default_model(name),
                    error=err,
                )
            )

        # Live providers: ping in parallel
        async def probe(name: str, adapter: LLMAdapter) -> ProviderStatus:
            ok = await adapter.health()
            models = await adapter.list_models() if ok else []
            return ProviderStatus(
                provider=name,
                available=ok,
                default_model=adapter.default_model,
                models=models,
                error=None if ok else "health check failed",
            )

        live = await asyncio.gather(
            *(probe(n, a) for n, a in self._adapters.items()),
            return_exceptions=False,
        )
        statuses.extend(live)

        # Stable ordering for the UI
        statuses.sort(key=lambda s: s.provider)
        self._cache = statuses
        self._cache_at = time.monotonic()
        return statuses


def _default_model(provider: str) -> str:
    """Surface a sensible default even when the adapter never initialised."""
    return {
        "openai": OpenAIAdapter.default_model,
        "anthropic": AnthropicAdapter.default_model,
        "ollama": OllamaAdapter.default_model,
    }.get(provider, "")
