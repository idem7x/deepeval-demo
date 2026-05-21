"""
The single contract every LLM adapter implements.

This module is intentionally provider-agnostic — it knows nothing about
OpenAI, Anthropic, or Ollama. That separation is the whole point:
- The chat endpoint (Phase 4) only knows about `LLMAdapter`.
- The eval matrix (Phase 7) treats SUT and judge identically.
- The DeepEval wrapper (Phase 3.6) wraps any adapter, so we can grade GPT
  with Claude, Claude with Ollama, etc.

Async-first because FastAPI is. The DeepEval wrapper bridges back to sync
via `asyncio.run()` for callers that need it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Protocol


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ChatMessage:
    role: Role
    content: str
    name: str | None = None  # tool name, when role == "tool"


@dataclass(slots=True)
class ChatOptions:
    """Knobs every provider accepts in some form.

    Provider-specific extras (e.g. OpenAI's `seed`) live in `extra` and are
    forwarded verbatim by the adapter that understands them. The adapter
    must silently ignore extras it doesn't recognise — never crash on them,
    because the same options object travels across providers in /arena.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    stop: list[str] | None = None
    # "text" or "json" — adapters translate to the provider's flag if any.
    response_format: Literal["text", "json"] | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ChatResponse:
    text: str
    model: str
    finish_reason: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    # The provider name as we identify it internally, e.g. "openai".
    provider: str = ""


@dataclass(slots=True)
class ModelInfo:
    """Returned by list_models(). Kept minimal — only what UI/registry needs."""

    id: str                  # the string callers pass to ChatOptions.model
    provider: str
    label: str | None = None # nice display name, falls back to id


class LLMAdapter(Protocol):
    """Every concrete provider implements this."""

    provider: str
    default_model: str

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatResponse: ...

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[str]:
        """Yield incremental text deltas. Implementations must `async def`+`yield`."""
        ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def health(self) -> bool:
        """Cheap liveness check: credentials present + endpoint reachable."""
        ...


class AdapterError(RuntimeError):
    """Raised by adapters on misconfiguration (missing key, bad URL, etc.)."""
