"""Anthropic adapter — Messages API via the official `anthropic` SDK."""

from __future__ import annotations

from typing import AsyncIterator

from apps.backend.core.settings import settings
from apps.backend.llm.base import (
    AdapterError,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ModelInfo,
    TokenUsage,
)


# Latest Claude 4.x family as of 2026. IDs are versionless aliases that
# Anthropic keeps pointing at the latest minor; if you need pinned snapshots,
# pass them explicitly via ChatOptions.model.
KNOWN_CHAT_MODELS: list[tuple[str, str]] = [
    ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ("claude-opus-4-7", "Claude Opus 4.7"),
]


class AnthropicAdapter:
    provider = "anthropic"
    default_model = "claude-haiku-4-5"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        key = api_key or settings.anthropic_api_key
        if not key:
            raise AdapterError("Anthropic adapter requires ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=key)

    # --- message translation --------------------------------------------

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        """Anthropic carries the system prompt outside the messages array.

        We collapse any 'system' role messages into a single string. If
        there are several (rare but legal in our schema), we join with a
        blank line — Anthropic does the same internally.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n\n".join(system_parts) if system_parts else None

        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            # Tool messages aren't supported here yet — fall back to user
            # role so we never silently drop a message.
            role = m.role if m.role in ("user", "assistant") else "user"
            out.append({"role": role, "content": m.content})
        return system, out

    def _common_kwargs(self, options: ChatOptions) -> dict:
        # Anthropic *requires* max_tokens. Pick a sane default if caller
        # didn't supply one.
        kw: dict[str, object] = {
            "model": options.model,
            "max_tokens": options.max_tokens or 1024,
            "temperature": options.temperature,
        }
        if options.stop:
            kw["stop_sequences"] = options.stop
        # No native json mode on Anthropic Messages; callers who want JSON
        # should add a system instruction. We honour `extra` for everything
        # provider-specific the SDK accepts.
        kw.update(options.extra)
        return kw

    # --- chat / stream ---------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatResponse:
        system, msgs = self._split_system(messages)
        resp = await self._client.messages.create(
            messages=msgs,
            system=system,
            **self._common_kwargs(options),
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = TokenUsage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return ChatResponse(
            text=text,
            model=resp.model,
            finish_reason=resp.stop_reason or "end_turn",
            usage=usage,
            provider=self.provider,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[str]:
        system, msgs = self._split_system(messages)
        async with self._client.messages.stream(
            messages=msgs,
            system=system,
            **self._common_kwargs(options),
        ) as stream:
            async for delta in stream.text_stream:
                yield delta

    # --- discovery -------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        # Anthropic exposes a /v1/models endpoint, but our curated list is
        # already what people actually want in the chat picker.
        return [
            ModelInfo(id=m_id, provider=self.provider, label=label)
            for m_id, label in KNOWN_CHAT_MODELS
        ]

    async def health(self) -> bool:
        try:
            # Cheapest meaningful call: ask for the model list. If the key
            # is valid the SDK returns; otherwise it raises.
            await self._client.models.list()
            return True
        except Exception:
            return False
