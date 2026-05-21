"""Anthropic adapter — Messages API via the official `anthropic` SDK.

TLS workaround
--------------
On macOS with Python 3.12 + OpenSSL 3.6 (which is the Homebrew default),
a TLS 1.3 ClientHello to Anthropic's edge is occasionally answered with a
TCP RST ("Connection reset by peer") — likely TLS fingerprinting upstream.
The same machine's `curl` works (it uses LibreSSL via macOS SecureTransport
and TLS 1.2). To make Python parity, we pin the SDK's underlying httpx
client to TLS 1.2 max. No-op on systems where TLS 1.3 already works.
"""

from __future__ import annotations

import ssl
from typing import AsyncIterator

import httpx

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


def _build_http_client() -> httpx.AsyncClient:
    """An httpx.AsyncClient with TLS 1.2 ceiling (see module docstring)."""
    ctx = ssl.create_default_context()
    ctx.maximum_version = ssl.TLSVersion.TLSv1_2
    return httpx.AsyncClient(verify=ctx, timeout=60.0)


class AnthropicAdapter:
    provider = "anthropic"
    default_model = "claude-haiku-4-5"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        key = api_key or settings.anthropic_api_key
        if not key:
            raise AdapterError("Anthropic adapter requires ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=key, http_client=_build_http_client())

    # --- message translation --------------------------------------------

    @staticmethod
    def _split_system(messages: list[ChatMessage]) -> tuple[list[dict] | None, list[dict]]:
        """Anthropic carries the system prompt outside the messages array.

        Current Messages API expects `system` as a list of content blocks,
        not a plain string — passing a string raises 400 "Input should be
        a valid array". We wrap each system message as its own text block.
        """
        system_parts = [m.content for m in messages if m.role == "system"]
        system: list[dict] | None = (
            [{"type": "text", "text": p} for p in system_parts] if system_parts else None
        )

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
        # The Messages API rejects `system=null` with "Input should be a valid
        # array" — pass the kwarg only when we actually have a system message.
        kw = self._common_kwargs(options)
        if system is not None:
            kw["system"] = system
        resp = await self._client.messages.create(messages=msgs, **kw)
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
        kw = self._common_kwargs(options)
        if system is not None:
            kw["system"] = system
        async with self._client.messages.stream(messages=msgs, **kw) as stream:
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
