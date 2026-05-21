"""OpenAI adapter — Chat Completions API via the official `openai` SDK."""

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


# Curated default short-list. We always also offer whatever `list_models()`
# returns from the API, but we never want the chat UI to show *every* fine-
# tune in the org — only models that make sense as chat backends.
KNOWN_CHAT_MODELS: list[tuple[str, str]] = [
    ("gpt-4o-mini", "GPT-4o mini"),
    ("gpt-4o", "GPT-4o"),
    ("gpt-4.1-mini", "GPT-4.1 mini"),
    ("gpt-4.1", "GPT-4.1"),
]


class OpenAIAdapter:
    provider = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        key = api_key or settings.openai_api_key
        if not key:
            raise AdapterError("OpenAI adapter requires OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=key)

    # --- chat / stream ---------------------------------------------------

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            msg: dict[str, object] = {"role": m.role, "content": m.content}
            if m.name:
                msg["name"] = m.name
            out.append(msg)
        return out

    def _common_kwargs(self, options: ChatOptions) -> dict:
        kw: dict[str, object] = {
            "model": options.model,
            "temperature": options.temperature,
        }
        if options.max_tokens is not None:
            kw["max_tokens"] = options.max_tokens
        if options.stop:
            kw["stop"] = options.stop
        if options.response_format == "json":
            kw["response_format"] = {"type": "json_object"}
        kw.update(options.extra)
        return kw

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatResponse:
        resp = await self._client.chat.completions.create(
            messages=self._to_openai_messages(messages),
            stream=False,
            **self._common_kwargs(options),
        )
        choice = resp.choices[0]
        usage = TokenUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        return ChatResponse(
            text=choice.message.content or "",
            model=resp.model,
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            provider=self.provider,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[str]:
        resp = await self._client.chat.completions.create(
            messages=self._to_openai_messages(messages),
            stream=True,
            **self._common_kwargs(options),
        )
        async for event in resp:
            if not event.choices:
                continue
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    # --- discovery -------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        # `client.models.list()` is fine but the UI doesn't need every
        # embedding/tts/dall-e id. Start with the curated short-list, then
        # union with anything from the API that looks chat-like.
        out: dict[str, ModelInfo] = {
            m_id: ModelInfo(id=m_id, provider=self.provider, label=label)
            for m_id, label in KNOWN_CHAT_MODELS
        }
        try:
            api = await self._client.models.list()
            for m in api.data:
                if m.id.startswith(("gpt-", "o1-", "o3-")) and m.id not in out:
                    out[m.id] = ModelInfo(id=m.id, provider=self.provider, label=m.id)
        except Exception:
            # list_models() must never blow up the registry — fall back
            # silently to the curated list.
            pass
        return list(out.values())

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False
