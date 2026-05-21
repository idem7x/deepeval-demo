"""Ollama adapter — local models via the official `ollama` Python SDK.

Ollama's strength here: it's the same chat shape as OpenAI/Anthropic but
runs on your machine, so the arena page can put a local llama3 next to
GPT-4o-mini and Claude-Haiku on the same prompt for free.
"""

from __future__ import annotations

from typing import AsyncIterator

from apps.backend.core.settings import settings
from apps.backend.llm.base import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ModelInfo,
    TokenUsage,
)


class OllamaAdapter:
    provider = "ollama"
    default_model = "llama3.1:8b"

    def __init__(self, base_url: str | None = None) -> None:
        # Imported lazily so the module is import-safe even if the package
        # isn't installed (e.g. on a CI runner that excludes [llm]).
        from ollama import AsyncClient

        self._client = AsyncClient(host=base_url or settings.ollama_base_url)

    # --- chat / stream ---------------------------------------------------

    def _to_ollama_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _common_options(self, options: ChatOptions) -> dict:
        # Ollama lumps generation params under `options`.
        ollama_opts: dict[str, object] = {"temperature": options.temperature}
        if options.max_tokens is not None:
            ollama_opts["num_predict"] = options.max_tokens
        if options.stop:
            ollama_opts["stop"] = options.stop
        ollama_opts.update(options.extra)
        return ollama_opts

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatResponse:
        kw: dict[str, object] = {
            "model": options.model,
            "messages": self._to_ollama_messages(messages),
            "options": self._common_options(options),
            "stream": False,
        }
        if options.response_format == "json":
            kw["format"] = "json"

        resp = await self._client.chat(**kw)
        usage = TokenUsage(
            prompt_tokens=resp.get("prompt_eval_count", 0) or 0,
            completion_tokens=resp.get("eval_count", 0) or 0,
        )
        return ChatResponse(
            text=resp["message"]["content"],
            model=options.model,
            finish_reason=resp.get("done_reason", "stop"),
            usage=usage,
            provider=self.provider,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[str]:
        async for chunk in await self._client.chat(
            model=options.model,
            messages=self._to_ollama_messages(messages),
            options=self._common_options(options),
            stream=True,
        ):
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta

    # --- discovery -------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        """Whatever the local Ollama server reports via /api/tags."""
        try:
            resp = await self._client.list()
        except Exception:
            return []
        items = []
        # ollama SDK returns either {"models":[...]} or a ListResponse
        # with a `.models` attribute, depending on version. Handle both.
        models = getattr(resp, "models", None) or resp.get("models", [])
        for m in models:
            name = getattr(m, "model", None) or m.get("model") or m.get("name", "")
            if name:
                items.append(ModelInfo(id=name, provider=self.provider, label=name))
        return items

    async def health(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            return False
