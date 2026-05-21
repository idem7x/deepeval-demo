"""
Bridge: any of our `LLMAdapter`s, used as a DeepEval judge.

DeepEval metrics need an instance of `DeepEvalBaseLLM`. By wrapping our
adapters in one, every provider in the project becomes a valid judge —
so the eval matrix can do GPT-judges-Claude, Claude-judges-Llama, etc.

DeepEval's contract:
- `load_model()`            -> the underlying client (returned verbatim;
                               DeepEval never actually calls this for chat,
                               it just stores the handle)
- `generate(prompt)`        -> sync str (DeepEval calls this for non-async
                               metrics)
- `a_generate(prompt)`      -> async str
- `get_model_name()`        -> short name shown in result logs

Our wrapper:
- Accepts an already-constructed adapter (so cross-provider configuration
  is decided once at the registry/factory layer, not here).
- Exposes a stable `judge_id` like "openai/gpt-4o-mini" for run-log
  attribution and the eval matrix dashboard.
"""

from __future__ import annotations

import asyncio

from deepeval.models import DeepEvalBaseLLM

from apps.backend.llm.base import ChatMessage, ChatOptions, LLMAdapter


class DeepEvalLLM(DeepEvalBaseLLM):
    """Wrap an LLMAdapter so DeepEval metrics can use it as a judge."""

    def __init__(
        self,
        adapter: LLMAdapter,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self._adapter = adapter
        self._model = model or adapter.default_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    # --- DeepEvalBaseLLM contract ----------------------------------------

    def load_model(self) -> LLMAdapter:
        # DeepEval stores this on the instance but never actually calls on
        # it for chat; returning the adapter is the most useful handle.
        return self._adapter

    def generate(self, prompt: str) -> str:
        # Sync entry point; bridge to async. Most DeepEval metrics call
        # this from sync evaluator code, so the trip through asyncio.run
        # is necessary.
        return asyncio.run(self.a_generate(prompt))

    async def a_generate(self, prompt: str) -> str:
        opts = ChatOptions(
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        resp = await self._adapter.chat(
            messages=[ChatMessage(role="user", content=prompt)],
            options=opts,
        )
        return resp.text

    def get_model_name(self) -> str:
        return self.judge_id

    # --- helpers ---------------------------------------------------------

    @property
    def judge_id(self) -> str:
        """Stable label for run logs: "provider/model"."""
        return f"{self._adapter.provider}/{self._model}"
