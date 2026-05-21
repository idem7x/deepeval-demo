"""
Smoke tests for the LLM layer — no real API calls.

We test the *contract* (shape, error handling, cross-provider judging),
not provider-specific quirks. Concrete adapters are exercised in the
eval matrix (Phase 7) when keys are present.
"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from apps.backend.llm.base import (
    AdapterError,
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ModelInfo,
    TokenUsage,
)
from apps.backend.llm.deepeval_wrap import DeepEvalLLM
from apps.backend.llm.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fake adapter used by every test that needs an LLMAdapter without network
# ---------------------------------------------------------------------------


class FakeAdapter:
    provider = "fake"
    default_model = "fake-model-1"

    def __init__(self, canned: str = "ok") -> None:
        self.canned = canned
        self.received: list[ChatMessage] = []

    async def chat(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> ChatResponse:
        self.received = messages
        return ChatResponse(
            text=self.canned,
            model=options.model,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=1, completion_tokens=2),
            provider=self.provider,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
    ) -> AsyncIterator[str]:
        for token in self.canned.split():
            yield token + " "

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=self.default_model, provider=self.provider)]

    async def health(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# base.py — data class shapes
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_token_usage_total():
    u = TokenUsage(prompt_tokens=10, completion_tokens=7)
    assert u.total == 17


@pytest.mark.smoke
def test_chat_options_defaults():
    opts = ChatOptions(model="x")
    assert opts.temperature == 0.7
    assert opts.max_tokens is None
    assert opts.extra == {}


# ---------------------------------------------------------------------------
# Registry — must report cleanly when nothing is configured
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_registry_without_credentials(monkeypatch):
    """Registry must report all providers, even when none are usable."""
    monkeypatch.setattr("apps.backend.core.settings.settings.openai_api_key", "")
    monkeypatch.setattr("apps.backend.core.settings.settings.anthropic_api_key", "")
    monkeypatch.setattr(
        "apps.backend.core.settings.settings.ollama_base_url",
        "http://127.0.0.1:1",  # guaranteed unreachable
    )

    reg = ModelRegistry()
    status = await reg.describe(force_refresh=True)
    names = {s.provider for s in status}
    assert names == {"openai", "anthropic", "ollama"}
    assert all(s.available is False for s in status)
    # Defaults must be filled even for providers that never initialised,
    # so the UI can still render them as "configure me".
    for s in status:
        assert s.default_model, f"empty default_model for {s.provider}"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_registry_caches_describe(monkeypatch):
    """Two back-to-back describe() calls must not hit the providers twice."""
    monkeypatch.setattr("apps.backend.core.settings.settings.openai_api_key", "")
    monkeypatch.setattr("apps.backend.core.settings.settings.anthropic_api_key", "")
    monkeypatch.setattr(
        "apps.backend.core.settings.settings.ollama_base_url",
        "http://127.0.0.1:1",
    )
    reg = ModelRegistry()
    a = await reg.describe(force_refresh=True)
    b = await reg.describe()                # within TTL → cached
    assert a is b                            # same list object → no re-ping


# ---------------------------------------------------------------------------
# DeepEvalLLM wrapper — the cross-provider judging bridge
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_deepeval_wrap_judge_id_and_generate():
    fake = FakeAdapter(canned="judgement-text")
    judge = DeepEvalLLM(fake, model="fake-x")
    assert judge.judge_id == "fake/fake-x"
    assert judge.get_model_name() == "fake/fake-x"
    assert judge.generate("hello") == "judgement-text"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_deepeval_wrap_a_generate_sends_message():
    fake = FakeAdapter(canned="answer")
    judge = DeepEvalLLM(fake)
    out = await judge.a_generate("explain ITP")
    assert out == "answer"
    assert len(fake.received) == 1
    assert fake.received[0].role == "user"
    assert fake.received[0].content == "explain ITP"


@pytest.mark.smoke
def test_deepeval_wrap_load_model_returns_adapter():
    fake = FakeAdapter()
    judge = DeepEvalLLM(fake)
    assert judge.load_model() is fake


# ---------------------------------------------------------------------------
# Real adapter constructors — must fail loudly without credentials
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_openai_adapter_requires_key(monkeypatch):
    monkeypatch.setattr("apps.backend.core.settings.settings.openai_api_key", "")
    from apps.backend.llm.openai import OpenAIAdapter

    with pytest.raises(AdapterError, match="OPENAI_API_KEY"):
        OpenAIAdapter()


@pytest.mark.smoke
def test_anthropic_adapter_requires_key(monkeypatch):
    monkeypatch.setattr("apps.backend.core.settings.settings.anthropic_api_key", "")
    from apps.backend.llm.anthropic import AnthropicAdapter

    with pytest.raises(AdapterError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter()
