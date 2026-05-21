"""
Shared API test fixtures.

We inject fakes via FastAPI `app.dependency_overrides`, not monkeypatch.
That gives every test a clean swap point + a guaranteed teardown via
the fixture finalizer, which is much safer than touching the singleton
registry/retriever instances.
"""

from __future__ import annotations

import time
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from apps.backend.api.deps import get_registry, get_retriever
from apps.backend.chat.session import store as session_store
from apps.backend.llm.base import (
    ChatMessage,
    ChatOptions,
    ChatResponse,
    ModelInfo,
    TokenUsage,
)
from apps.backend.llm.registry import ProviderStatus
from apps.backend.main import app
from apps.backend.rag.retriever import RetrievedChunk


# --- Fakes -----------------------------------------------------------------


class FakeAdapter:
    def __init__(self, provider: str, default_model: str, canned: str) -> None:
        self.provider = provider
        self.default_model = default_model
        self.canned = canned

    async def chat(self, messages, options) -> ChatResponse:
        return ChatResponse(
            text=self.canned,
            model=options.model,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=4),
            provider=self.provider,
        )

    async def stream(self, messages, options) -> AsyncIterator[str]:
        for word in self.canned.split():
            yield word + " "

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=self.default_model, provider=self.provider, label=self.default_model)]

    async def health(self) -> bool:
        return True


class FakeRegistry:
    """Pretends to be ModelRegistry; backed by a dict of FakeAdapters."""

    def __init__(self, adapters: dict[str, FakeAdapter]) -> None:
        self._adapters = adapters

    def get(self, provider: str):
        return self._adapters.get(provider)

    def all_adapters(self):
        return dict(self._adapters)

    async def describe(self, force_refresh: bool = False):
        out = []
        for name, a in self._adapters.items():
            models = await a.list_models()
            out.append(ProviderStatus(
                provider=name,
                available=True,
                default_model=a.default_model,
                models=models,
                error=None,
            ))
        return out


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.last_query: str | None = None
        self.last_kwargs: dict = {}

    def retrieve(self, query, k=5, **filters):
        self.last_query = query
        self.last_kwargs = {"k": k, **filters}
        return self._chunks[:k]


# --- Fixtures --------------------------------------------------------------


@pytest.fixture
def fake_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="# ITP\n\nITP is 6% in Madrid.",
            chunk_id="curated__itp::0000",
            source="curated",
            title="ITP — Transfer tax",
            distance=0.21,
            topic="tax",
        ),
        RetrievedChunk(
            text="# Regions overview\n\nMadrid has the lowest ITP rate.",
            chunk_id="curated__regions::0000",
            source="curated",
            title="Regions overview",
            distance=0.34,
            topic="region",
        ),
    ]


@pytest.fixture
def fake_registry() -> FakeRegistry:
    return FakeRegistry(
        adapters={
            "openai": FakeAdapter("openai", "gpt-4o-mini", "GPT answers ITP is 6%."),
            "anthropic": FakeAdapter("anthropic", "claude-haiku-4-5", "Claude says ITP is 6%."),
        }
    )


@pytest.fixture
def client(fake_registry, fake_chunks):
    fake_retriever = FakeRetriever(fake_chunks)
    app.dependency_overrides[get_registry] = lambda: fake_registry
    app.dependency_overrides[get_retriever] = lambda: fake_retriever
    # Clean session state — each test starts with no sessions.
    session_store._sessions.clear()
    try:
        with TestClient(app) as c:
            c.fake_retriever = fake_retriever  # for assertions
            yield c
    finally:
        app.dependency_overrides.clear()
