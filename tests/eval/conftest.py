"""
Shared fixtures for the DeepEval suite.

Every test that grades model output needs a *judge* model. We make that
one shared object (`judge`) so it's obvious what's doing the scoring
across all suites — and so cost-per-run is predictable. Default judge:
gpt-4o-mini, temperature=0 (cheap, deterministic-ish).

Tests that need a chat answer (RAG, conversational, custom) use the
`chat_adapter` fixture and the shared `service.answer()` helper. That
keeps the SUT identical to what the /chat HTTP endpoint produces, so
metric scores reflect production behaviour rather than test-specific
prompt drift.
"""

from __future__ import annotations

import os

import pytest

from apps.backend.llm.deepeval_wrap import DeepEvalLLM
from apps.backend.llm.openai import OpenAIAdapter


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
SUT_MODEL = os.environ.get("SUT_MODEL", "gpt-4o-mini")


def _require_openai_key() -> None:
    """Skip the suite cleanly if no OpenAI key is configured.

    These tests cost money — we never want them silently passing in CI
    without the budget being approved via the OPENAI_API_KEY secret.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        # Fall back to the .env loader (settings reads it on import).
        from apps.backend.core.settings import settings  # noqa: WPS433
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not set; skipping live eval suite.")


@pytest.fixture(scope="session")
def chat_adapter() -> OpenAIAdapter:
    _require_openai_key()
    return OpenAIAdapter()


@pytest.fixture(scope="session")
def judge(chat_adapter: OpenAIAdapter) -> DeepEvalLLM:
    """The model that grades. We reuse `chat_adapter`'s underlying client
    to keep connection pooling tight — but bind a different (cheaper,
    fixed-temperature) ChatOptions via the wrapper."""
    return DeepEvalLLM(chat_adapter, model=JUDGE_MODEL, temperature=0.0)


@pytest.fixture(scope="session")
def sut_model() -> str:
    """Model name used as the system-under-test in eval tests."""
    return SUT_MODEL
