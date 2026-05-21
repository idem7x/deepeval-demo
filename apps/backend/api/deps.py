"""
FastAPI dependency providers.

Routes do NOT import ModelRegistry or Retriever directly — they take them
via `Depends(...)`. That gives tests a clean seam: each test (or test
module) calls `app.dependency_overrides[get_registry] = lambda: FakeReg()`
to inject a fake without monkeypatching anything.

We hold one process-wide instance of each behind `@lru_cache` so the
production wiring stays cheap (no per-request reconstruction).
"""

from __future__ import annotations

from functools import lru_cache

from apps.backend.llm.registry import ModelRegistry
from apps.backend.rag.retriever import Retriever


@lru_cache(maxsize=1)
def _registry_singleton() -> ModelRegistry:
    return ModelRegistry()


@lru_cache(maxsize=1)
def _retriever_singleton() -> Retriever:
    return Retriever()


def get_registry() -> ModelRegistry:
    """Dependency: the live model registry. Override in tests via dependency_overrides."""
    return _registry_singleton()


def get_retriever() -> Retriever:
    """Dependency: the RAG retriever. Override in tests."""
    return _retriever_singleton()
