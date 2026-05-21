"""
FastAPI entrypoint.

Phase 0 — only a `/health` endpoint exists. Real routes (`/chat`, `/models`,
`/arena`, `/eval`) are added in Phase 4 once the LLM adapters and RAG
pipeline are in place.
"""

from __future__ import annotations

from fastapi import FastAPI

from apps.backend.core.settings import settings

app = FastAPI(
    title="DeepEval Lab",
    version="0.1.0",
    description="Multi-model chat with RAG and full DeepEval metric coverage.",
)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe + a tiny snapshot of what the app sees in its env.

    Returns which providers have credentials configured (we never return the
    keys themselves). Useful when you just installed the project and want to
    sanity-check that `.env` is being picked up.
    """
    return {
        "status": "ok",
        "providers": {
            "openai": bool(settings.openai_api_key),
            "anthropic": bool(settings.anthropic_api_key),
            "groq": bool(settings.groq_api_key),
            "ollama_url": settings.ollama_base_url,
        },
        "embeddings": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
    }
