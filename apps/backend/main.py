"""FastAPI entrypoint — wires all routers and CORS.

Phases:
- /health (Phase 0)
- /models (Phase 4)
- /chat   (Phase 4)
- /arena  (Phase 4)
- /eval   (Phase 4 stub; full impl in Phase 7)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.backend.api import arena_route, chat_route, eval_route, models_route
from apps.backend.core.settings import settings

app = FastAPI(
    title="DeepEval Lab",
    version="0.4.0",
    description="Multi-model chat with RAG and full DeepEval metric coverage.",
)

# CORS — Next.js dev server on 3000 is the only origin we expect in dev.
# Production wiring (when there is a deployed frontend) would pin the URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models_route.router)
app.include_router(chat_route.router)
app.include_router(arena_route.router)
app.include_router(eval_route.router)


@app.get("/health")
def health() -> dict[str, object]:
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
