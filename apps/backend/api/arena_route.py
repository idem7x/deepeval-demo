"""
POST /arena/compare — fan a single prompt out to 2-5 models in parallel.

This is the side-by-side comparison endpoint that powers the arena page
in the UI. Each candidate runs against the same prompt and the same RAG
context (retrieved ONCE and shared) so the comparison is fair.

The response is intentionally flat — text + latency + tokens per
candidate — because the UI renders a card per result and an optional
G-Eval verdict is computed client-side via /eval (Phase 7).
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.backend.api.chat_route import RetrievedChunkOut, SYSTEM_PROMPT_DEFAULT
from apps.backend.api.deps import get_registry, get_retriever
from apps.backend.llm.base import ChatMessage, ChatOptions
from apps.backend.llm.registry import ModelRegistry
from apps.backend.rag.retriever import Retriever


router = APIRouter(prefix="/arena", tags=["arena"])


class Candidate(BaseModel):
    provider: str
    model: str


class ArenaRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    candidates: list[Candidate] = Field(..., min_length=2, max_length=5)
    use_rag: bool = True
    rag_k: int = Field(5, ge=1, le=20)
    rag_filters: dict[str, str] = Field(default_factory=dict)
    system_prompt: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)


class CandidateResult(BaseModel):
    provider: str
    model: str
    text: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


class ArenaResponse(BaseModel):
    prompt: str
    retrieved_context: list[RetrievedChunkOut] = Field(default_factory=list)
    results: list[CandidateResult]


@router.post("/compare", response_model=ArenaResponse)
async def compare(
    req: ArenaRequest,
    registry: ModelRegistry = Depends(get_registry),
    retriever: Retriever = Depends(get_retriever),
) -> ArenaResponse:
    # Validate every candidate up-front; failing one halfway through after
    # spending tokens on others is a worse UX than a 400.
    for c in req.candidates:
        if registry.get(c.provider) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{c.provider}' is not configured.",
            )

    # Retrieve once; share the chunks across candidates so the comparison
    # measures the model, not retrieval variance.
    chunks = []
    if req.use_rag:
        chunks = retriever.retrieve(req.prompt, k=req.rag_k, **req.rag_filters)

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=req.system_prompt or SYSTEM_PROMPT_DEFAULT),
    ]
    if chunks:
        context = "Context (retrieved from knowledge base):\n\n" + "\n\n---\n\n".join(
            f"[source: {c.source} | title: {c.title}]\n{c.text}" for c in chunks
        )
        messages.append(ChatMessage(role="system", content=context))
    messages.append(ChatMessage(role="user", content=req.prompt))

    async def run_one(c: Candidate) -> CandidateResult:
        adapter = registry.get(c.provider)
        if adapter is None:  # already validated above; defensive
            return CandidateResult(provider=c.provider, model=c.model, error="not configured")
        opts = ChatOptions(
            model=c.model,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        t0 = time.perf_counter()
        try:
            resp = await adapter.chat(messages, opts)
        except Exception as e:
            return CandidateResult(
                provider=c.provider, model=c.model,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                error=f"{type(e).__name__}: {e}",
            )
        return CandidateResult(
            provider=c.provider,
            model=resp.model or c.model,
            text=resp.text,
            finish_reason=resp.finish_reason,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    results = await asyncio.gather(*(run_one(c) for c in req.candidates))

    return ArenaResponse(
        prompt=req.prompt,
        retrieved_context=[RetrievedChunkOut.from_chunk(c) for c in chunks],
        results=list(results),
    )
