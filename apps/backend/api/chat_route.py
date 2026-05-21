"""
POST /chat — single-model chat with optional RAG.

Two modes selected by `stream`:

- `stream=false` (default): one JSON response with full text + retrieved
  chunks. Easiest for first-time exploration and for tests.
- `stream=true`: Server-Sent Events. Events in order:
    event: context        — once, with retrieved chunks (if RAG used)
    event: delta          — many, each carrying one text chunk
    event: done           — once, with usage + finish_reason
    event: error          — instead of done, if something blew up

The retrieved context is always returned BEFORE the model output so the
UI can render "sources" while the answer streams in.
"""

from __future__ import annotations

import json
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from apps.backend.api.deps import get_registry, get_retriever
from apps.backend.chat.session import store as session_store
from apps.backend.llm.base import ChatMessage, ChatOptions
from apps.backend.llm.registry import ModelRegistry
from apps.backend.rag.retriever import RetrievedChunk, Retriever


router = APIRouter(prefix="/chat", tags=["chat"])


# --- request/response models ------------------------------------------------


class ChatRequest(BaseModel):
    provider: str = Field(..., description="openai | anthropic | ollama")
    model: str = Field(..., description="Model id from /models")
    message: str = Field(..., min_length=1)
    session_id: str | None = None
    use_rag: bool = True
    rag_k: int = Field(5, ge=1, le=20)
    rag_filters: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata filters: region/topic/source/lang",
    )
    system_prompt: str | None = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, ge=1, le=8192)
    stream: bool = False


class RetrievedChunkOut(BaseModel):
    text: str
    chunk_id: str
    source: str
    title: str
    distance: float
    url: str | None = None
    region: str | None = None
    topic: str | None = None

    @classmethod
    def from_chunk(cls, c: RetrievedChunk) -> "RetrievedChunkOut":
        return cls(
            text=c.text, chunk_id=c.chunk_id, source=c.source, title=c.title,
            distance=c.distance, url=c.url, region=c.region, topic=c.topic,
        )


class ChatResponse(BaseModel):
    session_id: str
    text: str
    provider: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    retrieved_context: list[RetrievedChunkOut] = Field(default_factory=list)
    latency_ms: int


# --- helpers ----------------------------------------------------------------


SYSTEM_PROMPT_DEFAULT = (
    "You are a knowledgeable assistant for foreign buyers of Spanish real "
    "estate. Answer questions accurately and concisely. When the user asks "
    "about taxes, legal procedures, or visa rules, cite specific Spanish "
    "laws or articles when known. If the answer is not in the provided "
    "context, say so plainly — do not invent figures or article numbers."
)


def _build_messages(
    req: ChatRequest,
    history: list[ChatMessage],
    chunks: list[RetrievedChunk],
) -> list[ChatMessage]:
    """Compose: [system, optional context, ...history, user message]."""
    msgs: list[ChatMessage] = []
    msgs.append(ChatMessage(role="system", content=req.system_prompt or SYSTEM_PROMPT_DEFAULT))
    if chunks:
        context = "Context (retrieved from knowledge base):\n\n" + "\n\n---\n\n".join(
            f"[source: {c.source} | title: {c.title}]\n{c.text}" for c in chunks
        )
        msgs.append(ChatMessage(role="system", content=context))
    msgs.extend(history)
    msgs.append(ChatMessage(role="user", content=req.message))
    return msgs


def _retrieve_if_enabled(
    req: ChatRequest, retriever: Retriever
) -> list[RetrievedChunk]:
    if not req.use_rag:
        return []
    return retriever.retrieve(req.message, k=req.rag_k, **req.rag_filters)


def _get_adapter(provider: str, registry: ModelRegistry):
    adapter = registry.get(provider)
    if adapter is None:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{provider}' is not configured "
                   f"(check /models for what is available).",
        )
    return adapter


# --- endpoints --------------------------------------------------------------


@router.post("", response_model=ChatResponse | None)
async def chat(
    req: ChatRequest,
    registry: ModelRegistry = Depends(get_registry),
    retriever: Retriever = Depends(get_retriever),
):
    """Single chat turn — streaming via SSE or full JSON depending on `stream`."""
    adapter = _get_adapter(req.provider, registry)
    session = session_store.get_or_create(req.session_id)
    session.provider = req.provider
    session.model = req.model

    chunks = _retrieve_if_enabled(req, retriever)
    messages = _build_messages(req, session.messages, chunks)
    options = ChatOptions(
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    if req.stream:
        return StreamingResponse(
            _sse_generator(adapter, messages, options, session.id, chunks, req.message),
            media_type="text/event-stream",
        )

    t0 = time.perf_counter()
    resp = await adapter.chat(messages, options)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    # Persist the user turn + assistant reply onto the session.
    session_store.append(session.id, ChatMessage(role="user", content=req.message))
    session_store.append(session.id, ChatMessage(role="assistant", content=resp.text))

    return ChatResponse(
        session_id=session.id,
        text=resp.text,
        provider=adapter.provider,
        model=resp.model,
        finish_reason=resp.finish_reason,
        prompt_tokens=resp.usage.prompt_tokens,
        completion_tokens=resp.usage.completion_tokens,
        retrieved_context=[RetrievedChunkOut.from_chunk(c) for c in chunks],
        latency_ms=latency_ms,
    )


async def _sse_generator(
    adapter,
    messages: list[ChatMessage],
    options: ChatOptions,
    session_id: str,
    chunks: list[RetrievedChunk],
    user_message: str,
):
    """Yields SSE-formatted strings per the protocol described at module top."""
    def sse(event: str, payload: dict | str) -> str:
        body = payload if isinstance(payload, str) else json.dumps(payload)
        return f"event: {event}\ndata: {body}\n\n"

    # Frontend needs session id immediately so it can re-use it on the
    # next turn even before the model finishes generating.
    yield sse("session", {"session_id": session_id})

    if chunks:
        yield sse(
            "context",
            {"chunks": [RetrievedChunkOut.from_chunk(c).model_dump() for c in chunks]},
        )

    t0 = time.perf_counter()
    collected: list[str] = []
    try:
        async for delta in adapter.stream(messages, options):
            collected.append(delta)
            yield sse("delta", {"text": delta})
    except Exception as e:
        yield sse("error", {"message": str(e)})
        return

    latency_ms = int((time.perf_counter() - t0) * 1000)
    full = "".join(collected)
    # Persist on success so a transport error doesn't poison the history.
    session_store.append(session_id, ChatMessage(role="user", content=user_message))
    session_store.append(session_id, ChatMessage(role="assistant", content=full))
    yield sse(
        "done",
        {
            "finish_reason": "stop",
            "latency_ms": latency_ms,
            "model": options.model,
            "provider": adapter.provider,
        },
    )


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, object]:
    """Inspect a session — useful for the UI and for multi-turn evals."""
    s = session_store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "id": s.id,
        "provider": s.provider,
        "model": s.model,
        "messages": [
            {"role": m.role, "content": m.content} for m in s.messages
        ],
    }


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    session_store.clear(session_id)
