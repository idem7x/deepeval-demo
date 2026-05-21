"""
A thin chat "service" that the /chat endpoint and the DeepEval tests both
call into. Keeps the prompt construction + retrieval in one place so when
we tweak the system prompt or context format, every consumer benefits.

`answer()` is async and returns both the model text and the retrieved
chunks — the latter is what DeepEval RAG metrics consume as
`retrieval_context`.
"""

from __future__ import annotations

from dataclasses import dataclass

from apps.backend.llm.base import ChatMessage, ChatOptions, LLMAdapter
from apps.backend.rag.retriever import RetrievedChunk, Retriever, default_retriever


DEFAULT_SYSTEM = (
    "You are a knowledgeable assistant for foreign buyers of Spanish real "
    "estate. Answer questions accurately and concisely. When the user asks "
    "about taxes, legal procedures, or visa rules, cite specific Spanish "
    "laws or articles when known. If the answer is not in the provided "
    "context, say so plainly — do not invent figures or article numbers."
)


@dataclass(slots=True)
class AnswerResult:
    text: str
    retrieval_context: list[str]   # raw chunk texts (what DeepEval expects)
    chunks: list[RetrievedChunk]   # full chunk objects (for the UI / dashboard)


def _build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    msgs: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt or DEFAULT_SYSTEM)]
    if chunks:
        context = "Context (retrieved from knowledge base):\n\n" + "\n\n---\n\n".join(
            f"[source: {c.source} | title: {c.title}]\n{c.text}" for c in chunks
        )
        msgs.append(ChatMessage(role="system", content=context))
    if history:
        msgs.extend(history)
    msgs.append(ChatMessage(role="user", content=question))
    return msgs


async def answer(
    adapter: LLMAdapter,
    question: str,
    *,
    model: str | None = None,
    use_rag: bool = True,
    rag_k: int = 5,
    rag_filters: dict[str, str] | None = None,
    history: list[ChatMessage] | None = None,
    system_prompt: str | None = None,
    retriever: Retriever | None = None,
    temperature: float = 0.0,
    max_tokens: int = 600,
) -> AnswerResult:
    """One-shot Q→A with optional RAG. Used by /chat (non-stream) and DeepEval tests.

    Defaults: `temperature=0` so multiple evaluation runs are reproducible.
    """
    chunks: list[RetrievedChunk] = []
    if use_rag:
        r = retriever or default_retriever()
        chunks = r.retrieve(question, k=rag_k, **(rag_filters or {}))

    messages = _build_messages(question, chunks, history=history, system_prompt=system_prompt)
    opts = ChatOptions(
        model=model or adapter.default_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    resp = await adapter.chat(messages, opts)
    return AnswerResult(
        text=resp.text,
        retrieval_context=[c.text for c in chunks],
        chunks=chunks,
    )
