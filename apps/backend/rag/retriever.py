"""
High-level retrieval API used by the chat layer and the DeepEval RAG tests.

`ChromaStore.query` is fine for diagnostics but exposes Chroma's `where`
clause directly. Most callers only need a couple of common filters
(by region, by topic, by language, by source), so this module wraps that
into a friendlier signature and a stable return type that won't leak Chroma
internals into the rest of the codebase.

`RetrievedChunk` mirrors what DeepEval RAG metrics expect as
`retrieval_context` — a flat list of strings, with their metadata
available separately for debugging / dashboard display.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from apps.backend.rag.store import ChromaStore, QueryResult


@dataclass(slots=True)
class RetrievedChunk:
    text: str
    chunk_id: str
    source: str
    title: str
    distance: float
    url: str | None = None
    region: str | None = None
    topic: str | None = None

    @classmethod
    def from_query_result(cls, r: QueryResult) -> "RetrievedChunk":
        m = r.metadata
        return cls(
            text=r.text,
            chunk_id=r.chunk_id,
            source=str(m.get("source", "unknown")),
            title=str(m.get("title", "")),
            distance=r.distance,
            url=m.get("url"),
            region=m.get("region"),
            topic=m.get("topic"),
        )


class Retriever:
    def __init__(self, store: ChromaStore | None = None) -> None:
        self._store = store or ChromaStore.default()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        region: str | None = None,
        topic: str | None = None,
        source: str | None = None,
        lang: str | None = None,
    ) -> list[RetrievedChunk]:
        """Top-k semantic search with optional metadata filters.

        When multiple filters are given they are AND-ed together (Chroma's
        default behaviour with multiple keys in `where`).
        """
        where: dict[str, str] = {}
        if region:
            where["region"] = region
        if topic:
            where["topic"] = topic
        if source:
            where["source"] = source
        if lang:
            where["lang"] = lang

        # Chroma requires multi-key where to use $and explicitly.
        if len(where) > 1:
            where = {"$and": [{k: v} for k, v in where.items()]}  # type: ignore[assignment]

        results = self._store.query(query, k=k, where=where or None)
        return [RetrievedChunk.from_query_result(r) for r in results]


@lru_cache(maxsize=1)
def default_retriever() -> Retriever:
    """Process-wide singleton for convenience in chat/eval code paths."""
    return Retriever()
