"""
ChromaDB-backed vector store.

One persistent client per process, one collection per
*(provider + model)*. Namespacing by provider matters because vectors from
different embedders live in different spaces and **cannot be compared** —
mixing them in a single collection silently corrupts retrieval. With this
naming scheme, switching `EMBEDDING_PROVIDER` just spawns a fresh
collection and the old one stays intact on disk in case you want to A/B.

Surface:

    store = ChromaStore.default()
    store.ingest(docs)                 # chunk + embed + upsert
    store.query("ITP rate Madrid", k=5)
    store.query("...", where={"topic": "tax"})
    store.stats()                      # for diagnostics
    store.reset()                      # drop the collection

Re-ingestion is *idempotent*: chunk IDs are stable (`doc.id::ordinal`),
so a re-run replaces the previous rows instead of duplicating them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import chromadb
from chromadb.api.types import EmbeddingFunction

from apps.backend.core.settings import settings
from apps.backend.rag.chunker import Chunk, chunk_documents
from apps.backend.rag.embeddings import EmbeddingProvider, get_embedder
from knowledge.loader import Document


# ChromaDB only accepts ASCII alnum + _ + - + . for collection names,
# and the name must be 3..63 chars. The provider:model strings break that.
_SAFE = re.compile(r"[^a-z0-9_.-]+")


def _safe_name(prefix: str, provider_name: str) -> str:
    raw = f"{prefix}_{provider_name}".lower()
    clean = _SAFE.sub("-", raw).strip("-")
    return clean[:63]


@dataclass(slots=True)
class QueryResult:
    chunk_id: str
    text: str
    metadata: dict[str, str]
    distance: float


class _ProviderEmbeddingFunction(EmbeddingFunction):
    """Adapts our EmbeddingProvider to Chroma's EmbeddingFunction protocol."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._p = provider

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - chroma API
        return self._p.embed(input)

    @classmethod
    def name(cls) -> str:
        # Chroma calls this on the class (without an instance) to identify
        # the embedding function type. Per-instance variation (model name)
        # is exposed via get_config() instead.
        return "deepeval_lab_provider"

    def get_config(self) -> dict[str, object]:
        # Persisted by Chroma alongside the collection so a fresh process
        # can rebuild the same embedder without our app being involved.
        provider, _, model = self._p.name.partition(":")
        return {"provider": provider, "model": model or ""}

    @classmethod
    def build_from_config(cls, config: dict[str, object]) -> "_ProviderEmbeddingFunction":
        provider = str(config.get("provider", "chromadb"))
        model = str(config.get("model") or "") or None
        return cls(get_embedder(provider, model))


class ChromaStore:
    """Thin wrapper that hides Chroma's low-level API from the rest of the app."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        path: str | None = None,
        base_name: str = "spain_real_estate",
    ) -> None:
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=str(path or settings.chroma_path))
        self._collection_name = _safe_name(base_name, embedder.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_ProviderEmbeddingFunction(embedder),
            metadata={"hnsw:space": "cosine"},
        )

    # --- Factories --------------------------------------------------------

    @classmethod
    def default(cls) -> "ChromaStore":
        """ChromaStore configured from settings (.env)."""
        return cls(embedder=get_embedder())

    # --- Mutation ---------------------------------------------------------

    def ingest(
        self,
        docs: Iterable[Document],
        chunk_tokens: int = 800,
        overlap_tokens: int = 100,
    ) -> int:
        """Chunk + upsert every doc. Returns the number of chunks written."""
        chunks = chunk_documents(docs, chunk_tokens, overlap_tokens)
        if not chunks:
            return 0
        self._upsert_chunks(chunks)
        return len(chunks)

    def _upsert_chunks(self, chunks: list[Chunk]) -> None:
        # Chroma will call our embedding function under the hood on `add`/
        # `upsert` when no `embeddings=` are passed, so we just send text.
        # We do batch in modest sizes for friendliness on large corpora.
        BATCH = 200
        for i in range(0, len(chunks), BATCH):
            slice_ = chunks[i : i + BATCH]
            self._collection.upsert(
                ids=[c.id for c in slice_],
                documents=[c.text for c in slice_],
                metadatas=[c.metadata for c in slice_],
            )

    def reset(self) -> None:
        """Drop and recreate the collection. Used by tests."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=_ProviderEmbeddingFunction(self.embedder),
            metadata={"hnsw:space": "cosine"},
        )

    # --- Query ------------------------------------------------------------

    def query(
        self,
        text: str,
        k: int = 5,
        where: dict[str, str] | None = None,
    ) -> list[QueryResult]:
        """Top-k semantic search, with optional metadata filter (Chroma syntax)."""
        res = self._collection.query(
            query_texts=[text],
            n_results=k,
            where=where or None,
        )
        # Chroma returns parallel arrays nested in [query_index][result_index];
        # we flatten the single-query case for ergonomic call sites.
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            QueryResult(chunk_id=i, text=d, metadata=m, distance=float(dist))
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]

    # --- Diagnostics ------------------------------------------------------

    def stats(self) -> dict[str, object]:
        return {
            "provider": self.embedder.name,
            "dim": self.embedder.dim,
            "collection": self._collection_name,
            "count": self._collection.count(),
            "path": str(settings.chroma_path),
        }
