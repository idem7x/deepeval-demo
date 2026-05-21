"""
Smoke tests for the full RAG pipeline (Phase 2).

Offline by design — uses ChromaDB's default ONNX embedder (~30 MB, downloaded
once into ~/.cache/chroma). No API keys, no Ollama, no network after the
first run.

Each test indexes only the **curated** subset (11 small files → ~11 chunks),
not the full Wikipedia+PDF corpus, so the suite stays under a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.backend.rag.chunker import chunk_documents
from apps.backend.rag.embeddings import get_embedder
from apps.backend.rag.retriever import Retriever
from apps.backend.rag.store import ChromaStore
from knowledge.loader import CURATED_DIR, parse_markdown


@pytest.fixture(scope="module")
def curated_docs():
    return [parse_markdown(p) for p in sorted(CURATED_DIR.rglob("*.md"))]


@pytest.fixture(scope="module")
def store(tmp_path_factory, curated_docs):
    """A throwaway ChromaStore at a tmp path, populated with curated docs."""
    tmp = tmp_path_factory.mktemp("chroma")
    s = ChromaStore(embedder=get_embedder("chromadb"), path=str(tmp))
    s.ingest(curated_docs)
    return s


# --- Pipeline structural tests -----------------------------------------------

@pytest.mark.smoke
@pytest.mark.rag
def test_chunker_handles_curated_corpus(curated_docs):
    chunks = chunk_documents(curated_docs)
    assert len(chunks) >= len(curated_docs), "expected ≥1 chunk per doc"
    # Every chunk must carry the parent doc id and a unique chunk id.
    ids = [c.id for c in chunks]
    assert len(set(ids)) == len(ids), "duplicate chunk ids"
    assert all("::" in c.id for c in chunks)


@pytest.mark.smoke
@pytest.mark.rag
def test_store_ingest_persists_chunks(store, curated_docs):
    stats = store.stats()
    assert stats["count"] >= len(curated_docs), (
        f"collection holds {stats['count']} chunk(s); expected ≥ {len(curated_docs)}"
    )
    assert stats["provider"] == "chromadb:default"


@pytest.mark.smoke
@pytest.mark.rag
def test_reingest_is_idempotent(store, curated_docs):
    """Same chunk ids → upsert, not duplicate."""
    before = store.stats()["count"]
    store.ingest(curated_docs)
    after = store.stats()["count"]
    assert after == before


# --- Retrieval quality tests -------------------------------------------------

@pytest.mark.smoke
@pytest.mark.rag
def test_itp_query_finds_itp_doc(store):
    r = Retriever(store)
    hits = r.retrieve("What is the ITP transfer tax rate in Madrid?", k=3)
    titles = [h.title for h in hits]
    assert any("ITP" in t for t in titles), (
        f"ITP doc not in top-3 results: {titles}"
    )


@pytest.mark.smoke
@pytest.mark.rag
def test_rental_query_finds_licensing_doc(store):
    r = Retriever(store)
    hits = r.retrieve("Can I rent my Barcelona apartment to tourists?", k=3)
    titles = [h.title for h in hits]
    assert any("Short-term" in t for t in titles), (
        f"rental licensing doc not in top-3: {titles}"
    )


@pytest.mark.smoke
@pytest.mark.rag
def test_topic_filter_narrows_results(store):
    """A `topic=tax` filter must only yield tax-tagged chunks."""
    r = Retriever(store)
    hits = r.retrieve("How does property taxation work?", k=5, topic="tax")
    assert hits, "expected some hits with topic=tax"
    assert all(h.topic == "tax" for h in hits), (
        f"non-tax topics leaked through filter: {[(h.title, h.topic) for h in hits]}"
    )


@pytest.mark.smoke
@pytest.mark.rag
def test_golden_visa_with_visa_filter(store):
    """The curated Golden Visa doc must win when filtered to topic=visa."""
    r = Retriever(store)
    hits = r.retrieve("Is the Golden Visa still available?", k=3, topic="visa")
    assert hits, "no hits with topic=visa"
    assert "Golden Visa" in hits[0].title
