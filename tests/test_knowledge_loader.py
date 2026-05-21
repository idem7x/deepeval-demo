"""
Smoke tests for the knowledge layer (Phase 1).

These tests are intentionally cheap and offline. They confirm:

- The loader picks up every curated markdown file (so we never accidentally
  ship the project with an empty corpus).
- Frontmatter parsing works for the fields RAG will rely on.
- ChromaDB metadata serialisation never returns non-scalars (ChromaDB will
  reject those).

No network. No LLM calls. Safe to run in CI.
"""

from __future__ import annotations

import pytest

from knowledge.loader import CURATED_DIR, Document, load_all, parse_markdown


@pytest.mark.smoke
def test_curated_dir_has_documents():
    md_files = list(CURATED_DIR.rglob("*.md"))
    assert len(md_files) >= 10, (
        f"Expected at least 10 curated markdown files in {CURATED_DIR}, "
        f"got {len(md_files)}"
    )


@pytest.mark.smoke
def test_loader_returns_documents():
    docs = load_all()
    assert docs, "loader returned 0 documents"
    assert all(isinstance(d, Document) for d in docs)
    assert all(d.text for d in docs), "some documents had empty body"
    # IDs must be unique — Phase 2 indexing relies on this for upsert.
    ids = [d.id for d in docs]
    assert len(ids) == len(set(ids)), "duplicate document ids found"


@pytest.mark.smoke
def test_frontmatter_fields_present_on_curated():
    """Every hand-written curated doc must have explicit metadata."""
    for path in CURATED_DIR.rglob("*.md"):
        doc = parse_markdown(path)
        assert doc.title and doc.title != path.stem.title(), (
            f"curated doc {path.name} missing explicit `title` in frontmatter"
        )
        assert doc.topic != "unknown", (
            f"curated doc {path.name} missing explicit `topic` in frontmatter"
        )
        assert doc.source == "curated"


@pytest.mark.smoke
def test_chroma_metadata_is_scalar_only():
    """ChromaDB rejects non-scalar metadata; never let one slip through."""
    for doc in load_all():
        meta = doc.to_chroma_metadata()
        for k, v in meta.items():
            assert isinstance(v, (str, int, float, bool)), (
                f"non-scalar metadata in {doc.id}: {k}={v!r} ({type(v).__name__})"
            )
