"""
Markdown-aware chunker.

Splits a `Document` into smaller `Chunk`s sized in **tokens** (not chars), so
the chunk size we set here is comparable to what the embedding/judge model
will actually see.

Strategy:

1. Token counter — `tiktoken` `cl100k_base` (the OpenAI/Anthropic-ish
   default). Good enough as a universal proxy.
2. Splitter — LangChain's `RecursiveCharacterTextSplitter` with separators
   ordered from most-structural to least: markdown headers → blank lines →
   newlines → sentences → spaces. So the chunker tries to break at H2/H3
   first, only falls back to mid-sentence as a last resort.
3. Each chunk carries the **title** of its parent document at the top, in a
   markdown header. This gives the embedding/retrieval model a topical
   anchor even when a chunk lands in the middle of a long article.
4. Stable per-chunk `id = "{doc.id}::{ordinal}"`, so re-ingestion upserts
   instead of duplicating rows in ChromaDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.loader import Document


DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100

# Markdown-aware separator list, most-structural first.
MD_SEPARATORS = [
    "\n## ",     # H2
    "\n### ",    # H3
    "\n#### ",   # H4
    "\n\n",      # paragraph
    "\n",        # line
    ". ",        # sentence
    " ",
    "",
]


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str]


def _token_len(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text, disallowed_special=()))


def chunk_document(
    doc: Document,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Turn one Document into a list of Chunks ready for embedding."""
    encoder = tiktoken.get_encoding("cl100k_base")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=lambda t: _token_len(t, encoder),
        separators=MD_SEPARATORS,
        is_separator_regex=False,
    )

    pieces = splitter.split_text(doc.text)
    base_meta = doc.to_chroma_metadata()

    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        # Always prepend the title — it gives the retriever a topical anchor
        # even when the slice lands deep inside a long article. Tiny cost
        # in token budget; large win on retrieval quality.
        body = f"# {doc.title}\n\n{piece.strip()}"
        chunk_meta = {
            **base_meta,
            "doc_id": doc.id,
            "chunk_ordinal": str(i),
            "chunk_count": str(len(pieces)),
        }
        chunks.append(
            Chunk(
                id=f"{doc.id}::{i:04d}",
                text=body,
                metadata=chunk_meta,
            )
        )
    return chunks


def chunk_documents(
    docs: Iterable[Document],
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Convenience: flatten many documents to one chunk list."""
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, chunk_tokens, overlap_tokens))
    return out
