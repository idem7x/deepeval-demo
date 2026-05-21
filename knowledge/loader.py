"""
Uniform document model + filesystem loader.

Every ingestion script — curated markdown, Wikipedia, gov't PDFs — produces
files on disk with the same shape: a markdown body, plus a YAML frontmatter
block carrying metadata. This module is the single place that knows how to
read them back into `Document` objects that the rest of the system consumes
(ChromaDB indexer in Phase 2, DeepEval Synthesizer in Phase 6, the UI chunk
inspector, etc.).

The metadata schema is intentionally small and stable. Add fields here when
multiple downstream consumers actually need them, not before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml


KNOWLEDGE_ROOT = Path(__file__).resolve().parent
CURATED_DIR = KNOWLEDGE_ROOT / "curated"
RAW_DIR = KNOWLEDGE_ROOT / "raw"


@dataclass(slots=True)
class Document:
    """A single source document, before chunking.

    `id` must be stable across re-ingestions — we derive it from the path so
    re-running the ingestion replaces existing entries in ChromaDB instead of
    duplicating them.
    """

    id: str
    text: str
    source: str           # "curated" | "wikipedia" | "pdf" | ...
    title: str
    topic: str            # "tax" | "region" | "process" | "visa" | "property-type"
    lang: str             # ISO 639-1, e.g. "en", "es"
    url: str | None = None
    region: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict[str, str]:
        """ChromaDB only accepts scalar metadata values — flatten accordingly."""
        meta = {
            "source": self.source,
            "title": self.title,
            "topic": self.topic,
            "lang": self.lang,
        }
        if self.url:
            meta["url"] = self.url
        if self.region:
            meta["region"] = self.region
        for k, v in self.extra.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = str(v)
        return meta


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_markdown(path: Path) -> Document:
    """Read a markdown file with YAML frontmatter into a Document.

    A file without frontmatter is still valid — we fall back to filename for
    title and "unknown" for topic, so dropping a random .md into the folder
    won't blow up the loader.
    """
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = match.group(2).strip()
    else:
        meta = {}
        body = raw.strip()

    rel = path.relative_to(KNOWLEDGE_ROOT)
    doc_id = str(rel).replace("/", "__").removesuffix(".md")

    return Document(
        id=doc_id,
        text=body,
        source=meta.get("source", _default_source_from_path(rel)),
        title=meta.get("title", path.stem.replace("-", " ").title()),
        topic=meta.get("topic", "unknown"),
        lang=meta.get("lang", "en"),
        url=meta.get("url"),
        region=meta.get("region"),
        extra={k: v for k, v in meta.items()
               if k not in {"source", "title", "topic", "lang", "url", "region"}},
    )


def _default_source_from_path(rel: Path) -> str:
    """Infer source from where the file lives, when frontmatter doesn't say."""
    parts = rel.parts
    if parts and parts[0] == "curated":
        return "curated"
    if parts and parts[0] == "raw" and len(parts) > 1:
        return parts[1]  # e.g. raw/wikipedia/... -> "wikipedia"
    return "unknown"


def iter_documents(roots: list[Path] | None = None) -> Iterator[Document]:
    """Yield every Document found under the given roots (defaults: curated + raw)."""
    if roots is None:
        roots = [CURATED_DIR, RAW_DIR]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            yield parse_markdown(path)


def load_all() -> list[Document]:
    """Materialise iter_documents() into a list. Convenient for scripts/tests."""
    return list(iter_documents())
