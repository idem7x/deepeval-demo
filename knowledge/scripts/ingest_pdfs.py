"""
Ingest PDF documents into the knowledge base.

Two-stage:

1. *Best-effort download* of a small, hand-curated list of stable public
   Spanish-government PDF URLs (BOE law texts, AEAT explanatory notes,
   Banco de España research). Failures are tolerated — the script prints
   a warning and continues, because URLs do rot.

2. *Process every PDF* found under `knowledge/raw/pdfs/`, extracting text
   with PyMuPDF and emitting a markdown file with YAML frontmatter next
   to it. This means you can also drop your own PDFs into that folder
   and just rerun the script.

Run:
    python -m knowledge.scripts.ingest_pdfs                # download + process
    python -m knowledge.scripts.ingest_pdfs --process-only # skip downloads
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx
import pymupdf  # PyMuPDF

from knowledge.loader import KNOWLEDGE_ROOT


PDF_DIR = KNOWLEDGE_ROOT / "raw" / "pdfs"
USER_AGENT = "deepeval-lab/0.1 (learning project; pdf ingestion)"


@dataclass(frozen=True)
class PdfSource:
    slug: str               # filename stem
    url: str
    title: str
    topic: str              # "law" | "tax" | "market" | ...
    lang: str = "es"


# Curated list. Each URL is *expected* to be stable, but the script is
# fault-tolerant if any one rots. Reference, not exhaustive: add more.
SOURCES: list[PdfSource] = [
    PdfSource(
        slug="boe-ley-14-2013-emprendedores",
        url="https://www.boe.es/boe/dias/2013/09/28/pdfs/BOE-A-2013-10074.pdf",
        title="Ley 14/2013, de apoyo a los emprendedores (texto original — Golden Visa)",
        topic="law",
        lang="es",
    ),
    PdfSource(
        slug="boe-ley-5-2019-credito-inmobiliario",
        url="https://www.boe.es/boe/dias/2019/03/16/pdfs/BOE-A-2019-3814.pdf",
        title="Ley 5/2019, reguladora de los contratos de crédito inmobiliario",
        topic="law",
        lang="es",
    ),
]


def download(src: PdfSource, timeout: float = 30.0) -> bool:
    """Download one PDF if not already cached. Returns True on success."""
    out = PDF_DIR / f"{src.slug}.pdf"
    if out.exists() and out.stat().st_size > 1024:
        print(f"  cached  {out.name}")
        return True

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with httpx.Client(
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=timeout,
        ) as client:
            r = client.get(src.url)
            r.raise_for_status()
        out.write_bytes(r.content)
        print(f"  fetched {out.name}  ({len(r.content) / 1024:.0f} KB)")
        return True
    except (httpx.HTTPError, OSError) as e:
        print(f"  FAIL    {src.slug}: {e}", file=sys.stderr)
        return False


def pdf_to_markdown(pdf_path: Path) -> str:
    """Extract text from a PDF as a single markdown string."""
    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                parts.append(f"## Page {i}\n\n{text}")
    return "\n\n".join(parts)


def process(pdf_path: Path, src: PdfSource | None) -> None:
    """Convert one PDF to .md with frontmatter, next to it."""
    md_path = pdf_path.with_suffix(".md")
    body = pdf_to_markdown(pdf_path)
    if not body:
        print(f"  EMPTY  {pdf_path.name} (no extractable text — likely scanned)")
        return

    if src is not None:
        title, topic, lang, url = src.title, src.topic, src.lang, src.url
    else:
        title = pdf_path.stem.replace("-", " ").title()
        topic, lang, url = "unknown", "es", ""

    frontmatter = (
        "---\n"
        f"title: {title}\n"
        f"topic: {topic}\n"
        f"lang: {lang}\n"
        "source: pdf\n"
        + (f"url: {url}\n" if url else "")
        + f"file: {pdf_path.name}\n"
        "---\n\n"
    )
    md_path.write_text(frontmatter + f"# {title}\n\n" + body, encoding="utf-8")
    print(f"  wrote   {md_path.name}  ({len(body):>7} chars)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the knowledge base.")
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip downloads; only convert PDFs already in raw/pdfs/.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="SLUG",
        help="Restrict download to specific slugs.",
    )
    args = parser.parse_args(argv)

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1 — downloads
    if not args.process_only:
        print("Downloads:")
        targets = SOURCES
        if args.only:
            targets = [s for s in SOURCES if s.slug in set(args.only)]
        for src in targets:
            download(src)
        if not targets:
            print("  (nothing matched)")

    # Stage 2 — process every PDF found, whether downloaded or user-provided
    print("\nProcessing:")
    by_slug = {s.slug: s for s in SOURCES}
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print("  (no PDFs in raw/pdfs/)")
        return 0
    for pdf in pdfs:
        process(pdf, by_slug.get(pdf.stem))

    print(f"\nDone. {len(pdfs)} PDF(s) in {PDF_DIR.relative_to(KNOWLEDGE_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
