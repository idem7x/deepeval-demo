"""
Pull a fixed list of Wikipedia articles relevant to Spanish real estate.

Writes one markdown file per article into `knowledge/raw/wikipedia/`, each
with YAML frontmatter so the standard loader picks them up. Re-running is
safe — existing files are overwritten.

Run manually:

    python -m knowledge.scripts.ingest_wikipedia

Or via Make:

    make ingest

Wikipedia content is CC-BY-SA, so we always keep the `url` in the
frontmatter and prepend an attribution line to the body.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wikipediaapi

from knowledge.loader import KNOWLEDGE_ROOT


# Topics we want indexed for the RAG demo. Each entry is the canonical
# Wikipedia page title; we hit `en.wikipedia.org` by default and fall back
# to the article if a redirect exists.
ARTICLES: list[tuple[str, str]] = [
    # (page title, our topic tag).
    # Titles verified to exist on en.wikipedia.org. Replace cautiously —
    # the script prints "MISS" but otherwise carries on, so a stale title
    # will silently shrink the corpus.
    ("Spanish property bubble", "market"),
    ("Economy of Spain", "market"),
    ("Costa del Sol", "region"),
    ("Costa Blanca", "region"),
    ("Costa Brava", "region"),
    ("Mallorca", "region"),
    ("Ibiza", "region"),
    ("Tenerife", "region"),
    ("Madrid", "region"),
    ("Barcelona", "region"),
    ("Valencia", "region"),
    ("Málaga", "region"),
    ("Marbella", "region"),
    ("Autonomous communities of Spain", "context"),
]

OUTPUT_DIR = KNOWLEDGE_ROOT / "raw" / "wikipedia"
USER_AGENT = "deepeval-lab/0.1 (https://github.com/-/deepeval-lab; learning project)"


def _slugify(title: str) -> str:
    """Wikipedia titles → safe filenames. Keeps it readable, ASCII-only."""
    out = []
    for ch in title.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
    return "".join(out).strip("-")


def fetch(title: str, lang: str = "en") -> tuple[str, str] | None:
    """Return (resolved_title, plain text body) for `title`, or None if missing."""
    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=lang)
    page = wiki.page(title)
    if not page.exists():
        return None
    return page.title, page.text


def write_markdown(title: str, body: str, topic: str, url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        "---\n"
        f"title: {title}\n"
        f"topic: {topic}\n"
        "lang: en\n"
        "source: wikipedia\n"
        f"url: {url}\n"
        "license: CC-BY-SA-4.0\n"
        "---\n\n"
    )
    attribution = (
        f"_Source: [Wikipedia, “{title}”]({url}), CC BY-SA 4.0._\n\n"
    )
    out.write_text(frontmatter + f"# {title}\n\n" + attribution + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles for RAG.")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="TITLE",
        help="Restrict to specific article titles (default: all in ARTICLES).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip articles that already have a local file (faster re-runs).",
    )
    args = parser.parse_args(argv)

    wanted = ARTICLES
    if args.only:
        wanted = [(t, topic) for t, topic in ARTICLES if t in set(args.only)]
        missing = set(args.only) - {t for t, _ in wanted}
        if missing:
            print(f"warning: unknown titles ignored: {sorted(missing)}", file=sys.stderr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    for title, topic in wanted:
        out = OUTPUT_DIR / f"{_slugify(title)}.md"
        if args.skip_existing and out.exists():
            print(f"  skip   {title}")
            continue

        result = fetch(title)
        if result is None:
            print(f"  MISS   {title}")
            continue
        resolved_title, body = result
        url = f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}"
        write_markdown(resolved_title, body, topic, url, out)
        print(f"  wrote  {out.relative_to(KNOWLEDGE_ROOT)}  ({len(body):>6} chars)")
        written += 1

    print(f"\nDone. {written} article(s) written to {OUTPUT_DIR.relative_to(KNOWLEDGE_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
