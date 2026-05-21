"""
Print a summary of every document the loader sees.

Useful as a quick sanity check before running the Phase 2 ChromaDB indexer:
if a document doesn't show up here, it won't show up in RAG either.

Run:
    python -m scripts.list_knowledge
    python -m scripts.list_knowledge --by source
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from knowledge.loader import load_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List knowledge base documents.")
    parser.add_argument(
        "--by",
        choices=["source", "topic", "lang"],
        help="Group output by source / topic / lang instead of listing flat.",
    )
    args = parser.parse_args(argv)

    docs = load_all()
    if not docs:
        print("No documents found. Run `make ingest` first.")
        return 1

    if args.by:
        groups: dict[str, list] = defaultdict(list)
        for d in docs:
            key = getattr(d, args.by) or "(none)"
            groups[key].append(d)
        for key in sorted(groups):
            items = groups[key]
            total_chars = sum(len(d.text) for d in items)
            print(f"\n[{args.by}={key}]  {len(items)} doc(s), {total_chars:>8} chars")
            for d in items:
                print(f"    {d.id:55s}  {len(d.text):>7} chars  {d.title}")
    else:
        for d in docs:
            print(
                f"[{d.source:>10s}] [{d.topic:14s}] {d.id:55s}  "
                f"{len(d.text):>7} chars  {d.title}"
            )

    total = sum(len(d.text) for d in docs)
    print(f"\nTotal: {len(docs)} document(s), {total:,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
