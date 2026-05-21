"""
Build / rebuild the ChromaDB index from everything under knowledge/.

This is the script wired to `make ingest`.

It is intentionally idempotent: re-running just upserts (same chunk IDs
overwrite their previous rows). Switching `EMBEDDING_PROVIDER` between
runs is also safe — the store namespaces collections by provider, so a
fresh provider starts with an empty collection rather than mixing
incompatible vector spaces.

Run:
    python -m scripts.seed_chroma
    python -m scripts.seed_chroma --reset       # drop collection first
    python -m scripts.seed_chroma --chunk 1200 --overlap 150
"""

from __future__ import annotations

import argparse
import time

from apps.backend.rag.store import ChromaStore
from knowledge.loader import load_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed ChromaDB from knowledge/.")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection before ingesting.")
    parser.add_argument("--chunk", type=int, default=800,
                        help="Target chunk size in tokens (default 800).")
    parser.add_argument("--overlap", type=int, default=100,
                        help="Inter-chunk overlap in tokens (default 100).")
    args = parser.parse_args(argv)

    docs = load_all()
    if not docs:
        print("No documents found. Run the ingest scripts first.")
        return 1

    store = ChromaStore.default()
    print(f"Embedder : {store.embedder.name}  (dim={store.embedder.dim})")
    print(f"Collection: {store.stats()['collection']}")
    print(f"Docs     : {len(docs)}")

    if args.reset:
        print("Resetting collection...")
        store.reset()

    t0 = time.perf_counter()
    written = store.ingest(docs, chunk_tokens=args.chunk, overlap_tokens=args.overlap)
    dt = time.perf_counter() - t0

    stats = store.stats()
    print(
        f"\nIngested {written} chunk(s) in {dt:.1f}s. "
        f"Collection now holds {stats['count']} chunk(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
