"""Build the index: PDFs -> blocks -> chunks -> embeddings -> Chroma.

Run once:  python ingest.py
"""
from __future__ import annotations

import argparse
import sys

from rag.chunk import chunk_corpus
from rag.config import settings
from rag.embed import get_embedder
from rag.extract import extract_corpus
from rag.logs import setup_logging
from rag.store import VectorStore


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the PDF corpus into Chroma.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Extract and chunk only; print chunks, no API calls.")
    ap.add_argument("--show", action="store_true", help="Print every chunk body.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show per-document extraction and chunking detail.")
    args = ap.parse_args()
    # Ingest logs at INFO by default: rejected tables and clipped lines are
    # things you want to see the moment they happen, not discover later.
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    print(f"Reading PDFs from {settings.pdf_dir}")
    blocks = extract_corpus(settings.pdf_dir)
    if not blocks:
        print("No PDFs found.", file=sys.stderr)
        return 1
    chunks = chunk_corpus(blocks)

    docs = sorted({c.doc for c in chunks})
    print(f"  {len(blocks)} pages -> {len(chunks)} chunks across {len(docs)} documents")
    print(f"  tokens: min {min(c.n_tokens for c in chunks)}, "
          f"avg {sum(c.n_tokens for c in chunks) // len(chunks)}, "
          f"max {max(c.n_tokens for c in chunks)}")
    flagged = [c for c in chunks if c.truncated]
    if flagged:
        print(f"  {len(flagged)} chunk(s) contain lines clipped in the source PDF:")
        for c in flagged:
            print(f"    - {c.doc} p.{c.page} | {c.section}")

    if args.show:
        for c in chunks:
            print("\n" + "-" * 70)
            print(c.embed_text)

    if args.dry_run:
        print("\nDry run: nothing embedded or stored.")
        return 0

    print(f"\nEmbedding with {settings.embedding_provider}:{settings.embedding_model}")
    embedder = get_embedder()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])

    store = VectorStore()
    store.reset()
    store.add(chunks, vectors)
    print(f"Stored {store.count()} chunks in {settings.chroma_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
