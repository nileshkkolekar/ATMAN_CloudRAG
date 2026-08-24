"""Terminal Q&A - the fastest way to sanity-check the pipeline.

  python cli.py "How much PTO do I accrue?"
  python cli.py                       # interactive
"""
from __future__ import annotations

import sys

from rag.answer import RagPipeline
from rag.config import settings
from rag.logs import LOG_DIR, setup_logging


def show(result) -> None:
    print("\n" + ("=" * 72))
    print(result.text)
    if result.citations:
        print("\nSources")
        for c in result.citations:
            sec = f" - {c.section}" if c.section else ""
            trunc = "   (clipped in source PDF)" if c.truncated else ""
            print(f"  [{c.marker}] {c.doc}, page {c.page}{sec}{trunc}")
    else:
        print(f"\n  (refused: {result.refusal_reason}, top score {result.top_score:.1f})")
    for w in result.warnings:
        print(f"  ! {w}")
    print(f"\n  {result.latency_ms} ms")
    print("=" * 72)


def main() -> int:
    # -v shows each pipeline stage; -vv adds every BM25/dense/rerank score.
    argv = [a for a in sys.argv[1:] if a not in ("-v", "--verbose", "-vv", "--debug")]
    level = "DEBUG" if {"-vv", "--debug"} & set(sys.argv) else (
        "INFO" if {"-v", "--verbose"} & set(sys.argv) else "WARNING")
    setup_logging(level=level)

    if not settings.has_key:
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    pipeline = RagPipeline()
    print(f"Indexed {len(pipeline.retriever.chunks)} chunks from "
          f"{len({c.doc for c in pipeline.retriever.chunks})} documents.")

    print(f"Full trace of every query -> {LOG_DIR / 'traces.jsonl'}")

    if argv:
        show(pipeline.answer(" ".join(argv)))
        return 0

    print("Ask a question (blank line or Ctrl-C to quit).\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        show(pipeline.answer(q))


if __name__ == "__main__":
    raise SystemExit(main())
