"""Ablation: does hybrid retrieval actually earn its complexity?

The README argues for BM25 + dense + RRF + reranking. That argument is worth
nothing unless the alternatives were measured, so this script runs the same
questions through four retrieval configurations and reports where each one
fails:

    1. BM25 only          lexical, no embeddings at all
    2. Dense only         embeddings, no lexical arm
    3. Hybrid (RRF)       both arms fused by rank, no reranker
    4. Hybrid + rerank    what actually ships

Retrieval-only: no answer is generated, so this is cheap and fast. Configs 1-3
cost nothing but a query embedding; only config 4 makes an LLM call.

The honest question this is designed to answer is not "is hybrid better" - it is
"is hybrid better ON THIS CORPUS, and if the difference is zero, say so."

Run:  python eval/ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.answer import RagPipeline          # noqa: E402
from rag.config import settings             # noqa: E402
from rag.schemas import Hit                 # noqa: E402

HERE = Path(__file__).resolve().parent
TOP_N = settings.rerank_top_n


def _rank_of(hits: list[Hit], expect_doc: str) -> int | None:
    for i, h in enumerate(hits, start=1):
        if h.chunk.doc == expect_doc:
            return i
    return None


def _chunk_rank(hits: list[Hit], expect_doc: str, needles: list[str]) -> int | None:
    """Rank of the specific CHUNK that actually contains the answer.

    Document-level Hit@k is close to meaningless on a 7-document corpus - it
    asks "did we find the right file", which is easy. The real question is
    whether the right PASSAGE outranked the other passages in that same file,
    which is where a retrieval design earns or loses its keep. Ground truth here
    is the chunk that literally contains the expected value.
    """
    if not needles:
        return None
    for i, h in enumerate(hits, start=1):
        if h.chunk.doc != expect_doc:
            continue
        text = h.chunk.text.lower()
        if all(n.lower() in text for n in needles):
            return i
    return None


def _metrics(ranks: list[int | None], n: int) -> dict:
    return {
        "hit1": sum(1 for r in ranks if r == 1) / n,
        "hit3": sum(1 for r in ranks if r and r <= 3) / n,
        "hit4": sum(1 for r in ranks if r and r <= TOP_N) / n,
        "mrr": sum((1 / r) if r else 0.0 for r in ranks) / n,
        "misses": sum(1 for r in ranks if r is None),
    }


def run(pipeline: RagPipeline, questions: list[dict]) -> dict:
    retr = pipeline.retriever
    names = ["BM25 only", "Dense only", "Hybrid (RRF)", "Hybrid + rerank"]
    results: dict[str, list[int | None]] = {k: [] for k in names}
    chunk_results: dict[str, list[int | None]] = {k: [] for k in names}
    per_question: list[dict] = []

    for q in questions:
        question, expect = q["question"], q["expect_doc"]

        bm25 = [Hit(chunk=c) for c in retr._lexical(question, settings.bm25_k)]
        dense = [Hit(chunk=c) for c in retr._dense(question, settings.dense_k)]
        fused = retr.fuse(question)
        reranked = retr.rerank(question, list(fused), client=pipeline.client)

        # Derived answers (arithmetic) have no chunk containing the value, so
        # they are excluded from chunk-level scoring rather than counted as
        # misses - the retrieval was correct, the ground truth is unmatchable.
        needles = [] if q.get("derived") else (q.get("expect_contains") or [])
        lists = {"BM25 only": bm25, "Dense only": dense,
                 "Hybrid (RRF)": fused, "Hybrid + rerank": reranked}
        row = {"question": question, "expect": expect, "needles": needles}
        for key, hl in lists.items():
            row[key] = _rank_of(hl, expect)
            row[f"chunk::{key}"] = _chunk_rank(hl, expect, needles)
        for key in results:
            results[key].append(row[key])
            if needles:
                chunk_results[key].append(row[f"chunk::{key}"])
        per_question.append(row)
        print(f"  {question[:52]:54} "
              + "  ".join(f"{k.split()[0][:5]}={str(row[k] or '-'):>2}" for k in results))

    n = len(questions)
    n_chunk = len(next(iter(chunk_results.values())))
    return {
        "metrics": {k: _metrics(v, n) for k, v in results.items()},
        "chunk_metrics": {k: _metrics(v, max(n_chunk, 1)) for k, v in chunk_results.items()},
        "rows": per_question,
        "n": n,
        "n_chunk": n_chunk,
    }


def main() -> int:
    questions = [
        q for q in yaml.safe_load((HERE / "questions.yaml").read_text(encoding="utf-8"))
        if q["answerable"] and q.get("expect_doc")
    ]
    print(f"Ablation over {len(questions)} answerable questions "
          f"(rank of the expected document; '-' = not retrieved at all)\n")

    pipeline = RagPipeline()
    out = run(pipeline, questions)

    print("\n" + "=" * 78)
    print(f"{'configuration':20} {'Hit@1':>7} {'Hit@3':>7} {'Hit@4':>7} {'MRR':>7} {'misses':>8}")
    print("-" * 78)
    for name, m in out["metrics"].items():
        print(f"{name:20} {m['hit1']:6.0%} {m['hit3']:6.0%} {m['hit4']:6.0%} "
              f"{m['mrr']:7.2f} {m['misses']:8}")
    print("=" * 78)

    print(f"\nCHUNK-level: rank of the passage that actually contains the answer "
          f"({out['n_chunk']} questions with a known value)")
    print(f"{'configuration':20} {'Hit@1':>7} {'Hit@3':>7} {'Hit@4':>7} {'MRR':>7} {'misses':>8}")
    print("-" * 78)
    for name, m in out["chunk_metrics"].items():
        print(f"{name:20} {m['hit1']:6.0%} {m['hit3']:6.0%} {m['hit4']:6.0%} "
              f"{m['mrr']:7.2f} {m['misses']:8}")
    print("=" * 78)

    # Where does each single-arm retriever fail that the hybrid does not?
    print("\nQuestions a single-arm retriever misses entirely:")
    any_found = False
    for row in out["rows"]:
        lost = [k for k in ("BM25 only", "Dense only") if row[k] is None]
        if lost:
            any_found = True
            print(f"  {row['question'][:60]}")
            print(f"     missed by: {', '.join(lost)}  |  hybrid rank: {row['Hybrid (RRF)']}")
    if not any_found:
        print("  none - both arms find every expected document on this corpus.")

    # Where reranking changes the outcome.
    moved = [r for r in out["rows"] if r["Hybrid (RRF)"] != r["Hybrid + rerank"]]
    print(f"\nReranking changed the rank of the expected document on "
          f"{len(moved)}/{out['n']} questions:")
    for r in moved:
        print(f"  {r['question'][:56]:58} {r['Hybrid (RRF)']} -> {r['Hybrid + rerank']}")

    write_report(out)
    return 0


def write_report(out: dict) -> None:
    lines = ["# Retrieval Ablation\n",
             f"Rank of the expected source document across {out['n']} answerable "
             "questions. Lower is better; `-` means the document was never retrieved.\n",
             "\n## Summary\n",
             "| Configuration | Hit@1 | Hit@3 | Hit@4 | MRR | Never retrieved |",
             "|---|---|---|---|---|---|"]
    for name, m in out["metrics"].items():
        lines.append(f"| {name} | {m['hit1']:.0%} | {m['hit3']:.0%} | {m['hit4']:.0%} "
                     f"| {m['mrr']:.2f} | {m['misses']} |")

    lines += ["\n## Chunk-level — rank of the passage containing the answer\n",
              "Document-level Hit@k is easy on a 7-document corpus. This asks the harder "
              "question: did the right *passage* outrank the other passages in the same "
              f"file? ({out['n_chunk']} questions with a known expected value.)\n",
              "| Configuration | Hit@1 | Hit@3 | Hit@4 | MRR | Never retrieved |",
              "|---|---|---|---|---|---|"]
    for name, m in out["chunk_metrics"].items():
        lines.append(f"| {name} | {m['hit1']:.0%} | {m['hit3']:.0%} | {m['hit4']:.0%} "
                     f"| {m['mrr']:.2f} | {m['misses']} |")

    lines += ["\n## Per question (document-level rank)\n",
              "| Question | Expected document | BM25 | Dense | Hybrid | +rerank |",
              "|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(
            f"| {r['question']} | {r['expect']} | {r['BM25 only'] or '-'} | "
            f"{r['Dense only'] or '-'} | {r['Hybrid (RRF)'] or '-'} | "
            f"{r['Hybrid + rerank'] or '-'} |"
        )
    (HERE / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {HERE / 'ablation.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
