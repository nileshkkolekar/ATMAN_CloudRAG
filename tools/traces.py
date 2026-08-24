"""Inspect the query traces written to logs/traces.jsonl.

The debugging workflow this is built for:

  1. A query returns a bad answer or an unexpected refusal.
  2. `python tools/traces.py` - find it in the list, note its trace id.
  3. `python tools/traces.py <id>` - see EVERY candidate considered, with its
     BM25 rank, dense rank, RRF score and rerank score; which gate fired and
     why; and what each LLM call cost.

That is usually enough to tell which of the three real failure modes it is:

  RETRIEVAL   the right chunk is missing from the candidate list entirely
              -> chunking or embedding problem, not a model problem
  RANKING     the right chunk is present but scored below the ones shown
              -> reranker prompt, or k too small
  GENERATION  the right chunk was shown and the answer still went wrong
              -> the answer prompt, or a genuinely ambiguous source

Usage:
  python tools/traces.py                 # recent queries, one line each
  python tools/traces.py <trace_id>      # full detail for one query
  python tools/traces.py --refused       # only refusals
  python tools/traces.py --cost          # spend summary
  python tools/traces.py --grep "PTO"    # questions matching a substring
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "logs" / "traces.jsonl"


def load() -> list[dict]:
    if not TRACES.exists():
        print(f"No traces yet at {TRACES}\nRun a query first: python cli.py \"...\"")
        raise SystemExit(1)
    out = []
    for line in TRACES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def summarise(records: list[dict]) -> None:
    print(f"{'trace':9} {'outcome':9} {'gate':22} {'top':>5} {'ms':>6} {'cost':>9}  question")
    print("-" * 110)
    for r in records:
        outcome = "ANSWERED" if r.get("answered") else "REFUSED"
        print(f"{r['trace_id']:9} {outcome:9} {str(r.get('gate')):22} "
              f"{r.get('top_score', 0):5.1f} {r.get('elapsed_ms', 0):6} "
              f"${r.get('cost_usd', 0):8.5f}  {r['question'][:44]}")


def detail(r: dict) -> None:
    print("=" * 100)
    print(f"trace {r['trace_id']}   {r['question']}")
    print("=" * 100)
    outcome = "ANSWERED" if r.get("answered") else "REFUSED"
    print(f"  outcome     {outcome}")
    print(f"  gate        {r.get('gate')}")
    if r.get("gate_detail"):
        print(f"  why         {r['gate_detail']}")
    print(f"  top score   {r.get('top_score', 0)}")
    print(f"  elapsed     {r.get('elapsed_ms', 0)} ms")
    print(f"  cost        ${r.get('cost_usd', 0):.5f}")
    for w in r.get("warnings", []):
        print(f"  WARNING     {w}")

    print("\n  LLM calls")
    for c in r.get("llm_calls", []):
        err = f"  ERROR: {c['error']}" if c.get("error") else ""
        print(f"    {c['stage']:14} {c['model']:22} {c['latency_ms']:5}ms  "
              f"{c['prompt_tokens']:5}+{c['completion_tokens']:<5}tok  "
              f"${c['cost_usd']:.5f}{err}")

    shown = set(r.get("shown", []))
    print(f"\n  Candidates considered ({len(r.get('candidates', []))}) "
          f"- '>' marks the ones shown to the answering model")
    print(f"    {'':2}{'rerank':>7} {'rrf':>8} {'bm25':>5} {'dense':>6}  source")
    for c in r.get("candidates", []):
        mark = ">" if c.get("id") in shown else " "
        trunc = " [clipped]" if c.get("truncated") else ""
        print(f"    {mark} {str(c.get('rerank')):>7} {c.get('rrf', 0):8.5f} "
              f"{str(c.get('bm25_rank') or '-'):>5} {str(c.get('dense_rank') or '-'):>6}  "
              f"{c['doc']} p{c['page']} | {c['section'][:38]}{trunc}")
        print(f"      {'':28}{c['preview'][:74]}")

    if r.get("answer_preview"):
        print(f"\n  Answer\n    {r['answer_preview'][:600]}")


def main() -> int:
    records = load()
    args = sys.argv[1:]

    if not args:
        summarise(records[-25:])
        print(f"\n{len(records)} trace(s) total. Detail: python tools/traces.py <trace_id>")
        return 0

    if args[0] == "--refused":
        summarise([r for r in records if not r.get("answered")])
        return 0

    if args[0] == "--grep" and len(args) > 1:
        needle = args[1].lower()
        summarise([r for r in records if needle in r["question"].lower()])
        return 0

    if args[0] == "--cost":
        total = sum(r.get("cost_usd", 0) for r in records)
        by_stage: dict[str, float] = {}
        for r in records:
            for c in r.get("llm_calls", []):
                by_stage[c["stage"]] = by_stage.get(c["stage"], 0) + c["cost_usd"]
        print(f"{len(records)} queries, ${total:.4f} total, "
              f"${total / max(len(records), 1):.5f} per query\n")
        for stage, amount in sorted(by_stage.items(), key=lambda x: -x[1]):
            print(f"  {stage:14} ${amount:.5f}  ({amount / max(total, 1e-9):.0%})")
        return 0

    match = [r for r in records if r["trace_id"].startswith(args[0])]
    if not match:
        print(f"No trace starting with {args[0]!r}")
        return 1
    detail(match[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
