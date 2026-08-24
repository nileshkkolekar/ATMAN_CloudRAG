"""Logging and per-query tracing.

Two outputs, because they answer different questions:

  CONSOLE  Human-readable, one line per pipeline stage. Answers "what is it
           doing right now" while you watch a query run.

  TRACE    One JSON object per query appended to logs/traces.jsonl, holding the
           COMPLETE record: every candidate with its BM25 rank, dense rank, RRF
           score and rerank score; which gate fired and why; every LLM call with
           its latency, token counts and cost. Answers "why did THAT query do
           THAT", days later, without reproducing it.

The trace is the one that matters for debugging RAG. A bad answer is almost
never a bad model - it is the wrong chunk retrieved, a gate firing at the wrong
threshold, or a chunk that looks right but says something subtly different. All
three are invisible in the answer text and obvious in the trace.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))

# The id that ties every log line and every trace record to one question.
_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")

# gpt-4o-mini and text-embedding-3-small, USD per 1M tokens.
PRICING = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = PRICING.get(model)
    if not rate:
        return 0.0
    return (prompt_tokens * rate["in"] + completion_tokens * rate["out"]) / 1_000_000


class _TraceFilter(logging.Filter):
    """Stamp every record with the current query's trace id."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace = _trace_id.get()
        return True


def setup_logging(level: str | None = None, log_dir: Path | None = None) -> None:
    """Idempotent - safe to call from the CLI, the API and Streamlit alike."""
    root = logging.getLogger("rag")
    if root.handlers:
        return

    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-5s [%(trace)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    console.addFilter(_TraceFilter())
    root.addHandler(console)

    directory = Path(log_dir or LOG_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # DEBUG always goes to the file even when the console is at INFO, so a
        # bug that only shows up in production is still fully recorded.
        detail = logging.FileHandler(directory / "rag.log", encoding="utf-8")
        detail.setLevel(logging.DEBUG)
        detail.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-5s [%(trace)s] %(name)s: %(message)s")
        )
        detail.addFilter(_TraceFilter())
        root.addHandler(detail)
        root.setLevel(min(getattr(logging, level, logging.INFO), logging.DEBUG))
        console.setLevel(level)
    except OSError:
        pass  # read-only filesystem: console logging still works


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"rag.{name}")


@dataclass
class LLMCall:
    stage: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


@dataclass
class QueryTrace:
    """The complete record of one question.

    Written as a single JSON line so `jq` can slice it: every candidate that was
    considered (not just the ones shown), the gate decision, and the cost.
    """

    question: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.perf_counter)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    shown: list[str] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    gate: str | None = None          # which gate decided the outcome
    gate_detail: str | None = None
    answered: bool | None = None
    top_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    answer_preview: str = ""

    def __enter__(self) -> QueryTrace:
        self._token = _trace_id.set(self.trace_id)
        get_logger("query").info("Q: %s", self.question)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        log = get_logger("query")
        if exc:
            self.gate, self.gate_detail = "exception", f"{exc_type.__name__}: {exc}"
            log.exception("query failed: %s", exc)
        self.write()
        _trace_id.reset(self._token)
        return False

    # -- recording ------------------------------------------------------
    def record_llm(self, stage: str, model: str, latency_ms: int, resp=None,
                   error: str | None = None) -> None:
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        call = LLMCall(stage, model, latency_ms, pt, ct, cost_usd(model, pt, ct), error)
        self.llm_calls.append(call)
        log = get_logger("llm")
        if error:
            log.warning("%s failed after %dms: %s", stage, latency_ms, error)
        else:
            log.debug("%s %s %dms  %d+%d tok  $%.5f",
                      stage, model, latency_ms, pt, ct, call.cost_usd)

    def record_candidates(self, hits) -> None:
        self.candidates = [
            {
                "id": h.chunk.id,
                "doc": h.chunk.doc,
                "page": h.chunk.page,
                "section": h.chunk.section,
                "bm25_rank": h.bm25_rank,
                "dense_rank": h.dense_rank,
                "rrf": round(h.rrf_score, 5),
                "rerank": h.rerank_score,
                "truncated": h.chunk.truncated,
                "preview": h.chunk.text[:120].replace("\n", " "),
            }
            for h in hits
        ]

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def write(self) -> None:
        record = {
            "trace_id": self.trace_id,
            "question": self.question,
            "answered": self.answered,
            "gate": self.gate,
            "gate_detail": self.gate_detail,
            "top_score": self.top_score,
            "elapsed_ms": self.elapsed_ms,
            "cost_usd": round(self.total_cost, 6),
            "shown": self.shown,
            "warnings": self.warnings,
            "answer_preview": self.answer_preview[:400],
            "llm_calls": [c.__dict__ for c in self.llm_calls],
            "candidates": self.candidates,
        }
        get_logger("query").info(
            "%s | gate=%s top=%.1f | %dms | $%.5f",
            "ANSWERED" if self.answered else "REFUSED",
            self.gate, self.top_score, self.elapsed_ms, self.total_cost,
        )
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (LOG_DIR / "traces.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # never let a logging failure break a query
