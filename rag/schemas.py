"""Typed records that flow through the pipeline.

Everything downstream depends on provenance (doc / page / section) surviving
from extraction all the way to the rendered citation, so provenance lives on
the dataclass rather than in a parallel dict that can drift out of sync.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """One extracted unit of a page, before chunking."""

    doc: str                 # "Employee_Handbook.pdf"
    doc_title: str           # "Employee Handbook"
    doc_code: str            # "HR-EH-2026"
    page: int                # 1-indexed, as printed in a citation
    text: str
    kind: str = "prose"      # prose | table | qa
    section: str = ""        # "3.1 Health Insurance"
    truncated: bool = False  # a source line was clipped by the PDF generator


@dataclass
class Chunk:
    """An indexed unit: what gets embedded, stored and cited."""

    id: str
    doc: str
    doc_title: str
    doc_code: str
    page: int
    section: str
    text: str                # the raw body, shown to the user
    kind: str = "prose"
    truncated: bool = False
    n_tokens: int = 0

    @property
    def header(self) -> str:
        """Provenance line prepended before embedding (Decision 3)."""
        head = f"[{self.doc_title} - {self.doc_code}]"
        return f"{head} > {self.section}" if self.section else head

    @property
    def embed_text(self) -> str:
        """What the embedder AND BM25 both see."""
        return f"{self.header}\n{self.text}"

    def metadata(self) -> dict[str, Any]:
        return {
            "doc": self.doc,
            "doc_title": self.doc_title,
            "doc_code": self.doc_code,
            "page": self.page,
            "section": self.section,
            "kind": self.kind,
            "truncated": self.truncated,
            "n_tokens": self.n_tokens,
        }


@dataclass
class Hit:
    """A retrieved candidate, carrying the scores that produced it."""

    chunk: Chunk
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """The relevance signal the refusal gate reads."""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    def source_label(self) -> str:
        loc = f"{self.chunk.doc} p.{self.chunk.page}"
        return f"{loc} - {self.chunk.section}" if self.chunk.section else loc


@dataclass
class Citation:
    marker: int              # the [1] the model wrote
    doc: str
    page: int
    section: str
    text: str
    truncated: bool = False


@dataclass
class Answer:
    question: str
    text: str
    answered: bool                       # False = refused
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    refusal_reason: str | None = None    # low_relevance | insufficient_context | ungrounded
    top_score: float = 0.0
    warnings: list[str] = field(default_factory=list)
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "answered": self.answered,
            "refusal_reason": self.refusal_reason,
            "top_score": round(self.top_score, 2),
            "warnings": self.warnings,
            "latency_ms": self.latency_ms,
            "sources": [
                {
                    "marker": c.marker,
                    "document": c.doc,
                    "page": c.page,
                    "section": c.section,
                    "truncated_in_source": c.truncated,
                    "chunk": c.text,
                }
                for c in self.citations
            ],
        }
