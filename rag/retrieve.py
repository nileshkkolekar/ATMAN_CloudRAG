"""Hybrid retrieval: BM25 + dense, fused by RRF, then reranked.

TWO RETRIEVERS, because this corpus asks two kinds of question and neither
retriever handles both:

  Dense wins on "how long do I have to report a breach?" - which shares almost
  no vocabulary with the sentence that answers it ("must be reported to
  security@atmancloud.com within 1 hour of discovery").

  BM25 wins on "what is SEC-POL-007?" and on literals like $0.08/GB or HTTP 429.
  Embedding models are unreliable on alphanumeric identifiers; BM25 matches them
  exactly and instantly. Every document in this corpus carries such a code.

FUSION BY RRF, NOT SCORE BLENDING. Cosine similarity and BM25 scores live on
incomparable scales, so a weighted sum needs a weight that must be recalibrated
whenever the corpus changes. RRF only reads ranks, so there is nothing to tune.

RERANKING because retrieval alone cannot resolve this corpus's central
ambiguity: the word "Standard" carries four unrelated senses across five
documents (a pricing tier, a rate-limit tier, an employee access level, and a
health plan). The Security Policy even flags the collision itself. A reranker
reading the full question against the full passage separates them; a similarity
score does not.
"""
from __future__ import annotations

import json
import re
import time

from rank_bm25 import BM25Okapi

from .config import settings
from .embed import Embedder, get_embedder
from .logs import get_logger
from .schemas import Chunk, Hit
from .store import VectorStore

log = get_logger("retrieve")

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\./$%@_]*")


def tokenize(text: str) -> list[str]:
    """Keep the punctuation that carries meaning here.

    A default \w+ tokenizer shatters exactly the tokens BM25 is here to catch:
    SEC-POL-007 becomes ['sec','pol','007'], $0.08/GB becomes ['0','08','gb'],
    and 99.95% becomes ['99','95']. Splitting on those destroys the one
    advantage lexical search has over embeddings.
    """
    return TOKEN_RE.findall(text.lower())


def _rrf(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


class Retriever:
    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.store = store or VectorStore()
        self._embedder = embedder
        self.chunks = self.store.all_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        # BM25 indexes the header-prefixed text, so the document code and
        # section title are exact-matchable on every chunk of that document.
        self._bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # -- stage 1: two independent candidate lists -------------------------
    def _lexical(self, question: str, k: int) -> list[Chunk]:
        toks = tokenize(question)
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [self.chunks[i] for i in order[:k] if scores[i] > 0]
        log.debug("bm25 tokens=%s", toks)
        for rank, i in enumerate(order[:k], start=1):
            if scores[i] > 0:
                log.debug("  bm25 #%d %.3f %s p%d %s", rank, scores[i],
                          self.chunks[i].doc, self.chunks[i].page,
                          self.chunks[i].section[:40])
        if not out:
            log.debug("bm25 matched nothing - dense arm is carrying this query")
        return out

    def _dense(self, question: str, k: int) -> list[Chunk]:
        started = time.perf_counter()
        vector = self.embedder.embed_query(question)
        log.debug("embedded query in %dms", int((time.perf_counter() - started) * 1000))
        out = self.store.query(vector, k)
        for rank, c in enumerate(out, start=1):
            log.debug("  dense #%d %s p%d %s", rank, c.doc, c.page, c.section[:40])
        return out

    # -- stage 2: fuse ----------------------------------------------------
    def fuse(self, question: str) -> list[Hit]:
        lexical = self._lexical(question, settings.bm25_k)
        dense = self._dense(question, settings.dense_k)

        hits: dict[str, Hit] = {}
        for rank, chunk in enumerate(dense, start=1):
            hit = hits.setdefault(chunk.id, Hit(chunk=chunk))
            hit.dense_rank = rank
            hit.rrf_score += _rrf(rank, settings.rrf_k)
        for rank, chunk in enumerate(lexical, start=1):
            hit = hits.setdefault(chunk.id, Hit(chunk=chunk))
            hit.bm25_rank = rank
            hit.rrf_score += _rrf(rank, settings.rrf_k)

        fused = sorted(hits.values(), key=lambda h: h.rrf_score, reverse=True)
        both = sum(1 for h in fused if h.bm25_rank and h.dense_rank)
        log.info("fused %d candidates (%d lexical, %d dense, %d found by both)",
                 len(fused), len(lexical), len(dense), both)
        return fused

    # -- stage 3: rerank --------------------------------------------------
    def rerank(self, question: str, hits: list[Hit], client=None, trace=None) -> list[Hit]:
        """One batched LLM call scoring every candidate 0-10.

        One call does two jobs: it orders the candidates, and its top score is
        the calibrated relevance signal the refusal gate reads. A cross-encoder
        (bge-reranker-base) is the right swap above ~30 candidates, where LLM
        reranking hits position bias and linear cost growth - but it reintroduces
        a 280MB local model for no gain at this size.
        """
        if not hits:
            return hits
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)

        listing = "\n\n".join(
            f"[{i}] {h.chunk.header}\n{h.chunk.text[:900]}"
            for i, h in enumerate(hits)
        )
        prompt = (
            "Score how well each passage answers the question, 0-10.\n"
            "10 = contains the complete answer. 7-9 = contains most of it. "
            "4-6 = related topic, partial or supporting information. "
            "1-3 = same document or vocabulary but does not address the question. "
            "0 = irrelevant.\n"
            "Judge only whether the passage ANSWERS THIS QUESTION. A passage about a "
            "different product model, a different plan tier, or a different meaning of "
            "the same word scores low even though it looks similar.\n\n"
            f"Question: {question}\n\nPassages:\n{listing}\n\n"
            'Reply with JSON only: {"scores": [{"id": 0, "score": 7}, ...]} '
            "covering every passage id."
        )
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            data = json.loads(resp.choices[0].message.content)
            for row in data.get("scores", []):
                idx = int(row["id"])
                if 0 <= idx < len(hits):
                    hits[idx].rerank_score = float(row["score"])
            if trace is not None:
                trace.record_llm("rerank", settings.llm_model,
                                 int((time.perf_counter() - started) * 1000), resp)
        except Exception as exc:
            # Degrade to fusion order rather than failing the query. The gate
            # then reads rrf_score, which is why Hit.score falls back cleanly.
            log.warning("rerank failed (%s) - falling back to RRF order", exc)
            if trace is not None:
                trace.record_llm("rerank", settings.llm_model,
                                 int((time.perf_counter() - started) * 1000),
                                 error=str(exc))
            return hits

        for h in hits:
            if h.rerank_score is None:
                h.rerank_score = 0.0
        ranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        for h in ranked:
            log.debug("  rerank %4.1f  %s p%d %s", h.rerank_score, h.chunk.doc,
                      h.chunk.page, h.chunk.section[:40])
        return ranked

    def retrieve(self, question: str, client=None, trace=None) -> list[Hit]:
        fused = self.fuse(question)
        ranked = self.rerank(question, fused, client=client, trace=trace)
        if trace is not None:
            trace.record_candidates(ranked)          # every candidate, not just shown
        shown = ranked[: settings.rerank_top_n]
        log.info("showing top %d: %s", len(shown),
                 ", ".join(f"{h.chunk.doc}:p{h.chunk.page}({h.score:.1f})" for h in shown))
        return shown
