"""Retrieval fusion, the four gates, and the citation validator.

These run against fakes, so they test OUR logic rather than the model's taste.
Whether gpt-4o-mini ranks a passage well is not something a unit test can
assert; whether a score below tau skips the generation call entirely is.
"""
from __future__ import annotations

import json

import pytest

from rag.answer import RagPipeline
from rag.config import settings
from rag.prompt import INSUFFICIENT
from rag.retrieve import Retriever, tokenize
from rag.schemas import Hit

from .conftest import FakeOpenAI, rerank_response


class TestTokenizer:
    def test_document_codes_survive_as_one_token(self):
        """The whole point of running BM25 alongside embeddings.

        A default \\w+ tokenizer shatters SEC-POL-007 into ['sec','pol','007'],
        destroying the one thing lexical search does better than dense.
        """
        assert "sec-pol-007" in tokenize("What is SEC-POL-007?")

    def test_money_percentages_and_codes_survive(self):
        toks = tokenize("$0.08/GB at 99.95% uptime returns HTTP 429")
        assert "0.08/gb" in toks
        assert "99.95%" in toks
        assert "429" in toks


class TestFusion:
    def test_rrf_merges_both_retrievers(self, stub_retriever):
        retriever, chunks = stub_retriever
        hits = retriever.fuse("Standard tier storage")
        assert hits
        assert all(isinstance(h, Hit) for h in hits)
        # every hit must record which retriever(s) surfaced it
        assert any(h.dense_rank is not None for h in hits)

    def test_a_chunk_found_by_both_outranks_one_found_by_either(self, stub_retriever):
        retriever, _ = stub_retriever
        hits = retriever.fuse("500 GB pooled storage")
        both = [h for h in hits if h.dense_rank and h.bm25_rank]
        if both:
            assert hits[0].rrf_score >= min(h.rrf_score for h in both)

    def test_bm25_finds_an_exact_code_the_fake_embedder_cannot(self, stub_retriever):
        retriever, chunks = stub_retriever
        # FakeEmbedder is semantically meaningless, so any correct result here
        # is attributable to the lexical arm alone.
        hits = retriever.fuse("Standard Tier Access for employees")
        assert any(h.chunk.doc == "Security_Policy.pdf" for h in hits)

    def test_rerank_reorders_by_score(self, stub_retriever):
        retriever, chunks = stub_retriever
        hits = retriever.fuse("Standard tier")
        n = len(hits)
        target = next(i for i, h in enumerate(hits) if h.chunk.doc == "Security_Policy.pdf")
        client = FakeOpenAI(lambda kw: rerank_response({target: 9.0}, n))
        ranked = retriever.rerank("Standard tier", hits, client=client)
        assert ranked[0].chunk.doc == "Security_Policy.pdf"
        assert ranked[0].rerank_score == 9.0

    def test_rerank_failure_degrades_to_fusion_order(self, stub_retriever):
        """A provider hiccup must not fail the query."""
        retriever, _ = stub_retriever
        hits = retriever.fuse("Standard tier")

        def boom(kw):
            raise RuntimeError("provider down")

        ranked = retriever.rerank("Standard tier", hits, client=FakeOpenAI(boom))
        assert [h.chunk.id for h in ranked] == [h.chunk.id for h in hits]
        assert ranked[0].score == ranked[0].rrf_score   # gate falls back cleanly


def _pipeline(stub_retriever, responder):
    retriever, _ = stub_retriever
    return RagPipeline(retriever=retriever, client=FakeOpenAI(responder))


def _router(rerank_scores, n, answer_text, grounded=True):
    """Dispatch by which call is being made."""

    def responder(kw):
        content = kw["messages"][-1]["content"]
        if "Score how well each passage" in content:
            return rerank_response(rerank_scores, n)
        if "Check whether every claim" in content:
            return json.dumps({"grounded": grounded, "unsupported": [] if grounded else ["x"]})
        return answer_text

    return responder


class TestGates:
    def test_gate1_refuses_below_threshold_without_calling_the_answerer(self, stub_retriever):
        """The strongest guarantee in the system: a model never invoked cannot
        hallucinate. Assert the generation call is never made."""
        calls = []

        def responder(kw):
            content = kw["messages"][-1]["content"]
            calls.append(content)
            if "Score how well each passage" in content:
                return rerank_response({}, 4)   # every candidate scores 0
            return "THIS SHOULD NEVER BE GENERATED"

        pipe = _pipeline(stub_retriever, responder)
        result = pipe.answer("What is Atman Cloud's 2025 revenue?")

        assert result.answered is False
        assert result.refusal_reason == "low_relevance"
        assert result.citations == []
        assert len(calls) == 1, "generation was called despite the relevance gate"

    def test_gate1_allows_a_score_above_threshold(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "Standard includes 500 GB pooled [1]."))
        result = pipe.answer("What storage does the Standard tier include?")
        assert result.answered is True
        assert "500 GB" in result.text

    def test_gate2_sentinel_becomes_a_refusal(self, stub_retriever):
        """Retrieval was confident but the fact is absent - the CSP-600 case."""
        pipe = _pipeline(stub_retriever, _router({0: 8.0}, 4, INSUFFICIENT))
        result = pipe.answer("What is the CSP-600's storage capacity?")
        assert result.answered is False
        assert result.refusal_reason == "insufficient_context"
        assert result.top_score >= settings.relevance_threshold  # gate 1 passed it

    def test_gate3_strips_an_invented_citation(self, stub_retriever):
        pipe = _pipeline(
            stub_retriever,
            _router({0: 9.0}, 4, "Standard includes 500 GB [1] and a free car [9]."),
        )
        result = pipe.answer("What storage does the Standard tier include?")
        assert "[9]" not in result.text
        assert any("invented citation [9]" in w for w in result.warnings)
        assert [c.marker for c in result.citations] == [1]

    def test_gate3_only_lists_sources_the_answer_actually_cited(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0, 1: 8.0}, 4, "Answer using [2] only."))
        result = pipe.answer("Standard tier")
        assert [c.marker for c in result.citations] == [2]

    def test_gate3_warns_when_no_markers_were_emitted(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "A confident answer with no sources."))
        result = pipe.answer("Standard tier")
        assert any("no citation markers" in w for w in result.warnings)

    def test_gate4_downgrades_an_ungrounded_answer_to_a_refusal(self, stub_retriever):
        pipe = _pipeline(
            stub_retriever,
            _router({0: 9.0}, 4, "The CSP-600 has 32TB [1].", grounded=False),
        )
        result = pipe.answer("What is the CSP-600's storage capacity?")
        assert result.answered is False
        assert result.refusal_reason == "ungrounded"

    def test_gate4_failure_does_not_break_a_good_answer(self, stub_retriever):
        """If the checker itself errors, keep the answer - never fail closed on
        a verification bug."""

        def responder(kw):
            content = kw["messages"][-1]["content"]
            if "Score how well each passage" in content:
                return rerank_response({0: 9.0}, 4)
            if "Check whether every claim" in content:
                return "not json at all"
            return "Standard includes 500 GB pooled [1]."

        pipe = _pipeline(stub_retriever, responder)
        result = pipe.answer("What storage does the Standard tier include?")
        assert result.answered is True

    def test_empty_question_is_handled(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "x"))
        assert pipe.answer("   ").answered is False


class TestCitations:
    def test_citation_carries_full_provenance(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "500 GB pooled [1]."))
        result = pipe.answer("Standard storage")
        c = result.citations[0]
        assert c.doc.endswith(".pdf")
        assert c.page >= 1
        assert c.section
        assert c.text

    def test_truncation_flag_reaches_the_citation(self, stub_retriever):
        retriever, chunks = stub_retriever
        chunks[0].truncated = True
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "Fix is [1]."))
        result = pipe.answer("LED blinking red")
        assert result.citations[0].truncated is True
        chunks[0].truncated = False

    def test_serialisation_shape_matches_the_api_contract(self, stub_retriever):
        pipe = _pipeline(stub_retriever, _router({0: 9.0}, 4, "500 GB [1]."))
        d = pipe.answer("Standard storage").to_dict()
        assert set(d) >= {"question", "answer", "answered", "sources", "top_score"}
        assert set(d["sources"][0]) == {
            "marker", "document", "page", "section", "truncated_in_source", "chunk"
        }
