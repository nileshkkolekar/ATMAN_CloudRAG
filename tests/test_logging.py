"""Tests for tracing.

The property that matters most: logging must never be able to break a query.
A trace write that raises would turn a debugging aid into an outage, so the
failure path is tested explicitly.
"""
from __future__ import annotations

import json

from rag.answer import RagPipeline
from rag.logs import QueryTrace, cost_usd

from .conftest import FakeOpenAI, rerank_response


def _router(scores, n, answer_text, grounded=True):
    def responder(kw):
        content = kw["messages"][-1]["content"]
        if "Score how well each passage" in content:
            return rerank_response(scores, n)
        if "Check whether every claim" in content:
            return json.dumps(
                {"grounded": grounded, "unsupported": [] if grounded else ["fabricated"]}
            )
        return answer_text

    return responder


class TestCost:
    def test_known_model_prices_are_applied(self):
        # 1M prompt tokens of gpt-4o-mini at $0.15/1M
        assert cost_usd("gpt-4o-mini", 1_000_000, 0) == 0.15

    def test_unknown_model_does_not_raise(self):
        assert cost_usd("some-future-model", 1000, 1000) == 0.0


class TestTrace:
    def test_trace_writes_one_json_line_per_query(self, tmp_path, monkeypatch, stub_retriever):
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        retriever, _ = stub_retriever
        pipe = RagPipeline(
            retriever=retriever,
            client=FakeOpenAI(_router({0: 9.0}, 4, "Standard includes 500 GB [1].")),
        )
        pipe.answer("What storage does the Standard tier include?")

        lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["answered"] is True
        assert rec["gate"] == "answered"
        assert rec["top_score"] == 9.0

    def test_trace_records_every_candidate_not_just_the_shown_ones(
        self, tmp_path, monkeypatch, stub_retriever
    ):
        """The whole point: when the right chunk was retrieved but not shown,
        the trace is the only place that fact survives."""
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        retriever, chunks = stub_retriever
        pipe = RagPipeline(
            retriever=retriever, client=FakeOpenAI(_router({0: 9.0}, 4, "Answer [1]."))
        )
        pipe.answer("Standard tier")

        rec = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip())
        assert len(rec["candidates"]) >= len(rec["shown"])
        for cand in rec["candidates"]:
            assert {"id", "doc", "page", "rrf", "rerank"} <= set(cand)

    def test_shown_markers_are_chunk_ids_not_page_keys(
        self, tmp_path, monkeypatch, stub_retriever
    ):
        """Regression: doc:page keys collide when a page holds several chunks,
        which made the trace viewer mark more candidates as shown than were."""
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        retriever, _ = stub_retriever
        pipe = RagPipeline(
            retriever=retriever, client=FakeOpenAI(_router({0: 9.0}, 4, "Answer [1]."))
        )
        pipe.answer("Standard tier")

        rec = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip())
        ids = {c["id"] for c in rec["candidates"]}
        assert set(rec["shown"]) <= ids
        assert len(set(rec["shown"])) == len(rec["shown"])   # no duplicates

    def test_refusal_records_which_gate_fired(self, tmp_path, monkeypatch, stub_retriever):
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        retriever, _ = stub_retriever
        pipe = RagPipeline(
            retriever=retriever,
            client=FakeOpenAI(_router({0: 9.0}, 4, "Wrong [1].", grounded=False)),
        )
        pipe.answer("What is the CSP-600's capacity?")

        rec = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip())
        assert rec["answered"] is False
        assert rec["gate"] == "4_ungrounded"
        assert "fabricated" in rec["gate_detail"]

    def test_llm_calls_are_costed(self, tmp_path, monkeypatch, stub_retriever):
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        retriever, _ = stub_retriever
        pipe = RagPipeline(
            retriever=retriever, client=FakeOpenAI(_router({0: 9.0}, 4, "Answer [1]."))
        )
        pipe.answer("Standard tier")

        rec = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip())
        stages = {c["stage"] for c in rec["llm_calls"]}
        assert {"rerank", "generate"} <= stages

    def test_a_failed_trace_write_never_breaks_the_query(
        self, tmp_path, monkeypatch, stub_retriever
    ):
        """A logging bug must degrade to no logs, never to a failed answer."""
        import rag.logs as logs_mod

        # point the trace at a path that cannot be written
        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path / "file_not_dir")
        (tmp_path / "file_not_dir").write_text("I am a file, not a directory")

        retriever, _ = stub_retriever
        pipe = RagPipeline(
            retriever=retriever,
            client=FakeOpenAI(_router({0: 9.0}, 4, "Standard includes 500 GB [1].")),
        )
        result = pipe.answer("What storage does the Standard tier include?")
        assert result.answered is True
        assert "500 GB" in result.text

    def test_exception_inside_a_query_is_recorded_and_reraised(self, tmp_path, monkeypatch):
        import rag.logs as logs_mod

        monkeypatch.setattr(logs_mod, "LOG_DIR", tmp_path)
        try:
            with QueryTrace("boom"):
                raise ValueError("kaboom")
        except ValueError:
            pass
        else:
            raise AssertionError("QueryTrace swallowed the exception")

        rec = json.loads((tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip())
        assert rec["gate"] == "exception"
        assert "kaboom" in rec["gate_detail"]
