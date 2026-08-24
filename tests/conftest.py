"""Fakes so the whole suite runs on a fresh clone with no API key and no network.

A test suite that requires a paid key is a test suite a reviewer will not run.
The fakes are deliberately dumb - they exist to exercise OUR control flow (the
gates, the fusion, the citation validator), not to simulate model quality.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

# Redirect logging BEFORE any rag module is imported.
#
# `log = get_logger(...)` runs at module import time in rag/extract.py and
# friends, and get_logger calls setup_logging(), which opens a FileHandler on
# the REAL logs/rag.log. Fixtures cannot undo that: by the time the first
# fixture runs, the handle is already open and session-scoped fixtures have
# already logged through it. The redirect has to happen here, at conftest import.
import rag.logs as _logs  # noqa: E402

_TEST_LOGS = Path(tempfile.mkdtemp(prefix="atman-rag-tests-"))
_logs.LOG_DIR = _TEST_LOGS
_logs.setup_logging(level="DEBUG", log_dir=_TEST_LOGS)

from rag.chunk import chunk_corpus
from rag.extract import extract_corpus
from rag.retrieve import Retriever
from rag.schemas import Chunk

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Never let a test write into the real logs/.

    BOTH outputs have to be redirected, and they redirect differently:

      traces.jsonl - LOG_DIR is read at write time, so monkeypatching it is
                     enough.
      rag.log      - the FileHandler is created ONCE inside setup_logging() and
                     holds an open handle to the real path. Patching LOG_DIR
                     afterwards does nothing to it. The handler itself has to be
                     swapped out.

    Missing the second half is why deliberate test failures ("kaboom",
    "provider down") showed up in the production log looking like real
    incidents.
    """
    import logging

    import rag.logs as logs_mod

    target = tmp_path / "logs"
    monkeypatch.setattr(logs_mod, "LOG_DIR", target)

    logger = logging.getLogger("rag")
    saved = logger.handlers[:]
    logger.handlers.clear()
    logs_mod.setup_logging(level="DEBUG", log_dir=target)
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.handlers.extend(saved)


@pytest.fixture(scope="session")
def blocks():
    return extract_corpus(PDF_DIR)


@pytest.fixture(scope="session")
def chunks(blocks):
    return chunk_corpus(blocks)


class FakeEmbedder:
    """Deterministic hash embedding. Meaningless semantically, stable across runs."""

    dim = 64

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in (h * ((self.dim // len(h)) + 1))[: self.dim]]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeChat:
    def __init__(self, responder):
        self._responder = responder

    class _Msg:
        def __init__(self, content):
            self.message = type("M", (), {"content": content})()

    def create(self, **kwargs):
        content = self._responder(kwargs)
        return type("R", (), {"choices": [FakeChat._Msg(content)]})()


class FakeOpenAI:
    """Scriptable stand-in for the OpenAI client.

    `responder(kwargs) -> str` receives the outgoing request so a test can
    branch on which call it is (rerank / answer / groundedness).
    """

    def __init__(self, responder):
        self.chat = type("C", (), {"completions": FakeChat(responder)})()


def rerank_response(scores: dict[int, float], n: int) -> str:
    return json.dumps(
        {"scores": [{"id": i, "score": scores.get(i, 0.0)} for i in range(n)]}
    )


def make_chunk(cid="D::000", doc="Doc.pdf", page=1, section="1 X", text="body", **kw) -> Chunk:
    return Chunk(
        id=cid, doc=doc, doc_title="Doc", doc_code="D-001",
        page=page, section=section, text=text, **kw
    )


class _StubStore:
    """Vector store stand-in: returns a fixed dense ordering."""

    def __init__(self, chunks, dense_order):
        self._chunks = chunks
        self._dense = dense_order

    def all_chunks(self):
        return self._chunks

    def count(self):
        return len(self._chunks)

    def query(self, vector, k):
        return self._dense[:k]


@pytest.fixture
def stub_retriever():
    chunks = [
        make_chunk("A::000", doc="Pricing_and_SLA.pdf", section="1 Plan Pricing",
                   text="Standard $12 / user / month, 500 GB pooled storage, up to 25 users."),
        make_chunk("B::000", doc="Security_Policy.pdf", section="1 Definitions",
                   text="Standard Tier Access is the default access level granted to all employees."),
        make_chunk("C::000", doc="Employee_Handbook.pdf", section="3.1 Health Insurance",
                   text="The company covers 100% of employee premiums for the Standard health plan."),
        make_chunk("D::000", doc="API_Reference.pdf", section="2 Rate Limits",
                   text="Standard tier allows 600 requests per minute with a burst of 100."),
    ]
    store = _StubStore(chunks, dense_order=list(chunks))
    return Retriever(store=store, embedder=FakeEmbedder()), chunks
