"""FastAPI service.

Thin by design: the pipeline is loaded once at startup and every route is a
call into RagPipeline.answer(). Retrieval state (Chroma handle, BM25 index) is
built once rather than per request - rebuilding BM25 on every query would be
the single most expensive thing in the service.

Run:  uvicorn app.api:app --reload
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Same guard as app/ui.py: keeps `import rag` working regardless of how the
# process was launched or what the working directory is.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.answer import RagPipeline
from rag.config import settings

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = RagPipeline()
    yield
    _state.clear()


app = FastAPI(
    title="Atman RAG",
    description="Source-grounded Q&A over Atman Cloud Consultancy documents.",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How much PTO do I accrue?"])


class SourceOut(BaseModel):
    marker: int
    document: str
    page: int
    section: str
    truncated_in_source: bool
    chunk: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    refusal_reason: str | None
    top_score: float
    warnings: list[str]
    latency_ms: int
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict:
    pipeline: RagPipeline = _state["pipeline"]
    return {
        "status": "ok",
        "chunks_indexed": len(pipeline.retriever.chunks),
        "documents": sorted({c.doc for c in pipeline.retriever.chunks}),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "api_key_configured": settings.has_key,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline: RagPipeline = _state["pipeline"]
    try:
        result = pipeline.answer(req.question)
    except RuntimeError as exc:            # missing key, misconfiguration
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:               # provider outage, malformed response
        raise HTTPException(status_code=502, detail=f"Upstream failure: {exc}") from exc
    return QueryResponse(**result.to_dict())
