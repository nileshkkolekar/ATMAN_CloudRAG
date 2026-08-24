"""Chroma wrapper.

WHY CHROMA OVER FAISS. The brief requires returning document name, page number
and chunk for every answer. Chroma stores metadata and the source text beside
the vector as first-class fields, so provenance is a property of the record.
With FAISS I would hand-roll a parallel sidecar keyed by row index and own the
risk of it drifting out of sync with the index on every re-ingest - a real class
of bug, traded for a speed advantage that is unmeasurable across 34 vectors.

This holds until roughly a million vectors or the first requirement for
per-user access filtering; then pgvector if Postgres already exists, Qdrant if
it does not.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from .config import settings
from .schemas import Chunk


class VectorStore:
    def __init__(self, path: Path | None = None, collection: str | None = None):
        self.path = Path(path or settings.chroma_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._name = collection or settings.collection
        # Embeddings are always supplied explicitly, never computed by Chroma -
        # otherwise the store would silently use its own default model and the
        # index would disagree with the query embedder.
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._col.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    def count(self) -> int:
        return self._col.count()

    def all_chunks(self) -> list[Chunk]:
        """Rehydrate every chunk - the BM25 index is built from these.

        BM25 is rebuilt in memory at startup rather than persisted: it takes
        milliseconds over 34 chunks, and a stale lexical index that disagrees
        with the vector index is a worse failure than a cold start.
        """
        got = self._col.get(include=["documents", "metadatas"])
        out: list[Chunk] = []
        for cid, text, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            out.append(_to_chunk(cid, text, meta))
        return sorted(out, key=lambda c: c.id)

    def query(self, vector: list[float], k: int) -> list[Chunk]:
        res = self._col.query(
            query_embeddings=[vector],
            n_results=min(k, max(self.count(), 1)),
            include=["documents", "metadatas"],
        )
        return [
            _to_chunk(cid, text, meta)
            for cid, text, meta in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0]
            )
        ]


def _to_chunk(cid: str, text: str, meta: dict) -> Chunk:
    return Chunk(
        id=cid,
        doc=meta.get("doc", ""),
        doc_title=meta.get("doc_title", ""),
        doc_code=meta.get("doc_code", ""),
        page=int(meta.get("page", 0)),
        section=meta.get("section", ""),
        text=text,
        kind=meta.get("kind", "prose"),
        truncated=bool(meta.get("truncated", False)),
        n_tokens=int(meta.get("n_tokens", 0)),
    )
