"""Embedding providers behind a two-method interface.

The interface is not speculative architecture - it is what makes the local
fallback a config change instead of a rewrite, and it costs ten lines. The
trigger for actually switching is not cost (indexing this corpus costs about
$0.0001); it is confidentiality. Client documents should not be sent to a
third-party embedding API without a data-processing agreement in place.
"""
from __future__ import annotations

from typing import Protocol

from .config import settings


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """text-embedding-3-small: 1536d, strong retrieval quality, one batched call.

    Chose -small over -large deliberately: -large is 3072d and roughly 6x the
    cost, which is worth it on hard technical corpora but unmeasurable across 34
    chunks. Spending 6x for a difference you cannot detect is not a trade-off,
    it is a reflex.
    """

    dim = 1536

    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or set EMBEDDING_PROVIDER=local to use the offline embedder."
            )
        self.model = model or settings.embedding_model
        self._client = OpenAI(api_key=key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # The whole corpus fits in one request; batch in case it stops fitting.
        out: list[list[float]] = []
        for i in range(0, len(texts), 256):
            resp = self._client.embeddings.create(
                model=self.model, input=texts[i : i + 256]
            )
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalEmbedder:
    """bge-small-en-v1.5 via sentence-transformers - no key, no network.

    Documented fallback for an air-gapped environment, a confidential corpus, or
    a reviewer with no API key. Costs a ~130MB model download on first use.
    """

    dim = 384

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge wants this instruction prefix on queries but not on documents.
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        return self._model.encode(
            [prefixed], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


def get_embedder(provider: str | None = None) -> Embedder:
    provider = (provider or settings.embedding_provider).lower()
    if provider == "local":
        return LocalEmbedder()
    return OpenAIEmbedder()
