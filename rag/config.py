"""Every tunable knob in one place.

Design note: the values that ship are sized for THIS corpus (~25 chunks, ~3k
tokens). The values that would be correct on a realistic corpus are recorded in
the `PRODUCTION_*` comments beside each field. Pretending the shipped numbers
were tuned at this scale would be dishonest; keeping both visible in one file
means the scale-up is a config edit, not a rewrite.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- provider ---------------------------------------------------------
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"   # 1536d
    llm_model: str = "gpt-4o-mini"
    embedding_provider: str = "openai"                # openai | local

    # ---- paths ------------------------------------------------------------
    pdf_dir: Path = ROOT / "data" / "pdfs"
    chroma_dir: Path = ROOT / "storage" / "chroma"
    collection: str = "atman_docs"

    # ---- chunking ---------------------------------------------------------
    # Documents here are 315-591 tokens TOTAL. A conventional 512-token window
    # would swallow an entire document as one chunk, collapsing citations to
    # "somewhere in Employee_Handbook.pdf". Hence a small merge floor.
    min_chunk_tokens: int = 120      # merge adjacent sections until this
    max_chunk_tokens: int = 350      # recursively sub-split above this
    chunk_overlap_ratio: float = 0.15

    # ---- retrieval --------------------------------------------------------
    # PRODUCTION_K: 50 lexical + 50 dense -> rerank -> 8.
    # At 25 chunks, 10+10 is most of the corpus and the reranker does the real
    # work. Shipped small, stated honestly.
    bm25_k: int = 10
    dense_k: int = 10
    rrf_k: int = 60                  # RRF damping constant, standard value
    rerank_top_n: int = 4            # chunks actually shown to the answerer

    # ---- gates ------------------------------------------------------------
    # Calibrated against eval/questions.yaml — see README "Calibrating tau".
    relevance_threshold: float = 4.0   # reranker score 0-10; below -> refuse
    enable_groundedness_check: bool = True

    # ---- observability ----------------------------------------------------
    log_level: str = "INFO"          # console verbosity; the file always gets DEBUG

    # ---- generation -------------------------------------------------------
    temperature: float = 0.0
    max_answer_tokens: int = 700

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key.strip())


settings = Settings()
