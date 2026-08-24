"""Streamlit reviewer UI.

Calls the pipeline in-process rather than over HTTP, so a reviewer can run the
UI alone without also starting the API. Both surfaces sit on the same
RagPipeline.answer(), so there is no second implementation to keep in step.

Run:  streamlit run app/ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/ui.py` executes this file with app/ as sys.path[0], so the
# project root is NOT importable and `import rag` fails with ModuleNotFoundError.
# (`python -m streamlit run ...` happens to work, because -m puts the CWD on the
# path - which is why this can pass in testing and fail for a reviewer.)
# Put the repo root on the path before importing anything local.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from rag.answer import RagPipeline  # noqa: E402
from rag.chunk import chunk_corpus  # noqa: E402
from rag.config import settings  # noqa: E402
from rag.embed import get_embedder  # noqa: E402
from rag.extract import extract_corpus  # noqa: E402
from rag.store import VectorStore  # noqa: E402

st.set_page_config(page_title="Atman RAG", layout="wide")


def ensure_index() -> None:
    store = VectorStore()
    if store.count() > 0:
        return

    blocks = extract_corpus(settings.pdf_dir)
    if not blocks:
        raise RuntimeError(f"No PDFs found in {settings.pdf_dir}")
    chunks = chunk_corpus(blocks)
    vectors = get_embedder().embed_documents([c.embed_text for c in chunks])
    store.reset()
    store.add(chunks, vectors)


@st.cache_resource(show_spinner="Loading or building index...")
def get_pipeline() -> RagPipeline:
    ensure_index()
    return RagPipeline()


st.title("Atman Cloud - Document Q&A")
st.caption(
    "Answers are generated only from the seven indexed documents, with a "
    "document, page and section citation for every claim."
)

if not settings.has_key:
    st.error("OPENAI_API_KEY is not set. Add it in Streamlit app secrets.")
    st.stop()

pipeline = get_pipeline()

with st.sidebar:
    st.subheader("Index")
    docs = sorted({c.doc for c in pipeline.retriever.chunks})
    st.metric("Chunks indexed", len(pipeline.retriever.chunks))
    for d in docs:
        st.caption(f"- {d}")
    st.subheader("Settings")
    st.caption(f"LLM: `{settings.llm_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model}`")
    st.caption(f"Refusal threshold: `{settings.relevance_threshold}`")

EXAMPLES = [
    "What storage does the Standard tier include?",
    "What does Standard Tier Access mean for employees?",
    "How long do I have to report a suspected data breach?",
    "What is the CSP-600's storage capacity?",
]
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["question"] = ex

question = st.text_input(
    "Ask a question", key="question", placeholder="e.g. How much PTO do I accrue?"
)

if question:
    with st.spinner("Retrieving and answering..."):
        result = pipeline.answer(question)

    if result.answered:
        st.success(result.text)
    else:
        st.warning(result.text)
        st.caption(
            f"Refused - reason: `{result.refusal_reason}`, "
            f"top relevance score {result.top_score:.1f} "
            f"(threshold {settings.relevance_threshold})"
        )

    for w in result.warnings:
        st.info(w)

    if result.citations:
        st.subheader("Sources")
        for c in result.citations:
            label = f"[{c.marker}] {c.doc} - page {c.page}"
            if c.section:
                label += f" - {c.section}"
            with st.expander(label):
                if c.truncated:
                    st.warning(
                        "One or more lines in this passage were clipped when the "
                        "source PDF was generated. The missing characters are not "
                        "recoverable from the file."
                    )
                st.text(c.text)

    with st.expander("Retrieval detail (all candidates considered)"):
        for h in result.hits:
            st.caption(
                f"{h.chunk.doc} p.{h.chunk.page} | {h.chunk.section} | "
                f"rerank={h.rerank_score} rrf={h.rrf_score:.4f} "
                f"dense_rank={h.dense_rank} bm25_rank={h.bm25_rank}"
            )
    st.caption(f"Answered in {result.latency_ms} ms")
