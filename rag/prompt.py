"""System prompt and context formatting.

The prompt does four things deliberately:
  1. Restricts the model to the numbered blocks, with no appeal to prior
     knowledge - the model knows plenty about cloud storage generally, and all
     of it is a hallucination risk here.
  2. Requires an inline [n] marker on every factual sentence, which is what
     makes the citation validator possible at all. A marker the model must emit
     is a marker we can check.
  3. Gives an explicit PARTIAL path. Without one, a model facing 80% of an
     answer will invent the last 20% rather than return nothing.
  4. Emits an exact sentinel when the context does not contain the answer, so
     refusal is detected by string equality rather than by guessing at phrasing
     like "I'm sorry, I don't".
  5. Preserves the SCOPE a fact was stated with. This one came from a real
     failure: asked "how much PTO do I accrue?", the model answered "You accrue
     1.75 days" from a chunk that says "FULL-TIME EMPLOYEES accrue 1.75 days".
     Dropping the qualifier turns a scoped policy into a universal claim, and a
     contractor asking that question gets a confidently wrong answer. Note this
     is the same failure the contractor question in the eval set tests - it just
     hides behind an ambiguous "I" instead of naming the group.
"""
from __future__ import annotations

from .schemas import Hit

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You answer questions about Atman Cloud Consultancy's internal \
documents. You are given numbered context blocks retrieved from those documents.

RULES
1. Use ONLY the context blocks. Never use outside knowledge, and never invent a \
value that is not written down.
2. Cite with inline markers [1], [2] matching the block numbers. Every factual \
sentence needs at least one marker.
3. Each block's header line states the document's title, its document code, its \
filename, page and section. That header is part of the evidence: if the question \
asks which document something is, or what a document code refers to, answer from \
the header.
4. Applying a stated rule to the case in the question is correct reasoning, not \
invention. Do the arithmetic, apply the threshold, and answer. What you must not \
do is supply a value the documents never state.
5. If the blocks discuss the right topic but do not contain the specific fact \
asked for, say exactly what the documents DO establish, then state precisely \
what is missing. If a question asks about a model, tier, or entity that the \
blocks never mention, say so - do not answer using a different one.
6. If the blocks do not address the question at all, reply with exactly \
{INSUFFICIENT} and nothing else.
7. If a block is marked TRUNCATED IN SOURCE, quote what is there and state that \
the source text is cut off at that point. Never complete a truncated sentence.
8. Preserve the SCOPE the documents attach to a fact. If a policy is stated for a specific group, plan, tier, or product model, name that group in your answer. Never restate a scoped fact as though it applied to the reader generally: "full-time employees accrue 1.75 days" must not become "you accrue 1.75 days", because the reader may not be in that group.
9. Be concise and concrete. Prefer the document's own numbers and wording.
"""


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks as numbered blocks.

    The block header carries the document TITLE and CODE, not just the filename.
    This matters more than it looks: the same provenance line is what gets
    embedded and BM25-indexed (see Chunk.embed_text), so omitting it here meant
    the retriever and the generator were reading different text. "What is
    document SEC-POL-007?" retrieved the right chunk at rank 1 and was then
    refused, because the very code the retriever matched on was never shown to
    the model. Whatever the retriever indexes, the generator must be able to see.
    """
    parts = []
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        flag = "  [TRUNCATED IN SOURCE]" if chunk.truncated else ""
        ident = chunk.doc_title or chunk.doc
        if chunk.doc_code:
            ident = f"{ident}, document code {chunk.doc_code}"
        section = f" - section {chunk.section}" if chunk.section else ""
        parts.append(
            f"[{i}] {ident} - file {chunk.doc} - page {chunk.page}{section}{flag}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def user_prompt(question: str, hits: list[Hit]) -> str:
    return f"Context blocks:\n\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer:"


GROUNDEDNESS_PROMPT = """Check whether every claim in the ANSWER is supported by the CONTEXT.

A claim is UNSUPPORTED only if the context does not establish it - a value that
appears nowhere, or a fact stated about a different entity, product model, tier,
or time period than the one being claimed.

A claim IS supported when it follows necessarily from what the context states.
Applying a stated rule to the specific case in the question is correct reasoning,
not fabrication. All of the following are SUPPORTED:
  - arithmetic over stated numbers (0.2% below guarantee at 5% per 0.1% = 10%)
  - applying a stated threshold to a specific value ("non-refundable after 14
    days" entails no refund at 20 days)
  - reading the document title, code, page or section from a block's header line
  - reporting that a value is absent, or that the source text is truncated

Do NOT flag a claim merely because the question's exact wording or exact number
does not appear verbatim in the context. Flag only genuine fabrication.

CONTEXT:
{context}

ANSWER:
{answer}

Reply with JSON only:
{{"grounded": true|false, "unsupported": ["claim", ...]}}"""

REFUSAL_TEXT = (
    "I could not find an answer to that in the provided documents. "
    "The knowledge base covers the CloudSync Pro manual, the employee handbook, "
    "the API reference, the support FAQ, the security policy, the onboarding "
    "guide, and pricing & SLA terms."
)
