"""Orchestration and the four grounding gates.

Four independent gates, because each has a blind spot the next one covers.
This is measured, not assumed. On the 20-question eval set, two unanswerable
questions score 10.0 - the maximum relevance score, identical to a perfectly
answerable question:

    "How much PTO do contractors accrue?"
        Retrieval returns the PTO section, and it is right to: that IS the
        relevant section. It just describes full-time employees only.

    "What is the penalty if Enterprise uptime drops below 99.0%?"
        Retrieval returns the SLA credit table, again correctly. The table caps
        credits at 30% and defines no penalty beyond them.

For both:
  Gate 1 (relevance) cannot catch them - the chunk genuinely is relevant, so the
                     score is maximal. NO value of tau separates these from a
                     real question.
  Gate 3 (citations) cannot catch them - the citation would point at a real block.
  Gates 2 and 4 are the only things standing between these questions and a
                     confidently fabricated answer. In the measured run, gate 2
                     caught the first and gate 4 caught the second.

Gate 1 alone scores 60% refusal recall on that set; all four score 80-100%.
That difference is what gates 2 and 4 are buying.

The range, not a point value, is deliberate: the uptime-penalty question is a
coin flip between refusing and returning a correct PARTIAL answer that names
what is missing. Both are non-fabricating; the eval's binary answerable flag is
what cannot express that. See the README, "A metric that is not stable".

Every query runs inside a QueryTrace (see rag/logs.py), which records each gate
decision plus every candidate considered - so when a gate fires unexpectedly,
the reason is already on disk.
"""
from __future__ import annotations

import json
import re
import time

from .config import settings
from .logs import QueryTrace, get_logger
from .prompt import (
    GROUNDEDNESS_PROMPT,
    INSUFFICIENT,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    format_context,
    user_prompt,
)
from .retrieve import Retriever
from .schemas import Answer, Citation, Hit

MARKER_RE = re.compile(r"\[(\d+)\]")

log = get_logger("answer")


class RagPipeline:
    """One entry point.

    FastAPI, Streamlit, the CLI and the eval harness all call `answer()`, so
    there is exactly one code path to reason about and the interfaces cannot
    drift apart.
    """

    def __init__(self, retriever: Retriever | None = None, client=None):
        self.retriever = retriever or Retriever()
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
                )
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    # ------------------------------------------------------------------
    def answer(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(
                question, "Please ask a question.", answered=False,
                refusal_reason="empty_question",
            )
        # Every query runs inside a trace: one JSON line in logs/traces.jsonl
        # holding every candidate, every gate decision and every LLM call.
        with QueryTrace(question) as trace:
            return self._answer(question, trace)

    def _answer(self, question: str, trace: QueryTrace) -> Answer:
        started = time.perf_counter()
        hits = self.retriever.retrieve(question, client=self.client, trace=trace)
        top = hits[0].score if hits else 0.0
        trace.top_score = top
        # Key on chunk id, not doc:page - several chunks share a page, and a
        # doc:page key made the trace viewer mark 7 of 12 candidates as "shown"
        # when only 4 were.
        trace.shown = [h.chunk.id for h in hits]

        # -- GATE 1: relevance ------------------------------------------
        # If nothing clears the threshold we refuse WITHOUT calling the
        # answering model at all. A model that is never invoked cannot
        # hallucinate - the cheapest and most reliable gate available.
        if not hits or top < settings.relevance_threshold:
            log.info("GATE 1 refuse: top score %.1f < tau %.1f "
                     "(generation model never called)", top,
                     settings.relevance_threshold)
            trace.gate = "1_relevance"
            trace.gate_detail = f"top {top:.1f} < tau {settings.relevance_threshold}"
            return self._refuse(question, hits, top, "low_relevance", started,
                                trace=trace)
        log.debug("GATE 1 pass: top score %.1f >= tau %.1f", top,
                  settings.relevance_threshold)

        # -- GATE 2: constrained generation -----------------------------
        text = self._generate(question, hits, trace)
        if INSUFFICIENT in text:
            log.info("GATE 2 refuse: model emitted %s at top score %.1f - "
                     "retrieval was confident but the fact is absent",
                     INSUFFICIENT, top)
            trace.gate = "2_insufficient_context"
            trace.gate_detail = f"sentinel emitted at top score {top:.1f}"
            return self._refuse(question, hits, top, "insufficient_context",
                                started, trace=trace)

        # -- GATE 3: citation validation --------------------------------
        text, citations, warnings = self._validate_citations(text, hits)
        log.debug("GATE 3: kept citation(s) %s", [c.marker for c in citations])
        for w in warnings:
            log.warning("GATE 3: %s", w)

        # -- GATE 4: groundedness pass ----------------------------------
        if settings.enable_groundedness_check:
            verdict = self._check_groundedness(text, hits, trace)
            if verdict and not verdict.get("grounded", True):
                unsupported = verdict.get("unsupported", [])
                if unsupported:
                    log.info("GATE 4 refuse: unsupported claim(s) %s",
                             unsupported[:3])
                    trace.gate = "4_ungrounded"
                    trace.gate_detail = "; ".join(unsupported[:3])
                    warnings.append(
                        "Groundedness check flagged unsupported claims: "
                        + "; ".join(unsupported[:3])
                    )
                    return self._refuse(
                        question, hits, top, "ungrounded", started,
                        warnings=warnings, trace=trace,
                    )
            log.debug("GATE 4 pass: answer is grounded in the retrieved chunks")

        trace.gate = "answered"
        trace.answered = True
        trace.answer_preview = text
        trace.warnings = warnings
        return Answer(
            question=question,
            text=text,
            answered=True,
            citations=citations,
            hits=hits,
            top_score=top,
            warnings=warnings,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    # ------------------------------------------------------------------
    def _generate(self, question: str, hits: list[Hit], trace=None) -> str:
        started = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=settings.llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_answer_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, hits)},
            ],
        )
        if trace is not None:
            trace.record_llm("generate", settings.llm_model,
                             int((time.perf_counter() - started) * 1000), resp)
        return (resp.choices[0].message.content or "").strip()

    def _validate_citations(self, text: str, hits: list[Hit]):
        """Strip markers that point at blocks the model was never given.

        Deterministic, so it catches the invented-citation failure with
        certainty rather than probability. It also drives the source list shown
        to the user: only blocks the answer actually cited are displayed, so the
        citation panel reflects the answer instead of the whole candidate set.
        """
        warnings: list[str] = []
        valid = set(range(1, len(hits) + 1))
        used: list[int] = []
        invented: set[int] = set()

        for m in MARKER_RE.finditer(text):
            n = int(m.group(1))
            if n in valid:
                if n not in used:
                    used.append(n)
            else:
                invented.add(n)

        for n in sorted(invented):
            warnings.append(f"Removed invented citation [{n}].")
            text = text.replace(f"[{n}]", "")

        citations = []
        for n in sorted(used):
            chunk = hits[n - 1].chunk
            citations.append(
                Citation(
                    marker=n,
                    doc=chunk.doc,
                    page=chunk.page,
                    section=chunk.section,
                    text=chunk.text,
                    truncated=chunk.truncated,
                )
            )
        if not citations and hits:
            warnings.append("Answer contained no citation markers.")
        return text.strip(), citations, warnings

    def _check_groundedness(self, text: str, hits: list[Hit], trace=None) -> dict | None:
        """A single verification pass, deliberately not a loop.

        An iterative critique loop needs a stopping condition, multiplies
        latency per turn, and a critic that errs the other way can talk the
        system out of a correct answer. One pass has deterministic termination
        and catches the failure that matters here.
        """
        started = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": GROUNDEDNESS_PROMPT.format(
                            context=format_context(hits), answer=text
                        ),
                    }
                ],
            )
            if trace is not None:
                trace.record_llm("groundedness", settings.llm_model,
                                 int((time.perf_counter() - started) * 1000), resp)
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:
            log.warning("GATE 4 checker errored (%s) - keeping the answer", exc)
            if trace is not None:
                trace.record_llm("groundedness", settings.llm_model,
                                 int((time.perf_counter() - started) * 1000),
                                 error=str(exc))
            return None  # never fail a good answer because the checker errored

    def _refuse(self, question, hits, top, reason, started, warnings=None,
                trace=None) -> Answer:
        if trace is not None:
            trace.answered = False
            trace.warnings = warnings or []
        return Answer(
            question=question,
            text=REFUSAL_TEXT,
            answered=False,
            citations=[],
            hits=hits,
            refusal_reason=reason,
            top_score=top,
            warnings=warnings or [],
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
