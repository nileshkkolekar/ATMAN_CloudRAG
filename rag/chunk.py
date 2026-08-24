"""Blocks -> chunks, split on the structure the documents already have.

WHY NOT FIXED-SIZE WINDOWS. This is arithmetic, not taste. These documents are
315-591 tokens each, end to end. A conventional 512-token window would swallow
an entire document as a single chunk: retrieval would return whole files,
citations would degrade to "somewhere in Employee_Handbook.pdf", and the
four-way "Standard" collision in this corpus would become unresolvable, because
all four senses would live inside the same vector.

So we split on the numbering the authors already wrote (1., 1.1, 2. ...) and on
the FAQ's Q:/A: pairs, then merge adjacent siblings up to a floor and
recursively sub-split anything over a ceiling. Sections are the unit a human
would cite, which is exactly what a citation needs to name.
"""
from __future__ import annotations

import re

import tiktoken

from .config import settings
from .logs import get_logger
from .schemas import Block, Chunk

log = get_logger("chunk")

_ENC = tiktoken.get_encoding("cl100k_base")

# "1. Authentication", "3.1 Upload a File" - a short, title-like numbered line.
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z(\u201c\"].{0,70})$")
QA_RE = re.compile(r"^Q:\s*(.+)$")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _is_heading(line: str) -> re.Match | None:
    line = line.strip()
    if len(line.split()) > 9 or line.endswith("."):
        return None
    return HEADING_RE.match(line)


class _Section:
    def __init__(self, title: str, page: int, kind: str = "prose"):
        self.title = title
        self.page = page          # where the heading was printed
        self.body_page = 0        # where the body actually starts (see add())
        self.kind = kind
        self.lines: list[str] = []
        self.truncated = False

    def add(self, line: str, page: int) -> None:
        """Record the page of the first body line.

        Product_Manual's "3. Storage & Retention" heading is printed at the foot
        of page 2 with every word of its body on page 3. Citing the heading's
        page would send a reviewer to a page that does not contain the answer,
        so a section is cited where its body starts.
        """
        if not self.lines:
            self.body_page = page
        self.lines.append(line)

    @property
    def cite_page(self) -> int:
        return self.body_page or self.page

    @property
    def body(self) -> str:
        return "\n".join(self.lines).strip()


def _sections(blocks: list[Block]) -> list[_Section]:
    """Walk one document's pages as a single line stream.

    Sections legitimately span page breaks here - Product_Manual's
    "3. Storage & Retention" heading sits at the foot of page 2 with its body on
    page 3 - so splitting per page would orphan the heading from its content.
    A section is cited at the page where its body starts, which is what a reader
    verifying the answer would actually turn to.
    """
    out: list[_Section] = []
    current: _Section | None = None

    for block in blocks:
        lines = block.text.split("\n")

        # A table-of-contents page is not seven empty sections; it is one
        # navigational chunk. Without this, every TOC entry parses as a heading
        # and shadows the real section of the same name later in the document.
        if lines and lines[0].strip().lower().startswith("table of contents"):
            sec = _Section("Table of Contents", block.page)
            for ln in lines[1:]:
                sec.add(ln, block.page)
            out.append(sec)
            current = None
            continue

        # The cover page carries the title and document code, both already in
        # every chunk header. Nothing to index.
        if block.page == 1 and len(block.text.split()) < 40:
            continue

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            qa = QA_RE.match(stripped)
            if qa:
                current = _Section(f"Q: {qa.group(1)}", block.page, kind="qa")
                current.add(stripped, block.page)
                out.append(current)
                continue

            head = _is_heading(stripped)
            if head:
                number, title = head.groups()
                current = _Section(f"{number} {title.strip()}", block.page)
                out.append(current)
                continue

            if current is None:  # prose before any heading
                current = _Section("", block.page)
                out.append(current)

            current.add(line, block.page)
            if stripped.startswith("|"):
                current.kind = "table"
            if "[...truncated in source]" in line:
                current.truncated = True

    return [s for s in out if s.body]


def _split_oversized(section: _Section) -> list[_Section]:
    """Recursive character split, used only inside a section that is too long.

    Tables are never split: a row separated from its header row is a set of
    values with nothing to say what they mean.
    """
    if n_tokens(section.body) <= settings.max_chunk_tokens or section.kind == "table":
        return [section]

    lines = section.lines
    overlap = max(1, int(len(lines) * settings.chunk_overlap_ratio))
    parts: list[_Section] = []
    budget = settings.max_chunk_tokens
    start = 0
    while start < len(lines):
        taken, total = [], 0
        i = start
        while i < len(lines) and (total + n_tokens(lines[i]) <= budget or not taken):
            taken.append(lines[i])
            total += n_tokens(lines[i])
            i += 1
        part = _Section(section.title, section.page, section.kind)
        for ln in taken:
            part.add(ln, section.cite_page)
        part.truncated = section.truncated
        parts.append(part)
        if i >= len(lines):
            break
        start = max(i - overlap, start + 1)
    return parts


def _merge_small(sections: list[_Section]) -> list[_Section]:
    """Merge adjacent siblings until a chunk clears the floor.

    A 20-token chunk is dominated by its own header prefix - the provenance line
    outweighs the content and skews the vector toward the document rather than
    the passage. The floor exists specifically to stop that.
    """
    merged: list[_Section] = []
    for sec in sections:
        if (
            merged
            and n_tokens(merged[-1].body) < settings.min_chunk_tokens
            and merged[-1].kind == sec.kind == "prose"
            and merged[-1].cite_page == sec.cite_page
        ):
            prev = merged[-1]
            prev.add(sec.title if sec.title else "", sec.cite_page)
            for ln in sec.lines:
                prev.add(ln, sec.cite_page)
            prev.truncated = prev.truncated or sec.truncated
            if prev.title and sec.title:
                prev.title = f"{prev.title}; {sec.title}"
            continue
        merged.append(sec)
    return merged


def chunk_document(blocks: list[Block]) -> list[Chunk]:
    if not blocks:
        return []
    head = blocks[0]
    sections = _merge_small(_sections(blocks))

    chunks: list[Chunk] = []
    for sec in sections:
        for part in _split_oversized(sec):
            body = part.body
            if not body.strip():
                continue
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{head.doc_code or head.doc}::{idx:03d}",
                    doc=head.doc,
                    doc_title=head.doc_title,
                    doc_code=head.doc_code,
                    page=part.cite_page,
                    section=part.title,
                    text=body,
                    kind=part.kind,
                    truncated=part.truncated,
                    n_tokens=n_tokens(body),
                )
            )
    return chunks


def chunk_corpus(blocks: list[Block]) -> list[Chunk]:
    by_doc: dict[str, list[Block]] = {}
    for b in blocks:
        by_doc.setdefault(b.doc, []).append(b)
    chunks: list[Chunk] = []
    for doc in sorted(by_doc):
        got = chunk_document(by_doc[doc])
        # Chunk-count-per-document is the canary for heading detection breaking:
        # if a parser change makes it 1, retrieval silently degrades to
        # whole-file citations. Logged at ingest so it is visible immediately.
        log.info("chunked %-26s %2d chunks (%d tok avg)", doc, len(got),
                 sum(c.n_tokens for c in got) // max(len(got), 1))
        if len(got) <= 1:
            log.warning("%s produced %d chunk(s) - heading detection may have failed",
                        doc, len(got))
        chunks.extend(got)
    return chunks
