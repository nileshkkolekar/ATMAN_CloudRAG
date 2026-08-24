"""PDF -> text, with two things a naive extractor gets wrong.

1. TABLES. Both PyMuPDF's `find_tables()` and pdfplumber's `extract_tables()`
   silently corrupt the two worst tables in this corpus, interleaving
   characters from overlapping cells into garbage like "leWd baoito 1t0
   minutes". So every extracted table is run past a GUARD: it is accepted only
   if every one of its cells also appears verbatim in the plain-text layer.
   Accepted tables are re-emitted as markdown pipe rows, which keeps a row on
   one line so chunking cannot separate a value from its row label. Rejected
   tables fall back to the plain text, which is merely awkward rather than
   wrong.

2. CLIPPED LINES. Six lines in this corpus were clipped when the PDFs were
   generated - their bounding boxes run past the right page edge and the lost
   characters are simply not in the content stream. No library recovers them
   and OCR cannot either, because they were never rendered. We detect them and
   tag the chunk, so the answer can say "the source row is truncated" instead
   of serving half a sentence as though it were whole.
"""
from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF

from .logs import get_logger
from .schemas import Block

log = get_logger("extract")

# A line whose right edge reaches the page edge was clipped at generation time.
CLIP_EPS = 1.0

DOC_CODE_RE = re.compile(r"Document Code:\s*([A-Z0-9\-]+)")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _page_lines(page: fitz.Page) -> list[dict]:
    """Lines in reading order, each tagged with whether it was clipped."""
    limit = page.rect.width - CLIP_EPS
    lines: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue
            lines.append(
                {
                    "text": text,
                    "bbox": line["bbox"],
                    "clipped": line["bbox"][2] >= limit,
                }
            )
    return lines


def _table_is_trustworthy(rows: list[list[str]], plain: str) -> bool:
    """The guard: every cell must appear verbatim in the plain-text layer.

    Character-interleaving corruption produces cells that exist nowhere in the
    real text, so this rejects a mangled table as a unit. It is a cheap,
    deterministic check - no heuristics about ruling lines or cell geometry.
    """
    plain_n = _norm(plain)
    cells = [_norm(c) for row in rows for c in row if c and _norm(c)]
    if len(cells) < 4:  # not really a table
        return False
    return all(cell in plain_n for cell in cells)


def _as_markdown(rows: list[list[str]]) -> str:
    """Re-emit an accepted table so one row = one line."""
    clean = [[_norm(c) for c in row] for row in rows]
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    head, *body = clean
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _extract_tables(page: fitz.Page, plain: str) -> list[dict]:
    """Accepted tables only, each with its bbox so prose can skip those lines."""
    accepted = []
    try:
        found = page.find_tables()
    except Exception:
        return accepted
    for table in found.tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        if _table_is_trustworthy(rows, plain):
            accepted.append({"bbox": fitz.Rect(table.bbox), "md": _as_markdown(rows)})
        else:
            # The interesting case: both extractors corrupt some tables, and a
            # rejected table silently falls back to plain text. Log it, or the
            # fallback is invisible when a table answer later looks wrong.
            log.warning(
                "table REJECTED by guard on page %d (cells absent from plain text) "
                "- falling back to plain text", page.number + 1,
            )
    return accepted


def _doc_identity(doc: fitz.Document, filename: str) -> tuple[str, str]:
    """Title and document code from the cover page.

    The title is taken as the lines set in the largest font on page 1, not the
    first line: one cover ("Information Security & Data Handling / Policy")
    wraps its title across two lines, and a first-line-only rule silently drops
    the word "Policy". Font size is the signal the layout actually encodes.

    The code (SEC-POL-007, PRC-SLA-021, ...) matters more than it looks: it is
    prepended to every chunk of the document, which makes it an exact-matchable
    BM25 token on all of them. Dense embeddings are famously weak on
    alphanumeric identifiers; BM25 nails them.
    """
    cover = doc[0]
    code_m = DOC_CODE_RE.search(cover.get_text())
    code = code_m.group(1) if code_m else ""

    sized: list[tuple[float, str]] = []
    for block in cover.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(sp["text"] for sp in line["spans"]).strip()
            if not text or text.startswith("Document Code:"):
                continue
            size = max(sp["size"] for sp in line["spans"])
            sized.append((round(size, 1), text))

    if not sized:
        return Path(filename).stem.replace("_", " "), code
    top = max(sz for sz, _ in sized)
    title = " ".join(text for sz, text in sized if sz == top)
    return title.strip(), code


def extract_document(path: Path) -> list[Block]:
    """One Block per page, carrying provenance and truncation flags."""
    doc = fitz.open(path)
    filename = path.name
    title, code = _doc_identity(doc, filename)
    blocks: list[Block] = []

    for pno, page in enumerate(doc, start=1):
        plain = page.get_text()
        tables = _extract_tables(page, plain)
        lines = _page_lines(page)

        parts: list[str] = []
        emitted: set[int] = set()
        truncated = False

        for line in lines:
            rect = fitz.Rect(line["bbox"])
            owner = next(
                (i for i, t in enumerate(tables) if rect.intersects(t["bbox"])), None
            )
            if owner is not None:
                if owner not in emitted:      # emit the whole table once, in place
                    parts.append(tables[owner]["md"])
                    emitted.add(owner)
                continue
            text = line["text"].rstrip()
            if line["clipped"]:
                truncated = True
                text = text + " [...truncated in source]"
            parts.append(text)

        page_text = "\n".join(parts).strip()
        if not page_text:
            continue
        if truncated:
            log.warning("%s p.%d contains line(s) clipped in the source PDF",
                        filename, pno)
        blocks.append(
            Block(
                doc=filename,
                doc_title=title,
                doc_code=code,
                page=pno,
                text=page_text,
                kind="table" if emitted else "prose",
                truncated=truncated,
            )
        )
    doc.close()
    return blocks


def extract_corpus(pdf_dir: Path) -> list[Block]:
    blocks: list[Block] = []
    for path in sorted(pdf_dir.glob("*.pdf")):
        got = extract_document(path)
        log.info("extracted %-26s %d page block(s)", path.name, len(got))
        blocks.extend(got)
    log.info("corpus: %d page blocks from %d PDFs", len(blocks),
             len({b.doc for b in blocks}))
    return blocks
