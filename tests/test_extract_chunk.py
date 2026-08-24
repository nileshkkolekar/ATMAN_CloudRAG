"""Ingestion tests: the properties that silently rot if a parser changes."""
from __future__ import annotations

from rag.chunk import _is_heading, n_tokens
from rag.config import settings
from rag.extract import _table_is_trustworthy


class TestExtraction:
    def test_all_seven_documents_extracted(self, blocks):
        assert len({b.doc for b in blocks}) == 7

    def test_every_block_has_provenance(self, blocks):
        for b in blocks:
            assert b.doc and b.page >= 1
            assert b.doc_code, f"{b.doc} lost its document code"

    def test_document_codes_are_exact(self, blocks):
        codes = {b.doc: b.doc_code for b in blocks}
        assert codes["Security_Policy.pdf"] == "SEC-POL-007"
        assert codes["Pricing_and_SLA.pdf"] == "PRC-SLA-021"
        assert codes["Product_Manual.pdf"] == "PM-CSP-001"

    def test_wrapped_title_is_not_truncated(self, blocks):
        """The security policy title wraps onto a second line on its cover."""
        title = next(b.doc_title for b in blocks if b.doc == "Security_Policy.pdf")
        assert title == "Information Security & Data Handling Policy"

    def test_clipped_lines_are_flagged(self, blocks):
        """Six lines were clipped when these PDFs were generated.

        The characters are absent from the content stream - no library and no
        OCR recovers them. Detecting them is the only correct response.
        """
        flagged = {b.doc for b in blocks if b.truncated}
        assert flagged == {"Product_Manual.pdf", "Security_Policy.pdf"}

    def test_truncation_marker_is_in_the_text(self, blocks):
        b = next(b for b in blocks if b.doc == "Product_Manual.pdf" and b.truncated)
        assert "[...truncated in source]" in b.text

    def test_corrupt_table_is_rejected_by_the_guard(self, blocks):
        """Both table extractors interleave this table into garbage.

        The guard must reject it so the plain-text fallback is used instead of
        cells like "leWd baoito 1t0 minutes" entering the index.
        """
        pm3 = next(b for b in blocks if b.doc == "Product_Manual.pdf" and b.page == 3)
        assert "leWd baoito" not in pm3.text
        assert "stNoraamgeed" not in pm3.text
        assert pm3.kind == "prose"          # table rejected, fell back to text

    def test_clean_table_is_accepted_and_row_aligned(self, blocks):
        """A good table is re-emitted as markdown so one row stays on one line."""
        pricing = next(b for b in blocks if b.doc == "Pricing_and_SLA.pdf" and b.page == 2)
        assert pricing.kind == "table"
        row = [ln for ln in pricing.text.split("\n") if ln.startswith("| Standard")]
        assert row, "Standard pricing row not found as a single markdown line"
        assert "500 GB pooled" in row[0] and "$12" in row[0]

    def test_guard_rejects_a_cell_that_is_not_in_the_plain_text(self):
        rows = [["Symptom", "Cause", "Fix"], ["LED red", "degraded", "leWd baoito 1t0"]]
        plain = "Symptom Cause Fix LED red degraded Wait 10 minutes"
        assert _table_is_trustworthy(rows, plain) is False

    def test_guard_accepts_a_faithful_table(self):
        rows = [["Tier", "Price"], ["Free", "$0"], ["Standard", "$12"]]
        plain = "Tier Price Free $0 Standard $12"
        assert _table_is_trustworthy(rows, plain) is True


class TestHeadingDetection:
    def test_numbered_headings_match(self):
        assert _is_heading("1. Authentication")
        assert _is_heading("3.1 Upload a File")
        assert _is_heading("2. Rate Limits")

    def test_prose_and_list_items_do_not_match(self):
        assert not _is_heading("1x Ethernet cable (Cat 6)")
        assert not _is_heading("Tokens expire after 3600 seconds.")
        assert not _is_heading("| Free | 60 | 10 |")


class TestChunking:
    def test_every_document_produces_multiple_chunks(self, chunks):
        """The failure this guards against.

        These documents are 315-591 tokens each. A 512-token window would emit
        one chunk per document, collapsing every citation to "somewhere in
        Employee_Handbook.pdf". If heading detection ever breaks, chunk count
        per document silently drops to 1 - so assert on it.
        """
        per_doc: dict[str, int] = {}
        for c in chunks:
            per_doc[c.doc] = per_doc.get(c.doc, 0) + 1
        for doc, count in per_doc.items():
            assert count >= 3, f"{doc} produced only {count} chunk(s)"

    def test_chunks_respect_the_size_ceiling(self, chunks):
        for c in chunks:
            assert c.n_tokens <= settings.max_chunk_tokens * 1.1, f"{c.id} is {c.n_tokens}t"

    def test_no_empty_or_header_only_chunks(self, chunks):
        for c in chunks:
            assert c.text.strip(), f"{c.id} has no body"
            assert n_tokens(c.text) >= 20, f"{c.id} is a {c.n_tokens}-token orphan"

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_header_prefix_carries_doc_code_and_section(self, chunks):
        c = next(c for c in chunks if c.doc == "Security_Policy.pdf")
        assert "SEC-POL-007" in c.embed_text
        assert c.embed_text.endswith(c.text)

    def test_section_spanning_a_page_break_cites_its_body_page(self, chunks):
        """Product_Manual "3. Storage & Retention" is headed on p2, body on p3.

        Citing the heading's page would send a reviewer to a page that does not
        contain the answer.
        """
        c = next(c for c in chunks if c.doc == "Product_Manual.pdf" and "16TB" in c.text)
        assert c.page == 3

    def test_faq_pairs_split_one_chunk_per_question(self, chunks):
        faq = [c for c in chunks if c.doc == "FAQ_Support.pdf"]
        assert len(faq) == 8
        assert all(c.section.startswith("Q:") for c in faq)

    def test_table_of_contents_is_one_chunk_not_seven_empty_sections(self, chunks):
        toc = [c for c in chunks if c.section == "Table of Contents"]
        assert len(toc) == 1

    def test_markdown_table_rows_are_never_split_from_their_header(self, chunks):
        for c in chunks:
            rows = [ln for ln in c.text.split("\n") if ln.strip().startswith("|")]
            if rows:
                assert any("---" in ln for ln in rows), (
                    f"{c.id} contains table rows without the header separator"
                )

    def test_truncation_flag_survives_into_the_chunk(self, chunks):
        flagged = [c for c in chunks if c.truncated]
        assert flagged, "clipped-line flag was lost between extraction and chunking"
        assert {c.doc for c in flagged} == {"Product_Manual.pdf", "Security_Policy.pdf"}
