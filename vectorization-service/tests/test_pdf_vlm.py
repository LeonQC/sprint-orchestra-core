"""PDF-to-VLM escalation tests; all external parsing/model calls are isolated behind fakes."""
import sys
from unittest.mock import patch

from app.config import Settings
from app.ingest.docling_parser import ParsedAttachmentSection, parse_attachment, parse_attachment_sections
from app.ingest.pdf_vlm import PdfPageCandidate


class _FakeVlm:
    def __init__(self, results: dict[bytes, str] | None = None, error: Exception | None = None):
        self._results = results or {}
        self._error = error
        self.calls: list[tuple[bytes, str]] = []

    async def describe(self, image_data: bytes, mime_type: str) -> str:
        self.calls.append((image_data, mime_type))
        if self._error:
            raise self._error
        return self._results[image_data]

    async def aclose(self) -> None:
        return None


def _settings(**overrides) -> Settings:
    return Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=True, **overrides)


async def test_scan_only_pdf_replaces_empty_docling_result_with_page_labelled_vlm_text():
    vlm = _FakeVlm({b"page-1": "Invoice number: INV-17", b"page-2": "Total: $42"})
    candidates = [
        PdfPageCandidate(1, b"page-1", "little_or_no_embedded_text"),
        PdfPageCandidate(2, b"page-2", "little_or_no_embedded_text"),
    ]
    with patch("app.ingest.docling_parser._convert_pdf_sections_sync", return_value=[]), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 2)
    ):
        result = await parse_attachment(b"pdf", "invoice.pdf", _settings(), vlm)

    assert result == (
        "[VLM extracted PDF page 1 (little_or_no_embedded_text)]\nInvoice number: INV-17\n\n"
        "[VLM extracted PDF page 2 (little_or_no_embedded_text)]\nTotal: $42"
    )
    assert vlm.calls == [(b"page-1", "image/png"), (b"page-2", "image/png")]


async def test_pdf_sections_keep_a_real_one_based_page_number_for_each_vlm_page():
    vlm = _FakeVlm({b"page-2": "Diagram: checkout flow retries once."})
    candidates = [PdfPageCandidate(2, b"page-2", "visual_content")]
    with patch("app.ingest.docling_parser._convert_pdf_sections_sync", return_value=[]), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 2)
    ):
        sections = await parse_attachment_sections(b"pdf", "flow.pdf", _settings(), vlm)

    assert len(sections) == 1
    assert sections[0].page_number == 2
    assert sections[0].content.startswith("[VLM extracted PDF page 2 (visual_content)]")


async def test_mixed_pdf_preserves_good_docling_text_and_appends_visual_page_facts():
    vlm = _FakeVlm({b"page-2": "Visual summary: Error rate rose from 1% to 7%."})
    candidates = [PdfPageCandidate(2, b"page-2", "visual_content")]
    docling_text = "Release notes\n\n" + ("normal selectable text " * 20)
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync",
        return_value=[ParsedAttachmentSection(docling_text)],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 3)
    ):
        result = await parse_attachment(b"pdf", "report.PDF", _settings(), vlm)

    assert result.startswith(docling_text.strip())
    assert "[VLM extracted PDF page 2 (visual_content)]" in result
    assert "Error rate rose from 1% to 7%" in result


async def test_page_aware_docling_sections_replace_only_the_selected_page_with_vlm():
    vlm = _FakeVlm({b"page-2": "VLM truth for page two"})
    candidates = [PdfPageCandidate(2, b"page-2", "visual_content")]
    docling_sections = [
        ParsedAttachmentSection("Docling page one", page_number=1),
        ParsedAttachmentSection("Docling page two", page_number=2),
    ]
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync", return_value=docling_sections
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 3)
    ):
        sections = await parse_attachment_sections(b"pdf", "report.pdf", _settings(), vlm)

    assert [section.page_number for section in sections] == [1, 2]
    assert sections[0].content == "Docling page one"
    assert "VLM truth for page two" in sections[1].content
    assert "Docling page two" not in sections[1].content


async def test_pdf_uses_vlm_when_docling_conversion_fails():
    vlm = _FakeVlm({b"scan": "SKU: W-7734"})
    candidates = [PdfPageCandidate(1, b"scan", "little_or_no_embedded_text")]
    with patch("app.ingest.docling_parser._convert_pdf_sections_sync", side_effect=RuntimeError("bad PDF")), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 1)
    ):
        result = await parse_attachment(b"pdf", "broken.pdf", _settings(), vlm)

    assert "SKU: W-7734" in result
    assert len(vlm.calls) == 1


async def test_pdf_page_vlm_failure_keeps_docling_result():
    vlm = _FakeVlm(error=RuntimeError("model unavailable"))
    candidates = [PdfPageCandidate(1, b"scan", "little_or_no_embedded_text")]
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync",
        return_value=[ParsedAttachmentSection("Docling fallback text")],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 1)
    ):
        result = await parse_attachment(b"pdf", "report.pdf", _settings(), vlm)

    assert result == "Docling fallback text"


async def test_pdf_vlm_is_not_rendered_without_both_feature_flags():
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync",
        return_value=[ParsedAttachmentSection("Docling text")],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates"
    ) as renderer:
        result = await parse_attachment(
            b"pdf", "report.pdf", Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False), _FakeVlm()
        )

    assert result == "Docling text"
    renderer.assert_not_called()


def test_pdf_page_selection_prioritizes_scans_before_visual_pages_and_respects_cap(monkeypatch):
    class _Pixmap:
        def __init__(self, index):
            self.index = index

        def tobytes(self, format):
            assert format == "png"
            return f"page-{self.index}".encode()

    class _Page:
        def __init__(self, index, text, images=0, drawings=0):
            self.index, self.text, self.images, self.drawings = index, text, images, drawings

        def get_text(self, kind):
            assert kind == "text"
            return self.text

        def get_images(self, full):
            assert full is True
            return [object()] * self.images

        def get_drawings(self):
            return [object()] * self.drawings

        def get_pixmap(self, matrix, alpha):
            assert alpha is False
            return _Pixmap(self.index)

    class _Document:
        def __init__(self):
            self.pages = [_Page(1, ""), _Page(2, "readable text", images=1), _Page(3, "more text")]
            self.closed = False

        def __iter__(self):
            return iter(self.pages)

        def __len__(self):
            return len(self.pages)

        def close(self):
            self.closed = True

    document = _Document()

    class _Fitz:
        @staticmethod
        def open(**kwargs):
            assert kwargs == {"stream": b"pdf", "filetype": "pdf"}
            return document

        @staticmethod
        def Matrix(x, y):
            return (x, y)

    monkeypatch.setitem(sys.modules, "fitz", _Fitz)
    from app.ingest.pdf_vlm import render_pdf_candidates

    candidates, total_pages = render_pdf_candidates(
        b"pdf", min_text_chars=10, include_visual_pages=True, max_pages=1, dpi=144
    )

    assert total_pages == 3
    assert candidates == [PdfPageCandidate(1, b"page-1", "little_or_no_embedded_text")]
    assert document.closed


def test_force_pages_are_rendered_even_when_heuristic_selection_is_disabled(monkeypatch):
    class _Pixmap:
        def __init__(self, index):
            self.index = index

        def tobytes(self, format):
            return f"page-{self.index}".encode()

    class _Page:
        def __init__(self, index, text):
            self.index, self.text = index, text

        def get_text(self, kind):
            return self.text

        def get_images(self, full):
            return []

        def get_drawings(self):
            return []

        def get_pixmap(self, matrix, alpha):
            return _Pixmap(self.index)

    class _Document:
        def __init__(self):
            # Page 1 has plenty of native text (would NOT be selected by the text/visual heuristics)
            # and page 2 has none (WOULD be selected). force_pages={1} + heuristic disabled should
            # render only page 1, proving the defect trigger is independent of the text-length guess.
            self.pages = [_Page(1, "plenty of readable text " * 20), _Page(2, "")]

        def __iter__(self):
            return iter(self.pages)

        def __len__(self):
            return len(self.pages)

        def close(self):
            pass

    document = _Document()

    class _Fitz:
        @staticmethod
        def open(**kwargs):
            return document

        @staticmethod
        def Matrix(x, y):
            return (x, y)

    monkeypatch.setitem(sys.modules, "fitz", _Fitz)
    from app.ingest.pdf_vlm import render_pdf_candidates

    candidates, total_pages = render_pdf_candidates(
        b"pdf",
        min_text_chars=10,
        include_visual_pages=True,
        max_pages=12,
        dpi=144,
        force_pages={1: "table_defect"},
        enable_heuristic_selection=False,
    )

    assert total_pages == 2
    assert candidates == [PdfPageCandidate(1, b"page-1", "table_defect")]


async def test_table_defect_page_escalates_to_vlm_even_when_vlm_pdf_enabled_is_false():
    # vlm_pdf_enabled=False (the reverted broad flag) must not suppress the narrow, defect-triggered
    # escalation — only the diagnosed-defective page should reach render_pdf_candidates/the VLM.
    defective_markdown = (
        "| Name | Country | % |\n|---|---|---|\n| (Formerly known as GTC Global Limited) | India | 100.00 % |\n"
    )
    vlm = _FakeVlm({b"page-1": "Western Express Industries Limited (Formerly known as GTC Global Limited)"})
    candidates = [PdfPageCandidate(1, b"page-1", "table_defect")]
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync",
        return_value=[ParsedAttachmentSection(defective_markdown, page_number=1)],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 1)
    ) as renderer:
        sections = await parse_attachment_sections(
            b"pdf", "report.pdf", Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False), vlm
        )

    assert renderer.call_args.kwargs["force_pages"] == {1: "table_defect"}
    assert renderer.call_args.kwargs["enable_heuristic_selection"] is False
    assert len(sections) == 1
    assert "Western Express Industries Limited" in sections[0].content


async def test_table_defect_escalation_is_skipped_when_its_own_flag_is_disabled():
    defective_markdown = (
        "| Name | Country | % |\n|---|---|---|\n| (Formerly known as GTC Global Limited) | India | 100.00 % |\n"
    )
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync",
        return_value=[ParsedAttachmentSection(defective_markdown, page_number=1)],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates"
    ) as renderer:
        result = await parse_attachment(
            b"pdf",
            "report.pdf",
            Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False, vlm_pdf_table_defect_detection_enabled=False),
            _FakeVlm(),
        )

    assert result == defective_markdown
    renderer.assert_not_called()


async def test_low_ocr_confidence_page_escalates_to_vlm_even_when_vlm_pdf_enabled_is_false():
    # Mirrors the table-defect test above for the second targeted signal: a page Docling's own OCR
    # scored badly on (garbled scan) must still reach the VLM even with the broad vlm_pdf_enabled
    # flag off — that's the actual answer to "doesn't turning vlm_pdf_enabled off mean we can't
    # handle a low-confidence attachment anymore?": no, because this evidence-based path is separate.
    vlm = _FakeVlm({b"page-1": "VLM recovered text"})
    candidates = [PdfPageCandidate(1, b"page-1", "ocr_low_confidence")]
    section = ParsedAttachmentSection("garbled docling ocr text", page_number=1, ocr_low_confidence_ratio=0.5)
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync", return_value=[section],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates", return_value=(candidates, 1)
    ) as renderer:
        sections = await parse_attachment_sections(
            b"pdf", "report.pdf", Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False), vlm
        )

    assert renderer.call_args.kwargs["force_pages"] == {1: "ocr_low_confidence"}
    assert renderer.call_args.kwargs["enable_heuristic_selection"] is False
    assert len(sections) == 1
    assert "VLM recovered text" in sections[0].content


async def test_ocr_confidence_escalation_requires_ratio_at_or_above_the_configured_threshold():
    # Default threshold is 0.15 — a page with only a little OCR noise (10%) is normal, not a struggle
    # signal, and must not trigger a VLM call.
    section = ParsedAttachmentSection("mostly clean text", page_number=1, ocr_low_confidence_ratio=0.10)
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync", return_value=[section],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates"
    ) as renderer:
        result = await parse_attachment(
            b"pdf", "report.pdf", Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False), _FakeVlm()
        )

    assert result == "mostly clean text"
    renderer.assert_not_called()


async def test_ocr_confidence_escalation_is_skipped_when_its_own_flag_is_disabled():
    section = ParsedAttachmentSection("garbled text", page_number=1, ocr_low_confidence_ratio=0.9)
    with patch(
        "app.ingest.docling_parser._convert_pdf_sections_sync", return_value=[section],
    ), patch(
        "app.ingest.docling_parser.render_pdf_candidates"
    ) as renderer:
        result = await parse_attachment(
            b"pdf",
            "report.pdf",
            Settings(vlm_ocr_enabled=True, vlm_pdf_enabled=False, vlm_pdf_ocr_confidence_detection_enabled=False),
            _FakeVlm(),
        )

    assert result == "garbled text"
    renderer.assert_not_called()


def test_real_pymupdf_rendering_produces_a_png_page_for_a_scan_like_pdf():
    # Dependency-level smoke test: this is deliberately not mocked, so a Docker image with the
    # declared PyMuPDF dependency catches API/ABI drift before a production attachment does.
    import fitz
    from app.ingest.pdf_vlm import render_pdf_candidates

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Scanned-like invoice")
    data = document.tobytes()
    document.close()

    candidates, total_pages = render_pdf_candidates(
        data, min_text_chars=1_000, include_visual_pages=True, max_pages=12, dpi=144
    )

    assert total_pages == 1
    assert candidates[0].page_number == 1
    assert candidates[0].reason == "little_or_no_embedded_text"
    assert candidates[0].image_data.startswith(b"\x89PNG")


def test_page_cap_logs_which_pages_kept_docling_text(caplog):
    """The cap must not drop pages silently — same failure shape as a truncated VLM transcription.

    A capped ingest returns a well-formed, non-empty parse; the pages past the ceiling quietly keep
    Docling's text, and nothing downstream can tell. The operator has to be told, because only they
    can decide whether the document is worth raising `vlm_pdf_max_pages` for.
    """
    import logging

    from app.ingest import pdf_vlm

    class _FakePage:
        def get_text(self, _mode):
            return ""

        def get_images(self, full=False):
            return []

        def get_drawings(self):
            return []

        def get_pixmap(self, matrix=None, alpha=False):
            class _Pix:
                def tobytes(self, _fmt):
                    return b"png"

            return _Pix()

    class _FakeDoc(list):
        def close(self):
            pass

    fake_doc = _FakeDoc(_FakePage() for _ in range(5))

    class _FakeFitz:
        Matrix = staticmethod(lambda *a: None)

        @staticmethod
        def open(stream=None, filetype=None):
            return fake_doc

    with patch.dict("sys.modules", {"fitz": _FakeFitz}), caplog.at_level(logging.WARNING):
        candidates, total = pdf_vlm.render_pdf_candidates(
            b"", min_text_chars=80, include_visual_pages=True, max_pages=2, dpi=72,
        )

    assert len(candidates) == 2
    assert total == 5
    record = next(r for r in caplog.records if "page cap reached" in r.message)
    assert record.dropped_pages == [3, 4, 5]
    assert record.selected_pages == 5
