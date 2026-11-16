"""Regression tests for the CSV/txt and image-OCR bypass fixes (app/ingest/docling_parser.py).

Found live against a real 132-file attachment corpus: Docling 2.14 has no CSV/plain-text
InputFormat at all and raises ConversionError for them — which the broad except-and-log in
parse_attachment silently turned into empty text for every CSV/txt attachment. These tests lock in
that CSV/txt now bypass Docling entirely rather than round-tripping through (and failing) it, and
that standalone images now go through direct EasyOCR instead of Docling's document/layout pipeline
(which was found, live, to classify real screenshot content as an un-OCR'd "Picture" region).
"""
from io import BytesIO
from unittest.mock import patch

import pytest

from app.config import Settings
from app.ingest.docling_parser import (
    _convert_structured_sections_sync,
    _OcrResult,
    _maybe_escalate_to_vlm,
    _ocr_reader,
    _page_ocr_confidence_ratios,
    _xlsx_used_ranges,
    has_orphan_parenthetical_cell,
    has_orphaned_labels,
    has_recognized_table,
    has_sparse_column_gap,
    parse_attachment,
    parse_attachment_sections,
)


def _fake_png_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class _FakeOcrReader:
    def __init__(self, detections):
        self._detections = detections

    def readtext(self, image):
        return self._detections


class _FakeVlmDescriber:
    def __init__(self, result=""):
        self._result = result
        self.calls = []

    async def describe(self, image_data, mime_type):
        self.calls.append((image_data, mime_type))
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def aclose(self):
        return None


async def test_csv_bypasses_docling_and_returns_raw_text():
    settings = Settings()
    data = b"issue,status\nATC-1,done\nATC-2,in_progress\n"
    result = await parse_attachment(data, "work-plan.csv", settings)
    assert result == "issue,status\nATC-1,done\nATC-2,in_progress\n"


async def test_txt_bypasses_docling_and_returns_raw_text():
    settings = Settings()
    result = await parse_attachment(b"plain notes here", "notes.txt", settings)
    assert result == "plain notes here"


async def test_csv_filename_is_case_insensitive():
    settings = Settings()
    result = await parse_attachment(b"a,b\n1,2\n", "DATA.CSV", settings)
    assert result == "a,b\n1,2\n"


async def test_invalid_utf8_in_plain_text_does_not_raise():
    settings = Settings()
    result = await parse_attachment(b"\xff\xfe not valid utf-8", "bad.txt", settings)
    assert isinstance(result, str)  # errors="replace" -> some decodable string, not an exception


async def test_log_extension_bypasses_docling_too():
    settings = Settings()
    result = await parse_attachment(b"2026-07-14T09:00Z checkout retry x3\n", "duplicate-checkout.log", settings)
    assert result == "2026-07-14T09:00Z checkout retry x3\n"


async def test_docling_rejecting_a_format_logs_and_returns_empty_without_crashing():
    """Regression test for a real bug found live: the except-branch's own logging call used to raise
    KeyError("Attempt to overwrite 'filename' in LogRecord") — `filename` collides with a reserved
    stdlib LogRecord attribute — which crashed the exception handler itself instead of the one bad
    attachment it was trying to skip past. This exercises that exact code path end to end (a
    genuinely Docling-unsupported extension, not one of the plain-text bypass ones) to prove the fix.
    """
    settings = Settings()
    result = await parse_attachment(b"whatever bytes", "mystery.xyz123", settings)
    assert result == ""


async def test_image_ocr_extracts_text_via_direct_easyocr_bypassing_docling():
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, docling_ocr_engine="easyocr")
    # bbox corners: top-left, top-right, bottom-right, bottom-left (EasyOCR's convention)
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "HELLO", 0.99),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "WORLD", 0.95),
    ]
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings)
    _ocr_reader.cache_clear()
    assert result == "HELLO WORLD"


async def test_image_ocr_sorts_into_reading_order_not_raw_detection_order():
    # Regression-relevant: real EasyOCR output ordering is not reading order (see module docstring's
    # "row_key" sort rationale) — detections here are deliberately shuffled (row 2 before row 1,
    # right-to-left within a row) to prove the sort, not just that concatenation happens.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, docling_ocr_engine="easyocr")
    detections = [
        ([[60, 20], [100, 20], [100, 30], [60, 30]], "WORLD", 0.9),  # row 2, right
        ([[50, 0], [90, 0], [90, 10], [50, 10]], "THERE", 0.9),      # row 1, right
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "HELLO", 0.9),      # row 2, left
        ([[0, 0], [40, 0], [40, 10], [0, 10]], "HI", 0.9),           # row 1, left
    ]
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings)
    _ocr_reader.cache_clear()
    assert result == "HI THERE HELLO WORLD"


async def test_image_ocr_drops_low_confidence_detections():
    # Found live on a synthetically degraded (rotated/blurred/low-contrast/re-compressed) test image:
    # garbled OCR fragments ("Mx", the mangled read of "SKU:"+"Pallet") scored 0.006-0.06, while real
    # words from the SAME image ("62", "SKU") scored 0.7-1.0 — a clean gap. Unfiltered, an agent
    # confidently presented "Mx" as a real SKU value in a live test; a faithfulness check can't catch
    # this class of error since the answer WAS faithful to what got retrieved.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, ocr_min_confidence=0.4, docling_ocr_engine="easyocr")
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "SKU", 0.72),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "Mx", 0.02),  # garbled — must be dropped
        ([[0, 40], [50, 40], [50, 50], [0, 50]], "62", 0.98),
    ]
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings)
    _ocr_reader.cache_clear()
    assert result == "SKU 62"
    assert "Mx" not in result


async def test_image_ocr_skipped_entirely_when_docling_do_ocr_disabled():
    # No text layer exists on a raw image without OCR, unlike a PDF that might have one — so this
    # correctly returns empty rather than attempting (and failing) some other extraction.
    settings = Settings(docling_do_ocr=False)
    result = await parse_attachment(_fake_png_bytes(), "shot.png", settings)
    assert result == ""


async def test_vlm_escalation_is_skipped_entirely_when_disabled_even_if_ocr_dropped_detections():
    # A dropped detection alone must not trigger a VLM call unless the feature is explicitly turned
    # on — this is a real per-image LLM cost. vlm_ocr_enabled is passed explicitly (not left to the
    # class default) because Settings reads app/../.env (VEC_ prefix) via pydantic-settings'
    # env_file, so a real local .env with VEC_VLM_OCR_ENABLED=true — a legitimate dev/prod config,
    # not a test fixture — would otherwise silently flip this test's premise out from under it. Every
    # other test in this file already does this explicitly for vlm_ocr_enabled=True; this is the one
    # that also needs it for =False, to test the disabled case regardless of ambient .env state.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, ocr_min_confidence=0.4, vlm_ocr_enabled=False, docling_ocr_engine="easyocr")
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "SKU", 0.72),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "Mx", 0.02),
    ]
    vlm = _FakeVlmDescriber(result="should never be used")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "SKU"
    assert vlm.calls == []


async def test_vlm_escalation_fires_and_replaces_the_result_when_ocr_dropped_a_detection():
    """A real scenario: a confidence-filtered OCR result still has a gap where
    the garbled detection got dropped. With VLM escalation on, the whole image is re-read and its
    (better) transcription replaces the partial OCR text rather than being appended to it.
    """
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, ocr_min_confidence=0.4, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "SKU", 0.72),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "Mx", 0.02),  # dropped -> triggers escalation
    ]
    vlm = _FakeVlmDescriber(result="SKU: W-7734\nPallet count: 62")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "SKU: W-7734\nPallet count: 62"
    assert len(vlm.calls) == 1
    image_data, mime_type = vlm.calls[0]
    assert mime_type == "image/png"
    assert image_data  # the real image bytes were forwarded, not some placeholder


async def test_vlm_escalation_fires_when_ocr_found_nothing_at_all():
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    vlm = _FakeVlmDescriber(result="WAREHOUSE INTAKE NOTE")
    with patch("easyocr.Reader", return_value=_FakeOcrReader([])):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "WAREHOUSE INTAKE NOTE"
    assert len(vlm.calls) == 1


async def test_incomplete_vlm_output_cannot_discard_a_full_dense_ocr_page():
    """Regression for a real two-column page where OCR contained the answer in 2,259 chars but the
    VLM returned only the first 219-char paragraph. Escalation is a fallback, not permission to erase
    ~90% of readable source content with a plausible-looking partial response.
    """
    ocr_text = " ".join(f"word{i}" for i in range(180))
    ocr_result = _OcrResult(text=ocr_text, dropped_count=1)
    settings = Settings(vlm_ocr_enabled=True, vlm_min_ocr_coverage_ratio=0.5)
    vlm = _FakeVlmDescriber(result="only the first paragraph was returned")

    result = await _maybe_escalate_to_vlm(b"image", "image/png", ocr_result, settings, vlm)

    assert result == ocr_text
    assert len(vlm.calls) == 1


async def test_complete_vlm_output_still_replaces_dense_ocr():
    ocr_text = " ".join(f"word{i}" for i in range(180))
    ocr_result = _OcrResult(text=ocr_text, dropped_count=1)
    settings = Settings(vlm_ocr_enabled=True, vlm_min_ocr_coverage_ratio=0.5)
    vlm_text = "V" * int(len(ocr_text) * 0.6)
    vlm = _FakeVlmDescriber(result=vlm_text)

    result = await _maybe_escalate_to_vlm(b"image", "image/png", ocr_result, settings, vlm)

    assert result == vlm_text


async def test_vlm_escalation_does_not_fire_when_ocr_succeeded_cleanly():
    # No dropped detections and non-empty text -> OCR looked fine, no reason to spend a VLM call.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = [([[0, 0], [50, 0], [50, 10], [0, 10]], "HELLO", 0.99)]
    vlm = _FakeVlmDescriber(result="should never be used")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "HELLO"
    assert vlm.calls == []


def _table_detections(rows: int = 4, columns: int = 5, height: int = 10) -> list:
    """Geometry of a plain grid: `columns` cells per row, separated by gaps far wider than a word
    space. Modelled on the real 7-column financial-disclosure page in the DocVQA fixtures whose flat
    OCR text lost every column association (see docling_parser._column_row_ratio)."""
    detections = []
    for row in range(rows):
        top = row * height * 3
        for column in range(columns):
            left = column * 200  # 200px pitch vs 10px text height -> unmistakably a column boundary
            detections.append((
                [[left, top], [left + 40, top], [left + 40, top + height], [left, top + height]],
                f"r{row}c{column}", 0.99,
            ))
    return detections


async def test_a_crisply_read_table_escalates_to_vlm_on_layout_even_with_no_dropped_detections():
    """The failure the confidence-based trigger was structurally unable to catch.

    A high-quality table scan produces NO low-confidence detections — it is the easiest thing on the
    page to read — and the old `dropped_count > 0` gate therefore never fired on it, while the flat
    space-joined serialization destroyed every cell's column. Measured live on 50 real scanned pages
    before this signal existed; see `_column_row_ratio`.
    """
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    vlm = _FakeVlmDescriber(result="| A | B |\n|---|---|\n| 1 | 2 |")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(_table_detections())):
        result = await parse_attachment(_fake_png_bytes(), "table.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert len(vlm.calls) == 1, "a confidently-read table must still escalate"
    assert result.startswith("| A | B |")


async def test_repeated_two_column_form_escalates_without_case_specific_labels():
    """Two-column forms lose label/value association despite having only one gap per row."""
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = []
    for row in range(6):
        top = row * 40
        detections.extend([
            ([[50, top], [180, top], [180, top + 20], [50, top + 20]], f"left-{row}", 0.99),
            ([[700, top], [850, top], [850, top + 20], [700, top + 20]], f"right-{row}", 0.99),
        ])
    vlm = _FakeVlmDescriber(result="PRIMARY OWNER: Nora Chen\nBACKUP OWNER: Hugo Park")

    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "two-column-form.png", settings, vlm)
    _ocr_reader.cache_clear()

    assert len(vlm.calls) == 1
    assert result.startswith("PRIMARY OWNER: Nora Chen")


async def test_prose_does_not_escalate_just_because_a_line_has_several_words():
    """The precision half: an ordinary text page is many rows of 3+ detections too. What separates it
    from a table is the GAPS, not the count — so word-spaced runs must score zero and cost nothing."""
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = []
    for row in range(6):
        top = row * 30
        for word in range(6):
            left = word * 45  # normal word spacing: gap (~5px) is under one 10px line height
            detections.append((
                [[left, top], [left + 40, top], [left + 40, top + 10], [left, top + 10]],
                f"word{word}", 0.99,
            ))
    vlm = _FakeVlmDescriber(result="should never be used")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        await parse_attachment(_fake_png_bytes(), "prose.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert vlm.calls == [], "prose must not be escalated as if it were tabular"


async def test_column_escalation_can_be_turned_off_without_disabling_vlm_entirely():
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, vlm_ocr_enabled=True, docling_ocr_engine="easyocr",
                        vlm_image_column_detection_enabled=False)
    vlm = _FakeVlmDescriber(result="should never be used")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(_table_detections())):
        await parse_attachment(_fake_png_bytes(), "table.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert vlm.calls == []


async def test_reading_order_row_grouping_is_independent_of_image_resolution():
    """Same page, same layout, two scan resolutions -> the same reading order.

    The previous grouping used a fixed 20px band, which is smaller than one line of text on a 300-DPI
    scan (splitting single lines) and larger than three lines on a small screenshot (merging them).
    Scaling the band to the median detection height makes the result depend on the layout instead of
    on how big the file happens to be.
    """
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, docling_ocr_engine="easyocr")

    def two_rows(scale: int) -> list:
        s = scale
        return [
            ([[60 * s, 0], [100 * s, 0], [100 * s, 12 * s], [60 * s, 12 * s]], "THERE", 0.9),
            ([[0, 0], [50 * s, 0], [50 * s, 12 * s], [0, 12 * s]], "HI", 0.9),
            ([[60 * s, 30 * s], [100 * s, 30 * s], [100 * s, 42 * s], [60 * s, 42 * s]], "WORLD", 0.9),
            ([[0, 30 * s], [50 * s, 30 * s], [50 * s, 42 * s], [0, 42 * s]], "HELLO", 0.9),
        ]

    results = []
    for scale in (1, 8):
        with patch("easyocr.Reader", return_value=_FakeOcrReader(two_rows(scale))):
            results.append(await parse_attachment(_fake_png_bytes(), "shot.png", settings))
        _ocr_reader.cache_clear()
    assert results[0] == "HI THERE HELLO WORLD"
    assert results[0] == results[1], f"resolution changed the reading order: {results}"


async def test_vlm_escalation_falls_back_to_the_ocr_result_when_the_vlm_says_no_text_found():
    # The VLM's explicit "nothing here" sentinel must not blank out a real (if partial) OCR result.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, ocr_min_confidence=0.4, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "SKU", 0.72),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "Mx", 0.02),
    ]
    vlm = _FakeVlmDescriber(result="(no text found)")
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "SKU"


async def test_vlm_escalation_falls_back_to_the_ocr_result_when_the_vlm_call_itself_fails():
    # A VLM outage (network error, 5xx, auth failure) must degrade to the OCR result, never crash
    # ingestion of the one attachment that happened to trigger escalation.
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, ocr_min_confidence=0.4, vlm_ocr_enabled=True, docling_ocr_engine="easyocr")
    detections = [
        ([[0, 0], [50, 0], [50, 10], [0, 10]], "SKU", 0.72),
        ([[0, 20], [50, 20], [50, 30], [0, 30]], "Mx", 0.02),
    ]
    vlm = _FakeVlmDescriber(result=RuntimeError("hosted VLM endpoint unreachable"))
    with patch("easyocr.Reader", return_value=_FakeOcrReader(detections)):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings, vlm)
    _ocr_reader.cache_clear()
    assert result == "SKU"


async def test_vlm_escalation_never_fires_for_non_image_attachments():
    settings = Settings(vlm_ocr_enabled=True)
    vlm = _FakeVlmDescriber(result="should never be used")
    result = await parse_attachment(b"issue,status\nATC-1,done\n", "data.csv", settings, vlm)
    assert result == "issue,status\nATC-1,done\n"
    assert vlm.calls == []


async def test_image_ocr_failure_returns_empty_without_crashing():
    _ocr_reader.cache_clear()
    settings = Settings(docling_do_ocr=True, docling_ocr_engine="easyocr")
    with patch("easyocr.Reader", side_effect=RuntimeError("model load failed")):
        result = await parse_attachment(_fake_png_bytes(), "shot.png", settings)
    _ocr_reader.cache_clear()
    assert result == ""


async def test_pptx_sections_preserve_slide_provenance_from_docling(monkeypatch):
    expected = [
        {"content": "## Slide 1\nIntro", "provenance": {"source_type": "pptx", "slide_number": 1}},
        {"content": "## Slide 2\nArchitecture", "provenance": {"source_type": "pptx", "slide_number": 2}},
    ]
    monkeypatch.setattr(
        "app.ingest.docling_parser._convert_structured_sections_sync",
        lambda *args: [
            __import__("app.ingest.docling_parser", fromlist=["ParsedAttachmentSection"]).ParsedAttachmentSection(**item)
            for item in expected
        ],
    )

    sections = await parse_attachment_sections(b"pptx", "deck.pptx", Settings())

    assert [section.provenance for section in sections] == [item["provenance"] for item in expected]


async def test_image_sections_expose_a_pixel_bbox(monkeypatch):
    monkeypatch.setattr(
        "app.ingest.docling_parser.parse_attachment",
        lambda *args: _async_text("OCR text"),
    )

    settings = Settings(docling_do_ocr=True, docling_ocr_engine="easyocr")
    sections = await parse_attachment_sections(_fake_png_bytes(), "screen.png", settings)

    assert sections[0].provenance["source_type"] == "image"
    assert sections[0].provenance["bbox"] == [0, 0, 10, 10]


async def _async_text(value):
    return value


def test_xlsx_provenance_uses_the_workbook_used_range():
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Q2 Revenue"
    sheet["B4"] = "Total"
    sheet["F18"] = 42
    output = BytesIO()
    workbook.save(output)
    workbook.close()

    assert _xlsx_used_ranges(output.getvalue()) == {"Q2 Revenue": "B4:F18"}


def test_each_xlsx_sheet_is_its_own_section_with_its_name():
    # Real Docling, not a fake: 2.150 renamed the sheet groups ("sheet: Prices" -> "Prices" with
    # label "sheet") and the whole workbook silently became one section with no sheet name.
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active.title = "Prices"
    workbook.active.append(["Part", "Price"])
    workbook.active.append(["CM4 108064", 195])
    stock = workbook.create_sheet("Stock")
    stock.append(["Part", "Qty"])
    stock.append(["CM4 108064", 5])
    output = BytesIO()
    workbook.save(output)

    sections = _convert_structured_sections_sync(output.getvalue(), "parts.xlsx", False, False)

    assert [s.provenance.get("sheet_name") for s in sections] == ["Prices", "Stock"]
    assert "195" in sections[0].content and "Qty" in sections[1].content


def test_orphan_parenthetical_cell_matches_the_diagnosed_defect_shape():
    # Real shape: TableFormer collapsed a two-line cell down to only the
    # parenthetical continuation, dropping "Western Express Industries Limited" entirely.
    markdown = (
        "| Name of the Subsidiary Company | Country | Share Holding Percentage |\n"
        "|---|---|---|\n"
        "| Golden Investment (Sikkim) Pvt. Limited | India | 99.97 % |\n"
        "| (Formerly known as GTC Global Limited) | India | 100.00 % |\n"
    )
    assert has_orphan_parenthetical_cell(markdown)


def test_orphan_parenthetical_cell_does_not_fire_on_a_normal_table():
    markdown = (
        "| Name | Country | Share % |\n"
        "|---|---|---|\n"
        "| Golden Investment (Sikkim) Pvt. Limited | India | 99.97 % |\n"
        "| Western Express Industries Limited | India | 100.00 % |\n"
    )
    assert not has_orphan_parenthetical_cell(markdown)


def test_orphan_parenthetical_cell_ignores_prose_parentheticals_outside_tables():
    # A parenthetical remark is completely normal in running prose (e.g. Notes to Accounts) — only a
    # table CELL that is nothing but a parenthetical is the signal, not the substring anywhere in text.
    markdown = (
        "Guarantees given by the holding Company (excluding Excise Duty referred to in Note (c) "
        "below): Rs.85,25,000 (Previous Year Rs.1,10,25,000)."
    )
    assert not has_orphan_parenthetical_cell(markdown)


def test_orphan_parenthetical_cell_ignores_markdown_separator_rows():
    markdown = "| A | B |\n|---|---|\n| (Also known as X) | y |\n"
    assert has_orphan_parenthetical_cell(markdown)  # sanity: the real data row still matches
    assert not has_orphan_parenthetical_cell("| --- | :--: |\n")  # separator-only: no data rows at all


def test_orphan_parenthetical_cell_ignores_negative_accounting_numbers():
    # Found live on the same real document (page 6): parentheses conventionally denote a negative /
    # prior-year figure in accounting notation, not a dropped line. A naive "any standalone
    # parenthetical cell" version of this heuristic false-positived on exactly this shape.
    markdown = (
        "| Expenses | Current | Previous |\n|---|---|---|\n"
        "| Remuneration | 59,87,060 | (59,95,296) |\n"
    )
    assert not has_orphan_parenthetical_cell(markdown)


def test_orphan_parenthetical_cell_ignores_split_unit_headers_and_labels():
    # Found live (page 3 / page 7): a header like "High (Rs.)" rendered as two adjacent cells, and
    # (page 5) a standalone cross-reference "(Refer Note hereinbelow)" — both legitimate, complete
    # content, not a dropped preceding line. Neither uses the formerly/previously/also-known-as
    # rename phrasing the real defect has, so neither should trigger escalation.
    markdown = (
        "| Month | High | (Rs.) |\n|---|---|---|\n"
        "| April, 2004 | 18.00 | (Rs.) |\n"
        "| Under Amnesty Scheme | 1,58,39,988 | (Refer Note hereinbelow) |\n"
        "| (a) | Guarantees given | 85,25,000 |\n"
    )
    assert not has_orphan_parenthetical_cell(markdown)


def test_sparse_column_gap_matches_the_diagnosed_defect_shape():
    # Real shape from a scanned page: the top scorer's name (24 out of 24 meetings) was
    # dropped while every other member's name, and the count itself, survived — confirmed live by
    # rendering the source page and reading it directly, and by a VLM re-read recovering
    # "Shri R. R. Kumar" for this exact row.
    markdown = (
        "| Name of member | Category | No. of meetings attended out of 24 |\n"
        "|---|---|---|\n"
        "|  | Non-Executive Independent Director | 24 |\n"
        "| Shri M. H. Patil | Non-Executive Independent Director | 8 |\n"
        "| Shri Viney Mehra | Executive Independent Director | 16 |\n"
        "| Shri A. K. Joshi | Executive Independent Director | 21 |\n"
    )
    assert has_sparse_column_gap(markdown)


def test_sparse_column_gap_does_not_fire_on_a_normal_table():
    markdown = (
        "| Name of member | Category | No. of meetings attended out of 24 |\n"
        "|---|---|---|\n"
        "| Shri R. R. Kumar | Non-Executive Independent Director | 24 |\n"
        "| Shri M. H. Patil | Non-Executive Independent Director | 8 |\n"
        "| Shri Viney Mehra | Executive Independent Director | 16 |\n"
        "| Shri A. K. Joshi | Executive Independent Director | 21 |\n"
    )
    assert not has_sparse_column_gap(markdown)


def test_sparse_column_gap_ignores_a_total_row_missing_its_label():
    # Real shape from the same document's remuneration table (page 2): the Total row has no director
    # name, which is correct — a naive version of this check would otherwise treat "Name" as a
    # usually-populated column with one inexplicable gap and escalate every table with a total row.
    markdown = (
        "| Sr. No | Name of the Directors | Total Rs. |\n"
        "|---|---|---|\n"
        "| 1 | Shri R. R. Kumar | 1,00,500 |\n"
        "| 3 | Shri Bharat B. Merchant | 30,000 |\n"
        "| 4 | Shri M. H. Patil | 40,500 |\n"
        "| 5 | Shri A.K. Joshi | 14,65,383 |\n"
        "|  | Total | 61,70,060 |\n"
    )
    assert not has_sparse_column_gap(markdown)


def test_sparse_column_gap_ignores_a_column_that_is_normally_sparse():
    # Real shape from the same document's board-attendance table: a "Chairman" committee-position
    # column that's blank for almost everyone must not trigger just because it stays blank for one
    # more row — the column was never usually-populated to begin with.
    markdown = (
        "| Name of the Directors | Category | Chairman |\n"
        "|---|---|---|\n"
        "| Shri J | Managing Director | 2 |\n"
        "| Shri Viney Mehra | Whole-time Director |  |\n"
        "| Shri M. H. Patil | Non-executive |  |\n"
        "| Shri Bharat B. Merchant | Non-executive |  |\n"
    )
    assert not has_sparse_column_gap(markdown)


def test_sparse_column_gap_requires_the_gap_row_to_have_other_content():
    # A row that's entirely blank across every column isn't a dropped-value defect — it's noise
    # (an artifact row Docling shouldn't have emitted at all), and escalating for it would be a call
    # spent on nothing.
    markdown = (
        "| Name | Category | Count |\n"
        "|---|---|---|\n"
        "| Shri A | Independent | 24 |\n"
        "| Shri B | Independent | 8 |\n"
        "| Shri C | Independent | 16 |\n"
        "|  |  |  |\n"
    )
    assert not has_sparse_column_gap(markdown)


def test_sparse_column_gap_ignores_short_tables_below_the_row_minimum():
    # Fewer than min_data_rows candidate rows means there isn't enough evidence that the column is
    # "usually populated" at all — three rows could just as easily be a small, legitimately partial
    # table.
    markdown = "| Name | Count |\n|---|---|\n|  | 24 |\n| Shri B | 8 |\n| Shri C | 16 |\n"
    assert not has_sparse_column_gap(markdown)


class _FakeOcrCell:
    def __init__(self, confidence):
        self.confidence = confidence


class _FakeDigitalCell:
    """A born-digital (non-OCR'd) cell: Docling's real `Cell` base class has no `confidence` field."""


class _FakePage:
    def __init__(self, page_no, cells):
        self.page_no = page_no  # 0-based, matching docling.datamodel.base_models.Page
        self.cells = cells


class _FakeConversionResult:
    def __init__(self, pages):
        self.pages = pages


def test_page_ocr_confidence_ratios_computes_fraction_below_threshold():
    result = _FakeConversionResult([
        _FakePage(0, [_FakeOcrCell(0.9), _FakeOcrCell(0.95), _FakeOcrCell(0.2)]),  # 1/3 below 0.5
        _FakePage(1, [_FakeOcrCell(0.9), _FakeOcrCell(0.95)]),  # 0/2 below 0.5
        _FakePage(2, [_FakeDigitalCell()]),  # no OCR at all — must be absent, not 0.0
    ])

    ratios = _page_ocr_confidence_ratios(result, min_confidence=0.5)

    # 0-based page_no -> 1-based page number
    assert ratios[1] == pytest.approx(1 / 3)
    assert ratios[2] == 0.0
    assert 3 not in ratios


def test_page_ocr_confidence_ratios_empty_when_no_pages():
    assert _page_ocr_confidence_ratios(_FakeConversionResult([]), min_confidence=0.5) == {}


class _FakeBBox:
    def __init__(self, l, t):  # noqa: E741 - matches Docling's real BoundingBox.l/.t attribute names
        self.l = l
        self.t = t


class _FakeTextCell:
    def __init__(self, text, l=0, t=0):  # noqa: E741
        self.text = text
        self.bbox = _FakeBBox(l, t)


def test_reading_order_text_from_cells_sorts_top_to_bottom_left_to_right():
    from app.ingest.docling_parser import _reading_order_text_from_cells

    # Docling's Cell.bbox is TOPLEFT-origin, y increasing DOWNWARD. Shuffled here to prove the sort
    # runs, not just that concatenation happens.
    page = _FakePage(0, [
        _FakeTextCell("WORLD", l=60, t=20),
        _FakeTextCell("HI", l=0, t=0),
        _FakeTextCell("HELLO", l=0, t=20),
        _FakeTextCell("THERE", l=50, t=0),
    ])
    assert _reading_order_text_from_cells(page) == "HI THERE HELLO WORLD"


def test_picture_cluster_recovered_text_requires_a_minimum_length():
    from app.ingest.docling_parser import _picture_cluster_recovered_text

    result = _FakeConversionResult([
        _FakePage(0, [_FakeTextCell("substantial recovered content here", l=0, t=0)]),
        _FakePage(1, [_FakeTextCell("hi", l=0, t=0)]),  # too short — likely real decorative content
    ])
    recovered = _picture_cluster_recovered_text(result, min_recovered_chars=20)
    assert 1 in recovered  # 1-based page number for 0-based page_no=0
    assert 2 not in recovered


def test_picture_only_page_is_recovered_from_raw_cells(monkeypatch):
    # End-to-end regression for the real bug found live: export_to_markdown returned
    # literally "<!-- image -->" for a page Docling's layout classifier mislabelled, even though OCR
    # had genuinely succeeded — 16 real cells recovered directly from page.cells.
    class _FakeDocument:
        def __init__(self, pages_map, markdown_by_page):
            self._pages_map = pages_map
            self._markdown = markdown_by_page

        @property
        def pages(self):
            return self._pages_map

        def export_to_markdown(self, page_no=None):
            return self._markdown[page_no]

    class _FakeResult:
        def __init__(self, document, pages):
            self.document = document
            self.pages = pages

    fake_cells = [_FakeTextCell(w, l=i * 10, t=0) for i, w in enumerate(
        ["To", "Date", "Time", "TELEPHONED", "URGENT", "PLEASE", "CALL", "Message", "operator", "name"]
    )]
    document = _FakeDocument(pages_map={1: object()}, markdown_by_page={1: "<!-- image -->"})
    result = _FakeResult(document, [_FakePage(0, fake_cells)])

    monkeypatch.setattr(
        "app.ingest.docling_parser._build_converter",
        lambda *a, **k: type("C", (), {"convert": lambda self, source: result})(),
    )
    from app.ingest.docling_parser import _convert_pdf_sections_sync

    sections = _convert_pdf_sections_sync(b"pdf-bytes", "slip.pdf", True, True)
    assert len(sections) == 1
    assert sections[0].provenance["extraction"] == "docling_cell_recovery"
    assert "TELEPHONED" in sections[0].content
    assert "To Date Time TELEPHONED" in sections[0].content


def test_mixed_page_with_a_small_real_fragment_still_recovers(monkeypatch):
    # Found live: a page can export SOMETHING real (e.g. a form-ID stamp) alongside a picture-labelled
    # cluster that swallowed everything else — exact-match-on-"only <!-- image -->" would miss this
    # case entirely and leave most of the page's real content discarded.
    class _FakeDocument:
        def __init__(self, pages_map, markdown_by_page):
            self._pages_map = pages_map
            self._markdown = markdown_by_page

        @property
        def pages(self):
            return self._pages_map

        def export_to_markdown(self, page_no=None):
            return self._markdown[page_no]

    class _FakeResult:
        def __init__(self, document, pages):
            self.document = document
            self.pages = pages

    fake_cells = [_FakeTextCell(w, l=i * 10, t=0) for i, w in enumerate(
        ["To", "Date", "Time", "TELEPHONED", "URGENT", "PLEASE", "CALL", "Message", "MC", "0343"]
    )]
    document = _FakeDocument(pages_map={1: object()}, markdown_by_page={1: "<!-- image -->\n\nMC 0343"})
    result = _FakeResult(document, [_FakePage(0, fake_cells)])

    monkeypatch.setattr(
        "app.ingest.docling_parser._build_converter",
        lambda *a, **k: type("C", (), {"convert": lambda self, source: result})(),
    )
    from app.ingest.docling_parser import _convert_pdf_sections_sync

    sections = _convert_pdf_sections_sync(b"pdf-bytes", "slip.pdf", True, True)
    assert sections[0].provenance["extraction"] == "docling_cell_recovery"
    assert "TELEPHONED" in sections[0].content


def test_a_genuinely_complete_export_is_not_overwritten_by_recovery(monkeypatch):
    # A page with a real, substantial export AND an incidental embedded figure reference must be left
    # alone — recovery is for pages the export failed on, not a general "prefer raw cells" policy.
    class _FakeDocument:
        def __init__(self, pages_map, markdown_by_page):
            self._pages_map = pages_map
            self._markdown = markdown_by_page

        @property
        def pages(self):
            return self._pages_map

        def export_to_markdown(self, page_no=None):
            return self._markdown[page_no]

    class _FakeResult:
        def __init__(self, document, pages):
            self.document = document
            self.pages = pages

    real_export = (
        "<!-- image -->\n\nThis is a complete, well-formed page with substantial real exported "
        "content that Docling's serializer produced correctly, alongside one incidental figure "
        "reference that legitimately has nothing more to say about it."
    )
    fake_cells = [_FakeTextCell("short", l=0, t=0)]  # far shorter than the real export above
    document = _FakeDocument(pages_map={1: object()}, markdown_by_page={1: real_export})
    result = _FakeResult(document, [_FakePage(0, fake_cells)])

    monkeypatch.setattr(
        "app.ingest.docling_parser._build_converter",
        lambda *a, **k: type("C", (), {"convert": lambda self, source: result})(),
    )
    from app.ingest.docling_parser import _convert_pdf_sections_sync

    sections = _convert_pdf_sections_sync(b"pdf-bytes", "slip.pdf", True, True)
    assert "extraction" not in sections[0].provenance
    assert sections[0].content == real_export


def test_build_converter_selects_the_configured_ocr_engine():
    from app.ingest.docling_parser import _build_converter

    # The engine is a measured choice (easyocr 23/31 vs tesseract
    # 18/31 facts extracted on the real DocVQA fixture), so a silently-ignored setting would quietly
    # cost real accuracy while the config file claims otherwise.
    _build_converter.cache_clear()
    easy = _build_converter(True, True, "easyocr", "fast")
    tess = _build_converter(True, True, "tesseract", "accurate")

    from docling.datamodel.base_models import InputFormat

    easy_opts = easy.format_to_options[InputFormat.PDF].pipeline_options
    tess_opts = tess.format_to_options[InputFormat.PDF].pipeline_options
    assert easy_opts.ocr_options.kind == "easyocr"
    assert tess_opts.ocr_options.kind == "tesseract"
    assert easy_opts.table_structure_options.mode.value == "fast"
    assert tess_opts.table_structure_options.mode.value == "accurate"


def test_build_converter_caches_per_config_not_globally():
    from app.ingest.docling_parser import _build_converter
    from docling.datamodel.base_models import InputFormat

    # Regression guard: the original lru_cache(maxsize=1) was keyed only on (do_ocr, do_tables), so
    # asking for a second engine returned the FIRST converter — the engine setting would have been
    # silently inert for every attachment after the first.
    _build_converter.cache_clear()
    first = _build_converter(True, True, "easyocr", "fast")
    second = _build_converter(True, True, "tesseract", "accurate")
    assert first is not second
    assert first is _build_converter(True, True, "easyocr", "fast")  # same config still cached
    assert second.format_to_options[InputFormat.PDF].pipeline_options.ocr_options.kind == "tesseract"


def test_build_converter_rejects_an_unknown_engine_instead_of_silently_using_the_default():
    from app.ingest.docling_parser import _build_converter

    _build_converter.cache_clear()
    with pytest.raises(ValueError, match="unknown docling_ocr_engine"):
        _build_converter(True, True, "not-a-real-engine", "fast")


def test_recognized_table_detected_and_protected():
    """A parsed table is the signal to leave a page alone, so it must be recognised as one."""
    markdown = (
        "## Committee attendance\n\n"
        "| Director | Meetings | Chairman |\n"
        "|---|---|---|\n"
        "| Shri R. R. Kumar | 24 | Yes |\n"
        "| Shri M. H. Patil | 8 | |\n"
        "| Shri A. K. Joshi | 11 | |\n"
    )
    assert has_recognized_table(markdown) is True


def test_flattened_form_is_not_a_recognized_table():
    """The failure this routing exists for: a form Docling flattened, no table, no visible defect.

    Taken from a real scanned page, where the reference answer ("5954 B") is not in the text at
    all and the surviving values have detached from their labels.
    """
    markdown = (
        "## CSF Run Sheet\n\nDate;\n\ncsF-222\n\nRun Nos.\n\n"
        "Net Pounds Infeed:\n\nNet Pounds Out:\\_\n\n893\n"
    )
    assert has_recognized_table(markdown) is False


def test_prose_with_pipe_characters_is_not_a_table():
    """"Any line with a pipe" would misfire on prose; a table needs its alignment row."""
    markdown = "The command is `grep foo | wc -l` and the shell pipes | between them.\n"
    assert has_recognized_table(markdown) is False


def test_two_line_table_fragment_is_not_worth_protecting():
    """A header plus one row carries no structure a re-read could damage; don't block escalation."""
    markdown = "| Field | Value |\n|---|---|\n| PD- | |\n"
    assert has_recognized_table(markdown) is False


def test_flattened_form_has_orphaned_labels():
    """The escalation case: labels stranded on their own lines, values detached."""
    markdown = (
        "## CSF Run Sheet\n\nDate;\n\ncsF-222\n\nRun Nos.\n\n"
        "Net Pounds Infeed:\n\nNet Pounds Out:\\_\n\n893\n"
    )
    assert has_orphaned_labels(markdown) is True


def test_intact_label_value_lines_are_protected():
    """Fields whose value sits on the SAME line are not broken and must not be escalated.

    Shape taken from this project's own product corpus, where an earlier length-based rule wrongly
    escalated 31 of 48 such pages because their lines are short.
    """
    markdown = (
        "## ATC-28 - Collect the beta shipping address - acceptance report\n\n"
        "## AT A GLANCE\n\nIssue: ATC-28 - Collect the beta shipping address\n\n"
        "Current result: PASS\n\nOwner: Lucas Martin\n\n"
        "- Scope and acceptance come from the Jira issue; this report adds no hidden requirements.\n"
    )
    assert has_orphaned_labels(markdown) is False


def test_near_empty_page_makes_no_claim():
    """Two lines say nothing either way; leave the judgement to the defect detectors."""
    assert has_orphaned_labels("ATLASCART / SYNTHETIC PROJECT RECORD\n\nSECTION 01:\n") is False


@pytest.mark.asyncio
async def test_corrupt_pdf_leaves_a_parse_error_breadcrumb():
    """A PDF Docling cannot open returns zero sections (the swallow-and-continue contract) but must
    leave a breadcrumb in the parse-error ContextVar so the pipeline can record `failed` not `empty`.
    Real bytes, real Docling — `%PDF` header, garbage body, no xref."""
    from app.ingest.docling_parser import pop_parse_errors

    settings = Settings(vlm_ocr_enabled=False)  # no VLM to "recover" it
    sections = await parse_attachment_sections(b"%PDF-1.4\nnot a real pdf body", "broken.pdf", settings)

    assert sections == []
    errors = pop_parse_errors()
    assert errors, "a corrupt PDF must record a parse error"
    assert "pdf" in errors[0].lower()
    # And the ContextVar is reset for the next call.
    assert pop_parse_errors() == []


@pytest.mark.asyncio
async def test_a_clean_parse_leaves_no_breadcrumb():
    """The other half: a CSV that parses fine records nothing, so the pipeline sees `indexed`."""
    from app.ingest.docling_parser import pop_parse_errors

    sections = await parse_attachment_sections(b"a,b,c\n1,2,3\n", "data.csv", Settings())

    assert sections and sections[0].content
    assert pop_parse_errors() == []


def test_office_zip_is_reordered_so_its_format_can_be_sniffed():
    # ATC-89's PPTX listed folders before [Content_Types].xml, read as a plain zip, and Docling
    # rejected it ("File format not allowed").
    import io
    import zipfile

    import filetype

    from app.ingest.docling_parser import _content_types_first

    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as z:
        z.writestr("_rels/", "")
        z.writestr("ppt/", "")
        z.writestr("ppt/presentation.xml", "<p:presentation/>")
        z.writestr("[Content_Types].xml", "<Types/>")
    assert filetype.guess_mime(raw.getvalue()) == "application/zip"

    fixed = _content_types_first(raw.getvalue())

    assert zipfile.ZipFile(io.BytesIO(fixed)).namelist()[0] == "[Content_Types].xml"
    assert filetype.guess_mime(fixed) == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert _content_types_first(b"not a zip") == b"not a zip"
