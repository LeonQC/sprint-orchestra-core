"""Attachment binary -> reading-order plain text, mostly via Docling.

Docling parses PDF/Office files and `export_to_markdown()` keeps tables in place next to the text
that introduces them. PDFs are exported page by page when Docling has page provenance, so page
numbers survive into chunks.

Exceptions, all found by running ingestion on a real 132-file attachment corpus:
  * CSV, .txt and .log skip Docling: it has no plain-text format and raised an error that used to be
    swallowed into empty text (~13% of that corpus).
  * Standalone images skip Docling and go straight to EasyOCR on the full image. Docling's layout
    step classified a whole bordered screenshot as a "Picture" and never OCR'd it (~10% of the text
    recovered, whatever OCR settings were used); raw EasyOCR got nearly all of it.
  * Log with `attachment_filename`, never `extra={"filename": ...}` — `filename` is reserved on
    LogRecord, and that KeyError inside the except handler crashed the whole run.

Also: OCR/table models are off by default (slow); parsing runs in `asyncio.to_thread` (CPU-heavy,
sync); Docling/EasyOCR are imported lazily so the service runs without them; struggling OCR can
escalate to a vision model (`VEC_VLM_OCR_ENABLED`, see `_maybe_escalate_to_vlm`).
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from io import BytesIO
from typing import Iterator

from app.config import Settings
from app.ingest.image_layout import analyze_layout
from app.ingest.pdf_vlm import PdfPageCandidate, render_pdf_candidates
from app.ingest.table_linearizer import (
    _ALIGNMENT_ROW,
    _split_row,
    carry_table_headers,
    linearize_markdown_tables,
)
from app.ingest.vlm_describer import NoopVlmImageDescriber, VlmImageDescriber
from app.models import AttachmentProvenance

logger = logging.getLogger(__name__)

# Every parse path swallows errors and returns empty text so one bad file can't stall the consumer.
# That makes a corrupt file look like a blank one, so real errors are recorded here and
# `handle_attachment` reads them to report `failed` vs `empty`. A ContextVar so concurrent
# attachments don't mix. `pop_parse_errors()` reads and resets it.
_parse_errors: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "docling_parse_errors", default=None
)


def _record_parse_error(stage: str, exc: BaseException) -> None:
    errors = _parse_errors.get()
    if errors is not None:
        errors.append(f"{stage}: {type(exc).__name__}: {exc}")


def pop_parse_errors() -> list[str]:
    """Return (and clear) parse errors since the last `parse_attachment_sections` call.

    Empty + zero sections = the file really had no text. Non-empty + zero sections = the parser broke.
    """
    errors = _parse_errors.get() or []
    _parse_errors.set([])
    return list(errors)


# VLM calls that raised during the current parse, where the OCR/Docling text was kept instead. The
# document is still indexed; `handle_attachment` queues it so the VLM pass runs again once the model
# is back (app/ingest/vlm_retry.py). Same ContextVar pattern as `_parse_errors`.
_vlm_failures: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "vlm_failures", default=None
)


def _record_vlm_failure(exc: BaseException) -> None:
    failures = _vlm_failures.get()
    if failures is not None:
        failures.append(f"{type(exc).__name__}: {exc}")


def pop_vlm_failures() -> list[str]:
    """Return (and clear) VLM calls that failed since the last `parse_attachment_sections` call."""
    failures = _vlm_failures.get() or []
    _vlm_failures.set([])
    return list(failures)

# Docling raises on these (no plain-text input format); they're already flat text.
_PLAIN_TEXT_EXTENSIONS = (".csv", ".txt", ".log")

# Images go straight to OCR (see module docstring). Only formats seen in the real corpus so far.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
_PDF_EXTENSION = ".pdf"

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}

# A table cell that is ONLY a rename parenthetical, e.g. "(Formerly known as GTC Global Limited)".
# Seen live: the company-name line above it in the same cell was never detected by
# Docling's text-line detection, so only the parenthetical got indexed. A VLM re-read of the page
# recovers it.
# Deliberately narrow: a looser "any (...) cell" matched normal content — accounting negatives
# "(2,89,65,150)", a split header "(Rs.)", "(Refer Note hereinbelow)" — and re-reading those pages
# risks the digit transposition this detector exists to avoid. A rename phrase on its own reliably
# means the name before it was dropped.
_ORPHAN_PARENTHETICAL_CELL = re.compile(
    r"^\((?:formerly|previously|also)\s+(?:known\s+as|called|named)\b.*\)$", re.IGNORECASE
)


def _table_data_rows(markdown: str) -> list[list[str]]:
    """Pipe-delimited table rows from Docling's Markdown export, skipping header-separator rows."""
    rows = []
    for line in markdown.splitlines():
        line = line.strip()
        if not (line.startswith("|") and line.endswith("|")):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(re.fullmatch(r"[-:\s]*", cell) for cell in cells):
            continue  # e.g. "|---|:--:|" — not data
        rows.append(cells)
    return rows


def has_orphan_parenthetical_cell(markdown: str) -> bool:
    """Whether any table cell on this page is only a rename parenthetical (see `_ORPHAN_PARENTHETICAL_CELL`)."""
    return any(
        _ORPHAN_PARENTHETICAL_CELL.match(cell)
        for cells in _table_data_rows(markdown)
        for cell in cells
    )


# Total/Subtotal rows legitimately leave identity columns blank (see has_sparse_column_gap). Matched
# in any cell, since the label column varies.
_TOTAL_ROW_RE = re.compile(r"^\s*(grand\s+)?(sub)?total\b", re.IGNORECASE)
_BLANK_CELL_VALUES = {"", "-", "—", "n/a", "na"}


def _iter_markdown_tables(markdown: str) -> Iterator[tuple[list[str], list[list[str]]]]:
    """Yield (headers, data_rows) for each Markdown table, skipping prose.

    A table = a header line followed by an alignment row, same definition as table_linearizer
    (reuses its `_ALIGNMENT_ROW`/`_split_row`).
    """
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        is_table = (
            index + 1 < len(lines)
            and "|" in lines[index]
            and "|" in lines[index + 1]
            and _ALIGNMENT_ROW.match(lines[index + 1])
        )
        if not is_table:
            index += 1
            continue
        headers = _split_row(lines[index])
        body_start = index + 2
        body_end = body_start
        while body_end < len(lines) and "|" in lines[body_end] and lines[body_end].strip():
            body_end += 1
        data_rows = [_split_row(line) for line in lines[body_start:body_end]]
        yield headers, data_rows
        index = body_end


def has_sparse_column_gap(
    markdown: str, *, min_populated_ratio: float = 0.6, min_data_rows: int = 4,
) -> bool:
    """Whether a column filled in most rows is blank in at least one non-total row.

    Seen live: an attendance table where the top attendee's name cell was dropped by Docling while
    every other column survived (legible on the page; a VLM re-read recovered it). The same document
    had the same shape on another page, hence a fill-ratio rule rather than a one-table rule.

    Guards: Total/Subtotal rows are excluded first, and `min_populated_ratio` means a normally sparse
    column (e.g. "Chairman") can't trigger just because one more row is blank.
    """
    for headers, data_rows in _iter_markdown_tables(markdown):
        rows = [row for row in data_rows if len(row) == len(headers)]
        candidate_rows = [row for row in rows if not any(_TOTAL_ROW_RE.match(cell) for cell in row)]
        if len(candidate_rows) < min_data_rows:
            continue
        for col in range(len(headers)):
            values = [row[col].strip() for row in candidate_rows]
            populated = [v for v in values if v.lower() not in _BLANK_CELL_VALUES]
            if len(populated) / len(candidate_rows) < min_populated_ratio:
                continue
            for row, value in zip(candidate_rows, values):
                if value.lower() in _BLANK_CELL_VALUES and any(
                    cell.strip() for i, cell in enumerate(row) if i != col
                ):
                    return True
    return False


def has_orphaned_labels(markdown: str, *, min_ratio: float = 0.10, min_lines: int = 4) -> bool:
    """Whether label/value pairs have come apart — labels left alone on their lines.

    Flattening a form strands the label and moves the value elsewhere:

        Net Pounds Infeed:      <- label, nothing after it
        Net Pounds Out:\\_       <- label, nothing after it
        893                     <- value, attached to neither

    Words-per-line was tried first and escalated 31 of 48 healthy product-report pages (short lines,
    but `Issue: ATC-28` keeps label and value together). Orphaned-label ratio: product pages median
    0.00, prose 0.07-0.17, a broken form 0.13-0.42. Prose overlaps forms, so this is one reason to
    escalate among several, not a separator on its own.
    """
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    orphaned = sum(1 for line in lines if line.rstrip().endswith((":", ";")))
    return orphaned / len(lines) >= min_ratio


def has_recognized_table(markdown: str, *, min_data_rows: int = 2) -> bool:
    """Whether Docling parsed a real table on this page — the signal to LEAVE THE PAGE ALONE.

    VLM escalation replaces the whole page's text, so it can damage what Docling got right: two
    features were reverted after a VLM re-transcribed correct tables with transposed digits (Case
    Studies 37, 38). A recognized table means the row/column structure is intact; pages without one
    are where flattening already lost it and a re-read has nothing to lose. Uses
    `_iter_markdown_tables`, so a sentence with a pipe in it doesn't count; `min_data_rows` skips
    two-line fragments.
    """
    return any(len(rows) >= min_data_rows for _, rows in _iter_markdown_tables(markdown))


@lru_cache(maxsize=4)
def _build_converter(do_ocr: bool, do_tables: bool, ocr_engine: str = "easyocr", table_mode: str = "fast"):
    """Build a Docling converter once per config and reuse it (model loading is expensive).

    PDF only — images don't use Docling. `do_ocr` matters for scanned PDFs without a text layer.

    OCR engine chosen by measurement (31 facts on a scanned doc):
    easyocr 23, tesseract 18. Apple Vision scored 27 but is macOS-only and crashed on another doc, so
    it was removed. Table FAST vs ACCURATE: 23 vs 23; English-only OCR: worse (21).

    Cached on all four inputs — caching on (do_ocr, do_tables) alone ignored the engine setting.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
        TesseractCliOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_tables
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE if table_mode == "accurate" else TableFormerMode.FAST
    )

    # "default" keeps Docling's own stock OCR options untouched, so an unset config behaves exactly
    # as before this became configurable.
    if ocr_engine == "tesseract":
        pipeline_options.ocr_options = TesseractCliOcrOptions(lang=["eng"])
    elif ocr_engine == "rapidocr":
        pipeline_options.ocr_options = RapidOcrOptions()
    elif ocr_engine == "easyocr":
        pipeline_options.ocr_options = EasyOcrOptions()
    elif ocr_engine != "default":
        raise ValueError(f"unknown docling_ocr_engine: {ocr_engine!r}")

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _convert_sync(
    data: bytes, filename: str, do_ocr: bool, do_tables: bool,
    ocr_engine: str = "default", table_mode: str = "fast",
) -> str:
    from docling.datamodel.base_models import DocumentStream

    converter = _build_converter(do_ocr, do_tables, ocr_engine, table_mode)
    source = DocumentStream(name=filename, stream=BytesIO(data))
    result = converter.convert(source)
    return result.document.export_to_markdown()


def _page_ocr_confidence_ratios(result, min_confidence: float) -> dict[int, float]:
    """PDF page number (1-based) -> fraction of that page's OCR cells below `min_confidence`.

    Same "OCR struggled" signal used for images, applied to PDF pages. Docling keeps per-cell OCR
    confidence in `result.pages[i].cells` even though the Markdown export drops it. Pages with no
    OCR cells (born-digital) are absent — "no OCR ran", not "OCR was perfect".
    """
    ratios: dict[int, float] = {}
    for page in getattr(result, "pages", None) or []:
        confidences = [cell.confidence for cell in page.cells if hasattr(cell, "confidence")]
        if not confidences:
            continue
        low = sum(1 for c in confidences if c < min_confidence)
        ratios[page.page_no + 1] = low / len(confidences)  # Page.page_no is 0-based; sections are 1-based
    return ratios


# The placeholder `export_to_markdown()` writes for a picture cluster. Stripped before comparing
# lengths below, or its own characters make a nearly empty page look substantial.
_PICTURE_PLACEHOLDER = re.compile(r"<!--\s*image\s*-->", re.IGNORECASE)


def _reading_order_text_from_cells(page) -> str:
    """Rebuild plain text from a page's raw OCR cells in rough reading order (same row grouping as
    `_ocr_image_sync`; both use top-left-origin boxes).
    """
    def row_key(cell):
        return (round(cell.bbox.t / 20), cell.bbox.l)

    cells = sorted(page.cells, key=row_key)
    return " ".join(cell.text for cell in cells if cell.text and cell.text.strip())


def _picture_cluster_recovered_text(result, min_recovered_chars: int = 20) -> dict[int, str]:
    """PDF page number -> text recovered from OCR cells, for pages Docling's export threw away.

    Seen live on a scanned phone-message slip: OCR worked (16 cells: "TELEPHONED", "URGENT", ...) but
    the page was labelled "picture", so the export was just "<!-- image -->". Different from the
    image bypass: OCR succeeded, and the text was lost one step later at serialization.

    `min_recovered_chars` avoids turning a real photo/logo's stray detections into a chunk. Also
    handles mixed pages (export "<!-- image -->\\n\\nMC 0343" with the rest lost): the caller compares
    lengths with the placeholder stripped.
    """
    recovered: dict[int, str] = {}
    for page in getattr(result, "pages", None) or []:
        text = _reading_order_text_from_cells(page)
        if len(text) >= min_recovered_chars:
            recovered[page.page_no + 1] = text
    return recovered


def _convert_pdf_sections_sync(
    data: bytes, filename: str, do_ocr: bool, do_tables: bool, ocr_min_confidence: float = 1.0,
    ocr_engine: str = "default", table_mode: str = "fast",
) -> list[ParsedAttachmentSection]:
    """Export Docling PDF output page by page when it has a page map.

    Keeps 1-based page numbers for chunks. Without a page map, return one unpaged section rather than
    guess. `ocr_min_confidence` defaults to 1.0 for callers that don't need the OCR signal; real
    callers pass `settings.ocr_min_confidence`.
    """
    from docling.datamodel.base_models import DocumentStream

    converter = _build_converter(do_ocr, do_tables, ocr_engine, table_mode)
    source = DocumentStream(name=filename, stream=BytesIO(data))
    result = converter.convert(source)
    document = result.document
    confidence_ratios = _page_ocr_confidence_ratios(result, ocr_min_confidence)
    recovered_text = _picture_cluster_recovered_text(result)
    page_numbers = sorted(getattr(document, "pages", {}) or {})
    if page_numbers:
        sections = []
        for page_number in page_numbers:
            content = document.export_to_markdown(page_no=page_number).strip()
            recovered = False
            candidate = recovered_text.get(page_number)
            if candidate:
                # Compare with placeholders stripped, and use a ratio rather than "is it empty" — a page
                # can be mixed (a stamp survived, the rest was swallowed by a picture cluster).
                real_export = _PICTURE_PLACEHOLDER.sub("", content).strip()
                if len(real_export) < 0.5 * len(candidate):
                    # Docling's export is missing most of what OCR found. Logged because it's a real
                    # Docling defect we're patching over.
                    logger.info(
                        "Recovered OCR text Docling's export discarded (picture-cluster page)",
                        extra={
                            "page_number": page_number,
                            "export_chars": len(real_export),
                            "recovered_chars": len(candidate),
                        },
                    )
                    content = candidate
                    recovered = True
            if content:
                provenance = {"source_type": "pdf", "page_number": page_number}
                if recovered:
                    provenance["extraction"] = "docling_cell_recovery"
                sections.append(
                    ParsedAttachmentSection(
                        content=content,
                        page_number=page_number,
                        provenance=provenance,
                        ocr_low_confidence_ratio=confidence_ratios.get(page_number),
                    )
                )
        if sections:
            return sections
    content = document.export_to_markdown().strip()
    return [ParsedAttachmentSection(content=content)] if content else []


def _content_types_first(data: bytes) -> bytes:
    """Rewrite an Office zip so `[Content_Types].xml` is its first entry.

    Docling sniffs the format from the first 8 KB. A file whose zip lists folders first (ATC-89's
    PPTX) reads as a plain zip, and Docling rejects it: "File format not allowed".
    """
    try:
        source = zipfile.ZipFile(BytesIO(data))
        names = source.namelist()
        if not names or names[0] == "[Content_Types].xml" or "[Content_Types].xml" not in names:
            return data
        out = BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
            target.writestr(source.getinfo("[Content_Types].xml"), source.read("[Content_Types].xml"))
            for name in names:
                if name != "[Content_Types].xml" and not name.endswith("/"):
                    target.writestr(source.getinfo(name), source.read(name))
        return out.getvalue()
    except zipfile.BadZipFile:
        return data


def _convert_structured_sections_sync(
    data: bytes, filename: str, do_ocr: bool, do_tables: bool,
    ocr_engine: str = "default", table_mode: str = "fast",
) -> list[ParsedAttachmentSection]:
    """Keep native locators for Office formats: page-like provenance for PPTX/DOCX when Docling has
    it, and per-sheet sections for XLSX so a chunk can point at a stable sheet name.
    """
    from docling.datamodel.base_models import DocumentStream

    converter = _build_converter(do_ocr, do_tables, ocr_engine, table_mode)
    source = DocumentStream(name=filename, stream=BytesIO(_content_types_first(data)))
    document = converter.convert(source).document
    lower_name = filename.lower()
    source_type = lower_name.rsplit(".", 1)[-1] if "." in lower_name else "document"

    if source_type in {"pptx", "docx"}:
        page_numbers = sorted(getattr(document, "pages", {}) or {})
        sections = []
        for page_number in page_numbers:
            content = document.export_to_markdown(page_no=page_number).strip()
            if content:
                locator = {"source_type": source_type}
                if source_type == "pptx":
                    locator["slide_number"] = page_number
                else:
                    locator["page_number"] = page_number
                sections.append(
                    ParsedAttachmentSection(
                        content=content,
                        # page_number is reserved for PDF pages; DOCX pages depend on the renderer, so
                        # they stay in the locator.
                        page_number=None,
                        provenance=locator,
                    )
                )
        if sections:
            return sections

    content = document.export_to_markdown().strip()
    if source_type == "xlsx":
        sheet_ranges = _xlsx_used_ranges(data)
        # Docling groups worksheet nodes per sheet: named `sheet: <name>` up to 2.14, named `<name>`
        # with label "sheet" from 2.150. Serialize each group directly so multiple sheets never
        # collapse into one unlocatable Markdown stream.
        try:
            from docling_core.transforms.serializer.markdown import MarkdownDocSerializer

            serializer = MarkdownDocSerializer(doc=document)
            sheet_groups = [
                group for group in getattr(document, "groups", [])
                if str(getattr(group, "name", "")).startswith("sheet:")
                or str(getattr(group, "label", "")) == "sheet"
            ]
        except Exception:  # noqa: BLE001 - fallback below still preserves whole-workbook text
            sheet_groups = []
        if sheet_groups:
            sections = []
            for group in sheet_groups:
                sheet_name = str(group.name).removeprefix("sheet:").strip()
                sheet_content = serializer.serialize(item=group).text.strip()
                if not sheet_content:
                    continue
                locator = {"source_type": "xlsx", "sheet_name": sheet_name}
                if sheet_ranges.get(sheet_name):
                    locator["cell_range"] = sheet_ranges[sheet_name]
                sections.append(ParsedAttachmentSection(content=sheet_content, provenance=locator))
            if sections:
                return sections
        # Older Docling versions may not expose worksheet groups. If they emit worksheet headings,
        # retain that stable name; otherwise fall back to one workbook-level locator.
        matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", content))
        if matches:
            sections = []
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
                sheet_name = match.group(1).strip()
                sheet_content = content[match.start():end].strip()
                if sheet_content:
                    locator = {"source_type": "xlsx", "sheet_name": sheet_name}
                    if sheet_ranges.get(sheet_name):
                        locator["cell_range"] = sheet_ranges[sheet_name]
                    sections.append(
                        ParsedAttachmentSection(
                            content=sheet_content,
                            provenance=locator,
                        )
                    )
            if sections:
                return sections
    return [
        ParsedAttachmentSection(
            content=content,
            provenance={"source_type": source_type},
        )
    ] if content else []


def _xlsx_used_ranges(data: bytes) -> dict[str, str]:
    """Worksheet used ranges via openpyxl, as citation coordinates (Docling still provides the text).
    Missing ranges are fine and never guessed.
    """
    try:
        import openpyxl

        workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=False)
        try:
            return {
                worksheet.title: worksheet.calculate_dimension()
                for worksheet in workbook.worksheets
            }
        finally:
            workbook.close()
    except Exception:  # noqa: BLE001 - coordinate enrichment must not break text ingestion
        return {}


@lru_cache(maxsize=1)
def _ocr_reader():
    """Build the EasyOCR reader once (model load takes ~1s) and reuse it across attachments."""
    import easyocr

    return easyocr.Reader(["en"], gpu=False)


@dataclass
class _OcrResult:
    text: str
    dropped_count: int  # detections discarded as low-confidence — a concrete "OCR struggled" signal
    # Fraction of rows carrying repeated positional structure (multi-column
    # tables or aligned two-column forms). See `_column_row_ratio`.
    column_row_ratio: float = 0.0


@dataclass(frozen=True)
class ParsedAttachmentSection:
    """Parsed text plus its source page when known. No page number means "not paginated" or
    "unknown" — never page one.
    """

    content: str
    page_number: int | None = None
    provenance: AttachmentProvenance | None = None
    # Fraction of low-confidence OCR cells on this PDF page (None if no OCR ran). Used only to decide
    # VLM escalation; deliberately not in `provenance`, so it never reaches the database.
    ocr_low_confidence_ratio: float | None = None


def _detection_rows(detections: list) -> list[list]:
    """Group EasyOCR detections into rows, left to right within each row.

    The row height comes from the median detection height. A fixed 20px band split lines in two on a
    2200px image and merged three lines on a 600px one.
    """
    if not detections:
        return []

    def y_center(bbox) -> float:
        return sum(point[1] for point in bbox) / 4

    def height(bbox) -> float:
        return max(point[1] for point in bbox) - min(point[1] for point in bbox)

    heights = sorted(height(bbox) for bbox, _text, _conf in detections)
    median_height = heights[len(heights) // 2] or 1.0
    # 0.6 of a line height: wide enough to keep a row together despite baseline jitter and
    # superscripts, narrow enough that two genuinely adjacent lines never merge.
    band = max(median_height * 0.6, 1.0)

    ordered = sorted(detections, key=lambda d: (y_center(d[0]), min(p[0] for p in d[0])))
    rows: list[list] = []
    for detection in ordered:
        center = y_center(detection[0])
        if rows and abs(center - y_center(rows[-1][-1][0])) <= band:
            rows[-1].append(detection)
        else:
            rows.append([detection])
    for row in rows:
        row.sort(key=lambda d: min(point[0] for point in d[0]))
    return rows


def _column_row_ratio(rows: list[list], median_height: float) -> float:
    """How much of the page is laid out in columns — what flattening to one string destroys.

    Different from `dropped_count` (hard to read). A crisp table OCRs at high confidence, so the old
    confidence-only rule never escalated it, yet joining the detections into one line wrecks it. Seen
    live: a 7-column creditor table became "CREDITOR NAME TYPE AMOUNT YEAR ... Whitaker Bank ...
    550,001", with "$50,001 - $100,000" mangled. Measures layout, not legibility; prose scores ~0.
    Uses boxes EasyOCR already returned, so no extra calls.
    """
    return analyze_layout(rows, median_height).structured_row_ratio


def _ocr_image_sync(data: bytes, min_confidence: float) -> _OcrResult:
    import numpy as np
    from PIL import Image

    reader = _ocr_reader()
    image = np.array(Image.open(BytesIO(data)).convert("RGB"))
    detections = reader.readtext(image)

    # Drop low-confidence detections: on a degraded image, garbage ("Mx", "Mfate") scored 0.006-0.06
    # and real words 0.7-1.0. Unfiltered, the agent presented "Mx" as a real SKU — and a faithfulness
    # check can't catch that, because the retrieved text itself was wrong.
    dropped = [(text, conf) for _bbox, text, conf in detections if conf < min_confidence]
    if dropped:
        logger.info(
            "dropped low-confidence OCR detections",
            extra={"dropped_count": len(dropped), "dropped_texts": [t for t, _ in dropped]},
        )
    detections = [d for d in detections if d[2] >= min_confidence]

    # EasyOCR's order isn't reading order; group into rows, left to right (see `_detection_rows`).
    rows = _detection_rows(detections)
    text = " ".join(text for row in rows for _bbox, text, _conf in row)

    heights = sorted(
        max(p[1] for p in bbox) - min(p[1] for p in bbox) for bbox, _text, _conf in detections
    )
    median_height = heights[len(heights) // 2] if heights else 1.0
    column_ratio = _column_row_ratio(rows, median_height)
    if column_ratio:
        logger.info(
            "OCR found column structure this flat text cannot preserve",
            extra={"column_row_ratio": round(column_ratio, 3), "rows": len(rows)},
        )
    return _OcrResult(text=text, dropped_count=len(dropped), column_row_ratio=column_ratio)




async def _maybe_escalate_to_vlm(
    data: bytes, mime_type: str, ocr_result: _OcrResult, settings: Settings, vlm_describer
) -> str:
    """Return `ocr_result.text`, or a VLM transcription when escalation is on and this image shows
    real evidence the flat OCR text is unfaithful. Most images are clean screenshots OCR handles fine
   , so this isn't run on everything.

    Two triggers:
    1. dropped_count / empty — OCR couldn't read it confidently.
    2. column_row_ratio — OCR read it fine, but the page is in columns that a flat string can't hold
       (see `_column_row_ratio`). Trigger 1 alone protected exactly the wrong documents.

    The VLM text replaces the OCR text, except on a dense page where OCR already got a lot and the VLM
    covers less than the configured fraction — then it's treated as an incomplete summary.
    """
    if not settings.vlm_ocr_enabled:
        return ocr_result.text
    struggled = ocr_result.dropped_count > 0 or not ocr_result.text.strip()
    structured_layout = (
        settings.vlm_image_column_detection_enabled
        and ocr_result.column_row_ratio >= settings.vlm_image_column_row_ratio
    )
    if not struggled and not structured_layout:
        return ocr_result.text
    if structured_layout and not struggled:
        logger.info(
            "escalating a confidently-read image to VLM for positional layout, not legibility",
            extra={"column_row_ratio": round(ocr_result.column_row_ratio, 3)},
        )
    try:
        vlm_text = await vlm_describer.describe(data, mime_type)
    except Exception as exc:  # noqa: BLE001 - a VLM outage must fall back to the OCR result, not crash ingestion
        logger.warning("VLM escalation failed; keeping the OCR result", extra={"error": str(exc)})
        _record_vlm_failure(exc)
        return ocr_result.text
    if not vlm_text or vlm_text.strip() == "(no text found)":
        return ocr_result.text
    ocr_chars = len(ocr_result.text.strip())
    vlm_chars = len(vlm_text.strip())
    if (
        ocr_chars >= 500
        and vlm_chars < ocr_chars * settings.vlm_min_ocr_coverage_ratio
    ):
        logger.warning(
            "VLM escalation output appears incomplete; keeping the fuller OCR result",
            extra={
                "ocr_text_len": ocr_chars,
                "vlm_text_len": vlm_chars,
                "coverage_ratio": round(vlm_chars / ocr_chars, 3),
                "minimum_ratio": settings.vlm_min_ocr_coverage_ratio,
            },
        )
        return ocr_result.text
    logger.info(
        "VLM escalation replaced a struggling OCR result",
        extra={"ocr_dropped_count": ocr_result.dropped_count, "ocr_text_len": ocr_chars},
    )
    return vlm_text


async def _pdf_sections_with_vlm(
    data: bytes,
    docling_sections: list[ParsedAttachmentSection],
    settings: Settings,
    vlm_describer: VlmImageDescriber,
) -> list[ParsedAttachmentSection]:
    """Add VLM-extracted text for PDF pages that look like scans or visuals.

    If every page needs it, the VLM pages replace Docling's (usually noise). For mixed PDFs, keep
    Docling's good pages and replace only the selected ones, labelled by page. Without page boundaries,
    keep Docling's text and append the VLM sections.
    """
    docling_text = "\n\n".join(section.content for section in docling_sections).strip()
    if not settings.vlm_ocr_enabled:
        return docling_sections

    # Two narrow triggers, independent of vlm_pdf_enabled: each needs evidence of a problem on THAT
    # page. ("Little native text" is true of every scanned page, which is why the broad flag
    # regressed accuracy.)
    force_pages: dict[int, str] = {}
    if settings.vlm_pdf_table_defect_detection_enabled:
        for section in docling_sections:
            if section.page_number is not None and has_orphan_parenthetical_cell(section.content):
                force_pages[section.page_number] = "table_defect"
    if settings.vlm_pdf_ocr_confidence_detection_enabled:
        for section in docling_sections:
            if (
                section.page_number is not None
                and section.page_number not in force_pages
                and section.ocr_low_confidence_ratio is not None
                and section.ocr_low_confidence_ratio >= settings.vlm_pdf_low_confidence_page_ratio
            ):
                force_pages[section.page_number] = "ocr_low_confidence"
    if settings.vlm_pdf_sparse_column_detection_enabled:
        for section in docling_sections:
            if (
                section.page_number is not None
                and section.page_number not in force_pages
                and has_sparse_column_gap(section.content)
            ):
                force_pages[section.page_number] = "sparse_column_gap"
    # Inverted routing: escalate every page that has nothing worth protecting (no recognized table).
    # The defect detectors escalated only 7 of 103 retrieved pages, so silently flattened forms never
    # reached the VLM; escalating every page scored 0.514 -> 0.753 on MP-DocVQA. This keeps that gain
    # while sparing parsed tables.
    if settings.vlm_pdf_difficulty_routing_enabled:
        for section in docling_sections:
            if (
                section.page_number is not None
                and section.page_number not in force_pages
                and not has_recognized_table(section.content)
                and has_orphaned_labels(section.content)
            ):
                force_pages[section.page_number] = "flattened_structure"
    if force_pages:
        logger.info(
            "Targeted PDF-page VLM escalation triggered",
            extra={"pages_by_reason": force_pages},
        )

    if not settings.vlm_pdf_enabled and not force_pages:
        return docling_sections
    try:
        candidates, total_pages = await asyncio.to_thread(
            render_pdf_candidates,
            data,
            min_text_chars=settings.vlm_pdf_min_text_chars,
            include_visual_pages=settings.vlm_pdf_include_visual_pages,
            max_pages=settings.vlm_pdf_max_pages,
            dpi=settings.vlm_pdf_render_dpi,
            force_pages=force_pages,
            enable_heuristic_selection=settings.vlm_pdf_enabled,
        )
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping PDF VLM fallback")
        return docling_sections
    except Exception as exc:  # noqa: BLE001 - preserve Docling text if inspection/rendering fails
        logger.warning("PDF VLM candidate rendering failed; keeping Docling result", extra={"error": str(exc)})
        return docling_sections

    if not candidates:
        return docling_sections

    rendered_pages: list[ParsedAttachmentSection] = []
    for candidate in candidates:
        try:
            text = await vlm_describer.describe(candidate.image_data, "image/png")
        except Exception as exc:  # noqa: BLE001 - one page failure must not discard its whole attachment
            logger.warning(
                "PDF page VLM extraction failed; keeping available source text",
                extra={"page_number": candidate.page_number, "error": str(exc)},
            )
            _record_vlm_failure(exc)
            continue
        if not text or text.strip() == "(no text found)":
            continue
        rendered_pages.append(
            ParsedAttachmentSection(
                content=f"[VLM extracted PDF page {candidate.page_number} ({candidate.reason})]\n{text.strip()}",
                page_number=candidate.page_number,
                provenance={
                    "source_type": "pdf",
                    "page_number": candidate.page_number,
                    "extraction": "vlm",
                },
            )
        )

    if not rendered_pages:
        return docling_sections
    # A fully selected PDF is scan/visual-only (within the configured safe page cap): VLM is the
    # better source of truth, rather than appending it to unusable Docling OCR garbage.
    if (
        len(candidates) == total_pages
        and len(docling_text.strip()) < total_pages * settings.vlm_pdf_min_text_chars
    ):
        return rendered_pages
    # With page-aware sections, replace only pages whose VLM extraction succeeded. Without a page
    # map, keep Docling's text and append the labelled VLM pages.
    rendered_numbers = {section.page_number for section in rendered_pages}
    if any(section.page_number is not None for section in docling_sections):
        retained = [
            section for section in docling_sections if section.page_number not in rendered_numbers
        ]
        return retained + rendered_pages
    return ([ParsedAttachmentSection(docling_text)] if docling_text else []) + rendered_pages


async def _parse_pdf_sections(
    data: bytes, filename: str, settings: Settings, vlm_describer: VlmImageDescriber
) -> list[ParsedAttachmentSection]:
    """Run Docling first, retaining real VLM page provenance whenever it is available."""
    docling_error: Exception | None = None
    try:
        docling_sections = await asyncio.to_thread(
            _convert_pdf_sections_sync,
            data,
            filename,
            settings.docling_do_ocr,
            settings.docling_do_table_structure,
            settings.ocr_min_confidence,
            settings.docling_ocr_engine,
            settings.docling_table_mode,
        )
    except ImportError:
        # VLM page rendering is independently useful for image-only PDFs even if Docling was not
        # installed in a lightweight deployment.
        logger.warning("docling not installed; trying configured PDF VLM fallback", extra={"attachment_filename": filename})
        docling_sections = []
    except Exception as exc:  # noqa: BLE001 - VLM can still recover a failed PDF conversion
        logger.warning("PDF parse failed; trying configured VLM fallback", extra={"attachment_filename": filename, "error": str(exc)})
        docling_sections = []
        docling_error = exc
    sections = await _pdf_sections_with_vlm(data, docling_sections, settings, vlm_describer)
    # Record only a real failure: Docling raised AND the VLM recovered nothing.
    if docling_error is not None and not sections:
        _record_parse_error("parse (pdf)", docling_error)
    return sections


async def parse_attachment(
    data: bytes, filename: str, settings: Settings, vlm_describer: VlmImageDescriber | None = None
) -> str:
    """Reading-order text for an attachment, or "" if it can't be parsed. Never raises for one bad
    file. `vlm_describer` is a no-op unless IngestPipeline passes its shared client.
    """
    vlm_describer = vlm_describer or NoopVlmImageDescriber()
    lower_name = filename.lower()

    if lower_name.endswith(_PLAIN_TEXT_EXTENSIONS):
        try:
            return data.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the consumer
            logger.warning(
                "plain-text attachment decode failed",
                extra={"attachment_filename": filename, "error": str(exc)},
            )
            _record_parse_error("parse (text decode)", exc)
            return ""

    if lower_name.endswith(_IMAGE_EXTENSIONS):
        if not settings.docling_do_ocr:
            return ""  # no OCR requested, and an image has no other extractable text layer
        try:
            ocr_result = await asyncio.to_thread(_ocr_image_sync, data, settings.ocr_min_confidence)
            ext = next(e for e in _IMAGE_EXTENSIONS if lower_name.endswith(e))
            return await _maybe_escalate_to_vlm(data, _MIME_TYPES[ext], ocr_result, settings, vlm_describer)
        except ImportError:
            logger.warning("easyocr not installed; skipping image OCR", extra={"attachment_filename": filename})
            return ""
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the consumer
            logger.warning("image OCR failed", extra={"attachment_filename": filename, "error": str(exc)})
            _record_parse_error("parse (image ocr)", exc)
            return ""

    if lower_name.endswith(_PDF_EXTENSION):
        sections = await _parse_pdf_sections(data, filename, settings, vlm_describer)
        return "\n\n".join(section.content for section in sections)

    try:
        return await asyncio.to_thread(
            _convert_sync, data, filename, settings.docling_do_ocr,
            settings.docling_do_table_structure, settings.docling_ocr_engine,
            settings.docling_table_mode,
        )
    except ImportError:
        logger.warning("docling not installed; skipping attachment parse", extra={"attachment_filename": filename})
        return ""
    except Exception as exc:  # noqa: BLE001 - one bad file must not kill the consumer
        logger.warning("attachment parse failed", extra={"attachment_filename": filename, "error": str(exc)})
        _record_parse_error("parse (docling)", exc)
        return ""


async def parse_attachment_sections(
    data: bytes, filename: str, settings: Settings, vlm_describer: VlmImageDescriber | None = None
) -> list[ParsedAttachmentSection]:
    """Parse an attachment, then restate Markdown tables row by row (app.ingest.table_linearizer).

    Done at this one shared exit so every format — TableFormer or VLM tables, and future formats —
    gets it.
    """
    _parse_errors.set([])
    _vlm_failures.set([])
    sections = await _parse_attachment_sections(data, filename, settings, vlm_describer)
    if len(sections) > 1 and all(s.page_number is not None for s in sections):
        carried = carry_table_headers([s.content for s in sections])
        sections = [replace(s, content=text) for s, text in zip(sections, carried)]
    return [
        replace(section, content=linearize_markdown_tables(section.content))
        if "|" in section.content else section
        for section in sections
    ]


async def _parse_attachment_sections(
    data: bytes, filename: str, settings: Settings, vlm_describer: VlmImageDescriber | None = None
) -> list[ParsedAttachmentSection]:
    """Attachment text as sections with provenance, for the ingestion pipeline.

    PDF output keeps a 1-based page number when Docling or the VLM provides one. Other formats put
    their own locator in `provenance` (slide, sheet/range, DOCX page, image bbox); `page_number`
    stays null for them.
    """
    vlm_describer = vlm_describer or NoopVlmImageDescriber()
    if filename.lower().endswith(_PDF_EXTENSION):
        return await _parse_pdf_sections(data, filename, settings, vlm_describer)
    lower_name = filename.lower()
    if lower_name.endswith(_IMAGE_EXTENSIONS):
        text = await parse_attachment(data, filename, settings, vlm_describer)
        if not text:
            return []
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as image:
                width, height = image.size
            provenance = {
                "source_type": "image",
                "bbox": [0, 0, width, height],
                "coordinate_system": "pixels",
            }
        except Exception:  # noqa: BLE001 - text remains useful when image metadata is unavailable
            provenance = {"source_type": "image"}
        return [ParsedAttachmentSection(text, provenance=provenance)]
    if lower_name.endswith(_PLAIN_TEXT_EXTENSIONS):
        text = await parse_attachment(data, filename, settings, vlm_describer)
        return [ParsedAttachmentSection(text, provenance={"source_type": lower_name.rsplit(".", 1)[-1]})] if text else []
    try:
        return await asyncio.to_thread(
            _convert_structured_sections_sync,
            data,
            filename,
            settings.docling_do_ocr,
            settings.docling_do_table_structure,
            settings.docling_ocr_engine,
            settings.docling_table_mode,
        )
    except ImportError:
        logger.warning("docling not installed; skipping attachment parse", extra={"attachment_filename": filename})
        return []
    except Exception as exc:  # noqa: BLE001 - preserve per-attachment failure isolation
        logger.warning("attachment structured parse failed", extra={"attachment_filename": filename, "error": str(exc)})
        _record_parse_error("parse (docling structured)", exc)
        return []
