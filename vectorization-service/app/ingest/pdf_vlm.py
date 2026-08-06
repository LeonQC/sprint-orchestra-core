"""Pick and render PDF pages for the VLM fallback.

Docling stays the main PDF parser (better reading order, cheaper). This only finds pages likely to
have lost content (scans, screenshots, image tables, charts), renders them to PNG, and lets the
caller send them to a VLM. PyMuPDF is imported lazily so text-only ingestion runs without it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfPageCandidate:
    """One PDF page selected for VLM extraction, numbered from one for user-facing citations."""

    page_number: int
    image_data: bytes
    reason: str


def render_pdf_candidates(
    data: bytes,
    *,
    min_text_chars: int,
    include_visual_pages: bool,
    max_pages: int,
    dpi: int,
    force_pages: Mapping[int, str] | None = None,
    enable_heuristic_selection: bool = True,
) -> tuple[list[PdfPageCandidate], int]:
    """Render only pages likely to be low quality, as PNG.

    Little native text on a page almost always means a scan. Optionally, pages with substantial
    visuals are included too, so chart/diagram facts aren't lost. Scan-like pages sort first, so a
    page cap spends calls on them before visual-only pages.

    `force_pages` (page -> reason) are pages another real signal already flagged as defective; always
    rendered and ranked first. `enable_heuristic_selection=False` renders only those, which keeps the
    narrow escalation paths independent of the broad `vlm_pdf_enabled` toggle.
    """
    if max_pages < 1:
        return [], 0
    if dpi < 72:
        raise ValueError("vlm_pdf_render_dpi must be at least 72")
    force_pages = force_pages or {}

    import fitz  # PyMuPDF

    document = fitz.open(stream=data, filetype="pdf")
    try:
        selected: list[tuple[int, str, object]] = []
        for zero_index, page in enumerate(document):
            page_number = zero_index + 1
            if page_number in force_pages:
                selected.append((zero_index, force_pages[page_number], page))
                continue
            if not enable_heuristic_selection:
                continue

            text_chars = len(page.get_text("text").strip())
            raster_images = len(page.get_images(full=True))
            # Vector-heavy pages commonly contain charts or diagrams but no embedded raster image.
            vector_drawings = len(page.get_drawings())

            if text_chars < min_text_chars:
                selected.append((zero_index, "little_or_no_embedded_text", page))
            elif include_visual_pages and (raster_images > 0 or vector_drawings >= 20):
                selected.append((zero_index, "visual_content", page))

        # Confirmed defects first (table-structure defects, then OCR-confidence struggle), then
        # scan-like pages, then visual-only — preserve page order within a class, making output
        # stable across re-ingestion and cache-friendly.
        _priority = {
            "table_defect": 0,
            "sparse_column_gap": 0,
            "ocr_low_confidence": 1,
            # Below the defect signals on purpose: those name a specific problem, this one only says
            # "nothing here worth protecting". When the cap forces a choice, the known defect wins.
            "flattened_structure": 2,
            "little_or_no_embedded_text": 3,
            "visual_content": 4,
        }
        selected.sort(key=lambda item: (_priority[item[1]], item[0]))
        if len(selected) > max_pages:
            # Log when the cap is hit: pages past it silently keep Docling's lossy text, and a later
            # "the document doesn't mention X" can be wrong. Not raised — the cap bounds cost, and only
            # the operator can decide a 40-page scan is worth 40 VLM calls.
            dropped = sorted(page_index + 1 for page_index, _, _ in selected[max_pages:])
            logger.warning(
                "PDF VLM page cap reached; remaining pages keep their Docling text",
                extra={
                    "selected_pages": len(selected),
                    "max_pages": max_pages,
                    "dropped_pages": dropped,
                },
            )
        zoom = dpi / 72
        candidates = []
        for zero_index, reason, page in selected[:max_pages]:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            candidates.append(
                PdfPageCandidate(
                    page_number=zero_index + 1,
                    image_data=pixmap.tobytes("png"),
                    reason=reason,
                )
            )
        return candidates, len(document)
    finally:
        document.close()
