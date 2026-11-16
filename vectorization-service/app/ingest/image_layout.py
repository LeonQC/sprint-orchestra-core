"""Pure geometry signals for deciding whether flat image OCR loses structure.

This module does not know document names, labels, or expected answers. It only
examines OCR bounding boxes, keeping the paid-VLM routing rule testable and
separate from parsing/network code.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutAnalysis:
    multi_column_row_ratio: float
    repeated_pair_row_ratio: float

    @property
    def structured_row_ratio(self) -> float:
        return max(self.multi_column_row_ratio, self.repeated_pair_row_ratio)

    @property
    def reason(self) -> str | None:
        if self.repeated_pair_row_ratio > self.multi_column_row_ratio:
            return "repeated_two_column_layout"
        if self.multi_column_row_ratio > 0:
            return "multi_column_layout"
        return None


def _left(bbox) -> float:
    return min(point[0] for point in bbox)


def _right(bbox) -> float:
    return max(point[0] for point in bbox)


def analyze_layout(rows: list[list], median_height: float) -> LayoutAnalysis:
    """Fraction of rows whose layout flat OCR would destroy. Multi-column table rows need two or more
    wide gaps; two-column rows count only when at least four share roughly the same divider, so a
    lone title/date pair doesn't count but label/value forms do.
    """
    if not rows:
        return LayoutAnalysis(0.0, 0.0)

    gap_threshold = max(median_height * 1.5, 1.0)
    divider_tolerance = max(median_height * 4.0, 1.0)
    multi_column_rows = 0
    pair_dividers: list[float] = []

    for row in rows:
        gaps = [
            (_left(right[0]) - _right(left[0]), (_right(left[0]) + _left(right[0])) / 2)
            for left, right in zip(row, row[1:])
        ]
        wide_gaps = [(gap, divider) for gap, divider in gaps if gap > gap_threshold]
        if len(wide_gaps) >= 2:
            multi_column_rows += 1
        if len(row) == 2 and len(wide_gaps) == 1:
            pair_dividers.append(wide_gaps[0][1])

    largest_aligned_cluster = 0
    for divider in pair_dividers:
        cluster_size = sum(abs(other - divider) <= divider_tolerance for other in pair_dividers)
        largest_aligned_cluster = max(largest_aligned_cluster, cluster_size)
    # Four repeated rows are strong page-level evidence. Fewer can be produced
    # by a title/status row plus a footer and are not worth a paid VLM call.
    repeated_pair_rows = largest_aligned_cluster if largest_aligned_cluster >= 4 else 0
    return LayoutAnalysis(
        multi_column_row_ratio=multi_column_rows / len(rows),
        repeated_pair_row_ratio=repeated_pair_rows / len(rows),
    )
