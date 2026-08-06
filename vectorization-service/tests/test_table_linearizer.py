"""Tests for app/ingest/table_linearizer.py.

The two tables reproduced here are the REAL extracted output of two scanned benchmark pages — one
labelled-column table whose column association the answering model got wrong live, and one
row-label lab table of the shape where a value's meaning depends on both its row and its column.
"""
from app.ingest.table_linearizer import linearize_markdown_tables
import pytest


def test_labelled_columns_get_their_header_attached_to_each_cell():
    # Extracted by the VLM. Asked "Interdepartmental study comes under which
    # heading", the agent answered with the page's H1 instead of "DESCRIPTION" — the fact was in
    # context, the cell-to-column association was not usable.
    text = (
        "# MAJOR STRATEGIC RESEARCH PROGRAMS\n"
        "\n"
        "| PROGRAM | DESCRIPTION | TIMING |\n"
        "|---|---|---|\n"
        "| Nicotine RSM Study | An interdepartmental study which:<br><br>1. Defines the role of "
        "nicotine. | Program complete for product optimization. |\n"
    )
    result = linearize_markdown_tables(text)

    assert "Table rows (one line each):" in result
    assert (
        "- PROGRAM: Nicotine RSM Study; DESCRIPTION: An interdepartmental study which: 1. Defines "
        "the role of nicotine.; TIMING: Program complete for product optimization." in result
    )
    # The original Markdown survives — it is what renders, and it reads better column-wise.
    assert "| PROGRAM | DESCRIPTION | TIMING |" in result


def test_a_blank_first_header_is_treated_as_a_row_label_not_a_value():
    # The first column names the row, so "5" only means something as
    # "Group 2's Dose". Restating it as a bare value would lose exactly the association that matters.
    text = (
        "# TABLE 8-Mouse, Female\n"
        "\n"
        "|  | Group 1 | Group 2 | Group 3 |\n"
        "|---|---:|---:|---:|\n"
        "| Dose (mg/kg) | 0 | 5 | 50 |\n"
        "| Survivors | 10 | 9 | 4 |\n"
    )
    result = linearize_markdown_tables(text)

    assert "- Dose (mg/kg) — Group 1: 0; Group 2: 5; Group 3: 50" in result
    assert "- Survivors — Group 1: 10; Group 2: 9; Group 3: 4" in result


def test_text_without_a_table_is_returned_byte_identical():
    text = "Just prose.\nIt mentions a | pipe character | but is not a table.\n\n## Heading\nMore."
    assert linearize_markdown_tables(text) == text


def test_a_row_whose_column_count_disagrees_with_the_header_is_skipped():
    """A merged/malformed row's header alignment is a guess, and a wrongly-labelled value is worse
    than an unlabelled one — the whole point of this module is trustworthy associations."""
    text = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| 1 | 2 | 3 |\n"
        "| only two | cells |\n"
    )
    result = linearize_markdown_tables(text)

    assert "- A: 1; B: 2; C: 3" in result
    assert "only two" not in result.split("Table rows (one line each):")[1]


def test_empty_cells_are_dropped_rather_than_restated_as_blanks():
    text = "| A | B | C |\n|---|---|---|\n| 1 |  | 3 |\n"
    result = linearize_markdown_tables(text)

    assert "- A: 1; C: 3" in result
    assert "B:" not in result.split("Table rows (one line each):")[1]


def test_two_tables_in_one_document_are_each_restated_after_themselves():
    text = (
        "| A | B |\n|---|---|\n| 1 | 2 |\n"
        "\n"
        "Some prose between them.\n"
        "\n"
        "| C | D |\n|---|---|\n| 3 | 4 |\n"
    )
    result = linearize_markdown_tables(text)

    assert result.count("Table rows (one line each):") == 2
    assert result.index("- A: 1; B: 2") < result.index("Some prose between them.")
    assert result.index("Some prose between them.") < result.index("- C: 3; D: 4")


def test_a_table_with_no_body_rows_is_left_untouched():
    text = "| A | B |\n|---|---|\n\nprose after"
    assert "Table rows" not in linearize_markdown_tables(text)


def test_an_oversized_cell_is_truncated_rather_than_bloating_the_index():
    text = f"| A | B |\n|---|---|\n| {'x' * 900} | ok |\n"
    result = linearize_markdown_tables(text)

    restated = result.split("Table rows (one line each):")[1]
    assert "…" in restated
    assert len(restated) < 700
    # The original cell is untouched in the Markdown above it; only the restatement is bounded.
    assert "x" * 900 in result


def test_a_mostly_empty_ocr_grid_is_not_restated():
    grid = "| 1 | 2 | 3 | 4 |\n|---|---|---|---|\n|  |  | 070 |  |\n|  | 128 |  |  |\n|  |  |  | 8,571 |"
    assert "Table rows" not in linearize_markdown_tables(grid)


# --- a table that runs over a page break ---------------------------------------------------------

from app.ingest.table_linearizer import carry_table_headers  # noqa: E402

_PAGE1 = """## 1. Summary

| Vendor | Category | Amount (USD) |
|---|---|---|
| Vendor-01 | Cloud | 1,250 |
| Vendor-02 | Support | 2,500 |"""

_PAGE2 = """| Vendor-03 | Cloud | 3,750 |
|---|---|---|
| Vendor-04 | Support | 5,000 |"""


def test_continued_table_gets_its_header_back():
    page2 = carry_table_headers([_PAGE1, _PAGE2])[1]
    assert page2.splitlines()[:4] == [
        "| Vendor | Category | Amount (USD) |",
        "|---|---|---|",
        "| Vendor-03 | Cloud | 3,750 |",
        "| Vendor-04 | Support | 5,000 |",
    ]
    restated = linearize_markdown_tables(page2)
    assert "- Vendor: Vendor-03; Category: Cloud; Amount (USD): 3,750" in restated
    assert "- Vendor: Vendor-04; Category: Support; Amount (USD): 5,000" in restated


def test_without_the_fix_the_first_row_became_the_header():
    # What Docling's per-page output did before: the proof the carry is worth having.
    assert "Vendor-03: Vendor-04" in linearize_markdown_tables(_PAGE2)


def test_header_is_carried_across_a_second_page_break():
    page3 = "| Vendor-05 | Cloud | 6,250 |\n|---|---|---|\n| Vendor-06 | Support | 7,500 |"
    assert carry_table_headers([_PAGE1, _PAGE2, page3])[2].startswith("| Vendor | Category | Amount (USD) |")


def test_a_new_table_after_a_heading_is_not_treated_as_a_continuation():
    page2 = "## 2. Other\n\n" + _PAGE2
    assert carry_table_headers([_PAGE1, page2])[1] == page2


def test_a_page_that_repeats_the_header_is_left_alone():
    repeated = "| Vendor | Category | Amount (USD) |\n|---|---|---|\n| Vendor-03 | Cloud | 3,750 |"
    assert carry_table_headers([_PAGE1, repeated])[1] == repeated


def test_a_different_column_count_is_left_alone():
    two_cols = "| Vendor-03 | 3,750 |\n|---|---|\n| Vendor-04 | 5,000 |"
    assert carry_table_headers([_PAGE1, two_cols])[1] == two_cols


def test_a_table_whose_rows_carry_no_digits_is_left_alone():
    text_only = "| Alpha | Beta | Gamma |\n|---|---|---|\n| Delta | Eps | Zeta |"
    assert carry_table_headers([_PAGE1, text_only])[1] == text_only


def test_a_footer_under_the_table_does_not_hide_it():
    page1 = _PAGE1 + "\n\nPage 1 of 9"
    assert carry_table_headers([page1, _PAGE2])[1].startswith("| Vendor | Category | Amount (USD) |")


def test_a_paragraph_under_the_table_breaks_the_continuation():
    page1 = _PAGE1 + "\n\n" + "All figures above are estimates prepared by finance and may change. " * 3
    assert carry_table_headers([page1, _PAGE2])[1] == _PAGE2


def test_a_continued_line_carries_the_header_even_without_digits():
    text_only = "Table 1 (continued)\n\n| Alpha | Beta | Gamma |\n|---|---|---|\n| Delta | Eps | Zeta |"
    out = carry_table_headers([_PAGE1, text_only])[1]
    assert "| Vendor | Category | Amount (USD) |\n|---|---|---|\n| Alpha | Beta | Gamma |" in out
    assert out.startswith("Table 1 (continued)")


def test_a_continued_line_with_a_different_column_count_is_left_alone():
    two_cols = "(continued)\n| A | B |\n|---|---|\n| C | D |"
    assert carry_table_headers([_PAGE1, two_cols])[1] == two_cols


def test_a_paragraph_without_a_continued_marker_still_breaks_the_continuation():
    page2 = "Some intro text.\n\n" + _PAGE2
    assert carry_table_headers([_PAGE1, page2])[1] == page2


def test_exact_density_boundary_keeps_output_and_its_whitespace():
    text = "| A | B | C | D | E |\n|---|---|---|---|---|\n| one | two | | | |\n"
    expected = text.rstrip("\n") + "\n\nTable rows (one line each):\n- A: one; B: two"
    assert linearize_markdown_tables(text) == expected


@pytest.mark.parametrize("width, restated", [(24, True), (25, False)])
def test_column_limit_preserves_original_table_without_guessing_labels(width, restated):
    headers = []
    cells = []
    for index in range(width):
        headers.append(f"Column {index}")
        cells.append(str(index))
    text = "|" + "|".join(headers) + "|\n"
    text += "|" + "|".join(["---"] * width) + "|\n"
    text += "|" + "|".join(cells) + "|"
    result = linearize_markdown_tables(text)
    assert result.startswith(text)
    assert ("Table rows (one line each):" in result) is restated


def test_an_empty_page_breaks_header_carry_instead_of_linking_unrelated_pages():
    pages = [_PAGE1, "", _PAGE2]
    assert carry_table_headers(pages) == pages


def test_two_short_leading_lines_with_continuation_marker_keep_both_lines():
    page2 = "Table 1 continued\nAmounts in USD\n\n" + _PAGE2
    restored = carry_table_headers([_PAGE1, page2])[1]
    assert restored.startswith("Table 1 continued\nAmounts in USD\n\n| Vendor | Category | Amount (USD) |")


def test_three_footer_lines_prevent_table_header_carryover():
    page1 = _PAGE1 + "\nConfidential\nPage 1\nCompany copy"
    assert carry_table_headers([page1, _PAGE2])[1] == _PAGE2
