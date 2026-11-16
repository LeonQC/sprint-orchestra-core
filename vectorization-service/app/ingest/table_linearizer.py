"""Restate Markdown table rows as self-describing "header: value" lines, appended after the table.

Markdown tables (from Docling or the VLM) encode meaning by position, which fails twice:
  * The model: with a table correctly in context, it still answered "which heading is X under?"
    with the page title instead of the column header.
  * Embeddings: a chunk boundary can separate body rows from the header, leaving rows of bare,
    unlabelled values that are nearly unretrievable.

Appended, not replacing the table: Markdown is better for column questions ("which group had the
highest dose") and renders in the UI; the restated lines are better for row lookups and survive
chunking.

Done here in code rather than by the VLM: it's a pure text transform, it would double output on the
pages already short of tokens, and this way it covers Docling and VLM tables alike.
"""
from __future__ import annotations

import re

# The "|---|---:|" alignment row between header and body — what makes it a real table, not prose
# with pipes in it.
_ALIGNMENT_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")

# Docling and the VLM both encode a line break inside a cell as a literal <br>; flattened to a space
# so a restated row stays one line (the whole point of the restatement is one self-contained line).
_CELL_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Guardrails. A "table" far outside these shapes is more likely to be prose or ASCII art that happens
# to satisfy the pattern than a real table, and restating it would inject noise into the index.
_MAX_COLUMNS = 24
_MAX_CELL_CHARS = 400
# OCR of a table image can yield a grid that is mostly empty cells with a few stray numbers; the
# "header" is then a data row and every restated label is wrong.
_MIN_FILLED_SHARE = 0.4


def _clean_cell(cell: str) -> str:
    """One cell on one line: `<br>` becomes a space, then whitespace runs are collapsed (a list cell
    `text<br><br>1. item` would otherwise leave double spaces).
    """
    return " ".join(_CELL_BREAK.sub(" ", cell).split())


def _split_row(line: str) -> list[str]:
    """Cells of one Markdown table row, without the leading/trailing pipe artifacts."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = []
    for cell in stripped.split("|"):
        cells.append(_clean_cell(cell))
    return cells


def _restate_row(headers: list[str], cells: list[str]) -> str | None:
    """One row as a self-describing line, or None if there's nothing to restate.

    * Labelled columns (`| PROGRAM | DESCRIPTION |`): each cell becomes `HEADER: value`.
    * Row-label tables (`|  | Group 1 | Group 2 |`, blank first header): the first cell is the
      subject, e.g. `Dose (mg/kg) — Group 1: 0; Group 2: 5`. Common in lab/finance tables, where a
      value depends on both its row and its column.
    """
    pairs = []
    for header, cell in zip(headers, cells):
        if not cell or cell in {"-", "—", "n/a", "N/A"}:
            continue
        if len(cell) > _MAX_CELL_CHARS:
            cell = cell[:_MAX_CELL_CHARS].rstrip() + "…"
        pairs.append((header, cell))
    if not pairs:
        return None

    subject = ""
    if headers and not headers[0] and cells and cells[0]:
        # Row-label table: the blank first header means column one names the row, not a value in it.
        subject = _clean_cell(cells[0])
        labelled_pairs = []
        for header, cell in pairs:
            if header:
                labelled_pairs.append((header, cell))
        pairs = labelled_pairs

    parts = []
    for header, cell in pairs:
        if header:
            parts.append(f"{header}: {cell}")
        else:
            parts.append(cell)
    body = "; ".join(parts)
    if not body:
        return None
    return f"{subject} — {body}" if subject else body


def _is_dense(body_lines: list[str]) -> bool:
    total_cells = 0
    filled_cells = 0
    for line in body_lines:
        for cell in _split_row(line):
            total_cells += 1
            if cell:
                filled_cells += 1
    if total_cells == 0:
        return False
    return filled_cells / total_cells >= _MIN_FILLED_SHARE


def linearize_markdown_tables(text: str) -> str:
    """Return `text` with a restated block after each Markdown table. Everything else is unchanged,
    and a table that can't be parsed confidently is left as is — this runs on every attachment.
    """
    if "|" not in text:
        return text

    lines = text.splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        # A table is a header line, an alignment line, then one or more body lines. Anything else is
        # copied through untouched.
        is_table = (
            index + 1 < len(lines)
            and "|" in lines[index]
            and _ALIGNMENT_ROW.match(lines[index + 1])
            and "|" in lines[index + 1]
        )
        if not is_table:
            out.append(lines[index])
            index += 1
            continue

        headers = _split_row(lines[index])
        body_start = index + 2
        body_end = body_start
        while body_end < len(lines) and "|" in lines[body_end] and lines[body_end].strip():
            body_end += 1

        restated = []
        if 1 < len(headers) <= _MAX_COLUMNS and _is_dense(lines[body_start:body_end]):
            for line in lines[body_start:body_end]:
                cells = _split_row(line)
                # A row with a different column count than the header is a malformed/merged row; its
                # header alignment is a guess, and a wrong label is worse than no label.
                if len(cells) != len(headers):
                    continue
                row = _restate_row(headers, cells)
                if row:
                    restated.append(row)

        out.extend(lines[index:body_end])
        if restated:
            # Labelled so a reader (and the answering model) can see this is the same table restated,
            # not additional rows of data that the table itself omitted.
            out.append("")
            out.append("Table rows (one line each):")
            for row in restated:
                out.append(f"- {row}")
        index = body_end

    return "\n".join(out)


# --- a table that runs over a page break ---------------------------------------------------------


_TRAILING_FOOTER_LINES = 2


def _starts_table(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and "|" in lines[index + 1]
        and bool(_ALIGNMENT_ROW.match(lines[index + 1]))
    )


def _trailing_table_header(lines: list[str]) -> tuple[str, str] | None:
    """Header and alignment lines of the table the text ends with, if it ends with one."""
    end = len(lines)
    # A page number or running footer under the table must not hide it.
    skipped = 0
    while end and skipped < _TRAILING_FOOTER_LINES and (
        not lines[end - 1].strip() or ("|" not in lines[end - 1] and len(lines[end - 1]) <= 80)
    ):
        skipped += bool(lines[end - 1].strip())
        end -= 1
    start = end
    while start and "|" in lines[start - 1] and lines[start - 1].strip():
        start -= 1
    if start == end or not _starts_table(lines, start):
        return None
    return lines[start], lines[start + 1]


def _looks_like_continued_row(line: str, header_line: str) -> bool:
    """A "header" that is really a data row: same column count, digits in it, none in the real
    header. Real headers ("Amount (USD)") rarely carry digits and data ("Vendor-33", "41,250") does;
    a table where that doesn't hold is left alone rather than guessed at.
    """
    cells = _split_row(line)
    headers = _split_row(header_line)
    if len(cells) != len(headers):
        return False
    return _has_digit(cells) and not _has_digit(headers)


def _has_digit(cells: list[str]) -> bool:
    for cell in cells:
        for character in cell:
            if character.isdigit():
                return True
    return False


_CONTINUED_MARKER = re.compile(r"continued|\bcont\.|續|续|接上頁|接上页", re.IGNORECASE)
_LEADING_LINES = 2


def _continuation_table_start(lines: list[str]) -> tuple[int, bool] | None:
    """Index of the table the page opens with, and whether a "(continued)" line says it continues.

    Up to two short non-table lines may come first; they only count if one says "continued".
    """
    nonempty_indices = []
    for index, line in enumerate(lines):
        if line.strip():
            nonempty_indices.append(index)
    leading_indices = nonempty_indices[: _LEADING_LINES + 1]

    for position, index in enumerate(leading_indices):
        if not _starts_table(lines, index):
            continue

        leading_lines = []
        for leading_index in leading_indices[:position]:
            leading_lines.append(lines[leading_index])
        if not leading_lines:
            return index, False

        for line in leading_lines:
            if "|" in line or len(line) > 80:
                return None
        for line in leading_lines:
            if _CONTINUED_MARKER.search(line):
                return index, True
        return None
    return None


def _same_columns_not_header(line: str, header_line: str) -> bool:
    """With a "(continued)" line there is no need for digits: same columns, and not the header again."""
    cells = _split_row(line)
    headers = _split_row(header_line)
    return len(cells) == len(headers) and cells != headers


def carry_table_headers(pages: list[str]) -> list[str]:
    """Put the header back on a table that continues from the previous page.

    Docling reads each page alone, so page 2 of a long table takes its first data row as the header,
    and every restated row on that page then carries wrong labels ("Vendor-33: Vendor-34; Cloud:
    Support"). Worse than no label. When a page opens straight into a table that fits the previous
    page's trailing table (same columns, first row is data), the previous header is restored and the
    mistaken header row becomes a data row again.
    """
    out: list[str] = []
    header: tuple[str, str] | None = None
    for text in pages:
        lines = text.splitlines()
        if header:
            continuation = _continuation_table_start(lines)
            if continuation is not None:
                first, marked = continuation
                if marked:
                    continues_previous_table = _same_columns_not_header(lines[first], header[0])
                else:
                    continues_previous_table = _looks_like_continued_row(lines[first], header[0])

                if continues_previous_table:
                    lines = lines[:first] + [header[0], header[1], lines[first]] + lines[first + 2 :]
        out.append("\n".join(lines))
        header = _trailing_table_header(lines)
    return out
