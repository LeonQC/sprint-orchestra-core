"""Text -> chunks, with two strategies.

One vector per chunk is the average meaning of its text, so a chunk mixing topics retrieves badly.
Both strategies keep chunks under `chunk_size` tokens with a small overlap, so a sentence on a
boundary appears whole somewhere. They differ in where they cut:

  * "token" — the original: fixed token windows. Cuts mid-sentence, mid-row, mid-word.
  * "structural" (default) — cuts at the strongest boundary available:
        markdown heading > blank line > newline > sentence end > token window
    then packs whole blocks up to the budget. Text with no structure chunks the same either way.

Why it matters here: Docling attachments have real headings and one-line table rows (after
table_linearizer), which a token window cuts through. And each chunk gets its heading path prepended
("Runbook: Checkout Service > Rollback"), so a chunk from deep in a document still says what it's
about — without an LLM call.

Short text (most issues and comments) stays a single chunk.

PDFs arrive page by page. `chunk_pages` chunks them as one document, so a paragraph or section that
runs over a page break stays together and keeps its heading path.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import deque
from dataclasses import dataclass, field, replace
from typing import List, Sequence

import tiktoken

from app.ingest.table_linearizer import _ALIGNMENT_ROW

# cl100k_base is a reasonable, provider-agnostic tokenizer for *sizing* chunks. It does not need to
# match the embedding model's exact tokenizer — we only use it to keep chunks in a sane token range.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# ATX markdown headings (what Docling emits). Up to three leading spaces, per CommonMark.
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*\S)\s*$")

# Sentence end + whitespace, including CJK full stops. The lookbehind skips "J. Smith"-style
# initials. Only a fallback inside an oversized block.
_SENTENCE_END = re.compile(r"(?<![A-Z])([.!?。！？])[ \t]*\n?|([。！？])")

_BREADCRUMB_SEPARATOR = " > "

# Headroom for block-join separators and BPE merges across a join — see `_structural_chunks`.
_JOIN_SLACK = 4

# How full a chunk must be before a heading is allowed to close it early. See `_structural_chunks`.
_HEADING_CLOSE_FILL = 0.6

# `_render` joins blocks with a blank line. Counting it keeps the estimate on the high side;
# under-counting pushed chunks just over budget, and the hard split left a fragment with no
# breadcrumb.
_BLOCK_JOIN_TOKENS = 2


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str, chunk_size: int, overlap: int, strategy: str = "structural"
) -> List[str]:
    """Split `text` into chunks of at most `chunk_size` tokens, overlapping by about `overlap`.

    Returns [] for blank input and [text] when it already fits. `strategy` is "structural" (default)
    or "token" (the original, kept so it can be compared and switched back by config).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []
    if count_tokens(text) <= chunk_size:
        return [text.strip()]
    if strategy == "token":
        return _token_windows(text, chunk_size, overlap)
    if strategy == "structural":
        return _structural_chunks(text, chunk_size, overlap)
    raise ValueError(f"unknown chunk strategy: {strategy!r}")


# --- strategy: token (original) ---------------------------------------------------------------


def _token_windows(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = _ENCODING.encode(text)
    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        piece = _ENCODING.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(tokens):
            break
    return chunks


# --- strategy: structural ----------------------------------------------------------------------


class _Block:
    """A block the packer won't cut, plus its heading path. A heading is a good place to end a chunk,
    but a chunk ending with only a heading is wasted.
    """

    __slots__ = (
        "text", "tokens", "headings", "section", "is_heading", "is_carried", "page", "starts_page"
    )

    def __init__(
        self,
        text: str,
        headings: tuple[str, ...],
        is_heading: bool = False,
        is_carried: bool = False,
        page: int | None = None,
        starts_page: bool = False,
        section: tuple[str, ...] = (),
    ):
        self.text = text
        self.tokens = count_tokens(text)
        self.headings = headings
        # The top of the heading path (levels 1-2): which parent section this block belongs to.
        self.section = section
        self.is_heading = is_heading
        # True for a block re-opened into the next chunk as overlap. It is tail-of-previous-section
        # text, so it must not be what the breadcrumb is computed from — see `_render`.
        self.is_carried = is_carried
        self.page = page
        # First block of a PDF page. A heading here is a section the page itself opens, so it closes
        # the chunk whatever its fill; a non-heading is a paragraph or table running over the break.
        self.starts_page = starts_page


def _blocks(
    text: str, page: int | None = None, heading_stack: List[tuple[int, str]] | None = None
) -> List[_Block]:
    """Split into paragraph/line blocks, tracking the heading stack above each. Headings are always
    their own block. Pass the same `heading_stack` for consecutive pages so a page that continues a
    section still knows which one.
    """
    heading_stack = [] if heading_stack is None else heading_stack
    blocks: List[_Block] = []
    pending: List[str] = []
    first_index = len(blocks)

    def flush() -> None:
        if not pending:
            return
        body = "\n".join(pending).strip()
        pending.clear()
        if body:
            blocks.append(
                _Block(
                    body, tuple(title for _level, title in heading_stack), page=page,
                    section=_top_section(heading_stack),
                )
            )

    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading:
            flush()
            level, title = len(heading.group(1)), heading.group(2)
            # Pop deeper-or-equal levels so `## B` after `## A` replaces A rather than nesting under it.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            blocks.append(
                _Block(
                    line.strip(), tuple(t for _l, t in heading_stack), is_heading=True, page=page,
                    section=_top_section(heading_stack),
                )
            )
        elif not line.strip():
            flush()
        else:
            pending.append(line)
    flush()
    if page is not None and len(blocks) > first_index:
        blocks[first_index].starts_page = True
    return blocks


def _top_section(heading_stack: List[tuple[int, str]]) -> tuple[str, ...]:
    """The level-1 and level-2 headings above a block: the document title and the section ("3.
    Findings"). Subsections ("3.1", "3.2") stay inside their section's parent.
    """
    return tuple(title for level, title in heading_stack if level <= 2)


def _breadcrumb(headings: Sequence[str], first_block: _Block | None) -> str:
    """The heading path to prepend, minus what the chunk already says. If the chunk starts with its
    own heading, drop just that last level — the ancestors still disambiguate ("Rate limits" in which
    runbook?).
    """
    if not headings:
        return ""
    path = list(headings)
    if first_block is not None and first_block.is_heading and path and path[-1] == headings[-1]:
        path.pop()
    return _BREADCRUMB_SEPARATOR.join(path)


def _render(blocks: Sequence[_Block]) -> str:
    """Join a chunk's blocks and prepend the breadcrumb for the section it's about.

    The breadcrumb comes from the first block this chunk owns, not blocks[0], which may be overlap
    from the previous section — that labelled a "Rate limits" chunk with only its parent section.
    """
    if not blocks:
        return ""
    body = "\n\n".join(b.text for b in blocks).strip()
    owner = next((b for b in blocks if not b.is_carried), blocks[0])
    crumb = _breadcrumb(owner.headings, owner)
    return f"{crumb}\n\n{body}" if crumb else body


def _split_oversized(block: _Block, budget: int, overlap: int) -> List[_Block]:
    """Split one block that's over budget, at the weakest boundary left: lines, then sentences, then
    token windows.

    The line step matters: a whole table is one block, and table rows have no sentence punctuation,
    so skipping straight to sentences fell through to a blind cut mid-row: a vendor-spend table lost
    a row that way, separating a vendor from its amount. Pieces are re-joined with the separator they were split on.
    """
    lines = _lines(block.text)
    header = _table_header(lines)
    if header is not None:
        return _split_table(lines, header, block, budget, overlap)
    for pieces, joiner in ((lines, "\n"), (_sentences(block.text), " ")):
        if len(pieces) <= 1:
            continue
        return _pack_pieces(pieces, joiner, block, budget, overlap)
    # No line or sentence structure at all (a base64 blob, a minified log line): the original blind
    # windower is the terminating case, so *something* always fits the budget.
    return [
        _Block(w, block.headings, page=block.page, section=block.section)
        for w in _token_windows(block.text, budget, overlap)
    ]


def _table_header(lines: Sequence[str]) -> tuple[int, int] | None:
    """Line range of the first Markdown table's header and `|---|` row, or None if there is no table."""
    for index in range(1, len(lines)):
        if "|" in lines[index - 1] and "|" in lines[index] and _ALIGNMENT_ROW.match(lines[index]):
            return index - 1, index + 1
    return None


def _split_table(
    lines: Sequence[str], header: tuple[int, int], block: _Block, budget: int, overlap: int
) -> List[_Block]:
    """Cut a table that is over budget between rows, and start every piece with the table's header,
    so rows in a later chunk still say which column each value is ("| CM4 004032 | 4GB | $125 |"
    alone does not say $125 is the price).
    """
    start, end = header
    head = "\n".join(lines[start:end])
    head_tokens = count_tokens(head + "\n")
    rows_end = end
    while rows_end < len(lines) and "|" in lines[rows_end]:
        rows_end += 1
    rows = list(lines[end:rows_end])
    if not rows or head_tokens > budget // 2:
        # No body rows, or a header too wide to repeat: cut between lines as before.
        return _pack_pieces(lines, "\n", block, budget, overlap)

    out: List[_Block] = []
    if start > 0:
        out.extend(_split_lines(lines[:start], block, budget, overlap))
    for piece in _pack_pieces(rows, "\n", block, budget - head_tokens, overlap):
        out.append(_Block(f"{head}\n{piece.text}", block.headings, page=block.page, section=block.section))
    if rows_end < len(lines):
        out.extend(_split_lines(lines[rows_end:], block, budget, overlap))
    return out


def _split_lines(lines: Sequence[str], block: _Block, budget: int, overlap: int) -> List[_Block]:
    """The lines before or after a table, as one block if they fit, else split like any other block."""
    rest = _Block("\n".join(lines), block.headings, page=block.page, section=block.section)
    return [rest] if rest.tokens <= budget else _split_oversized(rest, budget, overlap)


def _pack_pieces(
    pieces: Sequence[str], joiner: str, source: _Block, budget: int, overlap: int
) -> List[_Block]:
    out: List[_Block] = []
    buffer: List[str] = []
    buffered_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffered_tokens
        if buffer:
            out.append(
                _Block(
                    joiner.join(buffer).strip(), source.headings, page=source.page,
                    section=source.section,
                )
            )
            buffer, buffered_tokens = [], 0

    for piece in pieces:
        # Counted with its joiner: " cbc26 detail." is more tokens than "cbc26 detail.", and summing
        # the bare pieces undercounted a 26-sentence chunk by ~25 tokens, so it overflowed and the
        # hard split then cut it mid-sentence.
        piece_tokens = count_tokens(joiner + piece)
        if piece_tokens > budget:
            # One line/sentence that alone blows the budget: recurse down the ladder for it only,
            # so the rest of this block still gets the boundary it deserves.
            flush()
            piece_block = _Block(piece, source.headings, page=source.page, section=source.section)
            out.extend(_split_oversized(piece_block, budget, overlap))
            continue
        if buffered_tokens + piece_tokens > budget:
            flush()
        buffer.append(piece)
        buffered_tokens += piece_tokens
    flush()
    return [b for b in out if b.text]


def _lines(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.strip()]


def _sentences(text: str) -> List[str]:
    parts, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        parts.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _packed_tokens(blocks: Sequence[_Block]) -> int:
    """Token estimate for `_render(blocks)`'s body, including the blank lines between blocks."""
    if not blocks:
        return 0
    return sum(b.tokens for b in blocks) + _BLOCK_JOIN_TOKENS * (len(blocks) - 1)


def _overlap_blocks(blocks: Sequence[_Block], overlap: int, section: tuple[str, ...]) -> List[_Block]:
    """The last whole blocks of a finished chunk, to start the next one with. Whole blocks, since half
    a sentence doesn't keep a fact readable; a block bigger than the overlap budget is skipped.

    Only blocks of the section the next chunk continues: the tail of the previous section is noise
    in this one's vector (a disclaimer ahead of "1. Hematology").
    """
    if overlap <= 0:
        return []
    carried: List[_Block] = []
    total = 0
    for block in reversed(blocks):
        if block.is_heading:
            continue
        if block.headings != section or total + block.tokens > overlap:
            break
        carried.insert(
            0,
            _Block(block.text, block.headings, is_carried=True, page=block.page, section=block.section),
        )
        total += block.tokens
    return carried


@dataclass(frozen=True)
class PagedChunk:
    """A chunk and the source pages its own text comes from (not its overlap)."""

    text: str
    first_page: int | None
    last_page: int | None
    # Indexes into the parent list from `chunk_pages_with_parents`; empty when no parent adds anything.
    parent_indexes: tuple[int, ...] = ()
    # The blocks this chunk's own text came from (not its overlap). Only used to build parents.
    owned_blocks: tuple = field(default=(), compare=False, repr=False)


@dataclass(frozen=True)
class ParentSection:
    """A whole section (or a slice of a long one) that several chunks were cut from. The agent reads
    it when one chunk of the section is not enough.
    """

    text: str
    first_page: int | None
    last_page: int | None


def _structural_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    return [c.text for c in _pack_blocks(_blocks(text), chunk_size, overlap)]


def _pack_blocks(blocks: List[_Block], chunk_size: int, overlap: int) -> List[PagedChunk]:
    if not blocks:
        return []

    # Reserve room for the breadcrumb, sized from the document's longest heading path (capped at a
    # quarter of the budget). `_JOIN_SLACK` covers separators and BPE merges across joins; this is an
    # estimate, and `_enforce_budget` is what actually guarantees the limit.
    longest_path = max(
        (count_tokens(_BREADCRUMB_SEPARATOR.join(b.headings) + "\n\n") for b in blocks if b.headings),
        default=0,
    )
    # No headings = nothing to reserve, so structureless text chunks exactly like the token strategy
    # (test_structureless_text_chunks_identically_under_both_strategies).
    reserve = min(chunk_size // 4, longest_path + _JOIN_SLACK) if longest_path else 0
    budget = max(chunk_size - reserve, chunk_size // 2)

    expanded: List[_Block] = []
    for block in blocks:
        if block.tokens <= budget:
            expanded.append(block)
            continue
        pieces = _split_oversized(block, budget, overlap)
        if pieces:
            pieces[0].starts_page = block.starts_page
        expanded.extend(pieces)

    chunks: List[PagedChunk] = []
    current: List[_Block] = []
    current_tokens = 0
    # Don't close a chunk at a heading until it holds at least this share of the budget. At 0.25 it
    # split on nearly every heading (32 chunks vs 21) and scored worse than the token splitter at
    # chunk_size=800 (MRR 0.907 vs 0.941). At 0.6 chunk counts match, so only the cut positions differ.
    min_close = int(budget * _HEADING_CLOSE_FILL)

    def emit(blocks_: List[_Block]) -> None:
        owned_blocks = tuple(b for b in blocks_ if not b.is_carried)
        owned = [b.page for b in owned_blocks if b.page is not None]
        first, last = (min(owned), max(owned)) if owned else (None, None)
        for piece in _enforce_budget(_render(blocks_), chunk_size, overlap):
            chunks.append(PagedChunk(piece, first, last, owned_blocks=owned_blocks))

    queue = deque(expanded)
    while queue:
        block = queue.popleft()
        # Headings alone in the chunk leave less room than a full-budget piece needs: the piece would
        # overflow and the hard split would cut it mid-sentence with no breadcrumb. Re-split it to fit.
        room = budget - current_tokens - _BLOCK_JOIN_TOKENS
        if (
            current
            and not block.is_heading
            and not any(not b.is_heading for b in current)
            and block.tokens > room
        ):
            pieces = _split_oversized(block, max(room, budget // 4), overlap)
            if len(pieces) > 1 or (pieces and pieces[0].tokens < block.tokens):
                queue.extendleft(reversed(pieces))
                continue
        # A heading that opens a PDF page closes the chunk whatever its fill: the page already
        # separates it from what came before, which is how a one-section-per-page file chunked before
        # pages were merged. A page that opens with body text is a section running over, and stays.
        starts_section = block.is_heading and (current_tokens >= min_close or block.starts_page)
        # `has_body`: never emit a chunk that is only a heading.
        has_body = any(not b.is_heading for b in current)
        if has_body and (starts_section or current_tokens + block.tokens > budget):
            # A heading belongs to what follows, so a chunk must never end on one.
            orphans: List[_Block] = []
            while current and current[-1].is_heading:
                orphans.insert(0, current.pop())
            emit(current)
            current = _overlap_blocks(current, overlap, block.headings) + orphans
            current_tokens = _packed_tokens(current)
            # Re-check after reopening: the new chunk already holds the overlap (and maybe an orphan
            # heading), so the block may no longer fit. Overlap gives way, since the budget is a hard
            # limit (otherwise the hard split left a fragment with no breadcrumb). The heading stays.
            if current_tokens + block.tokens + _BLOCK_JOIN_TOKENS > budget:
                current = list(orphans)
                current_tokens = _packed_tokens(current)
        current.append(block)
        current_tokens += block.tokens + (_BLOCK_JOIN_TOKENS if len(current) > 1 else 0)

    # A trailing group that is nothing but headings (or re-carried overlap) adds no new text.
    if current and any(not b.is_heading and not b.is_carried for b in current):
        emit(current)
    return chunks


# --- PDF: pages as one document ------------------------------------------------------------------

# Lines this close to a page's top or bottom are candidates for a running header or footer.
_FURNITURE_EDGE_LINES = 3
# Needs this many pages and this share of them to call a line "running": two similar pages prove nothing.
_FURNITURE_MIN_PAGES = 3
_FURNITURE_PAGE_SHARE = 0.6


# Running headers and footers are short; a long line is body text, and normalising its digits would
# make "item 1 ..." and "item 2 ..." look identical.
_FURNITURE_MAX_CHARS = 80


def _furniture_key(line: str) -> str | None:
    text = line.strip()
    if not text or len(text) > _FURNITURE_MAX_CHARS or "|" in text or text.startswith(("- ", "* ")):
        return None
    # "PAGE 2 OF 3" and "PAGE 3 OF 3" are the same footer.
    return re.sub(r"\d+", "#", text.lower())


# Lines the extractors add around a page's text, not part of the document: the VLM's page marker and
# its code fence, and Docling's placeholder for a picture it did not read. Left in, the marker (and
# its fence) made page 2 look like a continuation of page 1's last paragraph.
_PAGE_WRAPPER_LINE = re.compile(
    r"^\s*(\[VLM extracted PDF page \d+[^\]]*\]|```(markdown)?|<!--\s*image\s*-->)\s*$"
)


def _strip_page_wrappers(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not _PAGE_WRAPPER_LINE.match(ln))


def _strip_running_furniture(pages: List[tuple[int, str]]) -> List[tuple[int, str]]:
    """Drop headers and footers that repeat on most pages, and keep a repeated heading once as the
    document title (H1) so it heads every chunk's path instead of being re-embedded per page.

    A line counts only if it sits at a page's top or bottom edge on enough pages, so a table row or
    a phrase that merely recurs in the body is never removed.
    """
    if len(pages) < _FURNITURE_MIN_PAGES:
        return pages
    edge_keys: List[set[str]] = []
    for _number, text in pages:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        edge = lines[:_FURNITURE_EDGE_LINES] + lines[-_FURNITURE_EDGE_LINES:]
        edge_keys.append({k for k in map(_furniture_key, edge) if k})
    needed = max(_FURNITURE_MIN_PAGES, math.ceil(len(pages) * _FURNITURE_PAGE_SHARE))
    counts: dict[str, int] = {}
    for keys in edge_keys:
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    running = {key for key, n in counts.items() if n >= needed}
    if not running:
        return pages

    title_seen = False
    out: List[tuple[int, str]] = []
    for number, text in pages:
        kept: List[str] = []
        for line in text.splitlines():
            key = _furniture_key(line)
            if key is None or key not in running:
                kept.append(line)
                continue
            heading = _HEADING.match(line)
            if heading and not title_seen:
                title_seen = True
                kept.append(f"# {heading.group(2)}")
            # Every other occurrence is dropped.
        out.append((number, "\n".join(kept)))
    return out


# --- PDF: heading levels from numbering -------------------------------------------------------
#
# Docling writes every PDF heading as `##`, so "3.1 Payment term" and "3. Payment" look like siblings
# and a chunk's path shows only the last one. Numbering says what the layout model doesn't.

_CJK_NUMERALS = "一二三四五六七八九十百"
_DOTTED = re.compile(r"^(\d+(?:\.\d+)+)(?=[.)、\s]|$)")  # 3.1  3.1.2
_SIMPLE_NUMBER = re.compile(r"^\d+[.)、]\s*\S")  # 3.  3)  3、
# "3 Dataset Construction", the style of academic papers. One or two digits and a capital, so a year
# ("2023 Annual Report") or "3 months later" is not taken for a section number.
_SPACED_NUMBER = re.compile(r"^\d{1,2}\s+[A-Z]")
# "-" because OCR reads the numeral 一 as a dash ("-、血液學").
_CJK_NUMBER = re.compile(rf"^[-{_CJK_NUMERALS}]+[、．.]")
_CJK_PART = re.compile(rf"^第[{_CJK_NUMERALS}\d]+([章部節节條条])")
_PART_LEVELS = {"章": 2, "部": 2, "節": 3, "节": 3, "條": 4, "条": 4}


def _numbered_level(title: str) -> int | None:
    """Heading level implied by the title's own numbering, or None if it has none. Level 1 is the
    document title, so a top-level number is 2 and each dot goes one deeper.
    """
    text = unicodedata.normalize("NFKC", title).strip()  # also folds ⼆ (a radical) into 二
    dotted = _DOTTED.match(text)
    if dotted:
        return min(dotted.group(1).count(".") + 2, 6)
    if _SIMPLE_NUMBER.match(text) or _SPACED_NUMBER.match(text) or _CJK_NUMBER.match(text):
        return 2
    part = _CJK_PART.match(text)
    return _PART_LEVELS[part.group(1)] if part else None


def _first_heading_is_title(pages: List[tuple[int, str]]) -> bool:
    """An unnumbered first `##` heading, with numbered `##` sections after it, is the document title."""
    levels = [
        (len(m.group(1)), m.group(2))
        for _n, text in pages
        for m in map(_HEADING.match, text.splitlines())
        if m
    ]
    if not levels or levels[0][0] != 2 or _numbered_level(levels[0][1]):
        return False
    return any(level == 2 and _numbered_level(title) for level, title in levels[1:])


def _infer_heading_levels(pages: List[tuple[int, str]]) -> List[tuple[int, str]]:
    """Re-level Docling's flat `##` headings by their numbering.

    An unnumbered heading straight after a numbered one, with no text between ("2. Requirements"
    then "AT A GLANCE"), is a label inside that section, so it nests one level down. So is one that
    comes later in the section while another numbered heading still follows ("1. Parsing", body,
    "Parser Customization", "2. Ingestion"). After the last numbered heading it stays a sibling, so
    "Appendix" after the final section's body is not filed under it. Only `##`
    is touched: other levels came from a source that knew them (the VLM).
    """
    after_numbered: int | None = None  # level of a numbered heading with no body text since
    current_numbered: int | None = None  # level of the latest numbered heading, body text or not
    title_pending = _first_heading_is_title(pages)
    # Headings after the last numbered one can't be told from a closing "Appendix", so only those
    # before it are filed under the section they sit in.
    flat_titles = [m.group(2) for _n, t in pages for m in map(_HEADING.match, t.splitlines()) if m and len(m.group(1)) == 2]
    last_numbered = max((i for i, t in enumerate(flat_titles) if _numbered_level(t)), default=-1)
    seen = -1
    out: List[tuple[int, str]] = []
    for number, text in pages:
        lines: List[str] = []
        for line in text.splitlines():
            heading = _HEADING.match(line)
            if not heading:
                if line.strip():
                    after_numbered = None
                lines.append(line)
                continue
            level, title = len(heading.group(1)), heading.group(2)
            if level != 2:
                after_numbered = None
                lines.append(line)
                continue
            seen += 1
            numbered = _numbered_level(title)
            if title_pending:
                title_pending = False
                lines.append(f"# {title}")
                continue
            if numbered:
                new_level, after_numbered, current_numbered = numbered, numbered, numbered
            elif after_numbered:
                new_level = min(after_numbered + 1, 6)
            elif current_numbered and seen < last_numbered:
                new_level = min(current_numbered + 1, 6)
            else:
                new_level = 2
            lines.append(f"{'#' * new_level} {title}")
        out.append((number, "\n".join(lines)))
    return out


def chunk_pages(pages: Sequence[tuple[int, str]], chunk_size: int, overlap: int) -> List[PagedChunk]:
    """Chunk a paginated document (PDF) as one: a paragraph, table or section that runs over a page
    break is not cut there, and the heading path carries across it. Each chunk reports the pages its
    own text came from.
    """
    return _chunk_pages(pages, chunk_size, overlap)


def chunk_pages_with_parents(
    pages: Sequence[tuple[int, str]], chunk_size: int, overlap: int, parent_max_tokens: int
) -> tuple[List[PagedChunk], List[ParentSection]]:
    """`chunk_pages`, plus the parent sections the chunks were cut from (parent-child retrieval).

    Chunks are cut exactly as `chunk_pages` cuts them, so search results don't change. A parent is
    a top-level section ("3. Findings", with its subsections), sliced at `parent_max_tokens` if it
    is longer. Only sections split over two or more chunks get a parent: for the rest, the chunk
    already holds the whole section.
    """
    chunks = _chunk_pages(pages, chunk_size, overlap)
    return _attach_parents(chunks, parent_max_tokens)


def chunk_text_with_parents(
    text: str, chunk_size: int, overlap: int, parent_max_tokens: int, page: int | None = None
) -> tuple[List[PagedChunk], List[ParentSection]]:
    """Structural `chunk_text` for a document that is not a paginated PDF (DOCX, HTML, a slide, a text
    file), plus its parent sections, so a long DOCX gets parents like a PDF does. Same chunks as
    `chunk_text(text, chunk_size, overlap, "structural")`.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return [], []
    if count_tokens(text) <= chunk_size:
        return [PagedChunk(text.strip(), page, page)], []
    chunks = _pack_blocks(_blocks(text, page=page), chunk_size, overlap)
    return _attach_parents(chunks, parent_max_tokens)


def _attach_parents(
    chunks: List[PagedChunk], parent_max_tokens: int
) -> tuple[List[PagedChunk], List[ParentSection]]:
    # Walk every chunk's own blocks in order and start a new group at each new section, or when the
    # group is full. A chunk that holds the end of one section and the start of the next joins both.
    groups: List[dict] = []
    group_of_block: dict[int, int] = {}
    for chunk_index, chunk in enumerate(chunks):
        for block in chunk.owned_blocks:
            # `_enforce_budget` can turn one packed chunk into several pieces that share their blocks;
            # a block goes into its group once.
            if id(block) not in group_of_block:
                last = groups[-1] if groups else None
                if (
                    last is None
                    or block.section != last["section"]
                    or last["tokens"] + block.tokens > parent_max_tokens
                ):
                    groups.append({"section": block.section, "blocks": [], "tokens": 0, "chunks": []})
                groups[-1]["blocks"].append(block)
                groups[-1]["tokens"] += block.tokens
                group_of_block[id(block)] = len(groups) - 1
            group = groups[group_of_block[id(block)]]
            if chunk_index not in group["chunks"]:
                group["chunks"].append(chunk_index)

    parents: List[ParentSection] = []
    parents_of_chunk: List[List[int]] = [[] for _ in chunks]
    for group in groups:
        if len(group["chunks"]) < 2:
            continue
        pages = [b.page for b in group["blocks"] if b.page is not None]
        parents.append(
            ParentSection(
                _render(group["blocks"]),
                min(pages) if pages else None,
                max(pages) if pages else None,
            )
        )
        for chunk_index in group["chunks"]:
            parents_of_chunk[chunk_index].append(len(parents) - 1)

    linked = [
        replace(chunk, parent_indexes=tuple(parents_of_chunk[i])) for i, chunk in enumerate(chunks)
    ]
    return linked, parents


def _chunk_pages(pages: Sequence[tuple[int, str]], chunk_size: int, overlap: int) -> List[PagedChunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    cleaned = _strip_running_furniture(
        [(n, _strip_page_wrappers(t)) for n, t in pages if t and t.strip()]
    )
    cleaned = _infer_heading_levels([(n, t) for n, t in cleaned if t.strip()])
    if not cleaned:
        return []
    heading_stack: List[tuple[int, str]] = []
    blocks: List[_Block] = []
    for number, text in cleaned:
        blocks.extend(_blocks(text, page=number, heading_stack=heading_stack))
    return _pack_blocks(blocks, chunk_size, overlap)


def _enforce_budget(chunk: str, chunk_size: int, overlap: int) -> List[str]:
    """Guarantee `count_tokens(chunk) <= chunk_size`.

    The packer's sums are off by a token or two (BPE merges, breadcrumbs), so anything still over
    budget falls back to the token windower. Rarely fires; it makes the limit an invariant.
    """
    if not chunk.strip():
        return []
    if count_tokens(chunk) <= chunk_size:
        return [chunk]
    return _token_windows(chunk, chunk_size, overlap)
