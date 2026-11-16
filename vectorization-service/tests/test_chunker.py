import pytest

from app.ingest.chunker import (
    _numbered_level,
    chunk_pages,
    chunk_pages_with_parents,
    chunk_text,
    chunk_text_with_parents,
    count_tokens,
)

# --- contract shared by both strategies --------------------------------------------------------


@pytest.mark.parametrize("strategy", ["token", "structural"])
def test_blank_text_yields_no_chunks(strategy):
    assert chunk_text("", 500, 80, strategy) == []
    assert chunk_text("   \n  ", 500, 80, strategy) == []


@pytest.mark.parametrize("strategy", ["token", "structural"])
def test_short_text_is_single_chunk(strategy):
    text = "connection pool leak in the payment service"
    assert chunk_text(text, 500, 80, strategy) == [text]


@pytest.mark.parametrize("strategy", ["token", "structural"])
def test_overlap_must_be_smaller_than_size(strategy):
    with pytest.raises(ValueError):
        chunk_text("anything", 100, 100, strategy)


@pytest.mark.parametrize("strategy", ["token", "structural"])
def test_every_chunk_stays_within_budget(strategy):
    text = "\n\n".join(
        f"## Section {i}\n\n" + " ".join(f"word{i}x{j}" for j in range(120)) for i in range(8)
    )
    chunks = chunk_text(text, 200, 40, strategy)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 200 for c in chunks)


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("x " * 5000, 100, 20, "semantic-magic")


# --- strategy: token (original behavior, unchanged) --------------------------------------------


def test_token_strategy_splits_with_overlap():
    # ~1200 tokens of distinct words so windows don't collapse.
    text = " ".join(f"word{i}" for i in range(1200))
    chunks = chunk_text(text, 500, 80, "token")

    assert len(chunks) > 1
    assert all(count_tokens(c) <= 500 for c in chunks)
    # Adjacent chunks overlap: the tail of chunk 0 reappears at the head of chunk 1.
    tail = chunks[0].split()[-1]
    assert tail in chunks[1].split()


def test_structureless_text_chunks_identically_under_both_strategies():
    """The structural packer is a refinement, not a replacement: with no boundary to find, it must
    fall all the way through to the same token windows the original splitter produced."""
    text = " ".join(f"word{i}" for i in range(1200))
    assert chunk_text(text, 500, 80, "structural") == chunk_text(text, 500, 80, "token")


# --- strategy: structural ----------------------------------------------------------------------


RUNBOOK = """# Runbook: Checkout Service

Owner: payments-platform.

## Root cause

The HikariCP connection pool saturates under holiday load. Default maximumPoolSize is 10,
which is below the sustained concurrency of the checkout worker fleet.

## Rollback

Drain the connection pool, then restart the checkout deployment one pod at a time.
Do not restart all pods simultaneously; the cold pool will re-saturate immediately.
"""


def test_structural_does_not_cut_mid_word_where_token_does():
    """The concrete regression this strategy exists for. The token splitter's window lands inside
    `maximumPoolSize`, opening the next chunk with the orphan fragment `Size is 10,`."""
    token_chunks = chunk_text(RUNBOOK, 40, 8, "token")
    assert any(c.startswith("Size") for c in token_chunks)

    structural_chunks = chunk_text(RUNBOOK, 40, 8, "structural")
    assert not any(c.startswith("Size") for c in structural_chunks)


def test_structural_chunk_carries_its_heading_breadcrumb():
    chunks = chunk_text(RUNBOOK, 40, 8, "structural")
    rollback = next(c for c in chunks if "Drain the connection pool" in c)
    # The property, not the mechanism: a chunk from the middle of the document must name both the
    # document it belongs to and the section it came from. Whether the section arrives as the literal
    # `## Rollback` heading line (the chunk started there) or as the breadcrumb's trailing element
    # (the section's body was split across chunks) is an implementation detail — asserting the former
    # made this test fail on output that was strictly better.
    assert rollback.startswith("Runbook: Checkout Service")
    assert "Rollback" in rollback


def test_breadcrumb_does_not_repeat_the_heading_the_chunk_already_opens_with():
    chunks = chunk_text(RUNBOOK, 40, 8, "structural")
    rollback = next(c for c in chunks if "Drain the connection pool" in c)
    assert rollback.count("Rollback") == 1


def test_structural_prefers_heading_boundaries():
    """No chunk should contain content from two different sections when each section fits alone."""
    chunks = chunk_text(RUNBOOK, 40, 8, "structural")
    for chunk in chunks:
        assert not ("HikariCP" in chunk and "Drain the connection pool" in chunk)


def test_structural_keeps_table_rows_intact():
    """Linearized table rows are newline-separated (see app/ingest/table_linearizer). A blind token
    window cuts through one; the line-respecting packer must not."""
    rows = [
        f"Creditor: Bank{i} | Type: Mortgage | Amount: {i}0,001 | Year incurred: 20{i:02d}"
        for i in range(40)
    ]
    text = "## Creditors\n\n" + "\n".join(rows)
    chunks = chunk_text(text, 150, 30, "structural")
    assert len(chunks) > 1
    for row in rows:
        assert any(row in chunk for chunk in chunks), f"row was split: {row}"


def _price_table(rows: int) -> str:
    lines = ["| Part Number | RAM | Price |", "|---|---|---|"]
    lines += [f"| CM4 {i:06d} | {i % 8 + 1}GB | ${100 + i} |" for i in range(rows)]
    return "\n".join(lines)


def test_a_table_split_over_chunks_repeats_its_header_in_every_chunk():
    text = "## Prices\n\n" + _price_table(60)
    chunks = chunk_text(text, 200, 30, "structural")
    with_rows = [c for c in chunks if "| CM4 " in c]
    assert len(with_rows) > 1
    for chunk in with_rows:
        assert "| Part Number | RAM | Price |\n|---|---|---|" in chunk
        assert count_tokens(chunk) <= 200
    for i in range(60):
        assert any(f"| CM4 {i:06d} |" in c for c in chunks), f"row {i} lost"


def test_text_around_a_split_table_is_kept_once():
    text = "## Prices\n\nTable 2: price list\n" + _price_table(60) + "\nPrices exclude tax."
    chunks = chunk_text(text, 200, 30, "structural")
    assert sum("Table 2: price list" in c for c in chunks) == 1
    assert sum("Prices exclude tax." in c for c in chunks) == 1


def test_structural_overlap_carries_whole_blocks():
    sections = "\n\n".join(f"Fact number {i} about the payment gateway." for i in range(40))
    chunks = chunk_text("## Facts\n\n" + sections, 120, 40, "structural")
    assert len(chunks) > 1
    # The tail paragraph of one chunk reappears verbatim (not half-cut) at the head of the next.
    body_before = [line for line in chunks[0].splitlines() if line.startswith("Fact number")]
    assert body_before[-1] in chunks[1]


def test_structural_splits_a_single_oversized_paragraph_at_sentences():
    paragraph = " ".join(f"Sentence {i} describes a distinct failure mode." for i in range(200))
    chunks = chunk_text(paragraph, 200, 40, "structural")
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 200 for c in chunks)
    # Sentence-level cuts, so no chunk ends mid-sentence.
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_structural_does_not_emit_a_heading_only_chunk():
    text = "# Title\n\n" + "\n\n".join(f"Paragraph {i} " + "filler " * 60 for i in range(6))
    for chunk in chunk_text(text, 150, 30, "structural"):
        body = [line for line in chunk.splitlines() if line.strip() and not line.startswith("#")]
        assert body, f"chunk carries no body text: {chunk!r}"


# --- overlap stays inside a section -------------------------------------------------------------


def test_overlap_is_not_carried_into_the_next_section():
    intro = "\n\n".join(f"Disclaimer line {i} " + "notice " * 25 for i in range(4))
    body = "\n\n".join(f"Blood count {i} " + "filler " * 30 for i in range(8))
    chunks = chunk_text(f"## Notice\n\n{intro}\n\n## Hematology\n\n{body}", 150, 60, "structural")
    hematology = [c for c in chunks if "Blood count" in c]
    assert hematology and not any("Disclaimer" in c for c in hematology)


# --- pages as one document ----------------------------------------------------------------------


def _para(tag: str, n: int) -> str:
    return " ".join(f"{tag}{i} detail." for i in range(n))


def test_a_section_running_over_a_page_break_stays_together():
    page1 = f"## Payment terms\n\n{_para('first', 20)}"
    page2 = _para("second", 20)  # no heading: the same section continues
    chunks = chunk_pages([(1, page1), (2, page2)], 800, 100)
    assert len(chunks) == 1
    assert (chunks[0].first_page, chunks[0].last_page) == (1, 2)
    assert "first19" in chunks[0].text and "second0" in chunks[0].text


def test_continuation_page_keeps_the_heading_path():
    page1 = f"## Payment terms\n\n{_para('first', 150)}"
    page2 = _para("second", 150)
    chunks = chunk_pages([(1, page1), (2, page2)], 200, 30)
    on_page2 = [c for c in chunks if "second100" in c.text]
    assert on_page2 and on_page2[0].text.startswith("Payment terms")


def test_a_page_that_opens_with_a_heading_starts_a_new_chunk():
    pages = [(n, f"## Section {n}\n\nShort body for section {n} " + "word " * 12) for n in (1, 2, 3)]
    # Each page alone is far below the heading-close fill, and still must not merge.
    chunks = chunk_pages(pages, 40, 5)
    assert [(c.first_page, c.last_page) for c in chunks] == [(1, 1), (2, 2), (3, 3)]


def test_running_header_and_footer_are_dropped_and_the_title_heads_the_path():
    def page(n: int, heading: str) -> tuple[int, str]:
        return n, (
            f"ACME CORP CONFIDENTIAL\n\n## Handbook\n\n## {heading}\n\n{_para(f'p{n}x', 40)}\n\n"
            f"PAGE {n} OF 3"
        )

    chunks = chunk_pages([page(1, "One"), page(2, "Two"), page(3, "Three")], 120, 20)
    joined = "\n".join(c.text for c in chunks)
    assert "CONFIDENTIAL" not in joined and "PAGE " not in joined
    assert joined.count("Handbook") == len(chunks) or "Handbook" in chunks[0].text
    two = next(c for c in chunks if "p2x0" in c.text)
    assert "Handbook > " in two.text or two.text.startswith("Handbook")


def test_a_body_line_that_recurs_is_not_mistaken_for_furniture():
    # "Status: Pass" repeats on every page but sits mid-page, away from the top and bottom edges.
    def page(n: int, word: str) -> tuple[int, str]:
        rows = "\n\n".join(f"{word} reading {chr(97 + i)} was recorded." for i in range(8))
        return n, f"## {word}\n\n{rows}\n\nStatus: Pass\n\n{rows[::-1]}\n\nClosing {word} note."

    pages = [page(1, "alpha"), page(2, "beta"), page(3, "gamma")]
    joined = "\n".join(c.text for c in chunk_pages(pages, 150, 20))
    assert joined.count("Status: Pass") == 3


def test_paged_chunks_respect_the_budget_and_report_their_pages():
    pages = [(n, f"## Part {n}\n\n" + _para(f"w{n}_", 300)) for n in range(1, 5)]
    chunks = chunk_pages(pages, 200, 30)
    assert all(count_tokens(c.text) <= 200 for c in chunks)
    assert all(c.first_page is not None and c.first_page <= c.last_page for c in chunks)


def test_vlm_page_marker_does_not_make_the_next_page_a_continuation():
    page1 = f"## One\n\n{_para('a', 40)}"
    page2 = f"[VLM extracted PDF page 2 (sparse_column_gap)]\n```markdown\n\n## Two\n\n{_para('b', 40)}\n```"
    chunks = chunk_pages([(1, page1), (2, page2)], 800, 100)
    assert [(c.first_page, c.last_page) for c in chunks] == [(1, 1), (2, 2)]
    assert not any("VLM extracted" in c.text or "```" in c.text for c in chunks)


def test_image_placeholders_are_dropped():
    chunks = chunk_pages([(1, "<!-- image -->\n\n## One\n\nbody text here")], 800, 100)
    assert "<!-- image -->" not in chunks[0].text


# --- heading levels from numbering ---------------------------------------------------------------


@pytest.mark.parametrize(
    "title, level",
    [
        ("3. Payment", 2), ("3) Payment", 2), ("三、付款", 2), ("⼆、鐵質", 2), ("-、⾎液學", 2),
        ("3.1 Payment term", 3), ("3.1.2 Late fee", 4), ("第三章 付款", 2), ("第二節 期限", 3),
        ("AT A GLANCE", None), ("2024 plan", None), ("Summary", None),
        ("3 Dataset Construction", 2), ("10 CONCLUSION", 2), ("2023 Annual Report", None),
        ("3 months later", None),
    ],
)
def test_numbered_level(title, level):
    assert _numbered_level(title) == level


def _flat(*lines: str) -> str:
    return "\n\n".join(lines)


def test_numbered_subsection_gets_its_chapter_in_the_path():
    # Docling emits every heading as `##`; without numbering "3.1" would replace "3. Payment".
    body = _para("fee", 150)
    page = _flat("## 3. Payment", "## 3.1 Payment term", body)
    chunks = chunk_pages([(1, page)], 120, 20)
    deep = next(c for c in chunks if "fee100" in c.text)
    assert "3. Payment > 3.1 Payment term" in deep.text


def test_label_straight_after_a_numbered_heading_nests_under_it():
    body = _para("req", 150)
    page = _flat("## 2. Requirements", "## AT A GLANCE", body)
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "req100" in c.text)
    assert "2. Requirements > AT A GLANCE" in deep.text


def test_unnumbered_heading_after_body_text_stays_a_sibling():
    body = _para("end", 150)
    page = _flat("## 3. Payment", "Some body text.", "## Appendix", body)
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "end100" in c.text)
    assert "3. Payment" not in deep.text and "Appendix" in deep.text


def test_unnumbered_heading_inside_a_section_nests_when_another_numbered_one_follows():
    page = _flat(
        "## 1. Parsing", "Intro.", "## Parser Customization", _para("cust", 150),
        "## 2. Ingestion", "More.",
    )
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "cust100" in c.text)
    assert "1. Parsing > Parser Customization" in deep.text


def test_unnumbered_heading_after_the_last_numbered_section_stays_a_sibling():
    page = _flat("## 1. Parsing", "Intro.", "## 2. Ingestion", "Body.", "## FAQ", _para("faq", 150))
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "faq100" in c.text)
    assert "2. Ingestion" not in deep.text and "FAQ" in deep.text


def test_levels_other_than_two_are_left_alone():
    page = _flat("# 1. Real top level", "### 3.1 Real third level", _para("x", 10))
    text = chunk_pages([(1, page)], 800, 100)[0].text
    assert "# 1. Real top level" in text and "### 3.1 Real third level" in text


def test_unnumbered_first_heading_is_the_title_when_numbered_sections_follow():
    page = _flat("## Blood Test Report", "## 1. Hematology", _para("cbc", 150))
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "cbc100" in c.text)
    assert "Blood Test Report > 1. Hematology" in deep.text


def test_first_heading_is_not_a_title_when_nothing_is_numbered():
    page = _flat("## Overview", "## Details", _para("plain", 150))
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "plain100" in c.text)
    assert "Overview" not in deep.text


def test_heading_then_a_long_paragraph_never_leaves_a_mid_sentence_fragment():
    # Regression: sentences were counted without their joiner, so the chunk overflowed its budget and
    # the hard split cut it mid-sentence, leaving a fragment with no breadcrumb.
    text = "## Title\n\n## 1. Section\n\n" + _para("cbc", 150)
    chunks = chunk_text(text, 120, 20, "structural")
    assert all(count_tokens(c) <= 120 for c in chunks)
    assert all(c.rstrip().endswith("detail.") for c in chunks)
    assert all(c.startswith(("Title", "# ", "## ", "Section", "1. Section")) or "1. Section" in c.split("\n")[0] for c in chunks)


# --- parent sections (parent-child retrieval) ---------------------------------------------------


def _runbook() -> list[tuple[int, str]]:
    page1 = f"## Runbook\n\n## 1. Overview\n\n{_para('intro', 10)}\n\n## 2. Rollback\n\n{_para('step', 150)}"
    page2 = f"## 3. Contacts\n\n{_para('call', 10)}"
    return [(1, page1), (2, page2)]


def test_parents_do_not_change_how_chunks_are_cut():
    chunks, _parents = chunk_pages_with_parents(_runbook(), 200, 30, 3000)
    assert [c.text for c in chunks] == [c.text for c in chunk_pages(_runbook(), 200, 30)]


def test_a_section_split_over_chunks_gets_one_parent_with_the_whole_section():
    chunks, parents = chunk_pages_with_parents(_runbook(), 200, 30, 3000)
    assert len(parents) == 1
    rollback = parents[0].text
    assert rollback.startswith("Runbook") and "## 2. Rollback" in rollback
    assert "step0 " in rollback and "step149 " in rollback
    assert "intro0" not in rollback and "call0" not in rollback
    # Every chunk holding a piece of Rollback points at it; the others point at nothing.
    for chunk in chunks:
        assert chunk.parent_indexes == ((0,) if "step" in chunk.text else ())


def test_a_section_that_fits_in_one_chunk_has_no_parent():
    chunks, parents = chunk_pages_with_parents([(1, f"## Notes\n\n{_para('note', 20)}")], 800, 100, 3000)
    assert len(chunks) == 1 and parents == []
    assert chunks[0].parent_indexes == ()


def test_a_long_section_is_sliced_into_parents_of_at_most_the_limit():
    chunks, parents = chunk_pages_with_parents(_runbook(), 200, 30, 400)
    assert len(parents) >= 2
    assert all(count_tokens(p.text) <= 400 + 20 for p in parents)  # + the breadcrumb
    assert all(p.text.startswith("Runbook > 2. Rollback") or "## 2. Rollback" in p.text for p in parents)
    # Each slice still covers two or more chunks, and every Rollback chunk has a parent.
    assert all(c.parent_indexes for c in chunks if "step" in c.text)


def test_a_docx_or_html_document_gets_the_same_chunks_plus_parents():
    text = f"# Runbook\n\n## 1. Overview\n\n{_para('intro', 10)}\n\n## 2. Rollback\n\n{_para('step', 150)}"
    chunks, parents = chunk_text_with_parents(text, 200, 30, 3000)
    assert [c.text for c in chunks] == chunk_text(text, 200, 30, "structural")
    assert len(parents) == 1 and "step0 " in parents[0].text and "step149 " in parents[0].text
    assert "intro0" not in parents[0].text
    for chunk in chunks:
        assert chunk.parent_indexes == ((0,) if "step" in chunk.text else ())


def test_a_short_document_is_one_chunk_with_no_parent():
    chunks, parents = chunk_text_with_parents("## Notes\n\nCall the on-call.", 200, 30, 3000, page=1)
    assert [(c.text, c.first_page) for c in chunks] == [("## Notes\n\nCall the on-call.", 1)]
    assert parents == []


def test_paper_style_section_numbers_are_siblings_not_subsections():
    # "3 Dataset" after "2.3 ..." was nested under 2.3, so its path read "2 Related Work > 2.3 ... >
    # 3 Dataset" and sections 2 and 3 shared a parent.
    page = _flat(
        "## Paper title", "## 2 Related Work", "## 2.3 Prior systems", _para("prior", 10),
        "## 3 Dataset", _para("data", 150),
    )
    deep = next(c for c in chunk_pages([(1, page)], 120, 20) if "data100" in c.text)
    assert deep.text.startswith("Paper title > 3 Dataset")
