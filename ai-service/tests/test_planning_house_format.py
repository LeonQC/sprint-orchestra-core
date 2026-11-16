"""Tests for detecting this board's own description layout (app/planning/house_format.py).

The detector exists because measuring "does it write like this team" with a judge came back a coin
 while a much more concrete thing was sitting in plain sight:
every finished issue on the board is built from the same three headings, and the planner was writing
flat prose. Sections are worth detecting precisely because compliance is countable.

What has to hold for that to be trustworthy:
  * a heading is told apart from ordinary prose by occurring ONCE per issue, not by a word list;
  * the indexer's own synthetic prefix is not mistaken for the team's layout;
  * nothing is claimed from a sample too small to establish a convention;
  * the layout survives a "no relevant history" verdict, because it is a property of the board.
"""
import pytest

from app.planning.house_format import (
    HouseFormat,
    detect_house_format,
    section_compliance,
)

# Real AtlasCart shape, as retrieval returns it: HTML stripped by the chunker, so the <h3> headings
# arrive as bare words between sentences, and each body carries the indexer's provenance prefix.
def _board_issue(key, parent, title, context, ac, reasoning):
    return (f"{key} Subtask of {parent} ({title}). {title} "
            f"Context {context} Acceptance criteria AC1 — {ac} "
            f"Estimate reasoning 3 — {reasoning}")


_BOARD = [
    _board_issue("ATC-74", "ATC-51", "Show catalog import progress",
                 "Give sellers a readable progress page for queued and applied imports.",
                 "The page shows current phase and processed counts.",
                 "The API already exposes job status."),
    _board_issue("ATC-29", "ATC-3", "Connect the payment sandbox",
                 "Create a small payment adapter for the provider sandbox.",
                 "Accepted and rejected responses map to internal payment states.",
                 "One adapter plus storage, no UI work."),
    _board_issue("ATC-72", "ATC-51", "Index products for catalog search",
                 "Keep the storefront search index aligned with accepted catalog changes.",
                 "An accepted change appears in search results.",
                 "Indexing pipeline exists; this is wiring plus a refresh path."),
]


def test_it_finds_the_headings_the_board_actually_uses():
    house = detect_house_format(_BOARD)

    assert house.sections == ["Context", "Acceptance criteria", "Estimate reasoning"]
    assert house.coverage == {"Context": 1.0, "Acceptance criteria": 1.0, "Estimate reasoning": 1.0}
    assert house.detected is True


def test_sections_come_back_in_the_order_a_reader_of_the_board_would_see():
    """Order matters: it is handed to the planner as the shape to produce, and a description with the
    right headings in the wrong order still does not look like the board's cards."""
    house = detect_house_format(_BOARD)

    assert house.sections.index("Context") < house.sections.index("Acceptance criteria")
    assert house.sections.index("Acceptance criteria") < house.sections.index("Estimate reasoning")


def test_a_heading_is_told_from_prose_by_occurring_once_per_issue():
    """The discriminator that removes the need for a hand-maintained stopword list. "Provider" appears
    in every one of these and repeatedly inside each, so it is vocabulary, not a heading.

    Also pins the coverage tie-break: "Context Provider work" occurs once per issue too, but in only
    two of the three, and a phrase that extends a heading into the prose after it can only lose
    documents. So "Context" wins on coverage — resolving that on length instead reports the heading
    as "Context Provider work", which is what this test caught."""
    # The word after each heading varies, as it does on any real board. A fixture where every issue's
    # Context opens with the same word makes "Context <word>" a genuine once-per-issue phrase that no
    # counting rule can rule out — see the module docstring's note on that limit.
    bodies = [
        "KEY-1 Context Shipping work. Provider limits and provider retries matter. Notes Provider again.",
        "KEY-2 Context Refund handling. The provider retries, the provider fails, provider again. Notes More.",
        "KEY-3 Context Catalog indexing. Provider timeouts, provider errors, provider retries. Notes Done.",
    ]

    house = detect_house_format(bodies)

    assert "Context" in house.sections
    assert "Notes" in house.sections
    assert "Provider" not in house.sections


def test_the_indexers_own_prefix_is_not_mistaken_for_the_teams_layout():
    """"ATC-74 Subtask of ATC-51 (...)" is prepended to every body by vectorization-service's chunker.
    It is in 100% of descriptions and appears once in each, so it passes both detection tests
    perfectly — and it describes how the text was INDEXED, not how the team writes."""
    house = detect_house_format(_BOARD)

    assert not any("Subtask" in section for section in house.sections)
    assert not any("ATC" in section for section in house.sections)


def test_a_list_marker_does_not_swallow_the_heading_in_front_of_it():
    """"AC1" loses its digits to the tokeniser and becomes "AC", which would otherwise turn the real
    heading into "Acceptance criteria AC" and hide it behind a longer false label."""
    house = detect_house_format(_BOARD)

    assert "Acceptance criteria" in house.sections
    assert "Acceptance criteria AC" not in house.sections


def test_one_issue_cannot_establish_a_convention():
    """Every phrase in a single description has 100% coverage. Reporting a layout from that would be
    claiming a team convention from one person's card."""
    assert detect_house_format([_BOARD[0]]).sections == []
    assert detect_house_format([]).sections == []


def test_a_board_with_no_shared_layout_reports_none_rather_than_inventing_one():
    bodies = [
        "KEY-1 Just some prose about shipping quotes and nothing structural at all.",
        "KEY-2 Background We tried a thing. Outcome It worked out fine in the end.",
        "KEY-3 A single sentence issue with no headings whatsoever.",
    ]

    house = detect_house_format(bodies)

    assert house.sections == []
    assert house.detected is False


def test_a_convention_two_thirds_of_the_board_follows_still_counts():
    """A real board has older issues predating its current template. A layout followed by most of the
    work is still the layout, so the bar is coverage, not unanimity."""
    bodies = _BOARD + ["ATC-9 An old issue from before the template, written as one flat sentence."]

    house = detect_house_format(bodies)

    assert house.sections == ["Context", "Acceptance criteria", "Estimate reasoning"]
    assert all(0.6 <= value < 1.0 for value in house.coverage.values())


# --- compliance: the countable half -------------------------------------------------------------


def test_compliance_scores_a_real_board_issue_at_one_and_flat_prose_at_zero():
    house = detect_house_format(_BOARD)

    assert section_compliance(_BOARD[0], house) == 1.0
    assert section_compliance(
        "Connect the provider's tokenized capability and persist only safe references.", house,
    ) == 0.0


def test_compliance_is_partial_when_a_section_is_missing():
    house = detect_house_format(_BOARD)

    # Rounded to 3dp by `section_compliance` so a response payload carries a readable number.
    assert section_compliance("<h3>Context</h3><p>Do the thing.</p>", house) == 0.333


def test_compliance_is_unscorable_rather_than_zero_when_there_is_no_layout_to_follow():
    """Zero would read as "it ignored the house format"; there wasn't one."""
    score = section_compliance("anything at all", HouseFormat())

    assert score != score  # NaN


def test_compliance_ignores_heading_case_so_html_and_flattened_text_both_score():
    house = detect_house_format(_BOARD)

    assert section_compliance(
        "<h3>context</h3><p>x</p><h3>ACCEPTANCE CRITERIA</h3><p>y</p><h3>Estimate reasoning</h3><p>z</p>",
        house,
    ) == 1.0


# --- the list-item convention, which the heading rules deliberately cannot see ------------------


def test_it_learns_how_the_board_labels_list_items():
    """`AC1` is not decoration on this board — other records cite it ("the screenshot supplied with
    ATC-74 AC1"), so criteria numbered "1., 2., 3." are criteria nothing else can refer to. It needs
    its own detection because the heading rules reject trailing all-caps tokens, which is exactly what
    a bare "AC" looks like once the tokeniser drops the digit."""
    house = detect_house_format(_BOARD)

    assert house.item_marker == "AC"


def test_the_marker_is_counted_once_per_issue_not_once_per_item():
    """A description with four AC items is one issue that follows the convention, not four."""
    bodies = [
        "K-1 Context Something. Acceptance criteria AC1 — a AC2 — b AC3 — c AC4 — d Estimate reasoning 3 — x",
        "K-2 Context Other. Acceptance criteria AC1 — a Estimate reasoning 5 — y",
        "K-3 Context Third. Acceptance criteria Nothing labelled here at all. Estimate reasoning 2 — z",
    ]

    house = detect_house_format(bodies)

    # 2 of 3 issues use it, which clears the coverage bar; counting items would have made it look
    # unanimous.
    assert house.item_marker == "AC"


def test_a_board_that_does_not_label_its_items_reports_no_marker():
    bodies = [
        "K-1 Context A thing. Acceptance criteria The page loads. Estimate reasoning 3 — small.",
        "K-2 Context Another. Acceptance criteria The form submits. Estimate reasoning 5 — medium.",
        "K-3 Context Third. Acceptance criteria The job runs. Estimate reasoning 2 — tiny.",
    ]

    house = detect_house_format(bodies)

    assert house.sections == ["Context", "Acceptance criteria", "Estimate reasoning"]
    assert house.item_marker is None


def test_a_year_after_a_capitalised_word_is_not_a_list_marker():
    """The prefix cap is what keeps "Reviewed 2026 — ..." from posing as a labelling convention."""
    bodies = [
        "K-1 Context Renewed in BERLIN2026 — nope. Notes x",
        "K-2 Context Renewed in BERLIN2026 — nope. Notes y",
    ]

    house = detect_house_format(bodies)

    assert house.item_marker is None


# --- how many items, not just which sections ----------------------------------------------------


def test_it_learns_how_many_items_a_description_usually_has():
    """Sections alone were not enough. The planner produced the right headings with the right
    numbering and then wrote three to four acceptance criteria per issue, on a board where 49% have
    one, 37% have two, 14% have three and none has four. Right structure, wrong volume, still visibly
    not one of theirs."""
    bodies = [
        "K-1 Context a. Acceptance criteria AC1 — one. Estimate reasoning 3 — x",
        "K-2 Context b. Acceptance criteria AC1 — one. Estimate reasoning 3 — x",
        "K-3 Context c. Acceptance criteria AC1 — one. AC2 — two. Estimate reasoning 3 — x",
        "K-4 Context d. Acceptance criteria AC1 — one. AC2 — two. Estimate reasoning 3 — x",
        "K-5 Context e. Acceptance criteria AC1 — one. AC2 — two. AC3 — three. Estimate reasoning 3 — x",
    ]

    house = detect_house_format(bodies)

    assert (house.item_count_low, house.item_count_high, house.item_count_max) == (1, 2, 3)


def test_the_typical_count_is_the_mode_not_the_median():
    """With 49/37/14 the median lands on 2, because 49% is just under half — and "usually 2" would
    push the planner to always write two where one is the single most common answer. Mode-to-80th
    spans the 86% that have one or two."""
    bodies = (
        ["K Context a. Acceptance criteria AC1 — one. Estimate reasoning 1 — x"] * 49
        + ["K Context b. Acceptance criteria AC1 — one. AC2 — two. Estimate reasoning 1 — x"] * 37
        + ["K Context c. Acceptance criteria AC1 — one. AC2 — two. AC3 — three. Estimate reasoning 1 — x"] * 14
    )

    house = detect_house_format(bodies)

    assert house.item_count_low == 1
    assert house.item_count_high == 2
    assert house.item_count_max == 3


def test_issues_with_no_items_do_not_drag_the_typical_count_down():
    """An issue with no criteria at all is a different kind of card. Counting it as zero would tell
    the planner to write fewer criteria than any real issue on the board has."""
    bodies = [
        "K-1 Context a. Acceptance criteria AC1 — one. AC2 — two. Estimate reasoning 3 — x",
        "K-2 Context b. Acceptance criteria AC1 — one. AC2 — two. Estimate reasoning 3 — x",
        "K-3 Context c. Acceptance criteria Nothing labelled at all here. Estimate reasoning 3 — x",
    ]

    house = detect_house_format(bodies)

    assert house.item_count_low == 2
    assert house.item_count_max == 2


def test_no_marker_means_no_count_to_report():
    bodies = [
        "K-1 Context a. Acceptance criteria The page loads. Estimate reasoning 3 — x",
        "K-2 Context b. Acceptance criteria The form submits. Estimate reasoning 3 — x",
    ]

    house = detect_house_format(bodies)

    assert house.item_count_low is None
    assert house.item_count_max is None


# --- an epic gets a shorter list than an issue ---------------------------------------------------


def test_an_epic_keeps_its_criteria_but_loses_the_estimate_section():
    """Only the vacuous section goes. An epic has no story points here — `EpicDraft` has no field for
    one — so that heading would introduce itself only to report there is no number. Acceptance
    criteria stay: the board's epics genuinely carry them, and an epic-level criterion is about the
    whole outcome rather than a restatement of any one issue's.

    The distinction is the point, and it holds beyond this board: a section is droppable when the data
    model makes it vacuous, not merely when its content also appears somewhere below."""
    house = detect_house_format(_BOARD)

    assert house.sections == ["Context", "Acceptance criteria", "Estimate reasoning"]
    assert house.epic_sections == ["Context", "Acceptance criteria"]


def test_it_matches_the_word_not_one_board_s_exact_label():
    """A board calling it "Sizing notes" or "Story point rationale" means the same thing."""
    assert HouseFormat(sections=["Background", "Sizing notes"]).epic_sections == ["Background"]
    assert HouseFormat(sections=["Context", "Story points rationale"]).epic_sections == ["Context"]
    assert HouseFormat(sections=["Context", "Acceptance tests"]).epic_sections == [
        "Context", "Acceptance tests",
    ]


def test_a_board_with_no_estimate_section_loses_nothing():
    assert HouseFormat(sections=["Context", "Notes"]).epic_sections == ["Context", "Notes"]
    assert HouseFormat().epic_sections == []


def test_a_section_that_merely_mentions_a_similar_word_is_kept():
    """Matched on whole words, so "Estimated impact" goes and "Points of contact" goes, but a label
    that only contains the letters — "Endpoints" — stays."""
    assert HouseFormat(sections=["Endpoints touched"]).epic_sections == ["Endpoints touched"]
