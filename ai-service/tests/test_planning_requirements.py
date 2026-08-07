"""Tests for per-requirement retrieval (app/planning/grounding.py::gather_requirement_exemplars).

The defect being fixed is measured, not supposed: with one search for the whole epic, 40% of estimated
issues cited nothing (anchor citation rate 0.60). An
epic proposal describes several pieces of work at once, so embedding all of it yields a vector for the
average of those pieces — which resembles none of them — and every issue then has to find its
comparable in cards retrieved for that average.

Properties worth defending:
  * a requirement with no precedent on the board is reported as having none, rather than being handed
    something that merely sounded close;
  * the whole-proposal pass still runs, because a requirement-shaped query says nothing about how the
    board writes or sizes work in general — and that pass is what the calibration result measured;
  * stating no requirements reproduces the old behaviour exactly;
  * a requirement number the model invents is cleared rather than pointing at the wrong requirement.
"""
from app.planning.grounding import GroundingContext, gather_requirement_exemplars
from app.planning.schemas import IssueDraft
from app.planning.service import drop_invalid_requirement_indexes
from tests.test_planning_grounding import (
    FakeRetrieval,
    _chunk,
    _detail,
    _row,
)


class _PerQueryRetrieval(FakeRetrieval):
    """Returns different hits depending on what was searched for, which is the whole point: one
    vector for "upload a spreadsheet and notify people by SMS" finds neither."""

    def __init__(self, hits_by_query, rows, details, recent=None):
        super().__init__(hits=[], rows=rows, details=details, recent=recent)
        self._hits_by_query = hits_by_query

    async def search_with_meta(self, query, space_ids, limit, mode="hybrid"):
        # No `self.queries.append` here — the base class records it. Appending in both counted every
        # search twice, which made "one search when no requirements are stated" look like two.
        for needle, hits in self._hits_by_query.items():
            if needle.lower() in query.lower():
                self._hits = hits
                break
        else:
            self._hits = []
        return await super().search_with_meta(query, space_ids, limit, mode)


def _retrieval():
    return _PerQueryRetrieval(
        hits_by_query={
            "spreadsheet": [_chunk("ATC-73", 0.95, 1)],
            "audit": [_chunk("ATC-75", 0.9, 2)],
            "onboarding": [_chunk("ATC-73", 0.7, 1)],   # the whole-proposal pass
        },
        rows=[_row("ATC-73", points=8), _row("ATC-75", points=5)],
        details=[_detail("ATC-73"), _detail("ATC-75")],
    )


async def test_each_requirement_is_searched_for_on_its_own():
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["Upload a spreadsheet", "Record an audit trail"],
        limit=8, per_requirement=3,
    )

    searched = [query for query, _, _ in retrieval.queries]
    assert "Employee onboarding" in searched        # the whole-proposal pass still runs
    assert "Upload a spreadsheet" in searched
    assert "Record an audit trail" in searched
    assert [r.index for r in context.requirements] == [1, 2]
    assert [e.issue_key for e in context.requirements[0].exemplars] == ["ATC-73"]
    assert [e.issue_key for e in context.requirements[1].exemplars] == ["ATC-75"]


async def test_a_requirement_with_no_precedent_is_reported_empty_rather_than_filled():
    """The honest outcome, and the one the old design could not express: before, an issue with no
    comparable was indistinguishable from a planner that had not bothered."""
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding",
        ["Upload a spreadsheet", "Send each new hire an SMS"],
        limit=8, per_requirement=3,
    )

    assert context.requirements[1].text == "Send each new hire an SMS"
    assert context.requirements[1].exemplars == []
    assert context.requirements[1].anchors == []


async def test_the_flat_exemplar_list_is_the_union_so_everything_downstream_keeps_working():
    """Citation verification, the estimate baseline and the layout detection all read the flat list
    and know nothing about requirements. Per-requirement hits have to reach them."""
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["Upload a spreadsheet", "Record an audit trail"],
        limit=8, per_requirement=3,
    )

    assert {e.issue_key for e in context.exemplars} >= {"ATC-73", "ATC-75"}
    # and no duplicates, even though ATC-73 was found by two different searches
    assert len(context.exemplars) == len({e.issue_key for e in context.exemplars})


async def test_stating_no_requirements_reproduces_the_previous_behaviour_exactly():
    """The whole-proposal path is what the calibration eval measured. It has to stay reachable
    unchanged, or that result stops describing anything that ships."""
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", [], limit=8, per_requirement=3,
    )

    assert context.requirements == []
    assert len(retrieval.queries) == 1


async def test_blank_requirement_lines_are_ignored_rather_than_searched():
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["Upload a spreadsheet", "   ", ""],
        limit=8, per_requirement=3,
    )

    assert [r.text for r in context.requirements] == ["Upload a spreadsheet"]


async def test_a_dead_index_does_not_turn_into_a_pile_of_failed_requirement_searches():
    retrieval = FakeRetrieval(hits=[], fail=True)

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["a", "b", "c"], limit=8, per_requirement=3,
    )

    assert context.status == "retrieval_failed"
    assert [item.text for item in context.requirements] == ["a", "b", "c"]
    assert all(not item.exemplars for item in context.requirements)
    assert len(retrieval.queries) == 1  # bailed after the first failure, not once per requirement


# --- the index the model cites back ---------------------------------------------------------------


def _issue(temp_id="1", index=None):
    return IssueDraft(
        temp_id=temp_id, title="t", description="d", issue_type="task",
        requirement_index=index, depends_on=[],
    )


def test_a_requirement_number_that_names_nothing_is_cleared():
    """A wrong number is worse than none: it tells a reviewer this card delivers requirement 3 when it
    delivers requirement 2. Cleared rather than remapped — there is nothing to remap it to."""
    cleaned, dropped = drop_invalid_requirement_indexes(
        [_issue("1", 1), _issue("2", 4)], requirement_count=2,
    )

    assert [i.requirement_index for i in cleaned] == [1, None]
    assert dropped == 1


def test_the_schema_already_refuses_a_non_positive_index():
    """The lower bound needs no runtime check: `ge=1` on the field means a 0 never reaches the code,
    and `instructor` re-prompts on it. Only the upper bound depends on how many were stated, which is
    why that one is checked in Python."""
    import pytest

    with pytest.raises(Exception):
        _issue("1", 0)


def test_with_no_requirements_stated_every_index_is_cleared():
    cleaned, dropped = drop_invalid_requirement_indexes([_issue("1", 1)], requirement_count=0)

    assert cleaned[0].requirement_index is None
    assert dropped == 1


def test_an_issue_that_serves_no_requirement_directly_keeps_its_null():
    cleaned, dropped = drop_invalid_requirement_indexes([_issue("1", None)], requirement_count=3)

    assert cleaned[0].requirement_index is None
    assert dropped == 0


async def test_a_requirement_gets_its_own_matches_not_the_boards_latest_work():
    """The bug this catches was found by running it, not by reading it. With the recency backfill left
    on, every requirement came back with the SAME three cards — the board's latest finished work —
    because those are all estimate anchors and ranking promotes anchors above everything else, so they
    displaced each requirement's narrower matches before the list was cut to three. The planner then
    correctly cited none of them and the citation rate went to zero, i.e. the change made the thing it
    was meant to fix worse."""
    retrieval = _PerQueryRetrieval(
        hits_by_query={
            "spreadsheet": [_chunk("ATC-73", 0.95, 1)],
            "onboarding": [_chunk("ATC-73", 0.7, 1)],
        },
        rows=[_row("ATC-73", points=8), _row("ATC-90", points=1), _row("ATC-91", points=1)],
        details=[_detail("ATC-73"), _detail("ATC-90"), _detail("ATC-91")],
        # Recent finished work, unrelated to any requirement — exactly what used to win.
        recent=[_row("ATC-90", points=1), _row("ATC-91", points=1)],
    )

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["Upload a spreadsheet", "Send an SMS"],
        limit=8, per_requirement=3,
    )

    assert [e.issue_key for e in context.requirements[0].exemplars] == ["ATC-73"]
    assert context.requirements[1].exemplars == []   # and stays honestly empty
    # The whole-proposal pass still gets the backfill, which is where it earns its place.
    assert {e.issue_key for e in context.exemplars} >= {"ATC-73", "ATC-90"}


async def test_progress_is_reported_per_requirement():
    """The slow, silent stretch of a Plan Epic call: one index round trip per requirement before the
    planning model is even reached. Without this the UI shows one unchanging label for all of it."""
    retrieval = _retrieval()
    seen: list[str] = []

    await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding",
        ["Upload a spreadsheet", "Record an audit trail", "Send an SMS"],
        limit=8, per_requirement=3, on_progress=seen.append,
    )

    assert seen == [
        "looking for past work like requirement 1 of 3",
        "looking for past work like requirement 2 of 3",
        "looking for past work like requirement 3 of 3",
    ]


async def test_no_progress_callback_is_fine():
    retrieval = _retrieval()

    context = await gather_requirement_exemplars(
        retrieval, [5], "Employee onboarding", ["Upload a spreadsheet"], limit=8, per_requirement=3,
    )

    assert len(context.requirements) == 1


# --- a refine round must not quietly strip what the first round earned -------------------------


def _full(temp_id, points=5, refs=("ATC-1",), origin="historical_pattern", patterns=("bulk import",),
          title=None, complexity="similar"):
    return IssueDraft(
        temp_id=temp_id, title=title or f"Issue {temp_id}", description="d", issue_type="task",
        estimate_story_points=points, estimate_rationale="because",
        estimate_reference=list(refs), complexity_vs_reference=complexity,
        origin=origin, pattern_basis=list(patterns), depends_on=[],
    )


def test_an_untouched_issue_keeps_the_citation_the_model_forgot_to_repeat():
    """Measured, not hypothetical: a real "add a QA task" round returned all six pre-existing issues
    with the same titles, points and descriptions — and every `estimate_reference` gone. The model had
    been shown "Estimate sized against: ATC-70 (similar)" for each and did not repeat it. A plea that
    fails needs a guarantee, not a firmer plea."""
    from app.planning.service import restore_untouched_provenance

    original = [_full("1")]
    refined = [_full("1", refs=(), complexity=None, origin="epic_requirement", patterns=())]

    (result,) = restore_untouched_provenance(refined, original)

    assert result.estimate_reference == ["ATC-1"]
    assert result.complexity_vs_reference == "similar"
    assert result.origin == "historical_pattern"
    assert result.pattern_basis == ["bulk import"]


def test_a_re_estimated_issue_does_not_get_its_old_comparison_back():
    """The citation justifies a SIZE. Once the size changes it is justifying someone else's number —
    the same rule the rollout applies when a human edits an estimate."""
    from app.planning.service import restore_untouched_provenance

    original = [_full("1", points=5)]
    refined = [_full("1", points=8, refs=(), complexity=None)]

    (result,) = restore_untouched_provenance(refined, original)

    assert result.estimate_story_points == 8
    assert result.estimate_reference == []


def test_a_rewritten_issue_does_not_get_its_old_origin_back():
    """`origin` says why the issue exists. A retitled issue may well be different work."""
    from app.planning.service import restore_untouched_provenance

    original = [_full("1", title="Validate rows")]
    refined = [_full("1", title="Validate and normalise rows", origin="epic_requirement", patterns=())]

    (result,) = restore_untouched_provenance(refined, original)

    assert result.origin == "epic_requirement"
    assert result.pattern_basis == []
    # the estimate is untouched, so its comparison still stands
    assert result.estimate_reference == ["ATC-1"]


def test_nothing_is_invented_for_an_issue_the_round_added():
    from app.planning.service import restore_untouched_provenance

    refined = [_full("99", refs=(), complexity=None, origin="epic_requirement", patterns=())]

    (result,) = restore_untouched_provenance(refined, [])

    assert result.estimate_reference == []
    assert result.pattern_basis == []


def test_a_value_the_model_deliberately_changed_is_left_alone():
    from app.planning.service import restore_untouched_provenance

    original = [_full("1", refs=("ATC-1",))]
    refined = [_full("1", refs=("ATC-2",))]

    (result,) = restore_untouched_provenance(refined, original)

    assert result.estimate_reference == ["ATC-2"]


def test_the_refine_request_model_and_the_plan_schema_stay_field_compatible():
    """A structural guard, added after the failure it would have caught.

    The refine route rebuilds `IssueDraft` from `IssueDraftIn`. It used to do that with a hand-written
    field list, which silently dropped every provenance field added since the list was written — so a
    plan's citations were destroyed on the way IN to a refine, before `_render_current_plan` or
    `restore_untouched_provenance` could protect them. Both exist for exactly that, and neither can
    help with data the route already discarded. `model_dump()` fixes it; this keeps the two shapes
    from drifting apart again.
    """
    from app.api.routes import IssueDraftIn
    from app.planning.schemas import IssueDraft

    assert set(IssueDraftIn.model_fields) == set(IssueDraft.model_fields)


def test_provenance_survives_the_request_to_schema_conversion():
    from app.api.routes import IssueDraftIn
    from app.planning.schemas import IssueDraft

    incoming = IssueDraftIn(
        tempId="1", title="t", description="d", estimateReference=["ATC-1"],
        complexityVsReference="simpler", origin="historical_pattern", patternBasis=["bulk import"],
        requirementIndex=2, estimateStoryPoints=3,
    )

    converted = IssueDraft(**incoming.model_dump())

    assert converted.estimate_reference == ["ATC-1"]
    assert converted.complexity_vs_reference == "simpler"
    assert converted.origin == "historical_pattern"
    assert converted.pattern_basis == ["bulk import"]
    assert converted.requirement_index == 2
