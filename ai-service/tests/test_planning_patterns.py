"""Tests for evidence evaluation + historical pattern extraction (app/planning/patterns.py).

The stage's whole reason to exist is that "retrieved" and "relevant" are different things, and the
old pipeline treated them as one. So the properties worth defending are the ones that keep that
distinction real:

  * a "none" verdict actually WITHHOLDS the history from the planner — otherwise the assessment is
    decoration and the planner pattern-matches against irrelevant issues anyway;
  * a pattern must recur across issues, or it is one issue wearing the word "pattern";
  * a pattern citing an issue nobody was shown is dropped, and an over-confident `strength` label is
    downgraded to match whatever survived that;
  * the estimate NUMBER comes from what the cited work really cost, not from the model;
  * every one of those degrades to the previous behaviour rather than failing the plan.
"""
from types import SimpleNamespace

import pytest

from app.planning.grounding import Exemplar, GroundingContext
from app.planning.pattern_schemas import (
    MIN_PATTERN_EVIDENCE,
    EvidenceAssessment,
    HistoricalPattern,
)
from app.planning.patterns import (
    assess_history,
    derive_estimate,
    estimate_confidence,
    historical_baseline,
    render_history_for_extraction,
    verify_assessment,
)
from app.planning.prompts import render_epic_plan_prompt
from app.planning.schemas import EpicDraft, EpicPlanDraft, IssueDraft
from app.planning.service import apply_historical_estimates, plan_epic
from tests.test_planning import FakeInstructorClient


def _exemplar(key, points=5, status="done", comments=(), title="t", body="d"):
    return Exemplar(key, title, "story", status, points, body, 0.9, "similar", list(comments))


def _context(*exemplars):
    return GroundingContext(status="grounded", exemplars=list(exemplars))


def _pattern(name="bulk import", keys=("A-1", "A-2"), steps=("validate first",), pitfalls=()):
    return HistoricalPattern(
        name=name, steps=list(steps), evidence_issue_keys=list(keys),
        why_relevant="This epic also ingests a file.", pitfalls=list(pitfalls),
    )


def _assessment(strength="strong", patterns=None, relevant=("A-1", "A-2")):
    return EvidenceAssessment(
        relevant_issue_keys=list(relevant),
        patterns=list(patterns) if patterns is not None else [_pattern()],
        strength=strength, reasoning="because",
    )


def _issue(temp_id="1", points=5, references=(), complexity=None):
    return IssueDraft(
        temp_id=temp_id, title=f"Issue {temp_id}", description="d", issue_type="task", labels=[],
        estimate_story_points=points, estimate_rationale="because" if points else None,
        estimate_reference=list(references), complexity_vs_reference=complexity, depends_on=[],
    )


# --- verification: what the model claims vs what it can support -------------------------------


def test_a_pattern_citing_an_issue_nobody_was_shown_loses_that_citation():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    assessment = _assessment(patterns=[_pattern(keys=["A-1", "A-2", "GHOST-9"])])

    cleaned, dropped, _ = verify_assessment(assessment, context)

    assert cleaned.patterns[0].evidence_issue_keys == ["A-1", "A-2"]
    assert dropped == 0


def test_a_shape_seen_in_only_one_issue_is_not_a_pattern():
    """"Pattern" is a claim that something RECURS. One issue cannot support it — and this is also
    what stops a single truncated comment thread from becoming a lesson."""
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    assessment = _assessment(patterns=[_pattern(keys=["A-1"])])

    cleaned, dropped, _ = verify_assessment(assessment, context)

    assert cleaned.patterns == []
    assert dropped == 1
    assert MIN_PATTERN_EVIDENCE == 2


def test_a_pattern_surviving_on_invented_keys_alone_is_dropped():
    context = _context(_exemplar("A-1"))
    assessment = _assessment(patterns=[_pattern(keys=["GHOST-1", "GHOST-2"])], relevant=["A-1"])

    cleaned, dropped, _ = verify_assessment(assessment, context)

    assert cleaned.patterns == []
    assert dropped == 1


def test_strength_is_downgraded_to_match_what_survived_verification():
    """The model reports `strength` because the judgement is semantic, but it does not get to keep a
    label the surviving evidence no longer supports."""
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    assessment = _assessment(strength="strong", patterns=[_pattern(keys=["GHOST-1", "GHOST-2"])])

    cleaned, _, downgraded = verify_assessment(assessment, context)

    assert cleaned.strength == "partial"  # relevant issues remain, but no reusable shape does
    assert downgraded is True


def test_losing_everything_downgrades_all_the_way_to_none():
    context = _context(_exemplar("A-1"))
    assessment = _assessment(strength="strong", patterns=[_pattern(keys=["GHOST-1", "GHOST-2"])],
                             relevant=["GHOST-1"])

    cleaned, _, downgraded = verify_assessment(assessment, context)

    assert cleaned.strength == "none"
    assert cleaned.relevant_issue_keys == []
    assert downgraded is True


def test_a_clean_assessment_passes_through_untouched():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))

    cleaned, dropped, downgraded = verify_assessment(_assessment(), context)

    assert cleaned.strength == "strong"
    assert [p.name for p in cleaned.patterns] == ["bulk import"]
    assert (dropped, downgraded) == (0, False)


# --- the verdict has to actually change what the planner sees ---------------------------------


def test_a_none_verdict_withholds_the_history_from_the_planner_entirely():
    """The point of assessing before planning. If irrelevant history still reached the prompt, the
    assessment would be decoration and the planner would pattern-match against it anyway."""
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    verdict = EvidenceAssessment(
        relevant_issue_keys=[], patterns=[], strength="none",
        reasoning="Nothing retrieved touches this epic's kind of work.",
    )

    prompt = render_epic_plan_prompt([], context, verdict)

    assert "PAST WORK" not in prompt
    assert "--- A-1 ---" not in prompt
    assert "ESTIMATE BY COMPARISON" not in prompt


def test_history_that_was_never_assessed_still_reaches_the_planner():
    """"Nobody looked" is not "somebody looked and found nothing" — only the second withholds. This
    keeps the measured behaviour intact when the stage is switched off."""
    context = _context(_exemplar("A-1"), _exemplar("A-2"))

    prompt = render_epic_plan_prompt([], context, None)

    assert "PAST WORK" in prompt
    assert "ESTIMATE BY COMPARISON" in prompt


def test_a_partial_verdict_tells_the_planner_not_to_pretend_an_equivalent_existed():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))

    prompt = render_epic_plan_prompt([], context, _assessment(strength="partial"))

    assert "evidence here is PARTIAL" in prompt
    assert "--- pattern: bulk import ---" in prompt


def test_patterns_reach_the_planner_with_their_evidence_and_their_pitfalls():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    assessment = _assessment(patterns=[_pattern(
        steps=["validate the file first", "report per-row errors"],
        pitfalls=["retries were added late"],
    )])

    prompt = render_epic_plan_prompt([], context, assessment)

    assert "seen in: A-1, A-2" in prompt
    assert "step: validate the file first" in prompt
    assert "trouble this team actually hit: retries were added late" in prompt


# --- comments feed pattern extraction, and only pattern extraction ------------------------------


def test_comments_are_labelled_as_what_happened_not_what_was_planned():
    """A description says what was intended; a comment says what actually went wrong. The framing is
    the reason comments are worth reading here at all."""
    rendered = render_history_for_extraction([
        _exemplar("A-1", comments=["we had to add a retry, the provider timed out"]),
    ])

    assert "comment (said while doing the work): we had to add a retry" in rendered


def test_the_planner_never_sees_comments():
    context = _context(_exemplar("A-1", comments=["a stray internal remark"]),
                       _exemplar("A-2", comments=["another"]))

    prompt = render_epic_plan_prompt([], context, _assessment())

    assert "stray internal remark" not in prompt


# --- estimation: code owns the number, the model owns the comparison ---------------------------


def test_the_number_comes_from_what_the_cited_work_actually_cost():
    context = _context(_exemplar("A-1", points=8))
    # The model guessed 3; the work it says this resembles really cost 8.
    issues = [_issue(points=3, references=["A-1"], complexity="similar")]

    adjusted, derived = apply_historical_estimates(issues, context)

    assert adjusted[0].estimate_story_points == 8
    assert derived == 1


def test_a_comparison_moves_exactly_one_rung_along_the_ladder():
    assert derive_estimate(5, "similar") == 5
    assert derive_estimate(5, "simpler") == 3
    assert derive_estimate(5, "more_complex") == 8
    assert derive_estimate(1, "simpler") == 1     # clamped at the bottom
    assert derive_estimate(13, "more_complex") == 13  # and at the top


def test_the_baseline_is_the_median_so_two_citations_take_the_smaller():
    """Story points are ladder positions, not a linear quantity: the mean of 3 and 8 is 5.5, which is
    not a size anyone estimates in. The median of two takes the lower, which is the conservative
    direction for a planner measured to over-estimate."""
    context = _context(_exemplar("A-1", points=3), _exemplar("A-2", points=8))

    assert historical_baseline(["A-1", "A-2"], context) == 3
    assert historical_baseline(["A-1", "A-2", "A-1"], context) == 3


def test_an_unfinished_issue_contributes_no_size_however_it_was_cited():
    context = _context(_exemplar("A-1", points=13, status="in_progress"))

    assert historical_baseline(["A-1"], context) is None


def test_with_nothing_comparable_the_model_keeps_its_own_estimate():
    """An estimate with no finished work behind it is a judgement call, and this codebase shows one
    honestly rather than dressing it up."""
    context = _context(_exemplar("A-1"))
    issues = [_issue(points=5, references=[], complexity=None)]

    adjusted, derived = apply_historical_estimates(issues, context)

    assert adjusted[0].estimate_story_points == 5
    assert derived == 0


def test_confidence_is_computed_from_the_evidence_not_claimed_by_the_model():
    assert estimate_confidence(["A-1", "A-2"], 5, "strong") == "high"
    assert estimate_confidence(["A-1"], 5, "strong") == "medium"
    assert estimate_confidence(["A-1"], 5, "partial") == "medium"
    assert estimate_confidence([], None, "strong") == "low"
    assert estimate_confidence(["A-1"], None, "strong") == "low"


# --- degradation --------------------------------------------------------------------------------


async def test_an_empty_space_costs_no_llm_call_to_conclude_nothing_about():
    client = FakeInstructorClient(RuntimeError("must not be called"))

    result = await assess_history(client, "fake-model", "a proposal", GroundingContext())

    assert result.assessment.strength == "none"
    assert client.completions.calls == []


async def test_a_failed_extraction_degrades_to_planning_without_patterns():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    client = FakeInstructorClient(RuntimeError("model exploded"))

    result = await assess_history(client, "fake-model", "a proposal", context)

    assert result.degraded is True
    assert result.assessment.strength == "none"
    assert result.assessment.patterns == []
    assert "model exploded" in result.error


async def test_a_successful_extraction_is_verified_before_it_is_returned():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    client = FakeInstructorClient(_assessment(patterns=[_pattern(keys=["A-1", "GHOST-2"])]))

    result = await assess_history(client, "fake-model", "a proposal", context)

    assert result.assessment.patterns == []   # one verifiable key is not a pattern
    assert result.dropped_patterns == 1
    assert result.assessment.strength == "partial"
    assert result.downgraded is True


async def test_the_extraction_call_is_shown_the_proposal_and_every_exemplar():
    context = _context(_exemplar("A-1", title="Bulk customer import"),
                       _exemplar("A-2", title="CSV product upload"))
    client = FakeInstructorClient(_assessment())

    await assess_history(client, "fake-model", "Support bulk employee onboarding", context)

    sent = client.completions.calls[0]["messages"][1]["content"]
    assert "Support bulk employee onboarding" in sent
    assert "Bulk customer import" in sent and "CSV product upload" in sent


async def test_plan_epic_carries_the_patterns_into_the_planning_prompt():
    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]),
        issues=[_issue("1", points=3, references=["A-1"], complexity="more_complex")],
    ))

    result = await plan_epic(client, "fake-model", "a proposal", [], context, _assessment())

    assert "--- pattern: bulk import ---" in client.completions.calls[0]["messages"][0]["content"]
    # A-1 really cost 5 and the model called this larger, so the derived number is one rung up.
    assert result.plan.issues[0].estimate_story_points == 8
    assert result.historically_derived_estimates == 1


async def test_the_durable_rollout_records_the_verdict_its_plan_was_written_from():
    """A rollout can sit at pending_approval overnight. The reviewer who opens it must see the
    reading the plan was written from, not a fresh one that would come out differently."""
    from app.planning.rollout_graph import plan_node
    from app.planning.rollout_schemas import initial_rollout_state
    from app.planning.patterns import AssessmentResult

    context = _context(_exemplar("A-1"), _exemplar("A-2"))
    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]),
        issues=[_issue("1", points=5, references=["A-1"], complexity="similar")],
    ))
    seen = []

    async def _gatherer(_space_id, _proposal, _requirements=None, _on_progress=None):
        return context

    async def _assessor(proposal, ctx):
        seen.append(proposal)
        return AssessmentResult(assessment=_assessment(), input_tokens=11, estimated_cost_usd=0.5)

    state = initial_rollout_state("bulk onboarding", [], None, None, 5, "u1", "alice")
    update = await plan_node(
        state, client, "fake-model", lambda: None, _gatherer, _assessor,
    )

    assert seen == ["bulk onboarding"]
    assert update["evidence"]["strength"] == "strong"
    assert update["evidence"]["patterns"][0]["name"] == "bulk import"
    # The extra reading call is billed with the plan, not lost. ("fake-model" has no rate card, so
    # the planning call itself prices at 0 and the assessment's 0.5 is the whole total — which is
    # exactly what makes this assert that the reading call was added rather than dropped.)
    assert update["input_tokens"] == 11 + 1234
    assert update["estimated_cost_usd"] == pytest.approx(0.5)


async def test_a_rollout_with_the_stage_switched_off_still_plans():
    from app.planning.rollout_graph import plan_node
    from app.planning.rollout_schemas import initial_rollout_state

    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]), issues=[_issue("1")],
    ))
    state = initial_rollout_state("anything", [], None, None, 5, "u1", "alice")

    update = await plan_node(state, client, "fake-model", lambda: None, None, None)

    assert update["evidence"]["strength"] == "unassessed"
    assert update["status"] == "pending_approval"


def test_the_durable_path_derives_the_same_estimate_facts_from_its_checkpoint():
    """The UI reads the rollout response, so the confidence badge and the baseline have to survive
    the checkpoint — computed from the exemplars the plan was WRITTEN against, not from a fresh
    retrieval that would come out differently by the time an overnight approval is opened."""
    from app.api.routes import _rollout_issue_out

    grounding = {"exemplars": [
        {"issue_key": "ATC-29", "story_points": 5, "estimate_anchor": True},
        {"issue_key": "ATC-36", "story_points": 3, "estimate_anchor": True},
        {"issue_key": "ATC-99", "story_points": 13, "estimate_anchor": False},
    ]}

    out = _rollout_issue_out(_issue("1", points=5, references=["ATC-29", "ATC-36"]), grounding, "strong")
    assert out.historical_baseline_points == 3   # median of 3 and 5 takes the lower
    assert out.estimate_confidence == "high"

    unanchored = _rollout_issue_out(_issue("2", points=8, references=["ATC-99"]), grounding, "strong")
    assert unanchored.historical_baseline_points is None  # never finished at a recorded size
    assert unanchored.estimate_confidence == "low"

    uncited = _rollout_issue_out(_issue("3", points=5), grounding, "strong")
    assert uncited.estimate_confidence == "low"


def test_a_checkpoint_from_before_this_feature_still_renders():
    from app.api.routes import _rollout_issue_out

    out = _rollout_issue_out(_issue("1", points=5, references=["ATC-29"]), {}, "unassessed")

    assert out.historical_baseline_points is None
    assert out.estimate_confidence == "low"


# --- the rollout status response, built from a real checkpoint --------------------------------


def _checkpoint_values(grounding=None):
    """A paused rollout as the checkpointer actually stores it."""
    from app.planning.rollout_schemas import initial_rollout_state
    from app.planning.rollout_graph import _grounding_to_state, _evidence_to_state

    values = dict(initial_rollout_state("p", [], None, None, 5, "u1", "alice", ["r1"]))
    values["epic"] = EpicDraft(title="T", description="D", goals=[])
    values["issues"] = [_issue("1", points=5, references=["ATC-1"])]
    values["grounding"] = (
        grounding if grounding is not None
        else _grounding_to_state(_context(_exemplar("ATC-1")), 0)
    )
    values["evidence"] = _evidence_to_state(None)
    return values


def test_a_paused_rollout_renders_from_its_checkpoint():
    """The regression that reached the browser as a bare "network error": `house_format` was added to
    the response model and not to the checkpointed state, so splatting the state into the model failed
    validation. The SSE stream emitted its progress events and then died before the result frame, and
    every test passed because none of them built the response from a realistic checkpoint."""
    from app.api.routes import _rollout_status_response

    response = _rollout_status_response("thread-1", _checkpoint_values())

    assert response.status == "pending_approval"
    assert response.plan.grounding.house_format.sections == []
    assert [e.issue_key for e in response.plan.grounding.exemplars] == ["ATC-1"]
    assert response.plan.issues[0].estimate_reference == ["ATC-1"]


def test_a_checkpoint_written_before_the_layout_field_existed_still_renders():
    """Defaults are filled per field, not by swapping in a whole default dict: an older `grounding`
    is PRESENT but incomplete, so an all-or-nothing fallback never fires."""
    from app.api.routes import _rollout_status_response

    values = _checkpoint_values(grounding={
        "status": "grounded", "exemplars": [], "estimate_anchor_count": 0,
        "dropped_estimate_references": 0,   # no house_format at all
    })

    response = _rollout_status_response("thread-1", values)

    assert response.plan.grounding.status == "grounded"
    assert response.plan.grounding.house_format.sections == []
    assert response.plan.grounding.house_format.item_marker is None


def test_the_layout_survives_the_checkpoint_round_trip():
    from app.api.routes import _rollout_status_response
    from app.planning.grounding import GroundingContext
    from app.planning.house_format import HouseFormat
    from app.planning.rollout_graph import _grounding_to_state

    context = GroundingContext(status="grounded", house_format=HouseFormat(
        sections=["Context", "Acceptance criteria"], item_marker="AC",
        coverage={"Context": 1.0, "Acceptance criteria": 1.0}, sample_size=20,
    ))
    response = _rollout_status_response("t", _checkpoint_values(_grounding_to_state(context, 0)))

    assert response.plan.grounding.house_format.sections == ["Context", "Acceptance criteria"]
    assert response.plan.grounding.house_format.item_marker == "AC"
    assert response.plan.grounding.house_format.sample_size == 20


def test_every_checkpointed_grounding_field_exists_on_the_response_model():
    """A structural guard, added after the failure it would have caught.

    `_rollout_status_response` splats the checkpointed grounding dict into `PlanGroundingOut`, so a
    field added to one and not the other breaks every rollout status call — which reached the browser
    as a bare "network error" while every unit test passed. Adding a field now fails here instead,
    at the moment the two shapes diverge, rather than at runtime on a paused workflow.
    """
    from app.api.routes import HouseFormatOut, PlanGroundingOut
    from app.planning.rollout_schemas import GroundingState, HouseFormatState

    assert set(GroundingState.__annotations__) <= set(PlanGroundingOut.model_fields), (
        "the checkpoint carries a grounding field the response model cannot accept"
    )
    assert set(HouseFormatState.__annotations__) <= set(HouseFormatOut.model_fields), (
        "the checkpoint carries a layout field the response model cannot accept"
    )
    # And the other direction: a required response field missing from the checkpoint is what actually
    # broke, so anything the model requires must either be checkpointed or have a default.
    required = {
        name for name, field in PlanGroundingOut.model_fields.items() if field.is_required()
    }
    assert required <= set(GroundingState.__annotations__), (
        "the response model requires a field the checkpoint does not store"
    )
