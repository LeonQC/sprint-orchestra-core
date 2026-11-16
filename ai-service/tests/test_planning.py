"""Tests for AI-assisted epic planning (app/planning/) — same fake-instructor-client, no-network
approach as tests/test_drafting.py.

Covers: (1) a successful call returns the validated plan untouched and degraded=False; (2) a failure
raises an error with a message for the user, never a stand-in plan; (3) `validate_and_order` breaks
dependency cycles deterministically instead of failing; (4) `allocate_sprints` bin-packs correctly
both in open-ended (capacity-driven) mode and fixed-sprint-count mode.
"""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.planning.prompts import render_epic_plan_prompt, render_epic_plan_refine_prompt
from app.planning.schemas import (
    EpicDraft,
    EpicPlanDraft,
    IssueDraft,
    MissingDependency,
    PlanCritique,
    UnrealisticEstimate,
)
from app.planning.service import (
    PlanGenerationError,
    allocate_sprints,
    critique_plan,
    critique_to_instruction,
    drop_unexplained_dependencies,
    plan_epic,
    plan_needs_revision,
    refine_plan,
    validate_and_order,
)


class _FakeCompletions:
    """Accepts either a single result (returned/raised for every call, the original behaviour every
    single-call test here relies on) or a dict of {response_model: [result, ...]} queues — the graph
    tests need the latter because planner and critic are each called several times across a revision
    loop, with a different answer each round.
    """

    def __init__(self, result):
        self._result = result if not isinstance(result, dict) else None
        self._queues = (
            {k: (list(v) if isinstance(v, list) else v) for k, v in result.items()}
            if isinstance(result, dict)
            else None
        )
        self.calls = []

    # Provider-reported token counts for one fake call. Deliberately non-round numbers that differ from
    # `_TOKENS_PER_LLM_CALL_ESTIMATE`, so a test asserting on real usage can't accidentally pass
    # against the flat pre-call estimate the budget guard still uses.
    fake_usage = SimpleNamespace(prompt_tokens=1234, completion_tokens=567, prompt_tokens_details=None)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._result
        if self._queues is not None:
            queued = self._queues[kwargs["response_model"]]
            result = queued if isinstance(queued, Exception) else queued.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def create_with_completion(self, **kwargs):
        """`instructor`'s other entrypoint: same parsed model, plus the raw provider response — which
        is where the real token counts live (see `_call_model` in app/sprint_recovery/graph.py, which
        uses this precisely so cost stops being a hardcoded estimate)."""
        parsed = await self.create(**kwargs)
        return parsed, SimpleNamespace(usage=self.fake_usage)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeInstructorClient:
    def __init__(self, result):
        self.completions = _FakeCompletions(result)
        self.chat = _FakeChat(self.completions)


def _issue(
    temp_id, depends_on=None, points=None, title=None,
    dependency_reasons=None, dependency_kinds=None,
):
    return IssueDraft(
        temp_id=temp_id,
        title=title or f"Issue {temp_id}",
        description=f"Description for {temp_id}",
        issue_type="task",
        labels=[],
        estimate_story_points=points,
        estimate_rationale="because reasons" if points is not None else None,
        depends_on=depends_on or [],
        dependency_reasons=dependency_reasons or {},
        dependency_kinds=dependency_kinds or {},
    )


async def test_plan_epic_returns_validated_plan_on_success():
    expected = EpicPlanDraft(
        epic=EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=["Toggle in settings"]),
        issues=[
            _issue("1"),
            # An explained edge, so `drop_unexplained_dependencies` leaves the plan untouched here.
            _issue(
                "2",
                depends_on=["1"],
                dependency_reasons={"1": "The toggle reads the theme variables defined in issue 1."},
                dependency_kinds={"1": "blocks"},
            ),
        ],
    )
    client = FakeInstructorClient(expected)

    result = await plan_epic(client, "fake-model", "Add dark mode support across the app")

    assert result.plan == expected
    assert result.degraded is False
    assert result.error is None
    assert client.completions.calls[0]["response_model"] is EpicPlanDraft
    assert result.input_tokens == 1234
    assert result.output_tokens == 567


async def test_plan_epic_raises_instead_of_returning_a_stand_in_plan():
    client = FakeInstructorClient(RuntimeError("unparseable"))

    with pytest.raises(PlanGenerationError, match="couldn't produce a usable plan"):
        await plan_epic(client, "fake-model", "Add dark mode\nmore detail nobody will read")


def test_epic_plan_schema_rejects_duplicate_temp_ids_before_ordering_can_drop_work():
    with pytest.raises(ValidationError, match="temp ids must be unique"):
        EpicPlanDraft(epic=EpicDraft(title="Epic", description="Scope"), issues=[
            _issue("1", title="First"),
            _issue("1", title="Second"),
        ])


def test_epic_plan_schema_rejects_blank_or_padded_temp_ids():
    with pytest.raises(ValidationError, match="temp ids cannot be blank"):
        EpicPlanDraft(
            epic=EpicDraft(title="Epic", description="Scope"), issues=[_issue("   ")],
        )
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        EpicPlanDraft(
            epic=EpicDraft(title="Epic", description="Scope"), issues=[_issue(" 1")],
        )


def test_render_with_existing_labels_includes_them_joined():
    prompt = render_epic_plan_prompt(["backend", "urgent"])
    assert "backend, urgent" in prompt


def test_unexplained_dependency_edges_are_dropped_rather_than_shown_to_the_human():
    """An edge the model couldn't justify is evidence against the edge, not reviewer homework.

    This replaced a regex "is the reason concrete enough" predicate that gated *approval*. That gate
    scored vocabulary rather than reasoning — it passed "aaaa... rather than bbbb" and failed real
    explanations — so the plan now carries only edges that arrived already explained.
    """
    issues = [
        _issue("1"),
        _issue(
            "2",
            depends_on=["1"],
            dependency_reasons={"1": "   "},
            dependency_kinds={"1": "blocks"},
        ),
    ]
    cleaned = drop_unexplained_dependencies(issues)
    assert cleaned[1].depends_on == []
    assert cleaned[1].dependency_reasons == {}
    assert cleaned[1].dependency_kinds == {}


def test_explained_dependency_edges_survive_with_their_reason_and_kind():
    reason = "The export job reads the approved permission scope to filter which orders it writes."
    issues = [
        _issue("1"),
        _issue(
            "2",
            depends_on=["1"],
            dependency_reasons={"1": reason},
            dependency_kinds={"1": "blocks"},
        ),
    ]
    cleaned = drop_unexplained_dependencies(issues)
    assert cleaned[1].depends_on == ["1"]
    assert cleaned[1].dependency_reasons == {"1": reason}
    assert cleaned[1].dependency_kinds == {"1": "blocks"}


def test_dependency_reason_wording_is_no_longer_judged_by_pattern_matching():
    """Real explanations the old regex gate rejected must now survive untouched."""
    rejected_by_the_old_gate = [
        "The audit log needs the job outcome fields, which only exist once the async job is built.",
        "The UI polls the job status endpoint introduced by the async job work.",
        "We need the agreed CSV column set before the download endpoint can serve a valid file.",
    ]
    issues = [_issue("1")] + [
        _issue(str(i + 2), depends_on=["1"], dependency_reasons={"1": reason})
        for i, reason in enumerate(rejected_by_the_old_gate)
    ]
    cleaned = drop_unexplained_dependencies(issues)
    assert [issue.depends_on for issue in cleaned[1:]] == [["1"], ["1"], ["1"]]


def test_render_requires_plain_language_dependency_reasons():
    prompt = render_epic_plan_prompt()
    assert "matching `dependency_reasons` entry" in prompt
    assert "never merely restate" in prompt


def test_render_decides_blocking_edges_by_naming_the_consumed_artifact():
    """"blocks" is settled by a nameable artifact, not by how confident the model feels.

    "genuinely cannot be done at all" is a judgement, and it measured differently across runs of the
    same proposal. Naming the consumed thing is checkable by the reviewer in seconds, and pushes the
    ambiguous cases to "after", where a wrong guess costs nothing.
    """
    prompt = render_epic_plan_prompt()
    assert "Name the specific thing the later issue consumes" in prompt
    assert "A completed *state* is not a named thing." in prompt
    # The asymmetry that makes "after" the safe default has to survive prompt edits.
    assert "choose \"after\"" in prompt


def test_render_requests_best_effort_estimates_for_sprint_planning():
    prompt = render_epic_plan_prompt()
    assert "Give every sufficiently scoped issue a best-effort" in prompt
    assert "Use null only when the proposal is" in prompt


def test_render_writes_for_a_mixed_non_engineering_planning_room():
    prompt = render_epic_plan_prompt()
    assert "product, design, QA, operations, and engineering" in prompt
    assert "Prefer everyday workplace language" in prompt
    assert "Use technical terms only when they are necessary" in prompt


async def test_refine_plan_returns_validated_plan_on_success():
    epic = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])
    issues = [_issue("1")]
    expected = EpicPlanDraft(epic=epic, issues=[_issue("1"), _issue("2", title="Add QA task")])
    client = FakeInstructorClient(expected)

    result = await refine_plan(client, "fake-model", epic, issues, "add a QA task")

    assert result.plan == expected
    assert result.degraded is False
    assert client.completions.calls[0]["response_model"] is EpicPlanDraft
    sent_user_message = client.completions.calls[0]["messages"][1]["content"]
    assert "Add dark mode" in sent_user_message
    assert "Instruction: add a QA task" in sent_user_message


async def test_refine_plan_raises_and_says_the_plan_is_unchanged():
    epic = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])
    issues = [_issue("1"), _issue("2", depends_on=["1"])]
    client = FakeInstructorClient(RuntimeError("unparseable"))

    with pytest.raises(PlanGenerationError, match="Your plan is unchanged"):
        await refine_plan(client, "fake-model", epic, issues, "add a QA task")


def test_render_refine_prompt_with_existing_labels_includes_them_joined():
    prompt = render_epic_plan_refine_prompt(["backend", "urgent"])
    assert "backend, urgent" in prompt


async def test_refine_shows_the_model_every_field_it_is_told_to_preserve():
    """A refine round must not be able to destroy a field just because it wasn't rendered.

    `labels` and `estimate_rationale` were omitted from the rendered plan while the prompt still
    said to preserve them, so one "adjust the plan" click blanked both on every pre-existing issue —
    measured live: four issues went from `labels=['frontend']` with rationales to `labels=[]` with
    none, while the newly added issue kept both.
    """
    epic = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])
    issues = [
        _issue("1", points=5).model_copy(update={
            "labels": ["frontend", "design"],
            "estimate_rationale": "Touches every themed surface.",
        }),
        _issue("2", points=3),  # no labels: must be rendered as an explicit empty, not omitted
    ]
    client = FakeInstructorClient(EpicPlanDraft(epic=epic, issues=issues))

    await refine_plan(client, "fake-model", epic, issues, "add a QA task")

    sent = client.completions.calls[0]["messages"][1]["content"]
    assert "labels: frontend, design" in sent
    assert "labels: none" in sent
    assert "Estimate rationale: Touches every themed surface." in sent


def test_validate_and_order_respects_dependencies():
    issues = [_issue("2", depends_on=["1"]), _issue("1"), _issue("3", depends_on=["2"])]

    ordered = validate_and_order(issues)

    assert [i.temp_id for i in ordered] == ["1", "2", "3"]


def test_validate_and_order_drops_unknown_temp_id_reference():
    issues = [_issue("1", depends_on=["does-not-exist"])]

    ordered = validate_and_order(issues)

    assert ordered[0].depends_on == []


def test_validate_and_order_does_not_invent_a_circular_reason_when_model_omits_one():
    issues = [_issue("1", title="Define export contract"), _issue("2", depends_on=["1"], title="Build exporter")]

    ordered = validate_and_order(issues)

    exporter = next(issue for issue in ordered if issue.temp_id == "2")
    assert exporter.dependency_reasons == {}


def test_validate_and_order_breaks_a_cycle_instead_of_hanging():
    issues = [_issue("1", depends_on=["2"]), _issue("2", depends_on=["1"])]

    ordered = validate_and_order(issues)

    assert {i.temp_id for i in ordered} == {"1", "2"}
    # the cycle must be broken: at least one edge involved in it no longer round-trips
    remaining_edges = {i.temp_id: set(i.depends_on) for i in ordered}
    assert not (remaining_edges["1"] == {"2"} and remaining_edges["2"] == {"1"})


def test_allocate_sprints_packs_by_capacity():
    issues = [_issue("1", points=5), _issue("2", points=5), _issue("3", points=5)]

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=None)

    assert [b.issue_temp_ids for b in buckets] == [["1"], ["2"], ["3"]]
    assert [b.total_points for b in buckets] == [5, 5, 5]


def test_allocate_sprints_packs_multiple_issues_into_one_bucket_when_capacity_allows():
    issues = [_issue("1", points=3), _issue("2", points=3), _issue("3", points=3)]

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=None)

    assert [b.issue_temp_ids for b in buckets] == [["1", "2"], ["3"]]


def test_allocate_sprints_first_fit_avoids_an_unnecessary_overflow_bucket():
    issues = [
        _issue("1", points=5),
        _issue("2", points=5),
        _issue("3", points=3),
        _issue("4", points=3),
    ]

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=2)

    assert [bucket.issue_temp_ids for bucket in buckets] == [["1", "3"], ["2", "4"]]
    assert [bucket.total_points for bucket in buckets] == [8, 8]


def test_allocate_sprints_never_places_a_prerequisite_after_its_dependent():
    issues = [
        _issue("1", points=5),
        _issue("2", points=5),
        _issue("3", depends_on=["1"], points=3),
        _issue("4", depends_on=["2"], points=3),
    ]

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=2)
    bucket_by_issue = {
        temp_id: bucket.sprint_index
        for bucket in buckets
        for temp_id in bucket.issue_temp_ids
    }

    for issue in issues:
        assert all(
            bucket_by_issue[dependency] <= bucket_by_issue[issue.temp_id]
            for dependency in issue.depends_on
        )


def test_allocate_sprints_places_unestimated_issues_without_blocking_on_missing_points():
    issues = [_issue("1", points=None), _issue("2", points=5)]

    buckets = allocate_sprints(issues, sprint_capacity_points=5, target_sprint_count=None)

    assert [b.issue_temp_ids for b in buckets] == [["1", "2"]]
    assert buckets[0].total_points == 5


def test_allocate_sprints_respects_target_sprint_count():
    issues = [_issue(str(i), points=2) for i in range(1, 7)]  # 6 issues, 12 points total

    buckets = allocate_sprints(issues, sprint_capacity_points=None, target_sprint_count=3)

    assert len(buckets) == 3
    assert sum(len(b.issue_temp_ids) for b in buckets) == 6


def test_allocate_sprints_capacity_wins_when_target_count_cannot_hold_the_work():
    issues = [_issue(str(i), points=5) for i in range(1, 8)]  # 35 points cannot fit in 2 x 8

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=2)

    assert len(buckets) == 7
    assert all(bucket.total_points <= 8 for bucket in buckets)
    assert [temp_id for bucket in buckets for temp_id in bucket.issue_temp_ids] == [
        str(i) for i in range(1, 8)
    ]


def test_allocate_sprints_splits_capacity_buckets_to_reach_a_larger_desired_count():
    issues = [_issue(str(i), points=2) for i in range(1, 7)]

    buckets = allocate_sprints(issues, sprint_capacity_points=10, target_sprint_count=3)

    assert len(buckets) == 3
    assert all(bucket.total_points <= 10 for bucket in buckets)
    assert [temp_id for bucket in buckets for temp_id in bucket.issue_temp_ids] == [
        str(i) for i in range(1, 7)
    ]


def test_allocate_sprints_keeps_one_oversized_issue_alone_for_human_review():
    issues = [_issue("1", points=13), _issue("2", points=3)]

    buckets = allocate_sprints(issues, sprint_capacity_points=8, target_sprint_count=2)

    assert [bucket.issue_temp_ids for bucket in buckets] == [["1"], ["2"]]
    assert [bucket.total_points for bucket in buckets] == [13, 3]


def test_allocate_sprints_returns_empty_list_for_no_issues():
    assert allocate_sprints([], sprint_capacity_points=10) == []


# --- Multi-agent path building blocks (see app/planning/graph.py for how they're wired together) ---

_EPIC = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])


async def test_critique_plan_returns_the_findings_and_sees_both_proposal_and_plan():
    critique = PlanCritique(coverage_gaps=["no issue covers data migration"])
    client = FakeInstructorClient(critique)

    result = await critique_plan(client, "fake-model", "Add dark mode everywhere", _EPIC, [_issue("1")])

    assert result.critique == critique
    assert result.error is None
    sent = client.completions.calls[0]["messages"][1]["content"]
    assert "Add dark mode everywhere" in sent  # the original proposal, needed to judge coverage
    assert "[1]" in sent  # the current plan's issues


async def test_critique_plan_treats_a_failed_review_as_approved_rather_than_blocking_the_plan():
    client = FakeInstructorClient(RuntimeError("model unreachable"))

    result = await critique_plan(client, "fake-model", "proposal", _EPIC, [_issue("1")])

    assert result.error == "model unreachable"
    assert plan_needs_revision(result.critique) is False


def test_plan_needs_revision_is_false_for_an_empty_critique():
    assert plan_needs_revision(PlanCritique()) is False


def test_plan_needs_revision_is_true_for_a_finding_in_any_single_category():
    assert plan_needs_revision(PlanCritique(coverage_gaps=["missing migration"])) is True
    assert plan_needs_revision(PlanCritique(redundant_issue_groups=[["1", "2"]])) is True
    assert plan_needs_revision(
        PlanCritique(unrealistic_estimates=[UnrealisticEstimate(temp_id="1", reason="way too small")])
    ) is True
    assert plan_needs_revision(
        PlanCritique(missing_dependencies=[MissingDependency(temp_id="2", likely_depends_on="1", reason="needs the API")])
    ) is True


def test_critique_to_instruction_renders_every_category_into_the_refine_instruction_shape():
    instruction = critique_to_instruction(
        PlanCritique(
            coverage_gaps=["no issue covers data migration"],
            redundant_issue_groups=[["2", "3"]],
            unrealistic_estimates=[UnrealisticEstimate(temp_id="4", reason="13 points for a config toggle")],
            missing_dependencies=[MissingDependency(temp_id="5", likely_depends_on="1", reason="needs the API first")],
        )
    )

    assert "no issue covers data migration" in instruction
    assert "2, 3" in instruction
    assert "13 points for a config toggle" in instruction
    assert "5" in instruction and "needs the API first" in instruction


def test_critique_to_instruction_skips_a_single_item_redundancy_group():
    # A one-element "group" is not a merge instruction — emitting "merge these: 2" would be nonsense.
    assert "Merge" not in critique_to_instruction(PlanCritique(redundant_issue_groups=[["2"]]))
