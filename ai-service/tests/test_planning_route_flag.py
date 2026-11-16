"""POST /plan-epic must behave identically to the caller whichever planning path produced the plan.

Two things are guarded here. First, that the flag actually switches paths (and that the default, no
flag, is still the original single-call one — a regression guard, since the multi-agent path costs at
least 2x the LLM calls and must never turn itself on). Second, that the response contract the frontend depends
on (`PlanEpicModal.tsx` has zero awareness of how a plan was produced) doesn't drift: a field renamed
or added here is a frontend break, so the field sets are snapshotted.
"""
from types import SimpleNamespace

from app.api.routes import (
    EpicDraftOut,
    IssueDraftOut,
    PlanEpicRequest,
    PlanEpicResponse,
    SprintBucketOut,
    plan_epic_endpoint,
)
from app.planning.graph import build_planning_graph
from app.stats import ServiceStats
from app.planning.schemas import EpicDraft, EpicPlanDraft, IssueDraft, PlanCritique
from tests.test_planning import FakeInstructorClient

_EPIC = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])
_ISSUE = IssueDraft(
    temp_id="1", title="Define theme variables", description="Add CSS custom properties.",
    issue_type="task", labels=[], estimate_story_points=3, estimate_rationale="small", depends_on=[],
)


def _request(client, planning_graph):
    return SimpleNamespace(
        headers={},
        app=SimpleNamespace(
            state=SimpleNamespace(
                instructor_client=client, instructor_model="fake-model", planning_graph=planning_graph,
                # What the endpoint needs to write a usage ledger entry (app/llm/usage_ledger.py):
                # the rate card to price the call against, and the counter it feeds.
                model_provider="Self-hosted", model_deployment="local",
                stats=ServiceStats(),
                # Grounding off: these tests are about which *planning path* runs, and a request with
                # no spaceId plans ungrounded regardless — see tests/test_planning_grounding.py for
                # the retrieval side.
                settings=SimpleNamespace(
                    epic_planning_grounding_enabled=False, epic_planning_grounding_exemplars=8,
                    epic_planning_pattern_extraction_enabled=False,
                ),
                retrieval=None,
            )
        )
    )


def _single_call_client():
    return FakeInstructorClient(EpicPlanDraft(epic=_EPIC, issues=[_ISSUE]))


def _multiagent_client():
    return FakeInstructorClient(
        {
            EpicPlanDraft: [EpicPlanDraft(epic=_EPIC, issues=[_ISSUE])],
            PlanCritique: [PlanCritique()],
        }
    )


async def test_flag_off_uses_the_single_call_path():
    client = _single_call_client()

    response = await plan_epic_endpoint(
        PlanEpicRequest(proposal="Add dark mode", requirements=["Support a persisted dark theme"]), _request(client, planning_graph=None)
    )

    assert len(client.completions.calls) == 1
    assert client.completions.calls[0]["response_model"] is EpicPlanDraft
    assert response.degraded is False


async def test_flag_on_routes_through_the_multi_agent_graph():
    client = _multiagent_client()
    graph = build_planning_graph(client, "fake-model")

    response = await plan_epic_endpoint(
        PlanEpicRequest(proposal="Add dark mode", requirements=["Support a persisted dark theme"]), _request(client, planning_graph=graph)
    )

    assert [c["response_model"] for c in client.completions.calls] == [EpicPlanDraft, PlanCritique]
    # the plan made it all the way out through the unchanged response assembly
    assert response.issues[0].estimate_story_points == 3


async def test_both_paths_produce_the_same_response_shape():
    single = await plan_epic_endpoint(
        PlanEpicRequest(proposal="Add dark mode", requirements=["Support a persisted dark theme"]), _request(_single_call_client(), planning_graph=None)
    )
    multi_client = _multiagent_client()
    multi = await plan_epic_endpoint(
        PlanEpicRequest(proposal="Add dark mode", requirements=["Support a persisted dark theme"]),
        _request(multi_client, planning_graph=build_planning_graph(multi_client, "fake-model")),
    )

    assert single.model_dump().keys() == multi.model_dump().keys()
    assert [i.temp_id for i in single.issues] == [i.temp_id for i in multi.issues]
    assert len(single.sprint_plan) == len(multi.sprint_plan)


def test_the_frontend_facing_response_contract_has_not_changed():
    """Fields were added deliberately, in three rounds, and this snapshot was updated with them each
    time: `grounding` + `estimate_reference` for retrieval-grounded planning, then `evidence` and the
    per-issue estimate/origin provenance for pattern extraction, then `change_summary` so a refine
    round reports what it actually did. All additive, so an older frontend ignores them; a *removal*
    or rename still fails here."""
    assert set(PlanEpicResponse.model_fields) == {
        "epic", "issues", "sprint_plan", "grounding", "evidence", "requirement_coverage",
        "change_summary", "degraded", "latency_seconds",
        "input_tokens", "output_tokens", "cached_input_tokens", "estimated_cost_usd",
    }
    assert set(EpicDraftOut.model_fields) == {"title", "description", "goals"}
    assert set(IssueDraftOut.model_fields) == {
        "temp_id", "title", "description", "issue_type", "labels", "estimate_story_points",
        "estimate_rationale", "estimate_reference", "complexity_vs_reference",
        "historical_baseline_points", "estimate_confidence", "origin", "pattern_basis",
        "requirement_index", "format_compliance", "depends_on", "dependency_kinds",
        "dependency_reasons",
    }
    assert set(SprintBucketOut.model_fields) == {"sprint_index", "issue_temp_ids", "total_points"}
