"""Creation requires explicit scope; missing history must never erase that scope."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api import routes
from app.planning.grounding import GroundingContext, with_stated_requirements
from app.planning.pattern_schemas import EvidenceAssessment
from app.planning.prompts import render_epic_plan_prompt
from app.planning.rollout_graph import plan_node
from app.planning.rollout_schemas import initial_rollout_state
from app.planning.service import _render_proposal
from tests.test_planning_grounding import FakeRetrieval, _plan_client, _route_request
from tests.test_planning_requirements import _retrieval


@pytest.mark.parametrize("path", ["/plan-epic", "/plan-epic/rollout", "/plan-epic/rollout/stream"])
@pytest.mark.parametrize("requirements", [Ellipsis, None, [], [""], [" \t\n"], ["Export orders", "  "]])
def test_creation_rejects_missing_empty_or_blank_requirements_before_doing_work(path, requirements):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.enforce_user_quota] = lambda: None
    graph = SimpleNamespace(ainvoke=AsyncMock())
    app.state.rollout_graph = graph
    payload = {"proposal": "Order exports", "spaceId": 7}
    if requirements is not Ellipsis:
        payload["requirements"] = requirements
    response = TestClient(app).post(path, json=payload)
    assert response.status_code == 422
    assert any(error["loc"][1] == "requirements" for error in response.json()["detail"])
    graph.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("model", [routes.PlanEpicRequest, routes.PlanRolloutRequest])
def test_one_requirement_is_sufficient_and_whitespace_is_trimmed(model):
    request = model(proposal="Order exports", space_id=7, requirements=["  Export orders  "])
    assert request.requirements == ["Export orders"]
    assert "requirements" in model.model_json_schema()["required"]


@pytest.mark.parametrize("enabled", [False, True])
async def test_stateless_planning_without_history_still_passes_requirements_to_model(enabled):
    client = _plan_client()
    request = _route_request(grounding_enabled=enabled, client=client)
    result = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(proposal="Order exports", requirements=["Only admins can export"]),
        request,
    )
    messages = client.completions.calls[0]["messages"]
    assert "Only admins can export" in messages[1]["content"]
    assert result.requirement_coverage[0].text == "Only admins can export"
    assert result.requirement_coverage[0].exemplar_keys == []


async def test_rollout_without_grounding_still_preserves_scope_in_model_input():
    client = _plan_client()
    state = initial_rollout_state(
        "Order exports", [], None, None, 7, "u1", "alice", requirements=["Only admins can export"],
    )
    result = await plan_node(state, client, "fake-model", lambda: None)
    assert result["status"] == "pending_approval"
    assert "Only admins can export" in client.completions.calls[0]["messages"][1]["content"]


async def test_failed_retrieval_still_generates_from_the_required_user_scope():
    client = _plan_client()
    request = _route_request(
        grounding_enabled=True, client=client, retrieval=FakeRetrieval(hits=[], fail=True),
    )
    response = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(
            proposal="Order exports", space_id=5,
            requirements=["Export orders", "Enforce permissions"],
        ),
        request,
    )
    assert response.grounding.status == "retrieval_failed"
    message = client.completions.calls[0]["messages"][1]["content"]
    assert "1. Export orders" in message
    assert "2. Enforce permissions" in message
    assert [item.text for item in response.requirement_coverage] == ["Export orders", "Enforce permissions"]


async def test_irrelevant_history_is_withheld_while_user_requirements_remain_visible():
    from app.planning.grounding import gather_requirement_exemplars

    context = await gather_requirement_exemplars(
        _retrieval(), [5], "Employee onboarding", ["Upload a spreadsheet"], limit=8,
    )
    assessment = EvidenceAssessment(
        strength="none", reasoning="The history does not apply.", relevant_issue_keys=[], patterns=[],
    )
    prompt = render_epic_plan_prompt([], context, assessment)
    assert "Upload a spreadsheet" in prompt
    assert "ATC-73" not in prompt
    assert "PAST WORK FROM THIS TEAM" not in prompt
    assert context.requirements[0].exemplars  # Rendering must not mutate the retrieval evidence.


def test_failed_retrieval_keeps_requirements_in_the_planning_instruction():
    context = with_stated_requirements(
        GroundingContext(status="retrieval_failed"), ["Export orders", "Enforce permissions"],
    )
    message = _render_proposal("Order exports", context)
    assert "1. Export orders" in message
    assert "2. Enforce permissions" in message
    assert context.status == "retrieval_failed"
