"""When no answer model can be reached, structured features say so instead of serving a stand-in."""
import json
import types
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from fastapi import HTTPException
from instructor.core.exceptions import InstructorRetryException
from langgraph.checkpoint.memory import InMemorySaver

from app.api import routes
from app.auth.space_membership import NoopSpaceMembershipChecker
from app.llm.instructor_client import _RequestRewriteTransport
from app.llm.circuit_breaker import (
    MODEL_UNAVAILABLE_MESSAGE,
    CircuitOpenError,
    ModelUnavailableError,
    is_model_unavailable,
)
from app.main import _model_unavailable, _plan_generation_failed
from app.planning.graph import planner_node
from app.planning.rollout_graph import build_rollout_graph
from app.planning.rollout_schemas import initial_rollout_state
from app.planning.schemas import EpicDraft, IssueDraft, PlanCritique
from app.planning.service import PLAN_FAILED_MESSAGE, PlanGenerationError, plan_epic, refine_plan
from app.sprint_recovery.graph import _call_model
from tests.test_planning import FakeInstructorClient

_REQUEST = httpx.Request("POST", "http://gateway.invalid/chat/completions")
_EPIC = EpicDraft(title="Add dark mode", description="Support a dark theme app-wide.", goals=[])
_ISSUES = [IssueDraft(
    temp_id="1", title="Theme tokens", description="Define them.", issue_type="task", labels=[],
    estimate_story_points=3, estimate_rationale="Small", depends_on=[],
)]


def _connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def _wrapped_by_instructor(exc: Exception) -> InstructorRetryException:
    try:
        raise InstructorRetryException("failed", n_attempts=1, total_usage=0) from exc
    except InstructorRetryException as wrapped:
        return wrapped


def test_recognizes_an_outage_through_instructor_and_sdk_wrapping():
    try:
        raise openai.APIConnectionError(request=_REQUEST) from CircuitOpenError("open")
    except openai.APIConnectionError as sdk_error:
        assert is_model_unavailable(_wrapped_by_instructor(sdk_error))

    server_error = openai.InternalServerError(
        "down", response=httpx.Response(500, request=_REQUEST), body=None,
    )
    assert is_model_unavailable(_wrapped_by_instructor(server_error))


def test_a_bad_request_or_a_parse_failure_is_not_an_outage():
    bad_request = openai.BadRequestError("bad", response=httpx.Response(400, request=_REQUEST), body=None)
    assert not is_model_unavailable(_wrapped_by_instructor(bad_request))
    assert not is_model_unavailable(RuntimeError("could not parse"))


class _ScriptedTransport(httpx.AsyncBaseTransport):
    def __init__(self, outcome):
        self._outcome = outcome
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return httpx.Response(self._outcome)


def _completion_request(model: str) -> httpx.Request:
    return httpx.Request(
        "POST", "http://gateway.invalid/chat/completions",
        content=json.dumps({"model": model, "messages": []}).encode(),
    )


async def test_instructor_transport_opens_the_breaker_after_repeated_upstream_failures():
    inner = _ScriptedTransport(502)
    transport = _RequestRewriteTransport(inner)
    model = f"breaker-instructor-{uuid.uuid4()}"

    for _ in range(3):
        await transport.handle_async_request(_completion_request(model))
    with pytest.raises(CircuitOpenError):
        await transport.handle_async_request(_completion_request(model))

    assert inner.calls == 3


async def test_instructor_transport_counts_connection_errors_but_not_bad_requests():
    model = f"breaker-instructor-{uuid.uuid4()}"
    bad_requests = _RequestRewriteTransport(_ScriptedTransport(400))
    for _ in range(5):
        await bad_requests.handle_async_request(_completion_request(model))

    unreachable = _RequestRewriteTransport(_ScriptedTransport(httpx.ConnectError("refused")))
    for _ in range(3):
        with pytest.raises(httpx.ConnectError):
            await unreachable.handle_async_request(_completion_request(model))
    with pytest.raises(CircuitOpenError):
        await unreachable.handle_async_request(_completion_request(model))


async def test_plan_epic_refuses_instead_of_serving_a_stub_plan():
    client = FakeInstructorClient(_wrapped_by_instructor(_connection_error()))

    with pytest.raises(ModelUnavailableError):
        await plan_epic(client, "fake-model", "Add dark mode")


async def test_refine_refuses_instead_of_reporting_an_unchanged_plan():
    client = FakeInstructorClient(_connection_error())

    with pytest.raises(ModelUnavailableError):
        await refine_plan(client, "fake-model", _EPIC, _ISSUES, "add a QA task")


async def test_an_outage_during_a_revision_round_keeps_the_first_plan():
    client = FakeInstructorClient(_connection_error())
    state = {
        "critique": PlanCritique(), "grounding": None, "assessment": None, "revision_round": 0,
        "epic": _EPIC, "issues": _ISSUES, "existing_labels": [], "proposal": "Add dark mode",
        "dropped_estimate_references": 0, "historically_derived_estimates": 0,
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "estimated_cost_usd": 0.0,
    }

    update = await planner_node(state, client, "fake-model")

    assert update["epic"] == _EPIC
    assert update["issues"] == _ISSUES
    assert update["degraded"] is True


async def test_rollout_ends_as_failed_with_the_message_and_never_reaches_review():
    graph = build_rollout_graph(
        FakeInstructorClient(_connection_error()), "fake-model", NoopSpaceMembershipChecker(),
        jira=None, planning_graph_getter=lambda: None,
    ).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = await graph.ainvoke(
        initial_rollout_state("Add dark mode", [], None, None, 7, "u1", "alice"), config=config,
    )

    assert result["status"] == "failed"
    assert result["failed_step"] == "plan"
    assert result["error"] == MODEL_UNAVAILABLE_MESSAGE
    assert result["epic"] is None
    assert not (await graph.aget_state(config)).next  # no pending approval interrupt


def _route_request(values: dict):
    graph = types.SimpleNamespace(aget_state=AsyncMock(return_value=types.SimpleNamespace(values=values)))
    state = types.SimpleNamespace(rollout_graph=graph, epic_rollout_checkpoint_db_url="postgresql://unused")
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state), headers={"x-user-id": "user-a"})


_PLAN_FAILED = {
    "status": "failed", "failed_step": "plan", "space_id": 7, "user_id": "user-a",
    "error": MODEL_UNAVAILABLE_MESSAGE,
}


async def test_a_plan_that_never_generated_is_not_offered_for_resume():
    with patch.object(routes, "_authorize_space_ids", new=AsyncMock()), \
         patch.object(routes, "find_plan_epic_active_thread_id", new=AsyncMock(return_value="t-1")):
        result = await routes.find_active_rollout_endpoint(space_id=7, request=_route_request(_PLAN_FAILED))

    assert result is None


async def test_a_plan_that_never_generated_cannot_be_retried():
    with patch.object(routes, "_authorize_workflow_state", new=AsyncMock()), \
         pytest.raises(HTTPException) as raised:
        await routes.retry_rollout_endpoint("t-1", _route_request(_PLAN_FAILED))

    assert raised.value.status_code == 409


def test_stream_errors_name_the_model_only_when_it_is_the_cause():
    assert routes._stream_error_detail(_wrapped_by_instructor(_connection_error()), "x") == MODEL_UNAVAILABLE_MESSAGE
    assert routes._stream_error_detail(RuntimeError("jira down"), "epic rollout failed") == "epic rollout failed"


async def test_sprint_recovery_model_call_raises_model_unavailable():
    client = FakeInstructorClient(_wrapped_by_instructor(_connection_error()))

    with pytest.raises(ModelUnavailableError):
        await _call_model(client, "fake-model", object, "prompt")


async def test_non_streaming_endpoints_answer_503_with_the_message():
    response = await _model_unavailable(None, ModelUnavailableError())

    assert response.status_code == 503
    assert json.loads(response.body) == {"detail": MODEL_UNAVAILABLE_MESSAGE}


async def test_rollout_ends_as_failed_when_the_model_answers_but_no_plan_validates():
    graph = build_rollout_graph(
        FakeInstructorClient(RuntimeError("unparseable")), "fake-model", NoopSpaceMembershipChecker(),
        jira=None, planning_graph_getter=lambda: None,
    ).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = await graph.ainvoke(
        initial_rollout_state("Add dark mode", [], None, None, 7, "u1", "alice"), config=config,
    )

    assert result["status"] == "failed"
    assert result["failed_step"] == "plan"
    assert result["error"] == PLAN_FAILED_MESSAGE
    assert result["epic"] is None


async def test_a_revision_round_that_produces_nothing_keeps_the_first_plan():
    client = FakeInstructorClient(RuntimeError("unparseable"))
    state = {
        "critique": PlanCritique(), "grounding": None, "assessment": None, "revision_round": 0,
        "epic": _EPIC, "issues": _ISSUES, "existing_labels": [], "proposal": "Add dark mode",
        "dropped_estimate_references": 0, "historically_derived_estimates": 0,
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "estimated_cost_usd": 0.0,
    }

    update = await planner_node(state, client, "fake-model")

    assert update["issues"] == _ISSUES


async def test_plan_failures_reach_the_user_as_their_own_message():
    error = PlanGenerationError(PLAN_FAILED_MESSAGE)
    assert routes._stream_error_detail(error, "epic rollout failed") == PLAN_FAILED_MESSAGE

    response = await _plan_generation_failed(None, error)
    assert response.status_code == 502
    assert json.loads(response.body) == {"detail": PLAN_FAILED_MESSAGE}
