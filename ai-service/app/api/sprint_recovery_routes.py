"""HTTP endpoints for the sprint-recovery workflow (app/sprint_recovery/graph.py), kept out of
app.api.routes because there are enough of them.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncIterator, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from openinference.instrumentation import using_tags

from app.api.routes import (
    _authorize_space_ids,
    _authorize_workflow_state,
    _caller_identity,
    _stream_error_detail,
    _tag_entrypoint_span,
    _tagged,
)
from app.auth.user_quota import enforce_user_quota
from app.llm.usage_ledger import FEATURE_SPRINT_RECOVERY
from app.observability import SPRINT_RECOVERY_INDEX_CATCH_UP_TIMEOUTS_TOTAL
from app.sprint_recovery.active_threads import find_active_thread_id, register_active_thread
from app.sprint_recovery.graph import (
    ON_STAGE_VAR,
    _describe_action,
    build_escalation_summary,
    flatten_escalation_summary,
    is_resumable_after_a_crash,
    list_checkpoint_history,
    retry_recovery_commit,
    stop_recovery,
    time_travel_resume,
    trigger_reevaluation,
)
from app.sprint_recovery.notifications import notify_escalation
from app.sprint_recovery.schemas import RecoveryAction
from app.sprint_recovery.state import initial_recovery_state
from app.api.route_tracing import TracedRoute

router = APIRouter(prefix="/sprint-recovery", route_class=TracedRoute)
logger = logging.getLogger(__name__)


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def _graph(request: Request):
    graph = getattr(request.app.state, "sprint_recovery_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="sprint recovery is disabled (AI_SPRINT_RECOVERY_ENABLED=false)")
    return graph


async def _register_active_thread_if_configured(request: Request, space_id: int, sprint_id: int, thread_id: str) -> None:
    """Best effort — failing to register must never block starting a check. Registered before the first
    diagnosis so a crashed run can still be found (see active_threads.py)."""
    db_url = getattr(request.app.state, "sprint_recovery_checkpoint_db_url", None)
    if not db_url:
        return
    try:
        await register_active_thread(db_url, space_id, sprint_id, thread_id)
    except Exception:  # noqa: BLE001 - see docstring
        logger.exception("failed to register active sprint-recovery thread (space=%s, sprint=%s)", space_id, sprint_id)


async def _auto_reevaluate_after_commit(graph, thread_id: str, values: dict) -> dict:
    """After approval finishes committing (status `waiting_reevaluation`), re-check once right away
    instead of making the user click "Re-check now" — "approve, then tell me what happened" is one
    action to them. "Re-check now" resumes the same pause if this one fails.

    Can't loop: reevaluate always ends in recovered, escalated or a new plan approval. Stale reads are
    handled by `_await_index_catch_up` (graph.py), which replaced a fixed 2-second sleep here.
    """
    if values.get("status") != "waiting_reevaluation":
        return values
    result = await _tagged(
        FEATURE_SPRINT_RECOVERY,
        trigger_reevaluation(graph, thread_id, source="auto_after_approve"),
    )
    return result


async def _notify_if_escalated(request: Request, thread_id: str, sprint_id: int, values: dict) -> None:
    """Notify once, when a `trigger_reevaluation` result newly shows "escalated" — never on a GET poll,
    which would notify on every page reload. See notifications.py.
    """
    if values.get("status") != "escalated":
        return
    graph = _graph(request)
    settings = request.app.state.settings
    summary = await build_escalation_summary(graph, thread_id)
    await notify_escalation(
        settings.sprint_recovery_escalation_webhook_url, thread_id, values["space_id"], sprint_id,
        values["sprint_name"], flatten_escalation_summary(summary),
    )


def _track_index_lag_if_timed_out(values: dict) -> None:
    """Count index catch-up timeouts, once per real re-evaluation (not in the node, where a crash-resume
    could count twice). The per-response flag tells one caller; this counter tells on-call whether the
    indexing pipeline is degraded.
    """
    if not values.get("index_catch_up_timed_out"):
        return
    SPRINT_RECOVERY_INDEX_CATCH_UP_TIMEOUTS_TOTAL.labels(str(values["space_id"])).inc()
    logger.warning(
        "sprint recovery reevaluation timed out waiting for the read model to catch up — reading anyway",
        extra={"space_id": values["space_id"], "sprint_id": values["sprint_id"]},
    )


class ActionOut(_CamelModel):
    action_type: str
    target_issue_key: str
    depends_on_issue_key: Optional[str]
    new_priority: Optional[str]
    comment_body: Optional[str]


class PlanOut(_CamelModel):
    plan_id: str
    name: str
    rationale: str
    impact_on_goal: str
    actions: List[ActionOut]


class HypothesisOut(_CamelModel):
    statement: str
    confidence: str
    supporting_evidence_ids: List[str]


class EvidenceOut(_CamelModel):
    citation_id: str
    issue_key: str
    source_type: str
    content: str


class EscalationRoundOut(_CamelModel):
    round: int
    plan_name: str
    rationale: str
    actions: List[str]


class RecoveryStatusResponse(_CamelModel):
    thread_id: str
    status: str
    risk_signal_count: int
    evidence: List[EvidenceOut]
    hypotheses: List[HypothesisOut]
    clarification_question: Optional[str]
    plans: List[PlanOut]
    committed_actions: Dict[str, str]
    escalation_round: int
    plan_revision_round: int
    # Lets the UI disable "request different plans" at the cap; past it, a revise turns into a reject
    # that looked identical to a normal rejection.
    max_plan_revision_rounds: int
    # Lets the UI say "Attempt 2 of 3" instead of a bare, meaningless "escalation round 1" — a reader
    # can't tell how much runway is left from a counter with no denominator.
    max_escalation_rounds: int
    token_usage: int
    # Real measured spend for this thread (see `_call_model`) — "what did this one check cost?"
    cost_usd: float = 0.0
    error: Optional[str]
    # Only when escalated. Structured per-round cards (see build_escalation_summary); one paragraph
    # was unreadable.
    escalation_rounds: List[EscalationRoundOut] = Field(default_factory=list)
    escalation_still_at_risk_reasons: List[str] = Field(default_factory=list)
    # Empty: every flagged issue was acted on, so what's left is real engineering work. Non-empty:
    # these issues were never acted on (e.g. budget ran out) — a real gap.
    escalation_unaddressed_issue_keys: List[str] = Field(default_factory=list)
    # True when this status was reached without confirming the index caught up with our writes.
    # Staleness only makes things look more at risk, never less; shown so a post-approval
    # "escalated" isn't confusing.
    index_catch_up_timed_out: bool = False
    # The plan actually approved and executing (after any human edits). `plans` only holds the
    # original proposals, so after an edit the UI's checkmarks and action text didn't match what was
    # sent to Jira. Set once a plan is approved; None while still choosing.
    approved_plan: Optional[PlanOut] = None
    # plan_node's direct reply to what the human just said — what changed, or why it couldn't. A note
    # that couldn't change anything used to just vanish. Only for the plans it's returned with.
    note_response: Optional[str] = None
    # Whether there's crashed work to continue (from `is_resumable_after_a_crash`, not a status string:
    # a clarify pause also shows "diagnosing" but must not be resumed this way). A crash at
    # "diagnosing" used to leave no Resume button. False when only `.values` is available (the
    # immediate /start and /clarify responses).
    resumable: bool = False


class ActiveRecoveryLookupResponse(_CamelModel):
    """Safe product-level answer to "can I start recovery for this sprint?".

    `busy` deliberately exposes neither the other user's checkpoint nor its thread id. The caller is
    already authorized for the sprint's space, so it is safe and useful to say that work is in
    progress while preserving per-workflow ownership.
    """

    availability: Literal["none", "resumable", "busy"]
    recovery: Optional[RecoveryStatusResponse] = None
    owner_username: Optional[str] = None
    current_status: Optional[str] = None
    # With "none": the sprint's last check was stopped, by whom and when. Without it, a stopped check
    # simply vanished and its owner couldn't tell why.
    stopped_by: Optional[str] = None
    stopped_at: Optional[str] = None


def _status_response(thread_id: str, values: dict, resumable: bool = False) -> RecoveryStatusResponse:
    return RecoveryStatusResponse(
        thread_id=thread_id,
        status=values.get("status", "diagnosing"),
        risk_signal_count=len(values.get("risk_signals") or []),
        evidence=[EvidenceOut(**e.model_dump()) for e in values.get("evidence") or []],
        hypotheses=[HypothesisOut(**h.model_dump()) for h in values.get("hypotheses") or []],
        clarification_question=values.get("clarification_question"),
        plans=[PlanOut(**p.model_dump()) for p in values.get("plans") or []],
        committed_actions=values.get("committed_actions") or {},
        escalation_round=values.get("escalation_round", 0),
        plan_revision_round=values.get("plan_revision_round", 0),
        max_plan_revision_rounds=values.get("max_plan_revision_rounds", 2),
        max_escalation_rounds=values.get("max_escalation_rounds", 2),
        token_usage=values.get("token_usage", 0),
        cost_usd=values.get("cost_usd", 0.0) or 0.0,
        error=values.get("error"),
        index_catch_up_timed_out=values.get("index_catch_up_timed_out", False),
        approved_plan=PlanOut(**values["approved_plan"].model_dump()) if values.get("approved_plan") else None,
        note_response=values.get("note_response"),
        resumable=resumable,
    )


async def _status_response_with_summary(thread_id: str, graph) -> RecoveryStatusResponse:
    """`_status_response`, plus the per-round summary when escalated (needs an async history walk).

    Read-only, so it's safe on every fetch, including reopening the modal later. Re-fetches state
    itself because callers' snapshots may predate a step that advanced it, and so `resumable` can be
    computed from the full snapshot.
    """
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    response = _status_response(thread_id, snap.values, resumable=is_resumable_after_a_crash(snap))
    if response.status == "escalated":
        summary = await build_escalation_summary(graph, thread_id)
        response.escalation_rounds = [EscalationRoundOut(**r) for r in summary["rounds"]]
        response.escalation_still_at_risk_reasons = summary["still_at_risk_reasons"]
        response.escalation_unaddressed_issue_keys = summary["unaddressed_issue_keys"]
    return response


def _sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


async def _stream_call(coro) -> AsyncIterator[str]:
    """Shared SSE progress stream for one graph call (same queue pattern as /ask/stream, via
    ON_STAGE_VAR since the graph is shared). Yields `stage` frames only; the caller sends the
    result/error frame, since only it knows the response shape.
    """
    stage_queue: asyncio.Queue[str] = asyncio.Queue()
    token = ON_STAGE_VAR.set(stage_queue.put_nowait)
    try:
        # using_tags around create_task (the task copies the context when created). Done here, the
        # one place all four SSE endpoints start graph work.
        with using_tags([FEATURE_SPRINT_RECOVERY]):
            task = asyncio.create_task(coro)
        while not task.done():
            try:
                label = await asyncio.wait_for(stage_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            yield _sse("stage", {"label": label})
        while not stage_queue.empty():
            yield _sse("stage", {"label": stage_queue.get_nowait()})
        task.result()  # re-raises whatever graph.ainvoke raised, for the caller's try/except to surface
    finally:
        ON_STAGE_VAR.reset(token)


async def _reject_if_already_active(request: Request, space_id: int, sprint_id: int) -> None:
    """Reject a second `/start` for a sprint that already has an unfinished workflow (409 with its id).

    Two starts close together (double click, two tabs, a retry) used to orphan one thread: still
    resumable, maybe waiting for approval, but unfindable via /by-sprint. Finished threads don't
    block.
    """
    db_url = getattr(request.app.state, "sprint_recovery_checkpoint_db_url", None)
    if not db_url:
        return
    existing_thread_id = await find_active_thread_id(db_url, space_id, sprint_id)
    if not existing_thread_id:
        return
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": existing_thread_id}})
    if snap.values and snap.values.get("status") not in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail="A Sprint Risk & Recovery check is already in progress for this sprint.",
        )


class StartRecoveryRequest(_CamelModel):
    space_id: int
    sprint_id: int
    sprint_name: str = Field(min_length=1)


# Starting or re-running a check is new AI work; clarify/decide/retry continue one.
_NEW_AI_WORK = [Depends(enforce_user_quota)]


@router.post("/start", response_model=RecoveryStatusResponse, dependencies=_NEW_AI_WORK)
async def start_recovery_endpoint(req: StartRecoveryRequest, request: Request) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    await _authorize_space_ids(request, [req.space_id])
    await _reject_if_already_active(request, req.space_id, req.sprint_id)
    graph = _graph(request)
    user_id, username = _caller_identity(request)
    thread_id = str(uuid.uuid4())
    await _register_active_thread_if_configured(request, req.space_id, req.sprint_id, thread_id)
    settings = request.app.state.settings
    initial = initial_recovery_state(
        req.space_id, req.sprint_id, req.sprint_name, user_id, username,
        max_clarification_rounds=settings.sprint_recovery_max_clarification_rounds,
        max_escalation_rounds=settings.sprint_recovery_max_escalation_rounds,
        max_token_budget=settings.sprint_recovery_max_token_budget,
        max_plan_revision_rounds=settings.sprint_recovery_max_plan_revision_rounds,
    )
    config = {"configurable": {"thread_id": thread_id}}
    await _tagged(FEATURE_SPRINT_RECOVERY, graph.ainvoke(initial, config=config))
    snap = await graph.aget_state(config)
    return _status_response(thread_id, snap.values)


@router.post("/start/stream", dependencies=_NEW_AI_WORK)
async def start_recovery_stream_endpoint(req: StartRecoveryRequest, request: Request) -> StreamingResponse:
    """SSE version of POST /start: `stage` events as diagnose and plan (the two LLM calls) run."""
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    await _authorize_space_ids(request, [req.space_id])
    await _reject_if_already_active(request, req.space_id, req.sprint_id)
    graph = _graph(request)
    user_id, username = _caller_identity(request)
    thread_id = str(uuid.uuid4())
    await _register_active_thread_if_configured(request, req.space_id, req.sprint_id, thread_id)
    settings = request.app.state.settings
    initial = initial_recovery_state(
        req.space_id, req.sprint_id, req.sprint_name, user_id, username,
        max_clarification_rounds=settings.sprint_recovery_max_clarification_rounds,
        max_escalation_rounds=settings.sprint_recovery_max_escalation_rounds,
        max_token_budget=settings.sprint_recovery_max_token_budget,
        max_plan_revision_rounds=settings.sprint_recovery_max_plan_revision_rounds,
    )
    config = {"configurable": {"thread_id": thread_id}}

    async def generate() -> AsyncIterator[str]:
        try:
            async for event in _stream_call(graph.ainvoke(initial, config=config)):
                yield event
        except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same as ask_stream's backstop
            logger.exception("sprint recovery start stream failed")
            yield _sse("error", {"detail": _stream_error_detail(exc, "sprint recovery request failed")})
            return
        snap = await graph.aget_state(config)
        yield _sse("result", _status_response(thread_id, snap.values).model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


# Finished: /by-sprint treats these as "no active thread". Their history is still reachable by
# thread_id; this only stops automatic resume.
_TERMINAL_STATUSES = {"recovered", "escalated", "rejected", "no_risk_found"}


@router.get("/by-sprint", response_model=ActiveRecoveryLookupResponse)
async def find_active_recovery_endpoint(
    space_id: int, sprint_id: int, request: Request,
) -> ActiveRecoveryLookupResponse:
    """Find this sprint's unfinished workflow, so the UI can resume after losing the thread id (closed
    modal, reload, crash). Called on mount, before offering "Analyze Sprint Health".

    Distinguishes "yours, resumable" from "busy, someone else's", so null doesn't mean both. Uses
    active_threads.py because the checkpoint tables have no space/sprint columns.
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    await _authorize_space_ids(request, [space_id])
    db_url = getattr(request.app.state, "sprint_recovery_checkpoint_db_url", None)
    if not db_url:
        return ActiveRecoveryLookupResponse(availability="none")
    thread_id = await find_active_thread_id(db_url, space_id, sprint_id)
    if not thread_id:
        return ActiveRecoveryLookupResponse(availability="none")
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        return ActiveRecoveryLookupResponse(availability="none")
    if snap.values.get("status") in _TERMINAL_STATUSES:
        if snap.values.get("stopped_by"):
            return ActiveRecoveryLookupResponse(
                availability="none", owner_username=snap.values.get("username") or None,
                stopped_by=snap.values["stopped_by"], stopped_at=getattr(snap, "created_at", None),
            )
        return ActiveRecoveryLookupResponse(availability="none")
    try:
        await _authorize_workflow_state(request, snap.values)
    except HTTPException as exc:
        if exc.status_code == 403:
            # Say "busy" without revealing the other user's checkpoint, identity or thread id.
            return ActiveRecoveryLookupResponse(
                availability="busy",
                owner_username=snap.values.get("username") or None,
                current_status=snap.values.get("status") or None,
            )
        raise
    return ActiveRecoveryLookupResponse(
        availability="resumable",
        recovery=await _status_response_with_summary(thread_id, graph),
    )


class StopRecoveryRequest(_CamelModel):
    space_id: int
    sprint_id: int


@router.post("/by-sprint/stop", response_model=ActiveRecoveryLookupResponse)
async def stop_active_recovery_endpoint(req: StopRecoveryRequest, request: Request) -> ActiveRecoveryLookupResponse:
    """Stop this sprint's unfinished workflow, so the sprint is free for a new check.

    Any member of the space may, not only the owner: otherwise a check whose owner is away holds the
    sprint for everyone. Keyed by sprint because a teammate's "busy" view never sees the thread id.
    A run that is mid-step when this lands can still write a later checkpoint; stop again once it
    pauses.
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    await _authorize_space_ids(request, [req.space_id])
    db_url = getattr(request.app.state, "sprint_recovery_checkpoint_db_url", None)
    if not db_url:
        return ActiveRecoveryLookupResponse(availability="none")
    thread_id = await find_active_thread_id(db_url, req.space_id, req.sprint_id)
    if not thread_id:
        return ActiveRecoveryLookupResponse(availability="none")
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values or snap.values.get("status") in _TERMINAL_STATUSES:
        return ActiveRecoveryLookupResponse(availability="none")
    _, username = _caller_identity(request)
    await stop_recovery(graph, thread_id, stopped_by=username or "a teammate")
    logger.info(
        "sprint-recovery stopped",
        extra={"thread_id": thread_id, "stopped_by": username, "owner": snap.values.get("username")},
    )
    return ActiveRecoveryLookupResponse(availability="none")


@router.get("/{thread_id}", response_model=RecoveryStatusResponse)
async def get_recovery_status_endpoint(thread_id: str, request: Request) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    return await _status_response_with_summary(thread_id, graph)


class ClarifyRequest(_CamelModel):
    answer: str = Field(min_length=1)


@router.post("/{thread_id}/clarify", response_model=RecoveryStatusResponse)
async def clarify_recovery_endpoint(thread_id: str, req: ClarifyRequest, request: Request) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if not snap.values.get("clarification_question"):
        raise HTTPException(status_code=409, detail=f"sprint-recovery {thread_id} is not awaiting clarification")
    await _tagged(FEATURE_SPRINT_RECOVERY, graph.ainvoke(Command(resume={"answer": req.answer}), config=config))
    final = await graph.aget_state(config)
    return _status_response(thread_id, final.values)


@router.post("/{thread_id}/clarify/stream")
async def clarify_recovery_stream_endpoint(thread_id: str, req: ClarifyRequest, request: Request) -> StreamingResponse:
    """SSE version of POST /{thread_id}/clarify: the answer goes back into diagnose, which may chain
    into plan — up to two LLM calls behind one request.
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if not snap.values.get("clarification_question"):
        raise HTTPException(status_code=409, detail=f"sprint-recovery {thread_id} is not awaiting clarification")
    sprint_id = snap.values["sprint_id"]

    async def generate() -> AsyncIterator[str]:
        try:
            async for event in _stream_call(graph.ainvoke(Command(resume={"answer": req.answer}), config=config)):
                yield event
        except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same as ask_stream's backstop
            logger.exception("sprint recovery clarification stream failed")
            yield _sse("error", {"detail": _stream_error_detail(exc, "sprint recovery request failed")})
            return
        final = await graph.aget_state(config)
        yield _sse("result", _status_response(thread_id, final.values).model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


class ActionIn(_CamelModel):
    action_type: Literal["link_dependency", "change_priority", "move_out_of_sprint", "add_comment"]
    target_issue_key: str
    depends_on_issue_key: Optional[str] = None
    new_priority: Optional[str] = None
    comment_body: Optional[str] = None


class DecisionRequest(_CamelModel):
    decision: Literal["approve", "edit", "reject", "revise"]
    # Needed for approve/edit (which plan). Not for reject/revise, which turn down all of them.
    plan_id: Optional[str] = None
    actions: Optional[List[ActionIn]] = None
    # Required for revise: what a human wants changed, folded into plan_node's next prompt verbatim —
    # see approval_node's docstring for why this loops back into "plan" instead of ending the workflow.
    feedback: Optional[str] = None


def _decision_resume_payload(req: DecisionRequest) -> dict:
    """Validate a decision and build its resume payload, shared by the plain and SSE endpoints."""
    if req.decision in ("approve", "edit") and not req.plan_id:
        raise HTTPException(status_code=422, detail=f"decision={req.decision} requires plan_id")
    if req.decision == "edit" and not req.actions:
        raise HTTPException(status_code=422, detail="decision=edit requires actions")
    if req.decision == "revise" and not (req.feedback and req.feedback.strip()):
        raise HTTPException(status_code=422, detail="decision=revise requires feedback")

    resume_payload: dict = {"decision": req.decision}
    if req.plan_id:
        resume_payload["plan_id"] = req.plan_id
    if req.decision == "edit":
        resume_payload["actions"] = [a.model_dump() for a in req.actions]
    if req.decision == "revise":
        resume_payload["feedback"] = req.feedback
    return resume_payload


def _require_awaiting_approval(thread_id: str, snap_values: dict) -> None:
    if not snap_values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    if snap_values.get("status") != "awaiting_plan_approval":
        raise HTTPException(
            status_code=409,
            detail=f"sprint-recovery {thread_id} is not awaiting plan approval (status={snap_values.get('status')})",
        )


@router.post("/{thread_id}/decision", response_model=RecoveryStatusResponse)
async def submit_recovery_decision_endpoint(
    thread_id: str, req: DecisionRequest, request: Request
) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    _require_awaiting_approval(thread_id, snap.values)
    resume_payload = _decision_resume_payload(req)

    await _tagged(FEATURE_SPRINT_RECOVERY, graph.ainvoke(Command(resume=resume_payload), config=config))
    final = await graph.aget_state(config)
    final_values = await _auto_reevaluate_after_commit(graph, thread_id, final.values)
    await _notify_if_escalated(request, thread_id, snap.values["sprint_id"], final_values)
    _track_index_lag_if_timed_out(final_values)
    return await _status_response_with_summary(thread_id, graph)


@router.post("/{thread_id}/decision/stream")
async def submit_recovery_decision_stream_endpoint(
    thread_id: str, req: DecisionRequest, request: Request
) -> StreamingResponse:
    """SSE version of POST /{thread_id}/decision. "revise" calls the LLM; "approve" can too, if the
    automatic re-check finds the sprint still at risk and plans again. Both stream on one connection.
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    _require_awaiting_approval(thread_id, snap.values)
    resume_payload = _decision_resume_payload(req)
    sprint_id = snap.values["sprint_id"]

    async def generate() -> AsyncIterator[str]:
        try:
            async for event in _stream_call(graph.ainvoke(Command(resume=resume_payload), config=config)):
                yield event
            final = await graph.aget_state(config)
            if final.values.get("status") == "waiting_reevaluation":
                async for event in _stream_call(
                    trigger_reevaluation(graph, thread_id, source="auto_after_approve")
                ):
                    yield event
        except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same as ask_stream's backstop
            logger.exception("sprint recovery decision stream failed")
            yield _sse("error", {"detail": _stream_error_detail(exc, "sprint recovery request failed")})
            return
        final = await graph.aget_state(config)
        await _notify_if_escalated(request, thread_id, sprint_id, final.values)
        _track_index_lag_if_timed_out(final.values)
        yield _sse("result", (await _status_response_with_summary(thread_id, graph)).model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/{thread_id}/retry", response_model=RecoveryStatusResponse)
async def retry_recovery_endpoint(thread_id: str, request: Request) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if not is_resumable_after_a_crash(snap):
        raise HTTPException(
            status_code=409,
            detail=f"sprint-recovery {thread_id} has nothing to resume (status={snap.values.get('status')})",
        )
    await _tagged(FEATURE_SPRINT_RECOVERY, retry_recovery_commit(graph, thread_id))
    final = await graph.aget_state(config)
    return _status_response(thread_id, final.values)


@router.post(
    "/{thread_id}/trigger-reevaluation", response_model=RecoveryStatusResponse, dependencies=_NEW_AI_WORK,
)
async def trigger_reevaluation_endpoint(thread_id: str, request: Request) -> RecoveryStatusResponse:
    """Manual re-check, for when the automatic one after approval didn't run (see
    `_auto_reevaluate_after_commit`).
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if snap.values.get("status") != "waiting_reevaluation":
        # Expected race, not an error: the automatic re-check, or another tab, may have resumed the
        # pause first. Return the current state (a 409 showed a raw error for a finished action).
        return await _status_response_with_summary(thread_id, graph)
    await _tagged(FEATURE_SPRINT_RECOVERY, trigger_reevaluation(graph, thread_id, source="manual"))
    final = await graph.aget_state(config)
    await _notify_if_escalated(request, thread_id, snap.values["sprint_id"], final.values)
    _track_index_lag_if_timed_out(final.values)
    return await _status_response_with_summary(thread_id, graph)


@router.post("/{thread_id}/trigger-reevaluation/stream", dependencies=_NEW_AI_WORK)
async def trigger_reevaluation_stream_endpoint(thread_id: str, request: Request) -> StreamingResponse:
    """SSE version of POST /{thread_id}/trigger-reevaluation. Re-evaluating is fast (no LLM), but a
    still-at-risk sprint goes straight into plan_node, which is what the stage events are for.
    """
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    sprint_id = snap.values["sprint_id"]

    if snap.values.get("status") != "waiting_reevaluation":
        # Same benign race as the non-streaming endpoint above — see its comment. Answered as a
        # normal `result` frame so the client's stream handler needs no special case for it.
        already = await _status_response_with_summary(thread_id, graph)

        async def already_done() -> AsyncIterator[str]:
            yield _sse("result", already.model_dump(by_alias=True))

        return StreamingResponse(already_done(), media_type="text/event-stream")

    async def generate() -> AsyncIterator[str]:
        try:
            async for event in _stream_call(trigger_reevaluation(graph, thread_id, source="manual")):
                yield event
        except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same as ask_stream's backstop
            yield _sse("error", {"detail": _stream_error_detail(exc, str(exc))})
            return
        final = await graph.aget_state(config)
        await _notify_if_escalated(request, thread_id, sprint_id, final.values)
        _track_index_lag_if_timed_out(final.values)
        yield _sse("result", (await _status_response_with_summary(thread_id, graph)).model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


class CheckpointOut(_CamelModel):
    checkpoint_id: str
    next_node: Optional[str]
    status: Optional[str]
    # When this checkpoint was written, so the time-travel list has an order.
    created_at: Optional[str] = None
    # Which action this step applied, for `commit_one_action` rows (otherwise a 3-action plan shows
    # three identical "Applying changes in Jira…" rows). None elsewhere.
    detail: Optional[str] = None
    # Jira actions committed after this checkpoint. A rewind doesn't undo them in Jira, but the rewound
    # thread forgets them (and might re-raise a priority already raised). Listed by name, since the
    # next question is always "which ones?". Empty for a clean rewind.
    real_actions_committed_after: List[str] = Field(default_factory=list)
    # What was being decided here (plans on the table, or the question), so revision 1 and revision 3
    # don't both just say "Waiting for your decision". None for other rows.
    revision_summary: Optional[str] = None
    # The human note that led to this decision. Notes were recorded and used, but invisible in the
    # timeline, so a plan that ignored "add 2 engineers" looked like the note was thrown away.
    triggered_by_note: Optional[str] = None


def _notes_that_triggered_each_revision(history_newest_first: list) -> Dict[str, str]:
    """Map each decision point to the human note that led to it (a clarification answer or a rewind
    note — both are stored in `clarification_answer`).

    Attached to the resulting decision, not the checkpoint where the note appears (that one is hidden
    from the timeline). Detects changes rather than reading the field, since it persists and would
    otherwise be claimed by every later revision.
    """
    result: Dict[str, str] = {}
    pending: Optional[str] = None
    previous_answer = None
    for snap in reversed(history_newest_first):  # oldest first
        answer = snap.values.get("clarification_answer")
        if answer and answer != previous_answer:
            pending = answer
        previous_answer = answer
        next_node = snap.next[0] if snap.next else None
        if next_node in ("approval", "clarify") and pending:
            result[snap.config["configurable"]["checkpoint_id"]] = pending
            pending = None
    return result


def _revision_summary(values: dict, next_node: Optional[str]) -> Optional[str]:
    if next_node == "approval":
        plans = values.get("plans") or []
        if not plans:
            return None
        descriptions = [f'"{p.name}" ({len(p.actions)} action{"s" if len(p.actions) != 1 else ""})' for p in plans]
        if len(descriptions) == 1:
            return f"Plan: {descriptions[0]}"
        return f"{len(descriptions)} plans to choose from: " + "; ".join(descriptions)
    if next_node == "clarify":
        question = values.get("clarification_question")
        return f'Question: "{question}"' if question else None
    return None


@router.get("/{thread_id}/history", response_model=List[CheckpointOut])
async def recovery_history_endpoint(thread_id: str, request: Request) -> List[CheckpointOut]:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    history = await list_checkpoint_history(graph, thread_id)
    history = [s for s in history if s.values]  # drop the empty pre-START snapshot
    details = _committing_step_details(history)
    after = _real_actions_committed_after(history)
    notes = _notes_that_triggered_each_revision(history)
    return [
        CheckpointOut(
            checkpoint_id=s.config["configurable"]["checkpoint_id"],
            next_node=s.next[0] if s.next else None,
            status=s.values.get("status"),
            created_at=getattr(s, "created_at", None),
            detail=details.get(s.config["configurable"]["checkpoint_id"]),
            real_actions_committed_after=after.get(s.config["configurable"]["checkpoint_id"], []),
            revision_summary=_revision_summary(s.values, s.next[0] if s.next else None),
            triggered_by_note=notes.get(s.config["configurable"]["checkpoint_id"]),
        )
        for s in history
    ]


def _real_actions_committed_after(history_newest_first: list) -> Dict[str, List[str]]:
    """For each checkpoint, the Jira actions committed after it (oldest first) — what a rewind there
    would make the thread forget (the writes stay in Jira).

    Walks every commit step, because `committed_actions` resets each escalation round. History comes
    newest-first, so each newly found step's actions are prepended.
    """
    result: Dict[str, List[str]] = {}
    running: List[str] = []
    for i, snap in enumerate(history_newest_first):
        result[snap.config["configurable"]["checkpoint_id"]] = list(running)
        this_step = _actions_landing_at(snap, history_newest_first, i)
        running = this_step + running
    return result


def _actions_landing_at(snap, history_newest_first: list, i: int) -> List[str]:
    """Actions this checkpoint added, by diffing `committed_actions` with the next-older one (i+1).

    A diff, not `status == "committing"`: the step that commits a plan's LAST action has status
    "waiting_reevaluation", so filtering by status hid every plan's final action (a 3-action plan
    showed 2). Also handles the reset to {} between rounds.
    """
    plan = snap.values.get("approved_plan")
    if plan is None:
        return []
    prev = history_newest_first[i + 1] if i + 1 < len(history_newest_first) else None
    before = set((prev.values.get("committed_actions") or {}).keys()) if prev else set()
    after = set((snap.values.get("committed_actions") or {}).keys())
    return [
        _describe_action(plan.actions[int(idx_str)])
        for idx_str in sorted(after - before, key=int)
        if 0 <= int(idx_str) < len(plan.actions)
    ]


def _committing_step_details(history_newest_first: list) -> Dict[str, str]:
    """Map each checkpoint that applied an action to that action, so the timeline can name it. One
    action per step, so a step adding more than one is ambiguous and gets no detail. Uses the same diff
    as `_actions_landing_at`.
    """
    result: Dict[str, str] = {}
    for i, snap in enumerate(history_newest_first):
        landed = _actions_landing_at(snap, history_newest_first, i)
        if len(landed) == 1:
            result[snap.config["configurable"]["checkpoint_id"]] = landed[0]
    return result


class TimeTravelRequest(_CamelModel):
    checkpoint_id: str = Field(min_length=1)
    note: str = Field(min_length=1)


@router.post("/{thread_id}/time-travel", response_model=RecoveryStatusResponse)
async def time_travel_endpoint(thread_id: str, req: TimeTravelRequest, request: Request) -> RecoveryStatusResponse:
    _tag_entrypoint_span(request, FEATURE_SPRINT_RECOVERY)
    graph = _graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no sprint-recovery workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    # Enforce the rewind rule on the server. The UI only disabled the option, so a stale frontend or
    # direct API call could rewind past 6 committed Jira writes and reset "Attempt 3 of 3" to 1.
    history = await list_checkpoint_history(graph, thread_id)
    history = [s for s in history if s.values]
    if req.checkpoint_id != _newest_revisable_checkpoint_id(history):
        already_committed = _real_actions_committed_after(history).get(req.checkpoint_id, [])
        if already_committed:
            detail = (
                f"can't rewind to checkpoint {req.checkpoint_id}: this thread already sent "
                f"{len(already_committed)} real action(s) to Jira after that point "
                f"({'; '.join(already_committed)}) — those changes stay in Jira, but rewinding here "
                "would make this check forget it made them."
            )
        else:
            detail = (
                f"can't rewind to checkpoint {req.checkpoint_id}: only the decision this check is "
                "currently waiting on can be revised. Anything older has already been superseded by a "
                "newer plan or question, and rewinding to it would discard everything in between "
                "without producing an outcome different from revising the current one."
            )
        raise HTTPException(status_code=409, detail=detail)
    result = await _tagged(
        FEATURE_SPRINT_RECOVERY,
        time_travel_resume(graph, thread_id, req.checkpoint_id, req.note),
    )
    return await _status_response_with_summary(thread_id, graph)


def _newest_revisable_checkpoint_id(history_newest_first: list) -> Optional[str]:
    """The one checkpoint /time-travel accepts: the decision currently being waited on (next is
    approval or clarify), and only if nothing was committed after it. None otherwise.

    Only one target: rewinding re-enters via clarify and diagnose re-reads live data whichever
    checkpoint is named, so older targets give the same result while silently dropping the revisions
    in between. Offering them was a false choice.
    """
    after = _real_actions_committed_after(history_newest_first)
    for snap in history_newest_first:  # newest first
        next_node = snap.next[0] if snap.next else None
        if next_node not in ("approval", "clarify"):
            continue
        checkpoint_id = snap.config["configurable"]["checkpoint_id"]
        return None if after.get(checkpoint_id) else checkpoint_id
    return None
