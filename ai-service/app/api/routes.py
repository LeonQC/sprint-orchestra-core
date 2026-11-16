"""HTTP surface: the agentic Q&A endpoint, plus a health check."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Annotated, AsyncIterator, Callable, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from openinference.instrumentation import using_tags
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic.alias_generators import to_camel

from app import degradation
from app.agent.crag_loop import is_answer_cacheable
from app.agent.history_summary import (
    SUMMARY_MAX_CHARS, CompactHistory, HistorySummary, apply_summary, valid_summary,
)
from app.agent.mode_router import ModeDecision, ModeRouter, manual_mode
from app.investigation.graph import MultiAgentRag
from app.feedback import attach_feedback_receipt
from app.auth.space_membership import SpaceMembershipError
from app.auth.user_quota import AUTHENTICATED_USER_SCOPE_KEY, enforce_user_quota
from app.comment_digest.schemas import CommentIn
from app.comment_digest.service import summarize_comments
from app.description_review.service import review_description
from app.precedent.service import find_precedents
from app.drafting.service import draft_task
from app.llm.circuit_breaker import MODEL_UNAVAILABLE_MESSAGE, is_model_unavailable
from app.llm.pricing import resolve_model
from app.llm.types import Usage
from app.llm.usage_ledger import (
    FEATURE_ASK_AI,
    FEATURE_DRAFT_TASK,
    FEATURE_IMPROVE_DESCRIPTION,
    FEATURE_LINK_SIMILAR_WORK_ITEMS,
    FEATURE_PLAN_EPIC,
    FEATURE_PLAN_EPIC_REFINE,
    FEATURE_SEMANTIC_SEARCH,
    FEATURE_SPRINT_PACE,
    FEATURE_SUMMARIZE_COMMENTS,
    Caller,
    ModelCharge,
    UsageLedger,
    caller_from_headers,
    charge_for,
)
from app.observability import (
    ASK_DEGRADED_TOTAL,
    CACHE_HITS_TOTAL,
    CACHE_LOOKUP_SECONDS,
    CACHE_WRITES_TOTAL,
    WATERMARK_FAILURES_TOTAL,
    get_request_id,
)
from app.planning.graph import plan_epic_multiagent
from app.planning.grounding import (
    GroundingContext, gather_exemplars, gather_requirement_exemplars, with_stated_requirements,
)
from app.planning.house_format import section_compliance
from app.planning.pattern_schemas import EvidenceAssessment
from app.planning.patterns import AssessmentResult, assess_history, estimate_confidence, historical_baseline
from app.planning.refine_intent import read_refine_intent
from app.planning.rollout_graph import ON_STAGE_VAR as ROLLOUT_ON_STAGE_VAR
from app.planning.rollout_graph import retry_rollout
from app.planning.rollout_schemas import initial_rollout_state
from app.planning.active_threads import (
    find_plan_epic_active_thread_id,
    register_plan_epic_active_thread,
)
from app.planning.schemas import EpicDraft, IssueDraft
from app.planning.service import (
    PlanGenerationError, allocate_sprints, plan_epic, refine_plan, validate_and_order,
)
from app.search.service import (
    DEFAULT_MIN_SCORE,
    RERANKED_MIN_SCORE,
    dedupe_by_issue,
    relevance_confidence,
)
from app.sprint_pace.schemas import FlaggedIssue, SprintStats
from app.sprint_pace.service import summarize_sprint_pace
from app.api.route_tracing import TracedRoute

logger = logging.getLogger(__name__)
router = APIRouter(route_class=TracedRoute)
_tracer = trace.get_tracer(__name__)


# Endpoints that start new AI work check the per-user limits; continuing a workflow doesn't.
_NEW_AI_WORK = [Depends(enforce_user_quota)]


@router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@router.get("/healthz/cache")
async def healthz_cache(request: Request) -> dict:
    """Report the cache settings in force and whether freshness can be checked right now.

    scripts/run_rag_eval.sh refuses to run while a cache is on, and this is the first thing to check
    when the hit rate is zero. `watermark_reachable` makes a live call.
    """
    settings = request.app.state.settings
    reachable = None
    if settings.cache_enabled:
        reachable = await _safe_watermark(request.app.state.index_watermark, [1]) is not None
    return {
        "cache_enabled": settings.cache_enabled,
        "similarity_threshold": settings.cache_similarity_threshold,
        "ttl_seconds": settings.cache_ttl_seconds,
        "max_entries_per_space": settings.cache_max_entries,
        "watermark_ttl_seconds": settings.cache_watermark_ttl_seconds,
        "watermark_reachable": reachable,
        # Part of every answer-cache key; changes whenever the code or a prompt does.
        "code_version": getattr(request.app.state.cache, "_code_version", None),
        "llm_response_cache_enabled": settings.llm_response_cache_enabled,
        # Not a cache setting, but evals check it here before an on/off run (get_section).
        "parent_child_tool_enabled": settings.parent_child_tool_enabled,
    }


@router.get("/stats")
async def stats(request: Request) -> dict:
    """Token usage and estimated cost since process start, plus which models produced it.

    Model names let an eval run record what answered it, since scores can differ a lot between
    models. They go through `resolve_model` because the config value is often a gateway alias that
    can be repointed later. Names only — no URLs or keys.
    """
    settings = request.app.state.settings
    router_model = (
        settings.agent_model if settings.router_llm_provider == "inherit" else settings.router_model
    )
    return {
        **request.app.state.stats.as_dict(),
        "models": {
            "answer": resolve_model(settings.agent_model),
            "answer_provider": settings.llm_provider,
            "router": resolve_model(router_model),
            "router_provider": settings.router_llm_provider,
        },
    }


class _CamelModel(BaseModel):
    """JSON in/out as camelCase — matches jira-backend's Jackson serialization (and, through it, the
    TypeScript frontend's convention) rather than Python's own snake_case. Same pattern
    vectorization-service's inbound Kafka messages already use (`app/models.py::_CamelModel` there)
    for the same reason: anything a JVM/TS consumer touches should speak camelCase at the wire, not
    make every caller translate.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ChatTurnIn(_CamelModel):
    role: Literal["user", "assistant"]
    content: str
    # The UI marks an assistant turn that was a clarifying question, so the agent asks at most once.
    clarification: bool = False


class HistorySummaryDto(_CamelModel):
    """A summary of the chat's earlier turns (app/agent/history_summary.py). The service returns one
    when it writes it; the caller sends it back with every later question in that chat."""
    text: str = Field(min_length=1, max_length=SUMMARY_MAX_CHARS)
    # How many history turns, counted from the start, the summary stands in for.
    covers_turns: int = Field(ge=1)


def _clarification_kwargs(history: List[ChatTurnIn]) -> dict:
    return {"clarification_asked": True} if any(t.clarification for t in history) else {}


class AskRequest(_CamelModel):
    question: str = Field(min_length=1)
    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    # Issue keys the UI already knows (e.g. "the title of this page"). Only narrows the question;
    # authorization still comes from the authenticated space_ids.
    attachment_issue_keys: List[str] = Field(default_factory=list, max_length=10)
    # Earlier turns' final text, oldest first. The frontend holds the conversation; this service
    # stays stateless.
    history: List[ChatTurnIn] = Field(default_factory=list)
    # "fast": one search agent. "deep": the router may use several (one per workspace, about 2x the
    # cost) when the question compares workspaces; otherwise it still answers with one.
    mode: Literal["fast", "deep"] = "fast"
    # The latest summary this chat got back from the service, if any. History is still sent whole;
    # the summary replaces only the turns it covers.
    history_summary: Optional[HistorySummaryDto] = None


class CitationOut(_CamelModel):
    issue_key: str
    chunk_type: str
    source_id: int
    content: str
    page_number: int | None = None
    provenance: dict = Field(default_factory=dict)


class StageTimingsOut(_CamelModel):
    """Wall-clock breakdown of one /ask call, for the demo UI's latency toggle — not a replacement
    for the Prometheus P95 histograms (agent_llm_call_seconds etc.), which answer "typical latency
    across all traffic"; this answers "where did *this* request's time actually go," including the
    cache-lookup step CragAgent itself never sees."""

    cache_lookup_ms: float
    retrieval_ms: float
    llm_ms: float
    total_ms: float


class ModelUsageOut(_CamelModel):
    # mode_router and history_summary run in routes.py, before the agent.
    role: Literal["router", "answer_agent", "mode_router", "history_summary"]
    provider: str
    model: str
    deployment: Literal["local", "hosted"]
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float
    pricing_status: Literal["free_local", "estimated", "unavailable"]


class ClarificationOptionOut(_CamelModel):
    """One candidate source the frontend can render as a picker button — see
    AgentAnswer.ClarificationOption for why `issue_key` is a plain (possibly empty) string."""

    label: str
    issue_key: str = ""


class AskResponse(_CamelModel):
    answer_id: Optional[str] = None
    feedback_receipt: Optional[str] = None
    answer: str
    abstained: bool
    # True when `answer` is a clarification question, not an answer — check this before rendering
    # `answer` as prose.
    needs_clarification: bool = False
    clarification_options: List[ClarificationOptionOut] = []
    retrieval_rounds: int
    queries_used: List[str]
    citations: List[CitationOut]
    # Text results of structured tools (query_issues etc.) this answer used, so evals can see facts
    # like "ATC-77 is blocked" that `citations` never carries.
    structured_evidence: List[str] = []
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float
    model_usage: List[ModelUsageOut] = Field(default_factory=list)
    route_source: Optional[str] = None
    route_tool: Optional[str] = None
    # Adaptive Ask AI path (app/agent/mode_router.py). None when an older cached blob is replayed.
    execution_mode: Optional[Literal["single", "multi"]] = None
    mode_route_source: Optional[str] = None
    mode_route_reason: Optional[str] = None
    # Set only when this request summarized earlier turns; the caller keeps it for the next question.
    history_summary: Optional[HistorySummaryDto] = None
    cache_hit: bool = False
    stage_timings: Optional[StageTimingsOut] = None
    # W3C traceparent of this request's span. The frontend sends it back as a `traceparent` header on
    # the next turn so a follow-up joins the same trace. Never cached.
    trace_parent: Optional[str] = None
    # Set when part of the pipeline ran in a reduced mode (app/degradation.py): BM25-only retrieval,
    # or no answer model at all. Degraded answers are never cached.
    degraded: bool = False
    degraded_reasons: List[str] = Field(default_factory=list)


def _tag_entrypoint_span(request: Request, feature: str, history_key: str = "") -> Caller:
    """Tag the inbound HTTP request span with the feature, session and user.

    Call first in a handler, while the current span is still the FastAPI server span. Context-based
    tagging (CallerAttributeMiddleware, `_tagged`) only reaches spans created later, and the server
    span already exists by then — so without this, `POST /ask` had no tag, user or session.

    Returns the Caller for reuse: with no session header and no history, a second
    `caller_from_headers` call would mint a different random session id.
    """
    caller = caller_from_headers(request.headers, history_key)
    span = trace.get_current_span()
    span.set_attribute(SpanAttributes.TAG_TAGS, [feature])
    span.set_attribute(SpanAttributes.SESSION_ID, caller.session_id)
    if caller.user_id:
        span.set_attribute(SpanAttributes.USER_ID, caller.user_id)
    return caller


def _current_traceparent() -> str:
    """W3C traceparent of the current span — call right after `_tag_entrypoint_span`.

    Each /ask turn is a separate HTTP request, so a clarification follow-up used to start a new trace.
    This goes back to the client as `trace_parent`; the frontend sends it as a `traceparent` header on
    the next turn and FastAPIInstrumentor makes it the parent. The old span can be long finished —
    only its ids matter.
    """
    ctx = trace.get_current_span().get_span_context()
    return f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"


def _first_user_turn(history: List[ChatTurnIn], question: str) -> str:
    """The conversation's opening question, used as the session key for its follow-ups.

    Falls back to `question` for turn 1 (empty history). Without that, turn 1 got a random session
    while turn 2 hashed the same question, so every conversation's first turn was orphaned.
    """
    return next((turn.content for turn in history if turn.role == "user"), question)


def _record_usage(
    request: Request,
    feature: str,
    charges: List[ModelCharge],
    detail: Optional[dict] = None,
    history_key: str = "",
    span=None,
    caller: Optional[Caller] = None,
) -> UsageLedger:
    """Record one AI feature call's cost against a user, session and rate card.

    Called even when the feature degraded — failed calls still burn tokens. Also the one place usage
    reaches GET /stats. A cache hit (no charges) goes to the trace but not to /stats, which counts LLM
    requests made, not questions answered.

    Pass the `caller` from `_tag_entrypoint_span` so both spans agree on the session id.
    """
    ledger = UsageLedger(
        feature=feature,
        caller=caller if caller is not None else caller_from_headers(request.headers, history_key),
        request_id=get_request_id() or "",
        charges=charges,
        detail=detail or {},
    )
    ledger.attach(span)
    if charges:
        request.app.state.stats.record(ledger.usage, ledger.cost_usd)
    return ledger


def _structured_output_charge(request: Request, result) -> ModelCharge:
    """Cost of one structured-output call (/draft-task, /plan-epic, /summarize-comments, ...).

    They share a client, model and result shape, so one adapter covers them all. No cache-write
    bucket: prompt caching is only set up on the CragAgent Anthropic path.
    """
    state = request.app.state
    return charge_for(
        "structured_output",
        state.model_provider,
        state.instructor_model,
        state.model_deployment,
        Usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_input_tokens=result.cached_input_tokens,
        ),
    )


def _agent_charges(model_usage) -> List[ModelCharge]:
    """CragAgent already split one `/ask` between its router and answer models; re-price each split
    from its token counts so the ledger carries the rate card, not just the totals."""
    return [
        charge_for(
            item.role, item.provider, item.model, item.deployment,
            Usage(
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                cache_read_input_tokens=item.cached_input_tokens,
                cache_creation_input_tokens=item.cache_write_tokens,
            ),
        )
        for item in model_usage
    ]


async def _tagged(feature: str, coro):
    """Run `coro` with `feature` as an OpenTelemetry tag on every span it creates.

    `ledger.attach()` tags only one span; the LLM generation spans under it (what Langfuse lists)
    stayed untagged. `using_tags` puts the tag in the context so SDK-created spans pick it up too.
    Done here, not in middleware, because the feature is only known after routing.
    """
    with using_tags([feature]):
        return await coro


def _attachment_scoped_question(question: str, issue_keys: List[str]) -> str:
    """Add caller-supplied attachment identity to the agent prompt, never answer content.

    The wording intentionally includes both the keys and ``attachments``: the existing deterministic
    router then selects ``get_issue_attachments`` and its code-owned input override extracts exactly
    these keys, bypassing similarity ranking for a source the caller already selected.
    """
    normalized = list(dict.fromkeys(key.strip().upper() for key in issue_keys if key.strip()))
    if not normalized:
        return question
    return f"Using only the attachments on {', '.join(normalized)}, {question}"



@router.get("/quota")
async def quota(request: Request) -> dict:
    """The signed-in user's AI limits and today's spend, for showing "used $0.40 of $2.00"."""
    user_quota = getattr(request.app.state, "user_quota", None)
    user_id = request.scope.get(AUTHENTICATED_USER_SCOPE_KEY)
    if user_quota is None or not user_id:
        return {"enabled": False}
    status = await user_quota.status(user_id)
    return {
        "enabled": True,
        "requestsPerMinute": status.requests_per_minute,
        "dailyBudgetUsd": status.daily_budget_usd,
        "spentTodayUsd": status.spent_today_usd,
        "resetsAt": status.resets_at,
    }

@router.post("/ask", response_model=AskResponse, dependencies=_NEW_AI_WORK)
async def ask(req: AskRequest, request: Request) -> AskResponse:
    caller = _tag_entrypoint_span(request, FEATURE_ASK_AI, _first_user_turn(req.history, req.question))
    trace_parent = _current_traceparent()
    await _authorize_space_ids(request, req.space_ids)
    notice = await _unselected_space_notice(request, req)
    if notice is not None:
        return notice.model_copy(update={"trace_parent": trace_parent})
    agent = request.app.state.agent
    cache = request.app.state.cache
    watermark_client = request.app.state.index_watermark
    route_start = time.perf_counter()

    # Mode first, then cache: entries are namespaced by execution_mode so a single-CRAG answer is
    # never served when the router chose multi (and the reverse).
    mode, investigator = await _choose_ask_mode(request, req)
    use_multi = investigator is not None

    # Cache key = (question, space_ids, data_version, execution_mode); data_version makes entries
    # unreachable after any indexing change (see semantic_cache.py).
    # Follow-ups are never written (meaningless without their thread) but can still hit an exact
    # string match ("show me that again") — no semantic match, since a paraphrase in a thread depends
    # on context. Attachment-scoped questions skip the cache entirely.
    cache_writable = not req.history and not req.attachment_issue_keys
    cache_readable = not req.attachment_issue_keys
    exact_only = bool(req.history)
    cache_ms = 0.0
    watermark = await _safe_watermark(watermark_client, req.space_ids) if cache_readable else None
    # No freshness token -> can't prove a cached answer is current, so always compute.
    cache_writable = cache_writable and watermark is not None
    cache_readable = cache_readable and watermark is not None
    if cache_readable:
        cache_start = time.perf_counter()
        cached = await _safe_cache_get(
            cache, req.question, req.space_ids, watermark, exact_only=exact_only,
            execution_mode=mode.mode,
        )
        cache_ms = (time.perf_counter() - cache_start) * 1000
        if cached is not None:
            CACHE_HITS_TOTAL.labels("hit").inc()
            # Replace the cached blob's timings with this request's own.
            timings = StageTimingsOut(
                cache_lookup_ms=round(cache_ms, 2), retrieval_ms=0.0, llm_ms=0.0,
                total_ms=round((time.perf_counter() - route_start) * 1000, 2),
            )
            # Still recorded (at $0) so the session shows the question and the cache's savings show up.
            _record_usage(
                request, FEATURE_ASK_AI, list(mode.charges),
                detail={"question": req.question, "cache_hit": True, "space_ids": req.space_ids,
                        "execution_mode": mode.mode},
                caller=caller,
            )
            return attach_feedback_receipt(
                request, req,
                AskResponse(**{**cached, "cache_hit": True, "stage_timings": timings,
                               "trace_parent": trace_parent, "execution_mode": mode.mode,
                               "mode_route_source": mode.source, "mode_route_reason": mode.reason}),
                watermark,
            )
        CACHE_HITS_TOTAL.labels("miss").inc()

    compact = await _tagged(FEATURE_ASK_AI, _compact_history(request, req))
    history = compact.turns
    try:
        # One parent span per /ask, so every LLM call in the loop nests under it instead of each one
        # becoming its own trace.
        span_name = "MultiAgentRag.ask" if use_multi else "CragAgent.ask"
        with degradation.collect() as degraded_reasons, _tracer.start_as_current_span(
            span_name,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.INPUT_VALUE: req.question,
                "execution_mode": mode.mode,
                "mode_route_source": mode.source,
            },
        ) as span:
            if use_multi:
                investigation = await _tagged(
                    FEATURE_ASK_AI,
                    investigator.ask(
                        _question_for_multi(req.question, history), req.space_ids,
                        space_labels=await _space_labels(request, req.space_ids),
                    ),
                )
                result = investigation.answer
                charges = list(mode.charges) + list(compact.charges) + investigation.charges
                if investigation.errors:
                    # The detailed errors are in the logs and on /investigate; one fixed reason keeps
                    # the Prometheus label set small.
                    degraded_reasons.add(degradation.MULTI_AGENT_PARTIAL)
            else:
                investigation = None
                result = await _tagged(FEATURE_ASK_AI, agent.ask(
                    _attachment_scoped_question(req.question, req.attachment_issue_keys),
                    req.space_ids,
                    history=history,
                    **_clarification_kwargs(req.history),
                    space_labels=await _space_labels(request, req.space_ids),
                ))
                charges = list(mode.charges) + list(compact.charges) + _agent_charges(result.model_usage)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.text)
            span.set_attribute("retrieval_rounds", result.retrieval_rounds)
            # On the agent span so the cost sits next to the question and answer. Must be inside the
            # `with`: an ended span ignores new attributes.
            _record_usage(
                request, FEATURE_ASK_AI, charges,
                detail={
                    "question": req.question,
                    "cache_hit": False,
                    "space_ids": req.space_ids,
                    "route_tool": result.route_tool,
                    "retrieval_rounds": result.retrieval_rounds,
                    "abstained": result.abstained,
                    "execution_mode": mode.mode,
                    "mode_route_source": mode.source,
                },
                caller=caller,
                span=span,
            )
    except Exception as exc:  # noqa: BLE001 - degrade to sources; 502 only if retrieval fails too
        # Reached after the gateway's own retries and provider fallback are exhausted.
        logger.exception("agent.ask() failed")
        response = await _retrieval_only_response(
            request, req, cache_ms, route_start, trace_parent, caller,
            list(mode.charges) + list(compact.charges),
        )
        if response is None:
            request_id = get_request_id()
            raise HTTPException(
                status_code=502,
                detail=f"AI request failed (request ID: {request_id})",
            ) from exc
        return attach_feedback_receipt(request, req, response, watermark)
    response = _build_ask_response(
        result, cache_ms, route_start, trace_parent, degraded_reasons, mode,
        extra_charges=list(mode.charges) + list(compact.charges),
    )
    await _cache_answer_if_allowed(
        cache, req, response, result, cache_writable, watermark, execution_mode=mode.mode,
    )
    # After the cache write: a summary belongs to this chat, never to a cached answer.
    response = _with_new_summary(response, compact)
    return attach_feedback_receipt(request, req, response, watermark)


#: Shown instead of a generated answer when every answer model failed. Says plainly that nothing was
#: generated, so a list of sources is never mistaken for the AI's conclusion.
MODEL_UNAVAILABLE_ANSWER = (
    "The AI assistant is temporarily unavailable, so this is not a generated answer. "
    "These are the most relevant items for your question:"
)
MODEL_UNAVAILABLE_NO_HITS = MODEL_UNAVAILABLE_MESSAGE


async def _retrieval_only_response(
    request: Request, req: "AskRequest", cache_ms: float, route_start: float, trace_parent: str,
    caller, charges: Optional[List[ModelCharge]] = None,
) -> Optional[AskResponse]:
    """The /ask response when the agent failed: retrieved sources, no generated answer.

    Search needs no answer model, so the user still gets somewhere to start. Returns None when
    retrieval fails too; the caller then answers 502.
    """
    settings = request.app.state.settings
    retrieval_start = time.perf_counter()
    with degradation.collect() as reasons:
        try:
            hits = await request.app.state.retrieval.search(
                req.question, req.space_ids, settings.retrieval_top_k
            )
        except Exception:  # noqa: BLE001 - nothing left to degrade to
            logger.exception("retrieval-only fallback failed")
            return None
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    reasons.add(degradation.ANSWER_MODEL_UNAVAILABLE)

    lines = []
    listed: set[str] = set()
    for hit in hits:
        if hit.issue_key in listed:
            continue
        listed.add(hit.issue_key)
        snippet = " ".join(hit.content.split())
        lines.append(f"- {hit.issue_key}: {snippet[:160]}{'…' if len(snippet) > 160 else ''}")
    answer = "\n".join([MODEL_UNAVAILABLE_ANSWER, *lines]) if lines else MODEL_UNAVAILABLE_NO_HITS

    for reason in reasons:
        ASK_DEGRADED_TOTAL.labels(reason).inc()
    _record_usage(
        request, FEATURE_ASK_AI, charges or [],
        detail={"question": req.question, "cache_hit": False, "space_ids": req.space_ids,
                "degraded_reasons": sorted(reasons)},
        caller=caller,
    )
    return AskResponse(
        answer=answer,
        abstained=False,
        retrieval_rounds=1,
        queries_used=[req.question],
        citations=[
            CitationOut(
                issue_key=h.issue_key, chunk_type=h.chunk_type, source_id=h.source_id,
                content=h.content, page_number=h.page_number, provenance=h.provenance,
            )
            for h in hits
        ],
        input_tokens=0,
        output_tokens=0,
        estimated_cost_usd=0.0,
        cache_hit=False,
        stage_timings=StageTimingsOut(
            cache_lookup_ms=round(cache_ms, 2), retrieval_ms=round(retrieval_ms, 2), llm_ms=0.0,
            total_ms=round((time.perf_counter() - route_start) * 1000, 2),
        ),
        trace_parent=trace_parent,
        degraded=True,
        degraded_reasons=sorted(reasons),
    )


def _stream_error_detail(exc: BaseException, fallback: str) -> str:
    """The `detail` of an SSE `error` frame: says so when the model is the cause, since that is the
    one failure where waiting and retrying is the right advice."""
    if isinstance(exc, PlanGenerationError):
        return str(exc)
    return MODEL_UNAVAILABLE_MESSAGE if is_model_unavailable(exc) else fallback


def _sse(event_name: str, data: dict) -> str:
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


def _build_ask_response(
    result, cache_ms: float, route_start: float, trace_parent: str, degraded_reasons: set[str],
    mode: ModeDecision | None = None, extra_charges: List[ModelCharge] = (),
) -> AskResponse:
    """The /ask response for a freshly computed answer (shared by /ask and /ask/stream).

    `extra_charges`: model calls made before the agent (mode router, history summary). Added to the
    usage the UI shows, so its cost matches what the ledger and the quota count.
    """
    for reason in degraded_reasons:
        ASK_DEGRADED_TOTAL.labels(reason).inc()
    stage_seconds = getattr(result, "stage_seconds", None) or {}
    stage_timings = StageTimingsOut(
        cache_lookup_ms=round(cache_ms, 2),
        retrieval_ms=round(stage_seconds.get("retrieval", 0.0) * 1000, 2),
        llm_ms=round(stage_seconds.get("llm", 0.0) * 1000, 2),
        total_ms=round((time.perf_counter() - route_start) * 1000, 2),
    )
    return AskResponse(
        answer=result.text,
        abstained=result.abstained,
        needs_clarification=result.needs_clarification,
        clarification_options=[
            ClarificationOptionOut(label=o.label, issue_key=o.issue_key)
            for o in result.clarification_options
        ],
        retrieval_rounds=result.retrieval_rounds,
        queries_used=result.queries_used,
        citations=[
            CitationOut(
                issue_key=c.issue_key, chunk_type=c.chunk_type, source_id=c.source_id,
                content=c.content, page_number=c.page_number,
                provenance=c.provenance,
            )
            for c in result.citations
        ],
        structured_evidence=result.structured_evidence,
        input_tokens=result.usage.input_tokens + sum(c.input_tokens for c in extra_charges),
        output_tokens=result.usage.output_tokens + sum(c.output_tokens for c in extra_charges),
        cached_input_tokens=(
            result.usage.cache_read_input_tokens + sum(c.cached_input_tokens for c in extra_charges)
        ),
        estimated_cost_usd=result.estimated_cost_usd + sum(c.cost_usd for c in extra_charges),
        model_usage=[
            ModelUsageOut(
                role=item.role,
                provider=item.provider,
                model=item.model,
                deployment=item.deployment,
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                cached_input_tokens=item.cached_input_tokens,
                estimated_cost_usd=item.estimated_cost_usd,
                pricing_status=item.pricing_status,
            )
            for item in result.model_usage
        ] + [
            ModelUsageOut(
                role=charge.role,
                provider=charge.provider,
                model=charge.model,
                deployment=charge.deployment,
                input_tokens=charge.input_tokens,
                output_tokens=charge.output_tokens,
                cached_input_tokens=charge.cached_input_tokens,
                estimated_cost_usd=charge.cost_usd,
                pricing_status=charge.pricing_status,
            )
            for charge in extra_charges
        ],
        route_source=result.route_source,
        route_tool=result.route_tool,
        execution_mode=mode.mode if mode else "single",
        mode_route_source=mode.source if mode else "disabled",
        mode_route_reason=mode.reason if mode else None,
        cache_hit=False,
        stage_timings=stage_timings,
        trace_parent=trace_parent,
        degraded=bool(degraded_reasons),
        degraded_reasons=sorted(degraded_reasons),
    )


def _question_for_multi(question: str, turns: List[dict]) -> str:
    """Multi's workers see only the question text, so a follow-up carries the earlier turns with it.

    `turns` is the compacted history (_compact_history), the same turns single gets.
    """
    if not turns:
        return question
    lines = []
    for turn in turns:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "Earlier in this chat:\n" + "\n".join(lines) + f"\n\nCurrent question: {question}"


async def _compact_history(
    request: Request, req: AskRequest, on_stage: Optional[Callable[[str], None]] = None,
) -> CompactHistory:
    """The chat history the model sees: long earlier turns summarized (app/agent/history_summary.py)."""
    history = [{"role": t.role, "content": t.content} for t in req.history]
    sent = req.history_summary
    summary = HistorySummary(sent.text, sent.covers_turns) if sent else None
    compactor = getattr(request.app.state, "history_compactor", None)
    if compactor is None:
        return CompactHistory(apply_summary(history, valid_summary(summary, len(history))))
    return await compactor.compact(history, summary, on_stage)


def _with_new_summary(response: AskResponse, compact: CompactHistory) -> AskResponse:
    if compact.new_summary is None:
        return response
    summary = HistorySummaryDto(
        text=compact.new_summary.text, covers_turns=compact.new_summary.covers_turns,
    )
    return response.model_copy(update={"history_summary": summary})


async def _choose_ask_mode(
    request: Request, req: AskRequest,
) -> tuple[ModeDecision, Optional[MultiAgentRag]]:
    """Single or multi for this /ask, plus the multi-agent runner (None means use CragAgent)."""
    mode = manual_mode(req.mode)
    if mode is None:
        mode_router: ModeRouter | None = getattr(request.app.state, "mode_router", None)
        if mode_router is None:
            return ModeDecision("single", "disabled", "mode router not installed"), None
        mode = await mode_router.choose(
            req.question,
            req.space_ids,
            has_history=bool(req.history),
            has_attachment_scope=bool(req.attachment_issue_keys),
        )
    if mode.mode == "single":
        return mode, None
    investigator = getattr(request.app.state, "investigator", None)
    if investigator is None:
        fallback = ModeDecision(
            "single", "fallback", "multi unavailable; using one search agent", mode.charges,
        )
        return fallback, None
    return mode, investigator


async def _cache_answer_if_allowed(
    cache, req: "AskRequest", response: AskResponse, result, cache_writable: bool, watermark,
    execution_mode: str = "single",
) -> None:
    """Write the answer to the cache when allowed, and count why or why not.

    Per-request fields (timings, trace_parent, feedback ids) are left out; they're replaced on a hit.
    """
    # A degraded answer would outlive the outage: the cache key only changes when the index does.
    used_live_data = bool(result.structured_evidence)
    cacheable = cache_writable and not response.degraded and is_answer_cacheable(
        result.route_tool, result.abstained, result.needs_clarification, used_live_data,
    )
    CACHE_WRITES_TOTAL.labels(
        "skipped_degraded" if cache_writable and response.degraded else
        _write_outcome(
            cache_writable, result.route_tool, result.abstained, result.needs_clarification,
            used_live_data,
        )
    ).inc()
    if cacheable:
        await _safe_cache_put(
            cache, req.question, req.space_ids,
            response.model_dump(exclude={"cache_hit", "stage_timings", "trace_parent", "answer_id", "feedback_receipt"}),
            watermark,
            execution_mode=execution_mode,
        )


@router.post("/ask/stream", dependencies=_NEW_AI_WORK)
async def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
    """SSE version of POST /ask: same agent call and final response, with progress shown as it happens.

    Events, in order:
      - `stage`: a progress label ("searching the knowledge base", ...) from `on_stage`.
      - `token`: a piece of the final answer as it's generated (`on_token`). Plain text, not partial
        JSON. Tool turns, routing, the verification rewrite and clarifications stream nothing.
      - `result`: the final AskResponse. Clients must replace the streamed text with `result.answer`,
        because verification can rewrite the draft, and citations/usage only exist here.

    `agent.ask()` runs as a background task while this generator drains the shared event queue, so
    events reach the client live.
    """
    caller = _tag_entrypoint_span(request, FEATURE_ASK_AI, _first_user_turn(req.history, req.question))
    trace_parent = _current_traceparent()
    await _authorize_space_ids(request, req.space_ids)
    agent = request.app.state.agent
    cache = request.app.state.cache
    watermark_client = request.app.state.index_watermark
    route_start = time.perf_counter()

    async def generate() -> AsyncIterator[str]:
        with degradation.collect() as degraded_reasons:
            async for frame in _generate(degraded_reasons):
                yield frame

    async def _generate(degraded_reasons: set[str]) -> AsyncIterator[str]:
        notice = await _unselected_space_notice(request, req)
        if notice is not None:
            notice = notice.model_copy(update={"trace_parent": trace_parent})
            yield _sse("result", notice.model_dump(by_alias=True))
            return
        # Same read/write contract as the non-streaming /ask handler — see its comment.
        mode, investigator = await _choose_ask_mode(request, req)
        use_multi = investigator is not None
        yield _sse("stage", {"label": (
            "deep investigation: one search agent per workspace" if use_multi else "using one search agent"
        )})

        cache_writable = not req.history and not req.attachment_issue_keys
        cache_readable = not req.attachment_issue_keys
        exact_only = bool(req.history)
        watermark = await _safe_watermark(watermark_client, req.space_ids) if cache_readable else None
        cache_writable = cache_writable and watermark is not None
        cache_readable = cache_readable and watermark is not None
        cache_ms = 0.0
        if cache_readable:
            cache_start = time.perf_counter()
            cached = await _safe_cache_get(
                cache, req.question, req.space_ids, watermark, exact_only=exact_only,
                execution_mode=mode.mode,
            )
            cache_ms = (time.perf_counter() - cache_start) * 1000
            if cached is not None:
                CACHE_HITS_TOTAL.labels("hit").inc()
                yield _sse("stage", {"label": "using a cached answer"})
                timings = StageTimingsOut(
                    cache_lookup_ms=round(cache_ms, 2), retrieval_ms=0.0, llm_ms=0.0,
                    total_ms=round((time.perf_counter() - route_start) * 1000, 2),
                )
                _record_usage(
                    request, FEATURE_ASK_AI, list(mode.charges),
                    detail={"question": req.question, "cache_hit": True, "space_ids": req.space_ids,
                            "execution_mode": mode.mode},
                    caller=caller,
                )
                payload = AskResponse(**{
                    **cached, "cache_hit": True, "stage_timings": timings, "trace_parent": trace_parent,
                    "execution_mode": mode.mode, "mode_route_source": mode.source,
                    "mode_route_reason": mode.reason,
                })
                payload = attach_feedback_receipt(request, req, payload, watermark)
                yield _sse("result", payload.model_dump(by_alias=True))
                return
            CACHE_HITS_TOTAL.labels("miss").inc()

        # One queue for both labels and tokens keeps them in the order they were produced.
        event_queue: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()

        def _emit(kind_payload: "tuple[str, str]") -> str:
            kind, payload = kind_payload
            if kind == "token":
                return _sse("token", {"text": payload})
            return _sse("stage", {"label": payload})

        # A task, so the "summarizing" label reaches the UI while the summary is being written.
        with using_tags([FEATURE_ASK_AI]):
            compact_task = asyncio.create_task(_compact_history(
                request, req, on_stage=lambda label: event_queue.put_nowait(("stage", label)),
            ))
        while not compact_task.done():
            try:
                item = await asyncio.wait_for(event_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            yield _emit(item)
        compact = compact_task.result()

        history = compact.turns

        # Open the span before create_task: the task copies the current context when it's created,
        # so LLM spans inside agent.ask() still nest under this one.
        with _tracer.start_as_current_span(
            "MultiAgentRag.ask" if use_multi else "CragAgent.ask",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
                SpanAttributes.INPUT_VALUE: req.question,
                "execution_mode": mode.mode,
                "mode_route_source": mode.source,
            },
        ) as span:
            # `using_tags` around create_task (not `_tagged` inside it), for the same reason: the tag
            # has to be in the context the task copies.
            with using_tags([FEATURE_ASK_AI]):
                if use_multi:
                    agent_task = asyncio.create_task(
                        investigator.ask(
                            _question_for_multi(req.question, history), req.space_ids,
                            on_stage=lambda label: event_queue.put_nowait(("stage", label)),
                            space_labels=await _space_labels(request, req.space_ids),
                        )
                    )
                else:
                    agent_task = asyncio.create_task(
                        agent.ask(
                            _attachment_scoped_question(req.question, req.attachment_issue_keys),
                            req.space_ids,
                            history=history,
                            **_clarification_kwargs(req.history),
                            on_stage=lambda label: event_queue.put_nowait(("stage", label)),
                            on_token=lambda text: event_queue.put_nowait(("token", text)),
                            space_labels=await _space_labels(request, req.space_ids),
                        )
                    )

            while not agent_task.done():
                try:
                    item = await asyncio.wait_for(event_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                yield _emit(item)
            # The task can finish before its last events were drained; flush them before the result.
            while not event_queue.empty():
                yield _emit(event_queue.get_nowait())

            try:
                raw = agent_task.result()
            except Exception as exc:  # noqa: BLE001 - see the non-streaming /ask handler's identical backstop
                logger.exception("agent.ask() failed (stream)")
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                fallback = await _retrieval_only_response(
                    request, req, cache_ms, route_start, trace_parent, caller,
                    list(mode.charges) + list(compact.charges),
                )
                if fallback is not None:
                    fallback = attach_feedback_receipt(request, req, fallback, watermark)
                    yield _sse("result", fallback.model_dump(by_alias=True))
                    return
                request_id = get_request_id()
                yield _sse(
                    "error",
                    {
                        "code": "AI_REQUEST_FAILED",
                        "detail": f"AI request failed (request ID: {request_id})",
                        "requestId": request_id,
                    },
                )
                return
            if use_multi:
                if raw.errors:
                    degraded_reasons.add(degradation.MULTI_AGENT_PARTIAL)
                result = raw.answer
                charges = list(mode.charges) + list(compact.charges) + raw.charges
            else:
                result = raw
                charges = list(mode.charges) + list(compact.charges) + _agent_charges(result.model_usage)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, result.text)
            span.set_attribute("retrieval_rounds", result.retrieval_rounds)
            # Explicit span, unlike everywhere else: this runs inside an async generator Starlette
            # iterates after the handler returned — see `UsageLedger.attach`.
            _record_usage(
                request, FEATURE_ASK_AI, charges,
                detail={
                    "question": req.question,
                    "cache_hit": False,
                    "space_ids": req.space_ids,
                    "route_tool": result.route_tool,
                    "retrieval_rounds": result.retrieval_rounds,
                    "abstained": result.abstained,
                    "streamed": True,
                    "execution_mode": mode.mode,
                    "mode_route_source": mode.source,
                },
                caller=caller,
                span=span,
            )

        response = _build_ask_response(
            result, cache_ms, route_start, trace_parent, degraded_reasons, mode,
            extra_charges=list(mode.charges) + list(compact.charges),
        )
        await _cache_answer_if_allowed(
            cache, req, response, result, cache_writable, watermark, execution_mode=mode.mode,
        )
        response = _with_new_summary(response, compact)
        response = attach_feedback_receipt(request, req, response, watermark)
        yield _sse("result", response.model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


class SemanticSearchRequest(_CamelModel):
    query: str = Field(min_length=1)
    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    limit: int = Field(10, ge=1, le=50)
    min_score: Optional[float] = Field(
        None, description="Relevance floor below which a result isn't returned at all, rather than "
                           "always handing back the top N regardless of actual relevance. No fixed "
                           "range: the meaningful scale depends on whether vectorization-service "
                           "reranks results (0-1 cosine similarity when it doesn't; an unbounded "
                           "cross-encoder logit when it does — see app/search/service.py). Null uses "
                           "the scale-appropriate service default, chosen automatically based on "
                           "whether this specific response was reranked."
    )


class SemanticSearchHitOut(_CamelModel):
    issue_id: int
    issue_key: str
    # Raw score: cosine (0-1) normally, an unbounded logit when reranked. For debugging and custom
    # thresholds only — not for display.
    score: float
    # The same signal as a real 0-1 confidence — show this one. Only the server knows the scale
    # (leaving it to callers once produced a "640% match").
    confidence: float
    snippet: str


class SemanticSearchResponse(_CamelModel):
    results: List[SemanticSearchHitOut]


@router.post("/search", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(req: SemanticSearchRequest, request: Request) -> SemanticSearchResponse:
    """Retrieval only, no LLM. Returns issues (best chunk per issue), not passages — what "find related
    issues" and duplicate detection need.

    Vector mode on purpose: its score is real cosine similarity, which `min_score` can filter on (an
    RRF score can't). If vectorization-service reranked, the score is a cross-encoder logit instead
    (seen live: 5.4 / 1.7), so the matching floor `RERANKED_MIN_SCORE` is used.
    """
    # The embedding call is still billable, so tag it like any other feature.
    _tag_entrypoint_span(request, FEATURE_SEMANTIC_SEARCH)
    await _authorize_space_ids(request, req.space_ids)
    retrieval = request.app.state.retrieval
    try:
        # Fetch extra chunks because several can belong to one issue and some get filtered out.
        # Capped at 50, vectorization-service's own limit — without the cap, limit=20 asked for 100
        # and got a 502.
        fetch_limit = min(max(req.limit * 5, 30), 50)
        result = await _tagged(
            FEATURE_SEMANTIC_SEARCH,
            retrieval.search_with_meta(req.query, req.space_ids, limit=fetch_limit, mode="vector"),
        )
    except Exception as exc:  # noqa: BLE001 - same clean-502 contract as /ask, don't leak internals
        logger.exception("semantic search failed")
        raise HTTPException(status_code=502, detail="search request failed") from exc

    if req.min_score is not None:
        effective_min_score = req.min_score
    elif result.reranked:
        effective_min_score = RERANKED_MIN_SCORE
    else:
        effective_min_score = DEFAULT_MIN_SCORE
    ranked = dedupe_by_issue(result.hits, req.limit, min_score=effective_min_score)

    return SemanticSearchResponse(
        results=[
            SemanticSearchHitOut(
                issue_id=h.issue_id,
                issue_key=h.issue_key,
                score=h.score,
                confidence=relevance_confidence(h.score, result.reranked),
                snippet=h.content[:280],
            )
            for h in ranked
        ]
    )


class DraftTaskRequest(_CamelModel):
    description: str = Field(min_length=1, description="Rough, informal description of the work.")
    existing_labels: List[str] = Field(default_factory=list, description="Labels already in use, for the model to prefer reusing over inventing new ones.")


class TaskDraftOut(_CamelModel):
    title: str
    issue_type: str
    labels: List[str]
    estimate_story_points: Optional[int]
    estimate_rationale: Optional[str]
    dependencies: List[str]


class DraftTaskResponse(_CamelModel):
    draft: TaskDraftOut
    degraded: bool
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    #: Of `input_tokens`, the portion served from the provider's prompt cache (billed ~10%).
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0


@router.post("/draft-task", response_model=DraftTaskResponse, dependencies=_NEW_AI_WORK)
async def draft_task_endpoint(req: DraftTaskRequest, request: Request) -> DraftTaskResponse:
    """AI-assisted task creation: one structured-output call -> title/labels/estimate/dependencies.

    Never 500s on a model/provider failure — see app/drafting/service.py's degradation strategy.
    `degraded=True` tells the caller the draft is an unassisted fallback, not a model failure hidden
    behind a normal-looking response.
    """
    caller = _tag_entrypoint_span(request, FEATURE_DRAFT_TASK)
    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    result = await _tagged(FEATURE_DRAFT_TASK, draft_task(client, model, req.description, req.existing_labels))
    _record_usage(
        request, FEATURE_DRAFT_TASK, [_structured_output_charge(request, result)],
        detail={"degraded": result.degraded},
        caller=caller,
    )
    return DraftTaskResponse(
        draft=TaskDraftOut(**result.draft.model_dump()),
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


RequirementText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PlanEpicRequest(_CamelModel):
    proposal: str = Field(min_length=1, description="High-level description of the work to plan.")
    existing_labels: List[str] = Field(default_factory=list, description="Labels already in use, for the model to prefer reusing over inventing new ones.")
    sprint_capacity_points: Optional[float] = Field(
        None, gt=0, allow_inf_nan=False,
        description="Real historical sprint velocity (e.g. average completedPoints of recent "
                    "sprints), computed by the caller. Null means open-ended bin-packing with "
                    "no capacity cap."
    )
    target_sprint_count: Optional[int] = Field(
        None, ge=1, description="Desired sprint count. Capacity and dependency order may require "
                                 "additional buckets, which the UI asks the user to accept."
    )
    requirements: List[RequirementText] = Field(
        min_length=1,
        description="The epic's requirements, one per entry — what it must actually deliver. An epic "
                    "title is a direction; these are the pieces. Each one is searched for separately, "
                    "so an issue written for a requirement is sized against the past work that "
                    "resembles THAT piece rather than the epic as a whole. At least one nonblank "
                    "requirement is required.",
    )
    space_id: Optional[int] = Field(
        None, description="The space this epic belongs to. Supplying it lets the planner read this "
                          "team's own past issues and write the plan in their style, sizing "
                          "estimates against work they actually finished (see "
                          "app/planning/grounding.py). Optional: without it the plan is generated "
                          "from the proposal and stated requirements without historical context."
    )


class EpicDraftOut(_CamelModel):
    title: str
    description: str
    goals: List[str]


class IssueDraftOut(_CamelModel):
    temp_id: str
    title: str
    description: str
    issue_type: str
    labels: List[str]
    estimate_story_points: Optional[int]
    estimate_rationale: Optional[str]
    #: Past issue keys this estimate was sized against, each verified against the exemplars.
    #: Empty = a judgement call, which is normal.
    estimate_reference: List[str] = Field(default_factory=list)
    #: The model's only estimate call: smaller / same / larger than the cited issues. The number is
    #: then derived from what those issues actually cost.
    complexity_vs_reference: Optional[str] = None
    #: Median points of the verified cited issues, before the one-step adjustment. Null if none.
    historical_baseline_points: Optional[int] = None
    #: "high" | "medium" | "low", computed from whether the estimate rests on finished work —
    #: never claimed by the model. Per issue, since some issues have a real comparable and some don't.
    estimate_confidence: str = "low"
    #: Why this issue exists: "epic_requirement" | "historical_pattern" | "planning_necessity".
    origin: str = "epic_requirement"
    #: Names of the historical patterns that led to it. Empty for most issues.
    pattern_basis: List[str] = Field(default_factory=list)
    #: Which stated requirement this issue delivers (1-based). Null if none were stated or it's
    #: supporting work.
    requirement_index: Optional[int] = None
    #: Fraction of the board's description sections this description contains — the countable part
    #: of "written like this team". Null when no layout was detected.
    format_compliance: Optional[float] = None
    depends_on: List[str]
    dependency_kinds: Dict[str, str]
    dependency_reasons: Dict[str, str]


class SprintBucketOut(_CamelModel):
    sprint_index: int
    issue_temp_ids: List[str]
    total_points: int


class GroundingExemplarOut(_CamelModel):
    """One past issue the planner was shown, returned in full so a reviewer can open it and check the
    comparison (was ATLAS-118 really a 5-pointer, and is it really similar?).
    """

    issue_key: str
    title: str
    issue_type: Optional[str]
    status: Optional[str]
    story_points: Optional[int]
    #: Normalized 0-1, never the raw retrieval score (see relevance_confidence).
    confidence: float
    #: Finished at a recorded size, so usable as an estimate anchor, not just a writing sample.
    estimate_anchor: bool
    #: "similar" = resembles this proposal. "recent" = backfilled from latest finished work.
    #: "Eight comparable issues" and "three, plus five recent" are different claims.
    source: str


class HistoricalPatternOut(_CamelModel):
    """One reusable pattern found in this team's finished work, with the past issues that support it
    (each verified against what was actually retrieved).
    """

    name: str
    steps: List[str]
    evidence_issue_keys: List[str]
    why_relevant: str
    pitfalls: List[str]


class PlanEvidenceOut(_CamelModel):
    """Whether the retrieved history was worth planning from, and what it taught.

    `PlanGroundingOut` says what was retrieved; this says whether it applies. strength "none" with
    exemplars present means history was found, judged irrelevant, and withheld from the planner.
    """

    #: "strong" | "partial" | "none", or "unassessed" when the stage did not run.
    strength: str
    reasoning: str
    relevant_issue_keys: List[str]
    patterns: List[HistoricalPatternOut]
    #: Patterns discarded for citing an issue that was never shown, or for resting on too few issues.
    dropped_patterns: int
    #: True when verification had to lower the model's own strength label to match what survived.
    downgraded: bool


class RequirementCoverageOut(_CamelModel):
    """One stated requirement, the past work found for it, and which planned issues cover it.

    Empty `exemplar_keys` means the board has no precedent for this piece — common, and why its issues
    come back uncited with low confidence.
    """

    index: int
    text: str
    #: Past issues retrieved for this requirement specifically, best first.
    exemplar_keys: List[str]
    #: How many of those were finished at a recorded size, so could anchor an estimate.
    anchor_count: int
    #: temp_ids of planned issues delivering this requirement. Empty = not covered; worth a look.
    issue_temp_ids: List[str]


class HouseFormatOut(_CamelModel):
    """Description layout detected from this board's finished issues, with each section's coverage so
    it can be checked. Empty `sections` = no consistent layout; the planner writes plain prose.
    """

    sections: List[str]
    #: List-item prefix, e.g. "AC" for `AC1 — …`. Null when there's no consistent convention.
    item_marker: Optional[str] = None
    #: How many labelled items a description here usually has, and the most it ever has.
    item_count_low: Optional[int] = None
    item_count_high: Optional[int] = None
    item_count_max: Optional[int] = None
    coverage: Dict[str, float]
    sample_size: int


class PlanGroundingOut(_CamelModel):
    """What the plan was grounded in. `disabled`, `no_history` and `retrieval_failed` all produce a
    generic plan but mean different things to a reviewer, so each is its own status.
    """

    status: str
    exemplars: List[GroundingExemplarOut]
    #: Here rather than under `evidence`: layout is a board property, applied even when the history
    #: was judged irrelevant.
    house_format: HouseFormatOut
    #: How many exemplars could anchor an estimate. 0 with exemplars present = "style_only".
    estimate_anchor_count: int
    #: Citations naming an issue the model was never shown, removed before the response. Non-zero
    #: means verification caught something.
    dropped_estimate_references: int


class ChangeSummaryOut(_CamelModel):
    """What one refine round changed, computed by diffing the plan before and after (not reported by
    the model). Null on /plan-epic, which has no "before". Lets a reviewer see which edits landed
    without rereading every card.
    """

    instruction: str
    #: Titles of issues present after this round that were not present before it.
    added: List[str]
    #: Titles of issues present before this round that are gone after it (dropped or merged away).
    removed: List[str]
    #: "New title (was: Old title)", for issues that survived but were renamed.
    retitled: List[str]
    #: "Title: 3 -> 5 pts", for issues that survived with a different estimate.
    re_estimated: List[str]
    #: Issues whose title and points both stayed the same (the description may still have changed).
    unchanged_count: int


class PlanEpicResponse(_CamelModel):
    epic: EpicDraftOut
    issues: List[IssueDraftOut]
    sprint_plan: List[SprintBucketOut]
    grounding: PlanGroundingOut
    evidence: PlanEvidenceOut
    #: One entry per stated requirement. Empty when none were stated.
    requirement_coverage: List[RequirementCoverageOut]
    #: Set only by `/plan-epic/refine` — see `ChangeSummaryOut`.
    change_summary: Optional[ChangeSummaryOut] = None
    degraded: bool
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0


async def _plan_grounding(
    request: Request,
    space_id: Optional[int],
    query_text: str,
    requirements: Optional[List[str]] = None,
    on_progress=None,
) -> GroundingContext:
    """Retrieve the past issues to ground this plan in, or say why there are none.

    Authorizes `space_id` whenever one is sent, even with the feature off — a caller-supplied space id
    is a request to read someone's data. Grounding failures never raise (the plan just goes
    ungrounded); a 403/502 from the membership check does.
    """
    if space_id is not None:
        await _authorize_space_ids(request, [space_id])
    settings = request.app.state.settings
    if not settings.epic_planning_grounding_enabled or space_id is None:
        return with_stated_requirements(GroundingContext(status="disabled"), requirements or [])
    include_comments = (
        settings.epic_planning_pattern_extraction_enabled
        and settings.epic_planning_pattern_comments_enabled
    )
    if requirements:
        return await gather_requirement_exemplars(
            request.app.state.retrieval, [space_id], query_text, requirements,
            limit=settings.epic_planning_grounding_exemplars,
            per_requirement=settings.epic_planning_requirement_exemplars,
            include_comments=include_comments,
            on_progress=on_progress,
        )
    return await gather_exemplars(
        request.app.state.retrieval, [space_id], query_text,
        limit=settings.epic_planning_grounding_exemplars,
        include_comments=include_comments,
    )


async def _assess_history(
    request: Request, feature: str, proposal: str, context: GroundingContext
) -> Optional[AssessmentResult]:
    """Judge whether the retrieved history is relevant and what it teaches, before planning from it.

    Returns None when the stage is off or there's nothing to read — reported as "unassessed", not
    "none": nobody looked is different from looked and found nothing.
    """
    settings = request.app.state.settings
    if not settings.epic_planning_pattern_extraction_enabled or not context.exemplars:
        return None
    return await _tagged(feature, assess_history(
        request.app.state.instructor_client, request.app.state.instructor_model, proposal, context,
    ))


def _evidence_out(result: Optional[AssessmentResult]) -> PlanEvidenceOut:
    if result is None:
        return PlanEvidenceOut(
            strength="unassessed", reasoning="", relevant_issue_keys=[], patterns=[],
            dropped_patterns=0, downgraded=False,
        )
    assessment = result.assessment
    return PlanEvidenceOut(
        strength=assessment.strength,
        reasoning=assessment.reasoning,
        relevant_issue_keys=list(assessment.relevant_issue_keys),
        patterns=[HistoricalPatternOut(**pattern.model_dump()) for pattern in assessment.patterns],
        dropped_patterns=result.dropped_patterns,
        downgraded=result.downgraded,
    )


def _issue_out(issue, context: GroundingContext, strength: str) -> IssueDraftOut:
    """One planned issue plus the two things code knows about its estimate that the model does not:
    what the cited work really cost, and how much this number therefore rests on history."""
    baseline = historical_baseline(issue.estimate_reference, context)
    compliance = section_compliance(issue.description, context.house_format)
    return IssueDraftOut(
        **issue.model_dump(),
        historical_baseline_points=baseline,
        estimate_confidence=estimate_confidence(issue.estimate_reference, baseline, strength),
        # NaN (nothing to comply with) becomes null, not 0, which would read as "ignored the layout".
        format_compliance=compliance if compliance == compliance else None,
    )


_EMPTY_HOUSE_FORMAT = {
    "sections": [], "item_marker": None, "item_count_low": None, "item_count_high": None,
    "item_count_max": None, "coverage": {}, "sample_size": 0,
}


def _grounding_out_from_state(state: Optional[dict]) -> PlanGroundingOut:
    """Build the grounding block from a checkpoint, tolerating older ones.

    Defaults are filled per field: an old checkpoint has a `grounding` dict without `house_format`,
    and a whole-dict default never fired — a paused rollout came back as a browser "network error".
    """
    state = dict(state or {})
    state.setdefault("status", "disabled")
    state.setdefault("exemplars", [])
    state.setdefault("estimate_anchor_count", 0)
    state.setdefault("dropped_estimate_references", 0)
    house = {**_EMPTY_HOUSE_FORMAT, **(state.get("house_format") or {})}
    return PlanGroundingOut(**{**state, "house_format": house})


def _diff_plan(before: List[IssueDraft], after: List[IssueDraft], instruction: str) -> ChangeSummaryOut:
    """Diff a refine round's output against its input, matched by `temp_id`.

    Plain code — asking the model "what did you change" is unreliable. A merged-away issue counts as
    removed, and the survivor shows up under retitled/re_estimated if its title or points changed.
    """
    before_by_id = {issue.temp_id: issue for issue in before}
    after_by_id = {issue.temp_id: issue for issue in after}
    added = [issue.title for issue in after if issue.temp_id not in before_by_id]
    removed = [issue.title for issue in before if issue.temp_id not in after_by_id]
    retitled: List[str] = []
    re_estimated: List[str] = []
    unchanged = 0
    for temp_id, prior in before_by_id.items():
        current = after_by_id.get(temp_id)
        if current is None:
            continue
        changed = False
        if current.title != prior.title:
            retitled.append(f"{current.title} (was: {prior.title})")
            changed = True
        if current.estimate_story_points != prior.estimate_story_points:
            prior_points = prior.estimate_story_points if prior.estimate_story_points is not None else "unestimated"
            current_points = current.estimate_story_points if current.estimate_story_points is not None else "unestimated"
            re_estimated.append(f"{current.title}: {prior_points} -> {current_points} pts")
            changed = True
        if not changed:
            unchanged += 1
    return ChangeSummaryOut(
        instruction=instruction, added=added, removed=removed,
        retitled=retitled, re_estimated=re_estimated, unchanged_count=unchanged,
    )


def _requirement_coverage_out(context: GroundingContext, issues) -> List[RequirementCoverageOut]:
    return [
        RequirementCoverageOut(
            index=requirement.index,
            text=requirement.text,
            exemplar_keys=[exemplar.issue_key for exemplar in requirement.exemplars],
            anchor_count=len(requirement.anchors),
            issue_temp_ids=[
                issue.temp_id for issue in issues if issue.requirement_index == requirement.index
            ],
        )
        for requirement in context.requirements
    ]


def _grounding_out(context: GroundingContext, dropped_references: int) -> PlanGroundingOut:
    return PlanGroundingOut(
        status=context.status,
        exemplars=[
            GroundingExemplarOut(
                issue_key=exemplar.issue_key,
                title=exemplar.title,
                issue_type=exemplar.issue_type,
                status=exemplar.status,
                story_points=exemplar.story_points,
                confidence=round(exemplar.confidence, 4),
                estimate_anchor=exemplar.is_estimate_anchor,
                source=exemplar.source,
            )
            for exemplar in context.exemplars
        ],
        house_format=HouseFormatOut(
            sections=list(context.house_format.sections),
            item_marker=context.house_format.item_marker,
            item_count_low=context.house_format.item_count_low,
            item_count_high=context.house_format.item_count_high,
            item_count_max=context.house_format.item_count_max,
            coverage=dict(context.house_format.coverage),
            sample_size=context.house_format.sample_size,
        ),
        estimate_anchor_count=len(context.anchors),
        dropped_estimate_references=dropped_references,
    )


@router.post("/plan-epic", response_model=PlanEpicResponse, dependencies=_NEW_AI_WORK)
async def plan_epic_endpoint(req: PlanEpicRequest, request: Request) -> PlanEpicResponse:
    """AI epic planning: one structured-output call -> epic + issues, then dependency validation +
    topological sort, then sprint bin-packing.

    With `epic_planning_multiagent_enabled`, the first step is the LangGraph planner -> critic
    pipeline instead; both return the same PlanResult. With no plan, it answers 503 (no model
    reachable) or 502 (no valid plan), never a stand-in. Nothing is persisted — the caller reviews
    and commits through the normal APIs.
    """
    caller = _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    # Tagged too: retrieval opens its own spans, and they should carry the feature name.
    grounding = await _tagged(FEATURE_PLAN_EPIC, _plan_grounding(
        request, req.space_id, req.proposal, req.requirements,
    ))
    assessment_result = await _assess_history(request, FEATURE_PLAN_EPIC, req.proposal, grounding)
    assessment = assessment_result.assessment if assessment_result else None
    planning_graph = getattr(request.app.state, "planning_graph", None)
    if planning_graph is not None:
        result = await _tagged(FEATURE_PLAN_EPIC, plan_epic_multiagent(planning_graph, req.proposal, req.existing_labels, grounding, assessment))
    else:
        result = await _tagged(FEATURE_PLAN_EPIC, plan_epic(client, model, req.proposal, req.existing_labels, grounding, assessment))
    ordered_issues = validate_and_order(result.plan.issues)
    buckets = allocate_sprints(ordered_issues, req.sprint_capacity_points, req.target_sprint_count)
    _record_usage(
        request, FEATURE_PLAN_EPIC, [_structured_output_charge(request, result)],
        detail={
            "degraded": result.degraded,
            "multiagent": planning_graph is not None,
            "issue_count": len(ordered_issues),
            # Per request, not just a metric: "was THIS plan grounded?" is the first question when
            # someone disagrees with a plan.
            "grounding": grounding.status,
            "grounding_exemplars": len(grounding.exemplars),
            "dropped_estimate_references": result.dropped_estimate_references,
            "evidence_strength": assessment.strength if assessment else "unassessed",
            "patterns": len(assessment.patterns) if assessment else 0,
            "historically_derived_estimates": result.historically_derived_estimates,
        },
        caller=caller,
    )
    return PlanEpicResponse(
        epic=EpicDraftOut(**result.plan.epic.model_dump()),
        issues=[
            _issue_out(issue, grounding, assessment.strength if assessment else "unassessed")
            for issue in ordered_issues
        ],
        grounding=_grounding_out(grounding, result.dropped_estimate_references),
        evidence=_evidence_out(assessment_result),
        requirement_coverage=_requirement_coverage_out(grounding, ordered_issues),
        sprint_plan=[
            SprintBucketOut(
                sprint_index=b.sprint_index,
                issue_temp_ids=b.issue_temp_ids,
                total_points=b.total_points,
            )
            for b in buckets
        ],
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


class EpicDraftIn(_CamelModel):
    title: str = Field(min_length=1, max_length=120)
    description: str
    goals: List[str] = Field(default_factory=list)


class IssueDraftIn(_CamelModel):
    temp_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1)
    issue_type: Literal["story", "task", "bug"] = "task"
    labels: List[str] = Field(default_factory=list)
    estimate_story_points: Optional[int] = Field(None, ge=1, le=13)
    estimate_rationale: Optional[str] = None
    estimate_reference: List[str] = Field(default_factory=list)
    complexity_vs_reference: Optional[str] = None
    requirement_index: Optional[int] = None
    origin: str = "epic_requirement"
    pattern_basis: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    dependency_kinds: Dict[str, Literal["blocks", "after"]] = Field(default_factory=dict)
    dependency_reasons: Dict[str, str] = Field(default_factory=dict)


class RefinePlanRequest(_CamelModel):
    epic: EpicDraftIn
    issues: List[IssueDraftIn] = Field(min_length=1)
    instruction: str = Field(min_length=1, description="Free-text change request, e.g. 'add a QA task' or 'drop the inventory issue and its dependents'.")
    existing_labels: List[str] = Field(default_factory=list)
    sprint_capacity_points: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    target_sprint_count: Optional[int] = Field(None, ge=1)
    requirements: List[str] = Field(
        default_factory=list,
        description="The same requirement list the plan was generated from, so an added issue is sized "
                    "against its own requirement's history rather than the epic average.",
    )
    space_id: Optional[int] = Field(
        None, description="Same role as on /plan-epic: lets an added or re-estimated issue be written "
                          "and sized like the rest of this team's work rather than in the generic voice."
    )


@router.post("/plan-epic/refine", response_model=PlanEpicResponse, dependencies=_NEW_AI_WORK)
async def refine_plan_endpoint(req: RefinePlanRequest, request: Request) -> PlanEpicResponse:
    """Apply a free-text edit ("we also need a QA task") to a plan from /plan-epic or an earlier
    refine. Persists nothing; the caller still reviews before committing.
    """
    # Tag here, in the handler: only the handler can label the server span
    # (tests/test_trace_tag_coverage.py checks for this).
    caller = _tag_entrypoint_span(request, FEATURE_PLAN_EPIC_REFINE)
    return await _run_refine(request, req, caller)


async def _run_refine(
    request: Request, req: "RefinePlanRequest", caller, on_stage=None,
) -> PlanEpicResponse:
    """The refine pipeline, with an optional progress reporter, shared by the POST and SSE endpoints."""
    def stage(label: str) -> None:
        if on_stage:
            on_stage(label)

    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    epic = EpicDraft(title=req.epic.title, description=req.epic.description, goals=req.epic.goals)
    # model_dump(), not a hand-written field list: the hand-written list silently dropped every
    # provenance field added later, destroying citations on the way into a refine.
    issues = [IssueDraft(**i.model_dump()) for i in req.issues]
    # Read the instruction first. Most edits (re-estimate, merge, drop) add no new work, so there's
    # nothing to look up. When it does add work, its own words become the search queries.
    stage("reading what you asked for")
    intent = await _tagged(FEATURE_PLAN_EPIC_REFINE, read_refine_intent(client, model, req.instruction))
    if intent.adds_work or intent.degraded:
        grounding = await _tagged(FEATURE_PLAN_EPIC_REFINE, _plan_grounding(
            request, req.space_id, f"{epic.title}\n\n{epic.description}\n\n{req.instruction}",
            # Search for the new work, not the original requirements (already covered). If the
            # instruction couldn't be read, fall back to the original list.
            intent.intent.new_work or req.requirements,
            on_progress=stage,
        ))
    else:
        stage("adjusting the plan — no new work to look up")
        # Still authorize: a space id in the request names someone's data either way.
        if req.space_id is not None:
            await _authorize_space_ids(request, [req.space_id])
        grounding = GroundingContext(status="disabled")
    if grounding.exemplars:
        stage("working out which past work actually applies")
    assessment_result = await _assess_history(
        request, FEATURE_PLAN_EPIC_REFINE, f"{epic.title}\n\n{epic.description}", grounding,
    )
    assessment = assessment_result.assessment if assessment_result else None
    stage(
        f"applying the change against {len(grounding.exemplars)} past issues"
        if grounding.exemplars else "applying the change"
    )
    result = await _tagged(FEATURE_PLAN_EPIC_REFINE, refine_plan(client, model, epic, issues, req.instruction, req.existing_labels, grounding, assessment))
    ordered_issues = validate_and_order(result.plan.issues)
    buckets = allocate_sprints(ordered_issues, req.sprint_capacity_points, req.target_sprint_count)
    # Billed separately from plan_epic, so repeated refines don't hide inside the planning cost.
    _record_usage(
        request, FEATURE_PLAN_EPIC_REFINE, [_structured_output_charge(request, result)],
        detail={
            "degraded": result.degraded, "issue_count": len(ordered_issues),
            "grounding": grounding.status, "grounding_exemplars": len(grounding.exemplars),
            "dropped_estimate_references": result.dropped_estimate_references,
            "evidence_strength": assessment.strength if assessment else "unassessed",
            "historically_derived_estimates": result.historically_derived_estimates,
            # Whether the edit added work, i.e. whether history was read at all.
            "adds_work": intent.adds_work,
            "new_work": len(intent.intent.new_work),
        },
        caller=caller,
    )
    return PlanEpicResponse(
        epic=EpicDraftOut(**result.plan.epic.model_dump()),
        issues=[
            _issue_out(issue, grounding, assessment.strength if assessment else "unassessed")
            for issue in ordered_issues
        ],
        grounding=_grounding_out(grounding, result.dropped_estimate_references),
        evidence=_evidence_out(assessment_result),
        requirement_coverage=_requirement_coverage_out(grounding, ordered_issues),
        change_summary=_diff_plan(issues, ordered_issues, req.instruction),
        sprint_plan=[
            SprintBucketOut(
                sprint_index=b.sprint_index,
                issue_temp_ids=b.issue_temp_ids,
                total_points=b.total_points,
            )
            for b in buckets
        ],
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


@router.post("/plan-epic/refine/stream", dependencies=_NEW_AI_WORK)
async def refine_plan_stream_endpoint(req: RefinePlanRequest, request: Request) -> StreamingResponse:
    """SSE version of POST /plan-epic/refine: same pipeline, with progress labels.

    A refine that adds work takes tens of seconds (read instruction, search per new piece, assess
    history, edit), and an edit with no new work says so instead of just spinning.
    """
    caller = _tag_entrypoint_span(request, FEATURE_PLAN_EPIC_REFINE)

    async def generate() -> AsyncIterator[str]:
        stage_queue: asyncio.Queue[str] = asyncio.Queue()
        # using_tags around create_task (the task copies the context when created).
        with using_tags([FEATURE_PLAN_EPIC_REFINE]):
            task = asyncio.create_task(
                _run_refine(request, req, caller, stage_queue.put_nowait)
            )
        while not task.done():
            try:
                label = await asyncio.wait_for(stage_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            yield _sse("stage", {"label": label})
        while not stage_queue.empty():
            yield _sse("stage", {"label": stage_queue.get_nowait()})
        try:
            response = task.result()
        except HTTPException as exc:
            # A 403 from the membership check is the caller being told no, not a failure to stream.
            yield _sse("error", {"detail": exc.detail})
            return
        except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same backstop as the rollout stream
            logger.exception("plan refine stream failed")
            yield _sse("error", {"detail": _stream_error_detail(exc, "plan refinement failed")})
            return
        yield _sse("result", response.model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


# --- Durable lifecycle for Plan Epic (app/planning/rollout_graph.py) ---
# Kept on the /rollout path for API compatibility. The Plan Epic UI starts this workflow, edits its
# preview, then submits the final epic/issues/sprints for durable publishing.


def _rollout_issue_out(issue, grounding: dict, strength: str) -> "RolloutIssueOut":
    """The durable path's issue, with the same derived estimate fields as `_issue_out`.

    Uses the checkpointed exemplars, not a fresh search: a rollout can wait overnight, and the
    reviewer should see the baseline the plan was actually written against.
    """
    anchors = {
        str(exemplar.get("issue_key", "")).upper(): exemplar.get("story_points")
        for exemplar in (grounding or {}).get("exemplars", [])
        if exemplar.get("estimate_anchor") and exemplar.get("story_points") is not None
    }
    points = sorted(
        anchors[key.upper()] for key in issue.estimate_reference if key.upper() in anchors
    )
    baseline = points[(len(points) - 1) // 2] if points else None
    return RolloutIssueOut(
        **issue.model_dump(),
        historical_baseline_points=baseline,
        estimate_confidence=estimate_confidence(issue.estimate_reference, baseline, strength),
    )


class RolloutIssueOut(_CamelModel):
    temp_id: str
    title: str
    description: str
    issue_type: str
    labels: List[str]
    estimate_story_points: Optional[int]
    estimate_rationale: Optional[str]
    estimate_reference: List[str] = Field(default_factory=list)
    complexity_vs_reference: Optional[str] = None
    historical_baseline_points: Optional[int] = None
    estimate_confidence: str = "low"
    requirement_index: Optional[int] = None
    origin: str = "epic_requirement"
    pattern_basis: List[str] = Field(default_factory=list)
    depends_on: List[str]
    dependency_kinds: Dict[str, str]
    dependency_reasons: Dict[str, str]


class RolloutSprintTargetOut(_CamelModel):
    sprint_index: int
    issue_temp_ids: List[str]
    mode: Literal["existing", "new"]
    sprint_id: Optional[int]
    sprint_name: Optional[str]


class RolloutPlanOut(_CamelModel):
    epic: Optional[EpicDraftOut]
    issues: List[RolloutIssueOut]
    sprint_plan: List[SprintBucketOut]
    sprint_targets: List[RolloutSprintTargetOut]
    grounding: PlanGroundingOut
    evidence: PlanEvidenceOut
    #: Rebuilt from the checkpoint, so a paused rollout still shows requirement coverage.
    requirement_coverage: List[RequirementCoverageOut]


class RolloutStatusResponse(_CamelModel):
    thread_id: str
    status: str
    plan: Optional[RolloutPlanOut]
    epic_issue_key: Optional[str]
    committed_issue_keys: Dict[str, str]
    degraded: bool
    error: Optional[str]
    sprint_capacity_points: Optional[float]
    target_sprint_count: Optional[int]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    planning_latency_seconds: float = 0.0


def _rollout_status_response(thread_id: str, values: dict) -> RolloutStatusResponse:
    epic = values.get("epic")
    issues = values.get("issues") or []
    plan = RolloutPlanOut(
        epic=EpicDraftOut(**epic.model_dump()) if epic else None,
        issues=[
            _rollout_issue_out(
                i, values.get("grounding") or {},
                (values.get("evidence") or {}).get("strength", "unassessed"),
            )
            for i in issues
        ],
        sprint_plan=[SprintBucketOut(**b) for b in values.get("sprint_buckets") or []],
        sprint_targets=[RolloutSprintTargetOut(**t) for t in values.get("sprint_targets") or []],
        requirement_coverage=[
            RequirementCoverageOut(
                index=index,
                text=text,
                # Per-requirement exemplars aren't checkpointed (only the merged set), so only
                # coverage by the plan is reported.
                exemplar_keys=[],
                anchor_count=0,
                issue_temp_ids=[i.temp_id for i in issues if i.requirement_index == index],
            )
            for index, text in enumerate(values.get("requirements") or [], start=1)
        ],
        # Defaults for checkpoints created before these fields existed, so a long-waiting approval
        # still renders.
        evidence=PlanEvidenceOut(**(values.get("evidence") or {
            "strength": "unassessed", "reasoning": "", "relevant_issue_keys": [],
            "patterns": [], "dropped_patterns": 0, "downgraded": False,
        })),
        grounding=_grounding_out_from_state(values.get("grounding")),
    )
    return RolloutStatusResponse(
        thread_id=thread_id,
        status=values.get("status", "pending_approval"),
        plan=plan,
        epic_issue_key=values.get("epic_issue_key"),
        committed_issue_keys=values.get("committed") or {},
        degraded=bool(values.get("degraded", False)),
        error=values.get("error"),
        sprint_capacity_points=values.get("sprint_capacity_points"),
        target_sprint_count=values.get("target_sprint_count"),
        input_tokens=values.get("input_tokens", 0) or 0,
        output_tokens=values.get("output_tokens", 0) or 0,
        cached_input_tokens=values.get("cached_input_tokens", 0) or 0,
        estimated_cost_usd=values.get("estimated_cost_usd", 0.0) or 0.0,
        planning_latency_seconds=values.get("planning_latency_seconds", 0.0) or 0.0,
    )


class PlanRolloutRequest(_CamelModel):
    proposal: str = Field(min_length=1)
    #: Same as on /plan-epic: each one is searched separately.
    requirements: List[RequirementText] = Field(
        min_length=1, description="At least one nonblank requirement describing what the epic must deliver.",
    )
    existing_labels: List[str] = Field(default_factory=list)
    sprint_capacity_points: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    target_sprint_count: Optional[int] = Field(None, ge=1)
    space_id: int = Field(description="The single space this epic's issues will be created in.")


def _rollout_graph(request: Request):
    graph = getattr(request.app.state, "rollout_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="epic rollout is disabled (AI_EPIC_ROLLOUT_ENABLED=false)")
    return graph


def _caller_identity(request: Request) -> tuple[str, str]:
    # Empty string, not a placeholder: the membership check skips falsy user ids (trusted internal
    # callers such as eval scripts). A placeholder would send a fake identity to jira-backend.
    user_id = request.headers.get("x-user-id") or ""
    username = request.headers.get("x-username") or ""
    return user_id, username


async def _register_plan_epic_thread_if_configured(
    request: Request, space_id: int, user_id: str, thread_id: str,
) -> None:
    """Best-effort discovery index; failing it must not prevent the workflow itself from starting."""
    db_url = getattr(request.app.state, "epic_rollout_checkpoint_db_url", None)
    if not db_url:
        return
    try:
        await register_plan_epic_active_thread(db_url, space_id, user_id, thread_id)
    except Exception:  # noqa: BLE001 - the checkpointed workflow remains usable by its direct id
        logger.exception(
            "failed to register active Plan Epic thread (space=%s, user=%s)", space_id, user_id,
        )


@router.post("/plan-epic/rollout", response_model=RolloutStatusResponse, dependencies=_NEW_AI_WORK)
async def start_rollout_endpoint(req: PlanRolloutRequest, request: Request) -> RolloutStatusResponse:
    """Start a durable rollout: generate a plan, then pause for approval. Returns "pending_approval"
    (or "failed" if planning failed). Never commits anything.
    """
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    await _authorize_space_ids(request, [req.space_id])
    graph = _rollout_graph(request)
    user_id, username = _caller_identity(request)
    thread_id = str(uuid.uuid4())
    await _register_plan_epic_thread_if_configured(request, req.space_id, user_id, thread_id)
    initial = initial_rollout_state(
        req.proposal, req.existing_labels, req.sprint_capacity_points, req.target_sprint_count,
        req.space_id, user_id, username, req.requirements,
    )
    config = {"configurable": {"thread_id": thread_id}}
    await _tagged(FEATURE_PLAN_EPIC, graph.ainvoke(initial, config=config))
    snap = await graph.aget_state(config)
    return _rollout_status_response(thread_id, snap.values)


@router.post("/plan-epic/rollout/stream", dependencies=_NEW_AI_WORK)
async def start_rollout_stream_endpoint(req: PlanRolloutRequest, request: Request) -> StreamingResponse:
    """SSE version of POST /plan-epic/rollout. Only `plan_node` calls an LLM, so this sends at most one
    `stage` label before `result` — still worth it, since that call is the whole wait.
    """
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    await _authorize_space_ids(request, [req.space_id])
    graph = _rollout_graph(request)
    user_id, username = _caller_identity(request)
    thread_id = str(uuid.uuid4())
    await _register_plan_epic_thread_if_configured(request, req.space_id, user_id, thread_id)
    initial = initial_rollout_state(
        req.proposal, req.existing_labels, req.sprint_capacity_points, req.target_sprint_count,
        req.space_id, user_id, username, req.requirements,
    )
    config = {"configurable": {"thread_id": thread_id}}

    async def generate() -> AsyncIterator[str]:
        stage_queue: asyncio.Queue[str] = asyncio.Queue()
        token = ROLLOUT_ON_STAGE_VAR.set(stage_queue.put_nowait)
        try:
            # using_tags around create_task (the task copies the context when created). Without it
            # the plan_node LLM span reached Langfuse untagged.
            with using_tags([FEATURE_PLAN_EPIC]):
                task = asyncio.create_task(graph.ainvoke(initial, config=config))
            while not task.done():
                try:
                    label = await asyncio.wait_for(stage_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                yield _sse("stage", {"label": label})
            while not stage_queue.empty():
                yield _sse("stage", {"label": stage_queue.get_nowait()})
            try:
                task.result()
            except Exception as exc:  # noqa: BLE001 - clean SSE error frame, same as ask_stream's backstop
                logger.exception("epic rollout stream failed")
                yield _sse("error", {"detail": _stream_error_detail(exc, "epic rollout failed")})
                return
        finally:
            ROLLOUT_ON_STAGE_VAR.reset(token)
        snap = await graph.aget_state(config)
        yield _sse("result", _rollout_status_response(thread_id, snap.values).model_dump(by_alias=True))

    return StreamingResponse(generate(), media_type="text/event-stream")


_ROLLOUT_TERMINAL_STATUSES = {"committed", "rejected"}


@router.get("/plan-epic/rollout/active", response_model=Optional[RolloutStatusResponse])
async def find_active_rollout_endpoint(space_id: int, request: Request) -> Optional[RolloutStatusResponse]:
    """This caller's latest unfinished Plan Epic workflow in the space, so a browser reload can find
    its thread id again. Finished/rejected workflows aren't returned, and another user's workflow
    looks the same as none.
    """
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    await _authorize_space_ids(request, [space_id])
    user_id, _ = _caller_identity(request)
    db_url = getattr(request.app.state, "epic_rollout_checkpoint_db_url", None)
    if not db_url:
        return None
    thread_id = await find_plan_epic_active_thread_id(db_url, space_id, user_id)
    if not thread_id:
        return None
    graph = _rollout_graph(request)
    snap = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snap.values or snap.values.get("status") in _ROLLOUT_TERMINAL_STATUSES:
        return None
    if snap.values.get("failed_step") == "plan":
        return None  # never produced a plan, so there's nothing to resume
    try:
        await _authorize_workflow_state(request, snap.values)
    except HTTPException as exc:
        if exc.status_code == 403:
            return None
        raise
    return _rollout_status_response(thread_id, snap.values)


@router.get("/plan-epic/rollout/{thread_id}", response_model=RolloutStatusResponse)
async def get_rollout_status_endpoint(thread_id: str, request: Request) -> RolloutStatusResponse:
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    graph = _rollout_graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no rollout workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    return _rollout_status_response(thread_id, snap.values)


class RolloutSprintTargetIn(_CamelModel):
    sprint_index: int = Field(ge=0)
    issue_temp_ids: List[str] = Field(min_length=1)
    mode: Literal["existing", "new"]
    sprint_id: Optional[int] = Field(None, ge=1)
    sprint_name: Optional[str] = None


class RolloutDecisionRequest(_CamelModel):
    decision: Literal["approve", "edit", "reject"]
    epic: Optional[EpicDraftIn] = None
    issues: Optional[List[IssueDraftIn]] = None
    sprint_targets: Optional[List[RolloutSprintTargetIn]] = None


@router.post("/plan-epic/rollout/{thread_id}/decision", response_model=RolloutStatusResponse)
async def submit_rollout_decision_endpoint(
    thread_id: str, req: RolloutDecisionRequest, request: Request
) -> RolloutStatusResponse:
    """Resume a paused rollout with a human decision. `edit` needs `epic` + `issues` (the edited
    plan); approve/reject ignore them. Runs the commit loop to the end before returning (it's
    crash-safe; see rollout_graph.py).
    """
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    graph = _rollout_graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no rollout workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if snap.values.get("status") != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail=f"rollout {thread_id} is not awaiting approval (status={snap.values.get('status')})",
        )

    resume_payload: dict = {"decision": req.decision}
    if req.decision == "edit":
        if req.epic is None or not req.issues:
            raise HTTPException(status_code=422, detail="decision=edit requires epic and issues")
        resume_payload["epic"] = req.epic.model_dump()
        resume_payload["issues"] = [i.model_dump() for i in req.issues]
        if req.sprint_targets is not None:
            resume_payload["sprint_targets"] = [t.model_dump() for t in req.sprint_targets]

    try:
        await _tagged(FEATURE_PLAN_EPIC, graph.ainvoke(Command(resume=resume_payload), config=config))
    except ValueError as exc:
        # Final-plan validation happens in review_node before the first irreversible Jira write.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    final = await graph.aget_state(config)
    return _rollout_status_response(thread_id, final.values)


@router.post("/plan-epic/rollout/{thread_id}/retry", response_model=RolloutStatusResponse)
async def retry_rollout_endpoint(thread_id: str, request: Request) -> RolloutStatusResponse:
    """Un-stick a rollout stuck at "committing" (likely a crash — reading status never advances it) or
    "failed" (e.g. jira-backend was down). See `retry_rollout` for how the two differ. 409 for any
    other status: this isn't a second way to submit the decision.
    """
    _tag_entrypoint_span(request, FEATURE_PLAN_EPIC)
    graph = _rollout_graph(request)
    config = {"configurable": {"thread_id": thread_id}}
    snap = await graph.aget_state(config)
    if not snap.values:
        raise HTTPException(status_code=404, detail=f"no rollout workflow with thread_id={thread_id}")
    await _authorize_workflow_state(request, snap.values)
    if snap.values.get("status") not in ("committing", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"rollout {thread_id} is not committing or failed (status={snap.values.get('status')})",
        )
    if snap.values.get("failed_step") == "plan":
        raise HTTPException(
            status_code=409, detail="this plan was never generated; start a new Plan Epic instead",
        )
    await _tagged(FEATURE_PLAN_EPIC, retry_rollout(graph, thread_id))
    final = await graph.aget_state(config)
    return _rollout_status_response(thread_id, final.values)


class FlaggedIssueIn(_CamelModel):
    issue_key: str
    title: str
    detail: Optional[str] = None


class SprintPaceRequest(_CamelModel):
    sprint_name: str = Field(min_length=1)
    risk_level: Literal["on_track", "at_risk", "behind", "incomplete", "unavailable"] = Field(
        description="Computed by the caller from real burndown math (points/days) — ai-service never guesses this."
    )
    days_remaining: Optional[int] = None
    committed_points: Optional[float] = None
    completed_points: Optional[float] = None
    total_points: Optional[float] = None
    issue_counts_by_status: Dict[str, int] = Field(default_factory=dict)
    blocked_issues: List[FlaggedIssueIn] = Field(default_factory=list)
    stale_issues: List[FlaggedIssueIn] = Field(default_factory=list)
    unestimated_issues: List[FlaggedIssueIn] = Field(default_factory=list)


class SprintPaceResponse(_CamelModel):
    summary: str
    recommendations: List[str]
    degraded: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float


@router.post("/sprint-pace", response_model=SprintPaceResponse, dependencies=_NEW_AI_WORK)
async def sprint_pace_endpoint(req: SprintPaceRequest, request: Request) -> SprintPaceResponse:
    """Turn pre-computed sprint stats and flagged issues into a short summary + recommendations.
    The model calculates nothing; on failure it falls back to a mechanical summary, never a 500.
    """
    caller = _tag_entrypoint_span(request, FEATURE_SPRINT_PACE)
    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    stats = SprintStats(
        sprint_name=req.sprint_name,
        risk_level=req.risk_level,
        days_remaining=req.days_remaining,
        committed_points=req.committed_points,
        completed_points=req.completed_points,
        total_points=req.total_points,
        issue_counts_by_status=req.issue_counts_by_status,
        blocked_issues=[FlaggedIssue(**i.model_dump()) for i in req.blocked_issues],
        stale_issues=[FlaggedIssue(**i.model_dump()) for i in req.stale_issues],
        unestimated_issues=[FlaggedIssue(**i.model_dump()) for i in req.unestimated_issues],
    )
    result = await _tagged(FEATURE_SPRINT_PACE, summarize_sprint_pace(client, model, stats))
    _record_usage(
        request, FEATURE_SPRINT_PACE, [_structured_output_charge(request, result)],
        detail={"degraded": result.degraded, "sprint_name": req.sprint_name},
        caller=caller,
    )
    return SprintPaceResponse(
        summary=result.insight.summary,
        recommendations=result.insight.recommendations,
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


class CommentIn_(_CamelModel):
    id: int
    author_name: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    content: str


# Generous on purpose: the server (not the caller) decides how many reach the model
# (_MAX_COMMENTS in app/comment_digest/service.py).
_MAX_COMMENTS_ACCEPTED = 200


class CommentDigestRequest(_CamelModel):
    comments: List[CommentIn_] = Field(max_length=_MAX_COMMENTS_ACCEPTED)


class AttributedPointOut(_CamelModel):
    text: str
    comment_ids: List[int]


class ActionItemOut_(AttributedPointOut):
    owner: Optional[str] = None


class CommentDigestResponse(_CamelModel):
    context: str
    discussion: str
    outcome: AttributedPointOut
    open_questions: List[AttributedPointOut]
    action_items: List[ActionItemOut_]
    degraded: bool
    skipped: bool
    truncated_count: int
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float


@router.post("/summarize-comments", response_model=CommentDigestResponse, dependencies=_NEW_AI_WORK)
async def comment_digest_endpoint(req: CommentDigestRequest, request: Request) -> CommentDigestResponse:
    """Summarize an issue's comment thread: context/discussion/outcome plus open questions and action
    items, each citing the comment(s) it came from.

    Takes comment bodies directly (the frontend already has them), so no membership check is needed —
    it's nothing the caller can't already see. Never 500s: `skipped` (too few comments) and
    `degraded` (model failure) both return an empty digest, flagged so the UI can say which.
    """
    caller = _tag_entrypoint_span(request, FEATURE_SUMMARIZE_COMMENTS)
    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    comments = [CommentIn(**c.model_dump()) for c in req.comments]
    result = await _tagged(FEATURE_SUMMARIZE_COMMENTS, summarize_comments(client, model, comments))
    _record_usage(
        request, FEATURE_SUMMARIZE_COMMENTS, [_structured_output_charge(request, result)],
        detail={
            "degraded": result.degraded,
            "skipped": result.skipped,
            "comment_count": len(comments),
            "truncated_count": result.truncated_count,
        },
        caller=caller,
    )
    return CommentDigestResponse(
        context=result.digest.context,
        discussion=result.digest.discussion,
        outcome=AttributedPointOut(**result.digest.outcome.model_dump()),
        open_questions=[AttributedPointOut(**p.model_dump()) for p in result.digest.open_questions],
        action_items=[ActionItemOut_(**a.model_dump()) for a in result.digest.action_items],
        degraded=result.degraded,
        skipped=result.skipped,
        truncated_count=result.truncated_count,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


class DescriptionReviewRequest(_CamelModel):
    # Sent as stored (HTML): the server strips it for the model but applies replacements to this
    # exact string, so it must be the real thing, not a pre-stripped version.
    description: str = Field(max_length=200_000)
    title: Optional[str] = Field(None, max_length=500, description="Context only; never reviewed.")


class FindingOut(_CamelModel):
    kind: str
    quote: str
    suggestion: str
    why: str


class DescriptionReviewResponse(_CamelModel):
    verdict: Literal["fine_as_is", "needs_changes"]
    findings: List[FindingOut]
    #: Null whenever there is nothing to apply — including the (common, expected) fine_as_is case.
    improved_description: Optional[str]
    #: Reported by the model but discarded because the quoted text wasn't in the description.
    #: Shown as a count: it's the honest signal for how grounded a run was.
    dropped_finding_count: int
    skipped: bool
    truncated: bool
    degraded: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float


@router.post("/description-review", response_model=DescriptionReviewResponse, dependencies=_NEW_AI_WORK)
async def description_review_endpoint(
    req: DescriptionReviewRequest, request: Request
) -> DescriptionReviewResponse:
    """"Improve description": a copy-edit pass that usually says the text is fine.

    Not a rewrite. The model only reports findings that quote exact spans; this service checks each
    quote against the real description and applies the replacements itself. That keeps the HTML and
    attachment chips intact and makes invented content impossible. Persists nothing, and
    `fine_as_is` is the common, correct result.
    """
    caller = _tag_entrypoint_span(request, FEATURE_IMPROVE_DESCRIPTION)
    client = request.app.state.instructor_client
    model = request.app.state.instructor_model
    result = await _tagged(FEATURE_IMPROVE_DESCRIPTION, review_description(client, model, req.description, req.title))
    _record_usage(
        request, FEATURE_IMPROVE_DESCRIPTION, [_structured_output_charge(request, result)],
        detail={
            "degraded": result.degraded,
            "skipped": result.skipped,
            "truncated": result.truncated,
            "verdict": result.review.verdict,
        },
        caller=caller,
    )
    return DescriptionReviewResponse(
        verdict=result.review.verdict,
        findings=[FindingOut(**f.model_dump()) for f in result.review.findings],
        improved_description=result.improved_description,
        dropped_finding_count=len(result.dropped_findings),
        skipped=result.skipped,
        truncated=result.truncated,
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


class PrecedentRequest(_CamelModel):
    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_key: str = Field(min_length=1, description="The current issue — excluded from its own results.")
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", max_length=200_000)


class PrecedentOut(_CamelModel):
    issue_key: str
    #: The real title from query_issues (the model never writes one).
    title: str
    relevance: str
    takeaway: Optional[str] = None


class PrecedentResponse(_CamelModel):
    verdict: Literal["found", "none_relevant"]
    precedents: List[PrecedentOut]
    #: How many candidates the model judged, so "none_relevant" over 8 reads as a real decision.
    candidates_considered: int
    #: Retrieval returned nothing above the similarity floor — a young space, not a failure.
    no_candidates: bool
    degraded: bool
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    #: Of `input_tokens`, the part served from the provider's prompt cache (billed ~10%).
    #: `input_tokens` is already the fresh count, so the raw provider total is input + cached + output.
    cached_input_tokens: int = 0
    estimated_cost_usd: float


@router.post("/link-similar-work-items", response_model=PrecedentResponse, dependencies=_NEW_AI_WORK)
async def precedents_endpoint(req: PrecedentRequest, request: Request) -> PrecedentResponse:
    """"Link similar work items" as a precedent finder: has this team already solved something
    close to this issue?

    It retrieves, so `space_ids` is a real permission boundary and is checked here. Not "similar
    text" (that's /search, already in Create Issue): it ranks by status and gives a reason to open
    each result. Uses title+description only, never comment threads (see app/precedent/service.py).
    Never 500s: any failure returns an empty report with `degraded=True`.
    """
    caller = _tag_entrypoint_span(request, FEATURE_LINK_SIMILAR_WORK_ITEMS)
    await _authorize_space_ids(request, req.space_ids)
    result = await _tagged(FEATURE_LINK_SIMILAR_WORK_ITEMS, find_precedents(
        request.app.state.instructor_client,
        request.app.state.instructor_model,
        request.app.state.retrieval,
        req.space_ids,
        req.issue_key,
        req.title,
        req.description,
    ))
    _record_usage(
        request, FEATURE_LINK_SIMILAR_WORK_ITEMS, [_structured_output_charge(request, result)],
        detail={
            "degraded": result.degraded,
            "issue_key": req.issue_key,
            "space_ids": req.space_ids,
            "candidates_considered": result.candidates_considered,
        },
        caller=caller,
    )
    return PrecedentResponse(
        verdict=result.report.verdict,
        precedents=[
            PrecedentOut(title=result.candidate_titles.get(p.issue_key, ""), **p.model_dump())
            for p in result.report.precedents
        ],
        candidates_considered=result.candidates_considered,
        no_candidates=result.no_candidates,
        degraded=result.degraded,
        latency_seconds=round(result.latency_seconds, 3),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_input_tokens=result.cached_input_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
    )


def _write_outcome(
    use_cache: bool, route_tool: str, abstained: bool, needs_clarification: bool = False,
    used_live_data: bool = False,
) -> str:
    """Label for CACHE_WRITES_TOTAL: why an answer was or wasn't cached.

    stored; skipped_route (a structured/exact read route, or any structured tool result in the
    answer, expected to be common); skipped_abstained (corpus
    didn't cover it — a content gap, not a cache issue); skipped_clarification; skipped_request
    (follow-up, attachment-scoped, or no watermark).
    """
    if not use_cache:
        return "skipped_request"
    if abstained:
        return "skipped_abstained"
    if needs_clarification:
        return "skipped_clarification"
    if not is_answer_cacheable(route_tool, abstained, needs_clarification, used_live_data):
        return "skipped_route"
    return "stored"


async def _safe_watermark(watermark_client, space_ids: List[int]) -> Optional[str]:
    """Freshness token for these spaces, or None if it can't be fetched right now.

    None means "can't check freshness", and callers then skip the cache and compute. Reusing without
    a check would bring back stale answers exactly when the system is already unhealthy.
    """
    try:
        return await watermark_client.watermark(space_ids)
    except Exception:  # noqa: BLE001
        logger.warning("index watermark lookup failed; bypassing the answer cache", exc_info=True)
        WATERMARK_FAILURES_TOTAL.inc()
        return None


async def _safe_cache_get(
    cache, question: str, space_ids: List[int], watermark: str, exact_only: bool = False,
    execution_mode: str = "single",
) -> Optional[dict]:
    # The cache is an optimization: if it fails (e.g. /embed timing out under load), treat it as a
    # miss rather than failing the request.
    start = time.perf_counter()
    try:
        return await cache.get(
            question, space_ids, watermark, exact_only=exact_only, execution_mode=execution_mode,
        )
    except Exception:  # noqa: BLE001
        logger.warning("cache.get() failed; treating as a cache miss", exc_info=True)
        return None
    finally:
        CACHE_LOOKUP_SECONDS.observe(time.perf_counter() - start)


async def _safe_cache_put(
    cache, question: str, space_ids: List[int], response: dict, watermark: str,
    execution_mode: str = "single",
) -> None:
    try:
        await cache.put(question, space_ids, response, watermark, execution_mode=execution_mode)
    except Exception:  # noqa: BLE001
        # The answer already went out; a failed cache write must not fail the request.
        logger.warning("cache.put() failed; response was still returned to the caller", exc_info=True)


async def _space_labels(request: Request, space_ids: List[int]) -> Dict[int, str]:
    """Space keys ("PRM") so an answer over several workspaces can name each one.

    The membership list was just fetched by `_authorize_space_ids` and is cached, so this is
    normally free. Best effort: without it the workspaces are called "workspace 1", "workspace 2".
    """
    user_id = request.headers.get("x-user-id")
    if not user_id or len(set(space_ids)) < 2:
        return {}
    try:
        spaces = await request.app.state.space_membership.authorized_spaces(
            user_id, request.headers.get("x-username") or "",
        )
    except Exception:  # noqa: BLE001 - labels are cosmetic; never fail the answer over them
        logger.warning("could not load space keys for the multi-agent answer", exc_info=True)
        return {}
    labels = {}
    for space in spaces or []:
        if space["id"] in space_ids and space.get("key"):
            labels[space["id"]] = space["key"]
    return labels


async def _unselected_space_notice(request: Request, req: AskRequest) -> Optional[AskResponse]:
    """A reply, with no model call, when the question names a workspace the user can open but didn't
    select ("What does PRM-20 do?" with only ATC selected).

    Searches only cover the selected workspaces, so the model never sees PRM; it answered that
    question with "PRM-20 appears to correspond to ATC-20". Only the user's own spaces are matched,
    so this never reveals a space they can't open. Best effort: no membership list, no check.
    """
    user_id = request.headers.get("x-user-id")
    if not user_id:
        return None
    try:
        spaces = await request.app.state.space_membership.authorized_spaces(
            user_id, request.headers.get("x-username") or "",
        )
    except Exception:  # noqa: BLE001 - a missed notice must not fail the ask
        logger.warning("could not load space keys for the scope check", exc_info=True)
        return None
    named = []
    for space in spaces or []:
        key = space.get("key")
        if key and space["id"] not in req.space_ids and _question_names_space(req.question, key):
            named.append(key)
    if not named:
        return None
    keys = ", ".join(named)
    if any("\u4e00" <= c <= "\u9fff" for c in req.question):
        text = f"問題提到 {keys}，但這次沒有勾選 {keys}，所以搜尋不到它的資料。請勾選 {keys} 後再問一次。"
    else:
        text = (f"Your question mentions {keys}, which isn't selected, so its issues can't be searched. "
                f"Select {keys} and ask again.")
    return AskResponse(
        answer=text, abstained=True, retrieval_rounds=0, queries_used=[], citations=[],
        input_tokens=0, output_tokens=0, estimated_cost_usd=0.0,
    )


def _question_names_space(question: str, key: str) -> bool:
    """An issue key of that space ("PRM-20", any case), or the space key itself as a word ("PRM")."""
    if re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}-\d+", question, re.IGNORECASE):
        return True
    # The bare key is matched case-sensitively, so a space keyed "WEB" doesn't catch "web".
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(key)}(?![A-Za-z0-9])", question) is not None


async def _authorize_space_ids(request: Request, space_ids: List[int]) -> None:
    """Reject a request for a space the authenticated user isn't a member of.

    Unlike the cache, this fails closed: if jira-backend can't be reached, return 502 rather than
    silently allowing access.
    """
    user_id = request.headers.get("x-user-id")
    username = request.headers.get("x-username")
    checker = request.app.state.space_membership
    try:
        await checker.validate(user_id, username, space_ids)
    except SpaceMembershipError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - jira-backend unreachable etc.
        logger.error("space membership check failed", exc_info=True)
        raise HTTPException(status_code=502, detail="space membership check failed") from exc


async def _authorize_workflow_state(request: Request, values: dict) -> None:
    """Only the user who started a durable workflow may read or resume it.

    A thread UUID identifies a workflow; it isn't a credential. Internal callers without user
    headers (eval scripts) are allowed, same as `_authorize_space_ids`.
    """
    user_id, _ = _caller_identity(request)
    if not user_id:
        return
    if str(values.get("user_id") or "") != user_id:
        raise HTTPException(status_code=403, detail="workflow is not owned by the authenticated user")
    await _authorize_space_ids(request, [values["space_id"]])
