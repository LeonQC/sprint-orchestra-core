"""FastAPI entrypoint for ai-service: wires the LLM client (app/llm/factory.py) and the retrieval
client into the CragAgent, served at POST /ask, plus the other AI features.
"""
from __future__ import annotations

import logging
from typing import List, Optional
from contextlib import AsyncExitStack, asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agent.crag_loop import CragAgent
from app.agent.history_summary import HistoryCompactor
from app.agent.mode_router import ModeRouter
from app.agent.retrieval_tool import RetrievalClient
from app.api.routes import router
from app.api.sprint_recovery_routes import router as sprint_recovery_router
from app.api.investigation_routes import router as investigation_router
from app.auth.request_auth import JwtVerifier, RequestAuthMiddleware
from app.auth.space_membership import build_space_membership_checker
from app.auth import user_quota
from app.auth.user_quota import UserQuota
from app.cache.embedding_client import EmbeddingClient
from app.cache.semantic_cache import build_cache
from app.cache.watermark_client import IndexWatermarkClient, NoopWatermarkClient
from app.config import get_settings
from app.drafting.instructor_client import build_instructor_client
from app.llm.circuit_breaker import ModelUnavailableError
from app.llm.factory import build_llm_client, build_router_llm_client, describe_llm_runtime
from app.investigation.graph import MultiAgentRag
from app.logging_config import configure_logging
from app.mcp_server.server import install_mcp_route, run_mcp_server
from app.observability import ObservabilityMiddleware, metrics_response
from app.planning.active_threads import ensure_plan_epic_active_threads_table
from app.planning.checkpoint import build_checkpointer
from app.planning.graph import build_planning_graph
from app.planning.grounding import GroundingContext, gather_exemplars, gather_requirement_exemplars
from app.planning.patterns import AssessmentResult, assess_history
from app.planning.jira_commit_client import JiraCommitClient
from app.planning.rollout_graph import build_rollout_graph
from app.planning.service import PlanGenerationError
from app.sprint_recovery.active_threads import ensure_active_threads_table
from app.sprint_recovery.graph import build_sprint_recovery_graph
from app.sprint_recovery.jira_actions_client import JiraActionsClient
from app.stats import ServiceStats
from app.tracing import CallerAttributeMiddleware, configure_tracing, instrument_app

logger = logging.getLogger(__name__)


def _workflow_checkpoint_db_url(settings) -> str | None:
    """Pick the checkpoint DSN for the enabled durable workflows. If both are on they share one
    checkpointer, so their DSNs must match — silently using one would connect to the wrong database.
    """
    if settings.epic_rollout_enabled and settings.sprint_recovery_enabled:
        if settings.epic_rollout_checkpoint_db_url != settings.sprint_recovery_checkpoint_db_url:
            raise ValueError(
                "AI_EPIC_ROLLOUT_CHECKPOINT_DB_URL and "
                "AI_SPRINT_RECOVERY_CHECKPOINT_DB_URL must match when both workflows are enabled"
            )
        return settings.epic_rollout_checkpoint_db_url
    if settings.epic_rollout_enabled:
        return settings.epic_rollout_checkpoint_db_url
    if settings.sprint_recovery_enabled:
        return settings.sprint_recovery_checkpoint_db_url
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    logger.info(
        "starting %s (llm_provider=%s, model=%s, router_provider=%s, router_model=%s)",
        settings.service_name,
        settings.llm_provider,
        settings.agent_model,
        settings.router_llm_provider,
        settings.router_model or settings.agent_model,
    )
    # Sets the real TracerProvider. Fine to do here even though `instrument_app` ran at import time:
    # OTel's ProxyTracer resolves to this provider once it's set.
    configure_tracing(settings)

    llm = build_llm_client(settings)
    router_llm, router_model = build_router_llm_client(settings, llm)
    model_provider, model_deployment = describe_llm_runtime(
        settings.llm_provider,
        settings.openai_base_url if settings.llm_provider == "openai_compatible" else "",
        settings.agent_model,
    )
    if settings.router_llm_provider == "inherit":
        router_provider, router_deployment = model_provider, model_deployment
    else:
        router_provider, router_deployment = describe_llm_runtime(
            settings.router_llm_provider,
            settings.router_openai_base_url
            if settings.router_llm_provider == "openai_compatible" else "",
            settings.router_model,
        )
    retrieval = RetrievalClient(settings.vectorization_service_url)
    agent = CragAgent(
        llm, retrieval, settings.max_tool_iterations, settings.retrieval_top_k, model=settings.agent_model,
        faithfulness_check_enabled=settings.faithfulness_check_enabled,
        parent_child_tool_enabled=settings.parent_child_tool_enabled,
        router_llm=router_llm, router_model=router_model,
        model_provider=model_provider, model_deployment=model_deployment,
        router_provider=router_provider, router_deployment=router_deployment,
    )
    embed_client = EmbeddingClient(settings.vectorization_service_url)
    # decode_responses=False: cache payloads are JSON strings we decode ourselves in
    # semantic_cache.py; list members (entry ids) come back as bytes there too, handled explicitly.
    redis_client = (
        redis.from_url(settings.redis_url)
        if settings.cache_enabled or settings.user_quota_enabled else None
    )
    cache = build_cache(settings, redis_client, embed_client)
    quota = (
        UserQuota(
            redis_client, settings.user_quota_requests_per_minute, settings.user_quota_daily_budget_usd,
        )
        if settings.user_quota_enabled else None
    )
    user_quota.install(quota)
    await cache.ensure_index()  # idempotent; creates the RediSearch vector index once at startup
    # Freshness token for the cache; only created when the cache is on.
    index_watermark = (
        IndexWatermarkClient(
            settings.vectorization_service_url,
            ttl_seconds=settings.cache_watermark_ttl_seconds,
        )
        if settings.cache_enabled
        else NoopWatermarkClient()
    )
    instructor_client, instructor_model = build_instructor_client(settings)
    space_membership = build_space_membership_checker(settings)
    # Compiled once, shared across concurrent requests — safe because per-request data flows through
    # ainvoke's state argument, not instance attributes (see app/planning/graph.py's docstring).
    planning_graph = (
        build_planning_graph(instructor_client, instructor_model)
        if settings.epic_planning_multiagent_enabled
        else None
    )

    # A plain async callable for `plan_node`, or None when the flag is off (so the node never calls
    # out). Keeps the graph and its tests unaware of the flag.
    async def _gather_planning_grounding(
        space_id: int,
        proposal: str,
        requirements: Optional[List[str]] = None,
        on_progress=None,
    ) -> GroundingContext:
        if requirements:
            return await gather_requirement_exemplars(
                retrieval, [space_id], proposal, requirements,
                limit=settings.epic_planning_grounding_exemplars,
                per_requirement=settings.epic_planning_requirement_exemplars,
                on_progress=on_progress,
                include_comments=(
                    settings.epic_planning_pattern_extraction_enabled
                    and settings.epic_planning_pattern_comments_enabled
                ),
            )
        return await gather_exemplars(
            retrieval, [space_id], proposal,
            limit=settings.epic_planning_grounding_exemplars,
            # Only fetched when something will read them, and only pattern extraction ever does.
            include_comments=(
                settings.epic_planning_pattern_extraction_enabled
                and settings.epic_planning_pattern_comments_enabled
            ),
        )

    planning_grounding_gatherer = (
        _gather_planning_grounding if settings.epic_planning_grounding_enabled else None
    )

    async def _assess_planning_history(proposal: str, context: GroundingContext) -> AssessmentResult:
        return await assess_history(instructor_client, instructor_model, proposal, context)

    planning_history_assessor = (
        _assess_planning_history if settings.epic_planning_pattern_extraction_enabled else None
    )

    exit_stack = AsyncExitStack()
    rollout_graph = None
    jira_commit_client = None
    # One checkpoint store for both durable workflows; each mints its own thread ids, so they can't
    # collide.
    checkpointer = None
    checkpoint_db_url = _workflow_checkpoint_db_url(settings)
    if checkpoint_db_url is not None:
        checkpointer = await build_checkpointer(checkpoint_db_url, exit_stack)
    if settings.epic_rollout_enabled:
        jira_commit_client = JiraCommitClient(settings.jira_backend_url, settings.internal_service_token)
        rollout_graph = build_rollout_graph(
            instructor_client,
            instructor_model,
            space_membership,
            jira_commit_client,
            # Read at call time, not captured now — see build_rollout_graph's docstring.
            lambda: app.state.planning_graph,
            planning_grounding_gatherer,
            planning_history_assessor,
        ).compile(checkpointer=checkpointer)
        if checkpoint_db_url is not None:
            await ensure_plan_epic_active_threads_table(checkpoint_db_url)

    sprint_recovery_graph = None
    jira_actions_client = None
    # Created before the graph so its real token/cost usage lands in the same /stats and Prometheus
    # counters as every other LLM call.
    stats = ServiceStats()
    if settings.sprint_recovery_enabled:
        jira_actions_client = JiraActionsClient(settings.jira_backend_url, settings.internal_service_token)
        sprint_recovery_graph = build_sprint_recovery_graph(
            instructor_client, instructor_model, space_membership, retrieval, jira_actions_client,
            on_usage=stats.record,
            llm_runtime=(model_provider, model_deployment),
        ).compile(checkpointer=checkpointer)
        # Lets the UI find "the thread for sprint X" after a crash; the checkpointer alone can't look
        # that up (see active_threads.py).
        if checkpoint_db_url is not None:
            await ensure_active_threads_table(checkpoint_db_url)

    app.state.settings = settings
    app.state.llm = llm
    app.state.retrieval = retrieval
    app.state.agent = agent
    app.state.investigator = MultiAgentRag(agent)
    # Same small classifier client as the capability router (the answer client when "inherit").
    app.state.mode_router = ModeRouter(
        router_llm,
        enabled=settings.adaptive_rag_routing_enabled,
        model=router_model, provider=router_provider, deployment=router_deployment,
    )
    # Long chats: the turns before the latest answer are summarized once by the answer model.
    app.state.history_compactor = HistoryCompactor(
        llm, model=settings.agent_model, provider=model_provider, deployment=model_deployment,
        over_tokens=settings.history_summary_over_tokens,
    )
    app.state.stats = stats
    app.state.cache = cache
    app.state.index_watermark = index_watermark
    app.state.instructor_client = instructor_client
    app.state.instructor_model = instructor_model
    # Pricing for structured-output features: same provider/model as the agent.
    app.state.model_provider = model_provider
    app.state.model_deployment = model_deployment
    app.state.space_membership = space_membership
    app.state.user_quota = quota
    app.state.planning_graph = planning_graph
    app.state.rollout_graph = rollout_graph
    app.state.epic_rollout_checkpoint_db_url = checkpoint_db_url
    app.state.sprint_recovery_graph = sprint_recovery_graph
    app.state.sprint_recovery_checkpoint_db_url = checkpoint_db_url
    if settings.mcp_enabled:
        await exit_stack.enter_async_context(run_mcp_server(app))

    try:
        yield
    finally:
        if router_llm is not llm:
            await router_llm.aclose()
        await llm.aclose()
        await retrieval.aclose()
        await embed_client.aclose()
        await instructor_client.client.close()
        await space_membership.aclose()
        if jira_commit_client is not None:
            await jira_commit_client.aclose()
        if jira_actions_client is not None:
            await jira_actions_client.aclose()
        await exit_stack.aclose()
        user_quota.install(None)
        await index_watermark.aclose()
        if redis_client is not None:
            await redis_client.aclose()
        logger.info("stopped %s", settings.service_name)


app = FastAPI(title="AI Service (Agentic RAG)", version="0.1.0", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
# Puts the caller in the OTel context before any span exists, so SDK spans with token counts also
# carry who spent them.
app.add_middleware(CallerAttributeMiddleware)
# Added last so it runs first: nothing else sees a request, or its X-User-Id, before it's verified.
# No CORS: browsers reach this through the same-origin dev proxy / CloudFront, never directly.
app.add_middleware(
    RequestAuthMiddleware,
    verifier=JwtVerifier(get_settings().jwt_jwks_url, get_settings().jwt_issuer),
    internal_token=get_settings().internal_service_token,
)
app.include_router(router)
app.include_router(sprint_recovery_router)
app.include_router(investigation_router)


@app.exception_handler(ModelUnavailableError)
async def _model_unavailable(_: Request, exc: ModelUnavailableError) -> JSONResponse:
    # 503 + `detail`: the frontend shows `detail` as the error text.
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PlanGenerationError)
async def _plan_generation_failed(_: Request, exc: PlanGenerationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})
install_mcp_route(app)

# Tracing must be installed here at import time, not in `lifespan`: FastAPIInstrumentor wraps the
# middleware stack, which is already built when lifespan runs (that was a silent no-op, and every
# LLM call became its own trace). The TracerProvider is still set in lifespan; ProxyTracer bridges it.
instrument_app(app, get_settings())


@app.get("/metrics")
async def metrics():
    return metrics_response()
