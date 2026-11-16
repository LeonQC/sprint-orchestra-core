"""Prometheus metrics and a per-request id for log correlation (mirrors
vectorization-service/app/observability.py). Stages here are LLM calls, retrieval calls and cache
lookups.
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
# The end user this request spends money for, sent to the provider as OpenAI's `user` field. A
# ContextVar because it's needed deep inside an httpx transport, far below anything that saw the HTTP
# request; threading it through every call site is how it went unused before. Empty for eval scripts (then
# the field is omitted).
_billing_user: ContextVar[str] = ContextVar("billing_user", default="")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests handled", ["method", "path", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds", "End-to-end HTTP request latency", ["method", "path"]
)
LLM_CALL_SECONDS = Histogram("agent_llm_call_seconds", "One LLMClient.next_turn() call's latency")
RETRIEVAL_CALL_SECONDS = Histogram("agent_retrieval_call_seconds", "One search_knowledge_base call's latency")
CACHE_LOOKUP_SECONDS = Histogram("agent_cache_lookup_seconds", "Semantic cache get() latency")
# How many corrective-retrieval rounds a question took — shows how often it actually fires.
RETRIEVAL_ROUNDS = Histogram(
    "agent_retrieval_rounds", "Corrective-retrieval rounds per /ask call", buckets=(0, 1, 2, 3, 4, 5)
)
TOOL_FAILURES_TOTAL = Counter(
    "agent_tool_failures_total",
    "Agent tool calls that failed before returning usable evidence",
    ["tool"],
)
ROUTING_DECISIONS_TOTAL = Counter(
    "agent_routing_decisions_total",
    "Initial Ask-AI data-source decisions, by decision layer and selected tool",
    ["source", "tool"],
)
# LLM calls the gateway answered with a fallback model because the requested one failed. Each one is
# also priced at the fallback model's rate, so a sustained rate is a cost change, not just an outage.
LLM_FALLBACKS_TOTAL = Counter(
    "llm_fallbacks_total",
    "LLM calls answered by a gateway fallback model instead of the requested one",
    ["requested", "served"],
)
# Ask AI responses that ran in a reduced mode, by reason (app/degradation.py).
ASK_DEGRADED_TOTAL = Counter(
    "agent_ask_degraded_total", "POST /ask responses served in a degraded mode", ["reason"]
)
# 1 while app/llm/circuit_breaker.py is failing calls to this gateway model without sending them.
LLM_CIRCUIT_OPEN = Gauge("llm_circuit_open", "1 while the circuit breaker for a gateway model is open", ["model"])
CACHE_HITS_TOTAL = Counter("agent_cache_hits_total", "POST /ask responses served from cache", ["hit_type"])
# Single agent LLM calls the LiteLLM gateway replayed from its response cache (no charge), as opposed
# to whole /ask answers from the semantic cache above.
LLM_RESPONSE_CACHE_HITS_TOTAL = Counter(
    "agent_llm_response_cache_hits_total", "Agent LLM calls replayed from the gateway response cache", ["model"]
)
# External agents calling the MCP server. `rejected` = a ToolError the caller can act on (not signed
# in, unknown space, issue not found); `error` = an unexpected failure.
MCP_TOOL_CALLS_TOTAL = Counter(
    "mcp_tool_calls_total", "MCP tool calls from external agents, by tool and outcome", ["tool", "outcome"]
)
MCP_TOOL_CALL_SECONDS = Histogram("mcp_tool_call_seconds", "One MCP tool call's latency", ["tool"])
# The freshness check itself failing — not a miss. It makes every request bypass the cache, which
# looks like a useless cache from the hit rate alone (and sends on-call to Redis, the wrong place).
WATERMARK_FAILURES_TOTAL = Counter(
    "agent_cache_watermark_failures_total",
    "Index-watermark lookups that failed, forcing the answer cache to be bypassed",
)
# Cache writes by outcome. Reads alone can't tell "nothing gets cached" from "nobody repeats a
# question" — both show 0% hits.
CACHE_WRITES_TOTAL = Counter(
    "agent_cache_writes_total",
    "POST /ask answers considered for caching, by outcome",
    ["outcome"],
)
# How often the multi-agent planner revised after a critique — is the critic finding real problems
# or rubber-stamping?
PLANNING_REVISION_ROUNDS = Histogram(
    "planning_revision_rounds", "Critique-driven revision rounds per multi-agent /plan-epic call",
    buckets=(0, 1, 2, 3),
)
# Grounding outcomes per /plan-epic call. Split by status: only "grounded" can anchor estimates, and
# "retrieval_failed" (an outage) would otherwise look like "no_history" (a young space).
PLANNING_GROUNDING_TOTAL = Counter(
    "planning_grounding_total",
    "Grounding retrievals for /plan-epic, by outcome",
    ["status"],
)
# How many examples each grounded plan got (eight vs one is a big difference).
PLANNING_GROUNDING_EXEMPLARS = Histogram(
    "planning_grounding_exemplars", "Past issues shown to the planner per /plan-epic call",
    buckets=(0, 1, 2, 3, 5, 8, 13),
)
# Counter of $ spent (only grows), the same number as GET /stats' total_cost_usd, so Grafana can graph it.
COST_USD_TOTAL = Counter("agent_cost_usd_total", "Cumulative estimated $ spent on LLM calls")
# Sprint-recovery workflows that hit the escalation cap — worth graphing over time.
SPRINT_RECOVERY_ESCALATIONS_TOTAL = Counter(
    "sprint_recovery_escalations_total",
    "Sprint-recovery workflows that exhausted their escalation cap and handed off to a human",
    ["space_id"],
)
# Index catch-up timeouts. A few is normal jitter; a rising rate means the Kafka/indexing pipeline is
# degraded.
SPRINT_RECOVERY_INDEX_CATCH_UP_TIMEOUTS_TOTAL = Counter(
    "sprint_recovery_index_catch_up_timeouts_total",
    "Reevaluations that gave up waiting for the read model to reflect this workflow's own writes",
    ["space_id"],
)


def get_request_id() -> str:
    return _request_id.get()


def get_billing_user() -> str:
    return _billing_user.get()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id.set(req_id)
        user_token = _billing_user.set(request.headers.get("x-user-id") or "")
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = req_id
            return response
        finally:
            path = request.url.path
            HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path).observe(time.perf_counter() - start)
            HTTP_REQUESTS_TOTAL.labels(request.method, path, str(status_code)).inc()
            _billing_user.reset(user_token)
            _request_id.reset(token)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
# Requests refused by the per-user AI limits (app/auth/user_quota.py). reason: rate | daily_budget
USER_QUOTA_REJECTIONS_TOTAL = Counter(
    "ai_user_quota_rejections_total", "AI requests refused by a per-user limit", ["reason"],
)
# Quota counts made in-process because Redis was unreachable. Non-zero = limits are per replica now.
USER_QUOTA_LOCAL_FALLBACK_TOTAL = Counter(
    "ai_user_quota_local_fallback_total", "Per-user limit steps counted locally because Redis was down", ["step"],
)
