"""Prometheus metrics (GET /metrics) and a per-request id for log correlation.

Metrics answer "what's P95 search latency this hour" as a query, not a grep. The request id lives in
a ContextVar set by the middleware, so log calls deep in the stack can include it without passing it
around. Cross-service tracing is in app/tracing.py.
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

_request_id: ContextVar[str] = ContextVar("request_id", default="-")

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests handled", ["method", "path", "status"]
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds", "End-to-end HTTP request latency", ["method", "path"]
)
# Stage-level breakdowns for POST /search specifically — the endpoint with a nontrivial pipeline
# (embed -> vector search -> lexical search -> optional rerank), where "which stage is slow" is the
# actually useful diagnostic question. See app/api/routes.py.
EMBED_SECONDS = Histogram("search_embed_seconds", "Query-embedding latency within /search")
# Labeled by mode rather than split into separate histograms: routes.py calls exactly one of
# search_vector/search_bm25/search_hybrid per request, so "which stage" and "which mode" are the
# same question here — one histogram with a label is simpler than three nearly-identical ones.
RETRIEVAL_SECONDS = Histogram("search_retrieval_seconds", "Store-level search latency", ["mode"])
RERANK_SECONDS = Histogram("search_rerank_seconds", "Cross-encoder reranking latency")
# Attachments indexed with OCR text because a VLM call failed, and what the retry worker did with
# them (app/ingest/vlm_retry.py).
VLM_DEGRADED_ATTACHMENTS_TOTAL = Counter(
    "ingest_vlm_degraded_attachments_total", "Attachments indexed without their VLM pass and queued for retry"
)
VLM_RETRIES_TOTAL = Counter("ingest_vlm_retries_total", "VLM retry attempts by outcome", ["outcome"])
INGEST_UNCHANGED_TOTAL = Counter(
    "ingest_unchanged_sources_total",
    "Re-ingested sources whose chunks matched what was stored, so embedding was skipped",
    ["chunk_type"],
)
SEARCH_DEGRADED_TOTAL = Counter(
    "search_degraded_total", "Searches served BM25-only because the query embedding failed",
    ["requested_mode"],
)


def get_request_id() -> str:
    return _request_id.get()


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id.set(req_id)
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
            _request_id.reset(token)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
