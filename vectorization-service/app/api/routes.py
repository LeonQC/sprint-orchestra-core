"""HTTP surface: health, ingestion stats, and the raw retrieval API.

This service owns the index, so it offers query primitives (/search, /issues/*). It does no
answer-level logic (synthesis, re-retrieval, citations for a chat UI) — ai-service composes these
as tools.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from datetime import date, datetime
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pydantic import BaseModel, Field, field_validator

from app.observability import (
    EMBED_SECONDS, RERANK_SECONDS, RETRIEVAL_SECONDS, SEARCH_DEGRADED_TOTAL, get_request_id,
)
from app.tracing import get_tracer

logger = logging.getLogger(__name__)
router = APIRouter()


def _blank_to_none(value: object) -> object:
    """Treat "" as "not provided" for an optional datetime filter.

    Strict-schema tool-calling models (gpt-5.6-luna) fill every property and use "" for "unset";
    without this the whole query got a 422.
    """
    return None if value == "" else value


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    """Liveness plus a ping of both retrieval backends (for Docker/K8s probes).

    `index_drift` = chunks in Postgres minus chunks in OpenSearch. The two writes aren't one
    transaction, so if OpenSearch was down during a write, BM25 silently stops finding that chunk.
    Non-zero drift means run POST /search/reindex.
    """
    consumer = request.app.state.consumer
    store = request.app.state.store
    db_ok = True
    pg_count = None
    try:
        pg_count = await store.count()
    except Exception:  # noqa: BLE001
        db_ok = False

    search_ok = True
    es_count = None
    try:
        es_count = await store.search_index.count()
    except Exception:  # noqa: BLE001
        search_ok = False

    healthy = db_ok and search_ok and consumer.stats.running
    return {
        "status": "ok" if healthy else "degraded",
        "consumer_running": consumer.stats.running,
        "db_ok": db_ok,
        "search_ok": search_ok,
        "chunk_count": pg_count,
        "indexed_count": es_count,
        "index_drift": (pg_count - es_count) if (pg_count is not None and es_count is not None) else None,
    }


@router.get("/stats")
async def stats(request: Request) -> dict:
    """Ingestion counters + current vector count, for dashboards and debugging."""
    consumer = request.app.state.consumer
    store = request.app.state.store
    s = consumer.stats
    try:
        vector_count = await store.count()
    except Exception:  # noqa: BLE001
        vector_count = None
    # Attachments indexed with OCR text while the VLM was down, waiting for app/ingest/vlm_retry.py.
    vlm_retry_pending = None
    if getattr(request.app.state, "vlm_retry_worker", None) is not None:
        try:
            vlm_retry_pending = await store.count_vlm_retries()
        except Exception:  # noqa: BLE001
            pass
    return {
        "topics": s.topics,
        "processed": s.processed,
        "skipped_poison": s.skipped_poison,
        "failed": s.failed,
        "last_error": s.last_error,
        "vector_count": vector_count,
        "vlm_retry_pending": vlm_retry_pending,
    }


class EmbedRequest(BaseModel):
    text: str = Field(min_length=1)


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


@router.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    """Expose the embedder, so ai-service's semantic cache can embed questions without its own
    embedding provider and credentials.
    """
    embedder = request.app.state.embedder
    (vector,) = await embedder.embed([req.text])
    return EmbedResponse(embedding=vector, dim=len(vector))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language or keyword query.")
    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    limit: int = Field(default=5, ge=1, le=50)
    # "bm25" is OpenSearch's lexical leg alone; "vector" is pgvector alone; "hybrid" fuses both.
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"


class SearchHitOut(BaseModel):
    id: str
    chunk_type: str
    issue_id: int
    issue_key: str
    space_id: int
    source_id: int
    content: str
    score: float
    retrievers: List[str]
    page_number: int | None = None
    provenance: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    hits: List[SearchHitOut]
    # The mode that actually ran: "bm25" on a hybrid/vector request whose query embedding failed.
    mode: str
    reranked: bool = False
    degraded: bool = False


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request) -> SearchResponse:
    """Raw retrieval: embed the query, run vector / bm25 / hybrid search, return ranked chunks.

    If the query embedding fails, a hybrid/vector request degrades to BM25 and says so
    (`degraded`, `mode`) instead of failing.

    Hits carry issue_key, source_id and chunk_type for citations, and `retrievers` shows which
    signal found each one. With reranking on, a wider candidate pool is fetched first so the
    cross-encoder has room to reorder (app/db/reranker.py).
    """
    store = request.app.state.store
    embedder = request.app.state.embedder
    reranker = request.app.state.reranker
    settings = request.app.state.settings

    rerank_active = settings.rerank_enabled
    fetch_limit = max(req.limit, settings.rerank_candidate_pool) if rerank_active else req.limit
    stage_ms: dict = {}

    # On every stage span: mode and the two limits explain most slow searches (rerank over 100
    # candidates is slower than over 20). The query text is left out on purpose — user input
    # shouldn't pile up in a trace collector.
    span_attributes = {
        "search.mode": req.mode,
        "search.limit": req.limit,
        "search.fetch_limit": fetch_limit,
        "search.space_count": len(req.space_ids),
    }

    # OpenInference span kind per stage, so traces show "embedding"/"retriever"/"reranker" instead of
    # a generic SPAN. A stage missing here fails loudly (KeyError).
    _STAGE_KIND = {
        "embed": OpenInferenceSpanKindValues.EMBEDDING.value,
        "retrieval": OpenInferenceSpanKindValues.RETRIEVER.value,
        "rerank": OpenInferenceSpanKindValues.RERANKER.value,
    }

    def _stage(name: str):
        # One timer per stage feeds the histogram, the log line and the span, so all three agree.
        attributes = {**span_attributes, SpanAttributes.OPENINFERENCE_SPAN_KIND: _STAGE_KIND[name]}
        return _StageTimer(name, stage_ms, attributes)

    served_mode = req.mode
    try:
        embedding = None
        if req.mode != "bm25":
            try:
                with _stage("embed") as t:
                    (embedding,) = await embedder.embed([req.query])
                EMBED_SECONDS.observe(t.seconds)
            except Exception as exc:  # noqa: BLE001 - degrade to keyword search, don't fail the request
                # Never fall back to another embedding model: its vectors live in a different space
                # from the index, so search would return confident garbage instead of an error.
                logger.warning(
                    "query embedding failed; serving BM25-only results",
                    extra={"request_id": get_request_id(), "requested_mode": req.mode, "error": str(exc)},
                )
                SEARCH_DEGRADED_TOTAL.labels(req.mode).inc()
                served_mode = "bm25"

        with _stage("retrieval") as t:
            if served_mode == "bm25":
                # No embedding on this path: BM25 is a pure keyword query.
                hits = await store.search_bm25(req.query, req.space_ids, fetch_limit)
            elif served_mode == "vector":
                hits = await store.search_vector(embedding, req.space_ids, fetch_limit)
            else:
                # Pass `candidate_pool` too: it defaults to 20, so raising
                # VEC_RERANK_CANDIDATE_POOL to 100 had no effect, and a chunk ranked #31 by vector
                # search could never reach fusion or the reranker.
                hits = await store.search_hybrid(
                    embedding, req.query, req.space_ids, fetch_limit, candidate_pool=fetch_limit
                )
        RETRIEVAL_SECONDS.labels(served_mode).observe(t.seconds)

        if rerank_active:
            with _stage("rerank") as t:
                hits = await reranker.rerank(req.query, hits, req.limit)
            RERANK_SECONDS.observe(t.seconds)
        else:
            hits = hits[: req.limit]

        # Build the response inside the try, so a schema mismatch is a clean 502. Outside it, it
        # became an unhandled 500 right after logging "search completed".
        response = SearchResponse(
            hits=[SearchHitOut(**asdict(hit)) for hit in hits],
            mode=served_mode,
            reranked=rerank_active,
            degraded=served_mode != req.mode,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a clean 502, don't leak internals
        # Note: to the caller, a 502 here looks like "nothing found" (ai-service abstains either way).
        # When "RAG can't find my attachment", check this exception before blaming OCR or embeddings —
        # e.g. provenance once came back as a string instead of a dict (see db/pool.py codecs).
        logger.exception("search failed")
        raise HTTPException(status_code=502, detail="search request failed") from exc

    logger.info(
        "search completed",
        extra={"request_id": get_request_id(), "mode": served_mode, "reranked": rerank_active,
               "hit_count": len(hits), "stage_ms": stage_ms},
    )
    return response


class IssueQueryRequest(BaseModel):
    """Structured issue query (counting/filtering/recency). Not natural language: ai-service already
    turned the question into these filters, so this is plain SQL — which is what makes "how many
    bugs" exact.
    """

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_keys: Optional[List[str]] = Field(None, description="Restrict to these exact issue keys.")
    parent_keys: Optional[List[str]] = Field(None, description="Only children of these issues, e.g. ['ATC-52'].")
    issue_types: Optional[List[str]] = Field(None, description="Filter to these issue types, e.g. ['bug'].")
    statuses: Optional[List[str]] = Field(None, description="Filter to these statuses, e.g. ['blocked'].")
    priorities: Optional[List[str]] = Field(None, description="Filter to these priorities, e.g. ['high'].")
    sprint_ids: Optional[List[int]] = Field(None, description="Filter to issues in these sprints, by id.")
    sprint_names: Optional[List[str]] = Field(None, description="Filter to issues in these sprints, by name.")
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None
    # Constrained, not free text — this is what gets formatted into the ORDER BY, so it must be a
    # closed enum (also enforced by jira-backend).
    order_by: Literal["created_at", "updated_at"] = "updated_at"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=20, ge=1, le=200)

    _blank_dates = field_validator(
        "created_after", "created_before", "updated_after", "updated_before", mode="before"
    )(_blank_to_none)


class IssueRowOut(BaseModel):
    issue_id: int
    issue_key: str
    space_id: int
    issue_type: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    sprint_id: Optional[int]
    sprint_name: Optional[str]
    title: Optional[str]
    parent_key: Optional[str]
    assignee_id: Optional[int]
    assignee_name: Optional[str]
    story_points: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class IssueQueryResponse(BaseModel):
    total_count: int
    counts_by_type: dict
    counts_by_status: dict
    counts_by_assignee: dict = {}
    issues: List[IssueRowOut]


class ReindexResponse(BaseModel):
    indexed: int
    index: str


@router.post("/search/reindex", response_model=ReindexResponse)
async def reindex(request: Request) -> ReindexResponse:
    """Rebuild the OpenSearch chunk index from Postgres (the source of truth).

    The repair path for the accepted gap: OpenSearch is written right after the Postgres commit, so
    an outage leaves chunks BM25 can't see (`/healthz` shows `index_drift`). Cheaper than a
    transactional outbox, and fine because a stale lexical index hurts ranking, not data.

    Also how the index is first filled and how mapping changes roll out (change mapping, delete
    index, call this). Idempotent. Doesn't remove documents whose Postgres row is gone — delete the
    index first for that.
    """
    store = request.app.state.store
    index = store.search_index
    await index.ensure_index()
    indexed = 0
    async for batch in store.iter_chunks_for_reindex():
        await index.index_chunks(batch)
        indexed += len(batch)
    await index.refresh()
    logger.info("reindex completed", extra={"request_id": get_request_id(), "indexed": indexed})
    return ReindexResponse(indexed=indexed, index=index.index_name)


@router.post("/issues/query", response_model=IssueQueryResponse)
async def query_issues(req: IssueQueryRequest, request: Request) -> IssueQueryResponse:
    """Structured issue query: exact count, type/status breakdown, ordered sample.

    /search answers "what's relevant"; this answers "how many / which / most recent", which top-K
    can't do honestly. Same space-scoped boundary (ai-service injects space_ids from the
    authenticated user, never the model).
    """
    try:
        payload = await request.app.state.structured.query("issues/query", req.model_dump(mode="json"))
        return IssueQueryResponse.model_validate(payload)
    except Exception as exc:
        logger.exception("authoritative issue query failed")
        raise HTTPException(status_code=502, detail="issue query failed") from exc


class IssueHistoryRequest(BaseModel):
    """Query over authoritative issue_history — transitions, not current state."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_keys: Optional[List[str]] = Field(None, description="Restrict to these issues, e.g. ['ATC-77'].")
    fields: Optional[List[str]] = Field(None, description="Restrict to changes of these fields, e.g. ['status','description'].")
    event_types: Optional[List[str]] = Field(None, description="Restrict to these history kinds, e.g. ['field_change'].")
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    # First-class because it's the transition question people actually ask, and its SQL shape
    # (from='done' AND to<>'done') isn't expressible through the generic filters above.
    reopened_only: bool = False
    limit: int = Field(default=20, ge=1, le=200)

    _blank_dates = field_validator("since", "until", mode="before")(_blank_to_none)


class IssueChangeOut(BaseModel):
    issue_key: str
    event_type: str
    field_name: Optional[str]
    from_value: Optional[str]
    to_value: Optional[str]
    description: Optional[str]
    actor_name: Optional[str]
    changed_at: datetime


class IssueHistoryResponse(BaseModel):
    total_count: int
    changes: List[IssueChangeOut]


@router.post("/issues/history", response_model=IssueHistoryResponse)
async def issue_history(req: IssueHistoryRequest, request: Request) -> IssueHistoryResponse:
    """Change history: who changed what, when, from -> to. Answers what a current-state snapshot
    can't ("which issues were reopened", "what did I change in ATC-77"). Space-scoped.
    """
    try:
        payload = await request.app.state.structured.query("issues/history", req.model_dump(mode="json"))
        return IssueHistoryResponse.model_validate(payload)
    except Exception as exc:
        logger.exception("authoritative issue history query failed")
        raise HTTPException(status_code=502, detail="issue history query failed") from exc


class IssueCommentsRequest(BaseModel):
    """Complete, unranked comment fetch for specific issues — see VectorStore.get_issue_comments."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_keys: List[str] = Field(min_length=1, description="Fetch every comment on these issues, e.g. ['ATC-30'].")
    limit: int = Field(default=200, ge=1, le=500)


class IssueCommentOut(BaseModel):
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    provenance: dict = Field(default_factory=dict)


class IssueCommentsResponse(BaseModel):
    total_count: int
    comments: List[IssueCommentOut]


@router.post("/issues/comments", response_model=IssueCommentsResponse)
async def issue_comments(req: IssueCommentsRequest, request: Request) -> IssueCommentsResponse:
    """Every comment on the named issues, not a search top-K, so a key comment (a reopen reason, a
    decision) can't be buried by ranking. Plain SELECT, no embedding.
    """
    store = request.app.state.store
    try:
        result = await store.get_issue_comments(req.space_ids, req.issue_keys, limit=req.limit)
    except Exception as exc:  # noqa: BLE001 - same clean-502 contract as /search
        logger.exception("issue comments query failed")
        raise HTTPException(status_code=502, detail="issue comments query failed") from exc

    logger.info(
        "issue comments fetch completed",
        extra={"request_id": get_request_id(), "total_count": result.total_count,
               "returned": len(result.comments), "issue_keys": req.issue_keys},
    )
    return IssueCommentsResponse(
        total_count=result.total_count,
        comments=[
            IssueCommentOut(issue_id=c.issue_id, issue_key=c.issue_key, source_id=c.source_id, content=c.content,
                            provenance=c.provenance)
            for c in result.comments
        ],
    )


class IssueDetailsRequest(BaseModel):
    """Complete, unranked issue-body fetch for specific issues — see VectorStore.get_issue_details."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_keys: List[str] = Field(min_length=1, description="Fetch the title+description of these issues, e.g. ['ATC-43'].")
    limit: int = Field(default=200, ge=1, le=500)


class IssueDetailOut(BaseModel):
    issue_id: int
    issue_key: str
    source_id: int
    content: str


class IssueDetailsResponse(BaseModel):
    total_count: int
    details: List[IssueDetailOut]


@router.post("/issues/details", response_model=IssueDetailsResponse)
async def issue_details(req: IssueDetailsRequest, request: Request) -> IssueDetailsResponse:
    """An issue's own title+description by exact key, not a search top-K. An issue's own chunk could
    rank below its comments; "what does issue X say" is a lookup, not a search. Plain SELECT.
    """
    store = request.app.state.store
    try:
        result = await store.get_issue_details(req.space_ids, req.issue_keys, limit=req.limit)
    except Exception as exc:  # noqa: BLE001 - same clean-502 contract as /search
        logger.exception("issue details query failed")
        raise HTTPException(status_code=502, detail="issue details query failed") from exc

    logger.info(
        "issue details fetch completed",
        extra={"request_id": get_request_id(), "total_count": result.total_count,
               "returned": len(result.details), "issue_keys": req.issue_keys},
    )
    return IssueDetailsResponse(
        total_count=result.total_count,
        details=[
            IssueDetailOut(issue_id=d.issue_id, issue_key=d.issue_key, source_id=d.source_id, content=d.content)
            for d in result.details
        ],
    )


class IssueAttachmentsRequest(BaseModel):
    """Complete, unranked attachment-text fetch for specific issues — see
    VectorStore.get_issue_attachments."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    issue_keys: List[str] = Field(min_length=1, description="Fetch every attachment chunk on these issues, e.g. ['ATC-46'].")
    limit: int = Field(default=200, ge=1, le=500)


class IssueAttachmentOut(BaseModel):
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    page_number: int | None = None
    provenance: dict = Field(default_factory=dict)


class IssueAttachmentsResponse(BaseModel):
    total_count: int
    attachments: List[IssueAttachmentOut]


@router.post("/issues/attachments", response_model=IssueAttachmentsResponse)
async def issue_attachments(req: IssueAttachmentsRequest, request: Request) -> IssueAttachmentsResponse:
    """Every parsed-attachment chunk for the named issues, by exact key. Specific facts (an exact
    SKU) are hard to phrase as a search; "what SKU does ATC-46's attachment use" flipped between
    answering and abstaining. Plain SELECT.
    """
    store = request.app.state.store
    try:
        result = await store.get_issue_attachments(req.space_ids, req.issue_keys, limit=req.limit)
    except Exception as exc:  # noqa: BLE001 - same clean-502 contract as /search
        logger.exception("issue attachments query failed")
        raise HTTPException(status_code=502, detail="issue attachments query failed") from exc

    logger.info(
        "issue attachments fetch completed",
        extra={"request_id": get_request_id(), "total_count": result.total_count,
               "returned": len(result.attachments), "issue_keys": req.issue_keys},
    )
    return IssueAttachmentsResponse(
        total_count=result.total_count,
        attachments=[
            IssueAttachmentOut(
                issue_id=a.issue_id,
                issue_key=a.issue_key,
                source_id=a.source_id,
                content=a.content,
                page_number=a.page_number,
                provenance=a.provenance,
            )
            for a in result.attachments
        ],
    )


class SectionsRequest(BaseModel):
    """Parent sections by id — the ids a search hit lists in provenance.section_ids."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    section_ids: List[str] = Field(min_length=1, max_length=10, description="e.g. ['attachment:42:section#3'].")


class SectionOut(BaseModel):
    id: str
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    page_number: int | None = None
    page_end: int | None = None


class SectionsResponse(BaseModel):
    sections: List[SectionOut]


@router.post("/sections", response_model=SectionsResponse)
async def sections(req: SectionsRequest, request: Request) -> SectionsResponse:
    """The whole section a chunk was cut from (parent-child retrieval), for when that chunk alone
    is not enough to answer. Plain SELECT, scoped to the caller's spaces.
    """
    store = request.app.state.store
    try:
        found = await store.get_sections(req.space_ids, req.section_ids)
    except Exception as exc:  # noqa: BLE001 - same clean-502 contract as /search
        logger.exception("sections fetch failed")
        raise HTTPException(status_code=502, detail="sections fetch failed") from exc

    logger.info(
        "sections fetch completed",
        extra={"request_id": get_request_id(), "requested": len(req.section_ids),
               "returned": len(found)},
    )
    return SectionsResponse(
        sections=[
            SectionOut(
                id=s.id, issue_id=s.issue_id, issue_key=s.issue_key, source_id=s.source_id,
                content=s.content, page_number=s.page_number, page_end=s.page_end,
            )
            for s in found
        ]
    )


class SprintQueryRequest(BaseModel):
    """Structured, non-semantic sprint query — the counterpart to IssueQueryRequest, over authoritative `sprints`. Answers "how many sprints", "which is active", velocity/points questions."""

    space_ids: List[int] = Field(min_length=1, description="Caller's authorized spaces (permission scope).")
    statuses: Optional[List[str]] = Field(None, description="Filter to these statuses, e.g. ['active'].")
    order_by: Literal["start_date", "end_date"] = "start_date"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=20, ge=1, le=200)


class SprintRowOut(BaseModel):
    sprint_id: int
    sprint_name: str
    space_id: int
    goal: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    status: Optional[str]
    initial_committed_points: Optional[int]
    initial_completed_points: Optional[int]
    final_scope_points: Optional[int]
    completed_points: Optional[int]
    initial_issue_count: Optional[int]
    completed_issue_count: Optional[int]
    final_issue_count: Optional[int]
    unestimated_issue_count: Optional[int]


class SprintQueryResponse(BaseModel):
    total_count: int
    counts_by_status: dict
    sprints: List[SprintRowOut]


@router.post("/sprints/query", response_model=SprintQueryResponse)
async def query_sprints(req: SprintQueryRequest, request: Request) -> SprintQueryResponse:
    """Structured sprint query: exact count, status breakdown, ordered sample. Same contract as
    /issues/query.
    """
    try:
        payload = await request.app.state.structured.query("sprints/query", req.model_dump(mode="json"))
        return SprintQueryResponse.model_validate(payload)
    except Exception as exc:
        logger.exception("authoritative sprint query failed")
        raise HTTPException(status_code=502, detail="sprint query failed") from exc


class _StageTimer:
    """Time a `with` block once, and feed that one number to the metric (`stage_ms`), the log and a
    tracing span, so they never disagree. With tracing off, the tracer is a no-op, so no
    `if tracing_enabled` branches are needed.
    """

    def __init__(self, name: str, stage_ms: dict, attributes: dict | None = None):
        self._name = name
        self._stage_ms = stage_ms
        self._attributes = attributes or {}
        self.seconds = 0.0

    def __enter__(self) -> "_StageTimer":
        self._span_cm = get_tracer().start_as_current_span(
            f"search.{self._name}", attributes=self._attributes
        )
        self._span = self._span_cm.__enter__()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.seconds = time.perf_counter() - self._start
        self._stage_ms[self._name] = round(self.seconds * 1000, 2)
        # Also stored as an attribute: exporters add their own clocks, and this is the number the
        # metric and log agree on.
        self._span.set_attribute("stage.duration_ms", self._stage_ms[self._name])
        # Propagates the exception so the span is marked with an error status by the SDK's own
        # context manager, rather than silently closing a failed stage as if it had succeeded.
        self._span_cm.__exit__(*exc_info)


# --- Ingestion status: per-document index tracking ------------------------------------------------
# /stats says how many failed; these say which document, and at which stage.


class IngestionStatusRequest(BaseModel):
    space_ids: Optional[List[int]] = Field(
        default=None,
        description="Restrict to these spaces (same permission scope as /search). Omit for all.",
    )
    source_types: Optional[List[Literal["issue", "comment", "attachment", "sprint"]]] = None
    states: Optional[
        List[Literal["in_progress", "indexed", "empty", "deleted", "skipped", "failed"]]
    ] = Field(default=None, description="e.g. ['failed','empty'] for everything needing attention.")
    issue_keys: Optional[List[str]] = Field(
        default=None, description="Exact issue keys, e.g. ['ATC-46'] — the usual starting point."
    )
    filename_contains: Optional[str] = Field(
        default=None,
        description="Substring of an attachment filename, for when the uploader knows the name only.",
    )
    stale_after_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Only documents stuck in_progress longer than this. Finds workers that died mid-stage — "
            "such a document never reaches a terminal state, so no state filter can surface it."
        ),
    )
    limit: int = Field(default=50, ge=1, le=500)


class IngestionStatusRowOut(BaseModel):
    source_type: str
    source_id: int
    space_id: int | None = None
    issue_key: str | None = None
    source_label: str | None = None
    state: str
    stage: str | None = None
    attempts: int
    chunk_count: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    event_id: str | None = None
    request_id: str | None = None
    duration_ms: int | None = None
    stage_ms: dict = Field(default_factory=dict)
    first_seen_at: datetime | None = None
    updated_at: datetime | None = None


class IngestionStatusResponse(BaseModel):
    total_count: int
    counts_by_state: dict
    rows: List[IngestionStatusRowOut]


@router.post("/ingestion/status", response_model=IngestionStatusResponse)
async def ingestion_status(
    req: IngestionStatusRequest, request: Request
) -> IngestionStatusResponse:
    """Per-document ingestion status (indexed, empty, skipped, stuck, failed). Counts cover every
    match; `rows` is a newest-first sample.
    """
    store = request.app.state.store
    result = await store.query_ingestion_status(
        space_ids=req.space_ids,
        source_types=req.source_types,
        states=req.states,
        issue_keys=req.issue_keys,
        source_label_contains=req.filename_contains,
        stale_after_seconds=req.stale_after_seconds,
        limit=req.limit,
    )
    return IngestionStatusResponse(
        total_count=result.total_count,
        counts_by_state=result.counts_by_state,
        rows=[IngestionStatusRowOut(**asdict(r)) for r in result.rows],
    )


class IndexWatermarkResponse(BaseModel):
    """Freshness token for a set of spaces' indexed content — see ``VectorStore.index_watermark``."""

    #: ISO-8601 timestamp of the most recent ingest state change in these spaces, or "never".
    last_change: str
    #: Documents tracked in these spaces. Paired with ``last_change`` so the token still changes if
    #: rows are ever removed, which ``max(updated_at)`` alone would not detect.
    document_count: int
    #: The two fields above collapsed into the single opaque string callers should key on. Callers
    #: must treat it as opaque: its only contract is "different content ⇒ different token".
    watermark: str


@router.get("/index/watermark", response_model=IndexWatermarkResponse)
async def index_watermark(request: Request, space_ids: str = "") -> IndexWatermarkResponse:
    """Freshness token: "has anything in these spaces changed since I cached an answer?"

    ai-service's answer cache puts this in its key, so any indexing change makes old answers for
    those spaces unreachable — no purge needed, unlike the old TTL-only cache. A GET because it's a
    cheap indexed read on every /ask.

    Searchable text only. Status, priority, sprint and history are not in it: a status drag would
    otherwise empty the whole space's cache, and ai-service doesn't cache answers that used those
    live facts anyway (is_answer_cacheable).

    Not permission-checked: it returns only a timestamp and a count, the caller has already
    authorized its space_ids, and it's only reachable on the private network.
    """
    parsed: List[int] = []
    for raw in space_ids.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(int(raw))
        except ValueError:
            raise HTTPException(status_code=422, detail=f"space_ids must be integers, got {raw!r}")

    store = request.app.state.store
    last_change, document_count = await store.index_watermark(space_ids=parsed or None)
    # "never", not "": an empty space is a real state to key on, and a falsy value means "the check
    # failed, skip the cache".
    rendered = last_change.isoformat() if last_change is not None else "never"
    return IndexWatermarkResponse(
        last_change=rendered,
        document_count=document_count,
        watermark=f"{rendered}:{document_count}",
    )


@router.get("/ingestion/health")
async def ingestion_health(request: Request) -> dict:
    """One-call triage of the index. `needs_attention` (failures + documents stuck over ten minutes)
    is the number to alert on. `empty` is shown but not counted: blank descriptions are normal.
    """
    store = request.app.state.store
    overall = await store.query_ingestion_status(limit=0)
    stuck = await store.query_ingestion_status(stale_after_seconds=600, limit=0)
    counts = overall.counts_by_state
    failed = counts.get("failed", 0)
    return {
        "total_documents": overall.total_count,
        "counts_by_state": counts,
        "stuck_in_progress": stuck.total_count,
        "needs_attention": failed + stuck.total_count,
        # Named so the next step is obvious from the response itself rather than from the README.
        "next": "POST /ingestion/status {\"states\":[\"failed\"]} for the failing documents",
    }
