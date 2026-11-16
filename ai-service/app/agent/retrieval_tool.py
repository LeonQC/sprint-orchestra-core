"""Thin client for vectorization-service's retrieval API. That service owns the index and offers
primitives; ai-service combines them into answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.degradation import RETRIEVAL_BM25_ONLY
from app.degradation import note as note_degraded
from app.observability import get_request_id


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    chunk_type: str
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    score: float
    retrievers: List[str]
    # 1-based source PDF page when parser provenance is available; null for non-page sources and
    # legacy chunks. Preserved end-to-end so UI citations can deep-link to the right PDF page.
    page_number: Optional[int] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IssueQueryResult:
    """Result of /issues/query: counts cover every match; `issues` is the ordered, limited sample. The
    agent uses this for exact counts and lists instead of semantic search.
    """

    total_count: int
    counts_by_type: dict
    counts_by_status: dict
    issues: List[dict]
    counts_by_assignee: dict = field(default_factory=dict)


@dataclass(slots=True)
class IssueHistoryResult:
    """Result of a change-history query — see vectorization-service /issues/history. `changes` are
    raw dicts (issue_key, event_type, field_name, from_value, to_value, actor_name, changed_at)."""

    total_count: int
    changes: List[dict]


@dataclass(slots=True)
class IssueCommentsResult:
    """All comments for named issues, unranked (/issues/comments) — for details a top-K search might
    miss. `comments` are raw dicts."""

    total_count: int
    comments: List[dict]


@dataclass(slots=True)
class IssueDetailsResult:
    """Issues' own title+description, unranked (/issues/details). `details` are raw dicts."""

    total_count: int
    details: List[dict]


@dataclass(slots=True)
class IssueAttachmentsResult:
    """Issues' attachment text, unranked (/issues/attachments) — for specific facts (an exact SKU)
    that are hard to search for. `attachments` are raw dicts."""

    total_count: int
    attachments: List[dict]


@dataclass(slots=True)
class SprintQueryResult:
    """Result of /sprints/query (name, goal, dates, status, velocity)."""

    total_count: int
    counts_by_status: dict
    sprints: List[dict]


@dataclass(slots=True)
class SearchResult:
    hits: List[RetrievedChunk]
    # Whether the server reranked (a server setting `mode` doesn't control). If true, `score` is an
    # unbounded logit (strong ~5, weak ~1-2, irrelevant < 0), not 0-1 cosine — a cosine-calibrated
    # `min_score` silently stops filtering. Check this before using `score` as a number.
    reranked: bool
    # True when the server fell back to BM25 because the query embedding failed.
    degraded: bool = False


class RetrievalClient:
    def __init__(self, base_url: str):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30.0)

    async def search(
        self, query: str, space_ids: Sequence[int], limit: int, mode: str = "hybrid"
    ) -> List[RetrievedChunk]:
        """For CragAgent, which never thresholds on `score`. Use `search_with_meta` if you need to
        interpret scores.
        """
        result = await self.search_with_meta(query, space_ids, limit, mode)
        return result.hits

    async def search_with_meta(
        self, query: str, space_ids: Sequence[int], limit: int, mode: str = "hybrid"
    ) -> SearchResult:
        # Pass the request id along so vectorization-service logs can be matched to this /ask.
        resp = await self._client.post(
            "/search",
            json={"query": query, "space_ids": list(space_ids), "limit": limit, "mode": mode},
            headers={"x-request-id": get_request_id()},
        )
        resp.raise_for_status()
        body = resp.json()
        hits = [
            RetrievedChunk(
                id=h["id"], chunk_type=h["chunk_type"], issue_id=h["issue_id"], issue_key=h["issue_key"],
                source_id=h["source_id"], content=h["content"], score=h["score"], retrievers=h["retrievers"],
                page_number=h.get("page_number"),
                provenance=h.get("provenance") or {},
            )
            for h in body["hits"]
        ]
        degraded = body.get("degraded", False)
        if degraded:
            note_degraded(RETRIEVAL_BM25_ONLY)
        return SearchResult(hits=hits, reranked=body.get("reranked", False), degraded=degraded)

    async def query_issues(
        self, space_ids: Sequence[int], filters: dict
    ) -> IssueQueryResult:
        """Structured issue query (/issues/query). `space_ids` comes from the authenticated caller,
        never the model; `filters` holds only model-controlled fields, so a prompt-injected model can't
        widen its own scope.
        """
        payload = {"space_ids": list(space_ids), **filters}
        resp = await self._client.post(
            "/issues/query", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return IssueQueryResult(
            total_count=body["total_count"],
            counts_by_type=body.get("counts_by_type", {}),
            counts_by_status=body.get("counts_by_status", {}),
            issues=body.get("issues", []),
            counts_by_assignee=body.get("counts_by_assignee", {}),
        )

    async def query_issue_history(
        self, space_ids: Sequence[int], filters: dict
    ) -> IssueHistoryResult:
        """Change-history query against vectorization-service /issues/history. Same security shape as
        query_issues: space_ids injected here from the authenticated caller, never model-supplied."""
        payload = {"space_ids": list(space_ids), **filters}
        resp = await self._client.post(
            "/issues/history", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return IssueHistoryResult(
            total_count=body["total_count"],
            changes=body.get("changes", []),
        )

    async def get_issue_comments(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueCommentsResult:
        """All comments for named issues (/issues/comments). space_ids from the caller, never the model."""
        payload = {"space_ids": list(space_ids), "issue_keys": list(issue_keys), "limit": limit}
        resp = await self._client.post(
            "/issues/comments", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return IssueCommentsResult(
            total_count=body["total_count"],
            comments=body.get("comments", []),
        )

    async def get_issue_details(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueDetailsResult:
        """Issue bodies by key (/issues/details). space_ids from the caller, never the model."""
        payload = {"space_ids": list(space_ids), "issue_keys": list(issue_keys), "limit": limit}
        resp = await self._client.post(
            "/issues/details", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return IssueDetailsResult(
            total_count=body["total_count"],
            details=body.get("details", []),
        )

    async def get_issue_attachments(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueAttachmentsResult:
        """Attachment text by key (/issues/attachments). space_ids from the caller, never the model."""
        payload = {"space_ids": list(space_ids), "issue_keys": list(issue_keys), "limit": limit}
        resp = await self._client.post(
            "/issues/attachments", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return IssueAttachmentsResult(
            total_count=body["total_count"],
            attachments=body.get("attachments", []),
        )

    async def get_sections(
        self, space_ids: Sequence[int], section_ids: Sequence[str]
    ) -> List[dict]:
        """Whole parent sections by id (/sections). space_ids from the caller, never the model."""
        payload = {"space_ids": list(space_ids), "section_ids": list(section_ids)}
        resp = await self._client.post(
            "/sections", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        return resp.json().get("sections", [])

    async def query_sprints(
        self, space_ids: Sequence[int], filters: dict
    ) -> SprintQueryResult:
        """Structured sprint query against vectorization-service /sprints/query. Same security shape
        as query_issues: space_ids injected here from the authenticated caller, never model-supplied."""
        payload = {"space_ids": list(space_ids), **filters}
        resp = await self._client.post(
            "/sprints/query", json=payload, headers={"x-request-id": get_request_id()}
        )
        resp.raise_for_status()
        body = resp.json()
        return SprintQueryResult(
            total_count=body["total_count"],
            counts_by_status=body.get("counts_by_status", {}),
            sprints=body.get("sprints", []),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
