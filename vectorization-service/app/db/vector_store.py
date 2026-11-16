"""pgvector store for chunk embeddings, which also keeps the OpenSearch lexical index in step.

This service owns the index: writes/deletes are keyed on deterministic ids, and callers get ranked
hits back, never raw SQL.

Retrieval uses two engines:
  * dense (`search_vector`) — pgvector in this service's Postgres;
  * lexical (`search_bm25`) — OpenSearch BM25 (app/search/chunk_index.py). Postgres had no real BM25
    (`ts_rank_cd` lacks IDF and length normalization).
`search_hybrid` fuses them with Reciprocal Rank Fusion, which combines ranks, so the two score scales
never need to match.

Postgres is the source of truth for chunk text; OpenSearch is derived from it (see
`ChunkSearchIndex` for consistency and reindexing).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Sequence

import asyncpg
import numpy as np

from app.db.rrf import fuse_rrf
from app.search.chunk_index import ChunkSearchIndex
from app.models import (
    Chunk,
    ChunkParent,
    IngestionStatusQueryResult,
    IngestionStatusRow,
    IssueAttachment,
    IssueAttachmentsQueryResult,
    IssueComment,
    IssueCommentsQueryResult,
    IssueDetail,
    IssueDetailsQueryResult,
    SearchHit,
    VlmRetryItem,
)

logger = logging.getLogger(__name__)


class VectorStore:
    """`search_index` is optional for maintenance and vector-only callers.
    Lexical/hybrid search raise a clear error without it — silently running vector-only
    would show up weeks later as an unexplained eval regression.
    """

    def __init__(self, pool: asyncpg.Pool, search_index: ChunkSearchIndex | None = None):
        self._pool = pool
        self._search_index = search_index

    @property
    def search_index(self) -> ChunkSearchIndex:
        if self._search_index is None:
            raise RuntimeError(
                "lexical retrieval requires an OpenSearch ChunkSearchIndex; this VectorStore "
                "was constructed without one"
            )
        return self._search_index

    async def replace_source(
        self,
        chunk_type: str,
        source_id: int,
        chunks: List[Chunk],
        vectors: List[List[float]],
        parents: Sequence[ChunkParent] = (),
    ) -> None:
        """Replace all chunks (and parent sections) for one source in Postgres, then OpenSearch.

        Delete-and-insert in one transaction, because a re-parse can produce a different number of
        chunks. OpenSearch is written after Postgres commits, so the failure mode is "OpenSearch is
        briefly behind" (fixed by POST /search/reindex), never "OpenSearch has rows that never
        existed".
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM chunks WHERE chunk_type = $1 AND source_id = $2",
                    chunk_type,
                    source_id,
                )
                if chunks:
                    await conn.executemany(
                        """
                        INSERT INTO chunks
                            (id, embedding, chunk_type, issue_id, issue_key, space_id,
                             source_id, chunk_index, content, page_number, provenance, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, now())
                        """,
                        [
                            (
                                c.id,
                                np.asarray(v, dtype=np.float32),
                                c.chunk_type,
                                c.issue_id,
                                c.issue_key,
                                c.space_id,
                                c.source_id,
                                c.chunk_index,
                                c.content,
                                c.page_number,
                                c.provenance,
                            )
                            for c, v in zip(chunks, vectors)
                        ],
                    )
                await conn.execute(
                    "DELETE FROM chunk_parents WHERE chunk_type = $1 AND source_id = $2",
                    chunk_type,
                    source_id,
                )
                if parents:
                    await conn.executemany(
                        """
                        INSERT INTO chunk_parents
                            (id, chunk_type, issue_id, issue_key, space_id, source_id, content,
                             page_number, page_end)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        [
                            (p.id, p.chunk_type, p.issue_id, p.issue_key, p.space_id, p.source_id,
                             p.content, p.page_number, p.page_end)
                            for p in parents
                        ],
                    )
                await _mark_content_changed(conn, chunk_type, source_id)
        if self._search_index is not None:
            await self._search_index.replace_source(chunk_type, source_id, chunks)
            # Move it again now that OpenSearch has the new text too. Otherwise a request that read the
            # watermark between the commit and this write could search OpenSearch's old copy and cache
            # that answer under the new watermark.
            await _mark_content_changed(self._pool, chunk_type, source_id)

    async def get_source_chunks(self, chunk_type: str, source_id: int) -> List[Chunk]:
        """The chunks currently stored for one source, in order — what a re-ingest compares against."""
        rows = await self._pool.fetch(
            """
            SELECT id, chunk_type, issue_id, issue_key, space_id, source_id, chunk_index, content,
                   page_number, provenance
            FROM chunks
            WHERE chunk_type = $1 AND source_id = $2
            ORDER BY chunk_index
            """,
            chunk_type,
            source_id,
        )
        return [Chunk(**{**dict(r), "provenance": r["provenance"] or {}}) for r in rows]

    async def refresh_lexical_index(self, chunk_type: str, source_id: int, chunks: List[Chunk]) -> None:
        """Rewrite one source's OpenSearch docs from chunks Postgres already holds (no embedding)."""
        if self._search_index is not None:
            await self._search_index.replace_source(chunk_type, source_id, chunks)

    async def delete_source(self, chunk_type: str, source_id: int) -> int:
        """Delete all chunks for one comment/attachment. Returns rows removed from Postgres."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM chunks WHERE chunk_type = $1 AND source_id = $2",
                    chunk_type,
                    source_id,
                )
                await conn.execute(
                    "DELETE FROM chunk_parents WHERE chunk_type = $1 AND source_id = $2",
                    chunk_type,
                    source_id,
                )
                if _rows_affected(result):
                    await _mark_content_changed(conn, chunk_type, source_id)
        if self._search_index is not None:
            await self._search_index.delete_source(chunk_type, source_id)
            if _rows_affected(result):
                # Same reason as replace_source.
                await _mark_content_changed(self._pool, chunk_type, source_id)
        return _rows_affected(result)

    async def upsert_ingestion_status(
        self,
        *,
        source_type: str,
        source_id: int,
        space_id: int | None = None,
        issue_key: str | None = None,
        source_label: str | None = None,
        state: str,
        stage: str | None = None,
        chunk_count: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        event_id: str | None = None,
        request_id: str | None = None,
        duration_ms: int | None = None,
        stage_ms: dict | None = None,
    ) -> None:
        """Record where one document is in the pipeline (app/ingest/status.py).

        `attempts` goes up only when a document restarts from a terminal state, so it counts
        redeliveries, not stages (attempts=4 + failed = retried four times, won't fix itself).
        `first_seen_at` is kept across upserts; errors are cleared on any non-failed state.
        """
        await self._pool.execute(
            """
            INSERT INTO ingestion_status
                (source_type, source_id, space_id, issue_key, source_label, state, stage,
                 attempts, chunk_count, error_type, error_message, event_id, request_id,
                 duration_ms, stage_ms, first_seen_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7,
                    1, $8, $9, $10, $11, $12, $13, $14, now(), now())
            ON CONFLICT (source_type, source_id) DO UPDATE SET
                space_id      = COALESCE(EXCLUDED.space_id, ingestion_status.space_id),
                issue_key     = COALESCE(EXCLUDED.issue_key, ingestion_status.issue_key),
                source_label  = COALESCE(EXCLUDED.source_label, ingestion_status.source_label),
                state         = EXCLUDED.state,
                stage         = EXCLUDED.stage,
                -- `attempts` is the retry count of a *currently failing* document — the number an
                -- operator reads as "will this fix itself". It counts consecutive failed passes:
                --   * `in_progress` from `failed`      → +1  (a retry is starting)
                --   * `in_progress` from `in_progress` → unchanged (a later stage of the same pass;
                --                                        the tracker writes one row per stage)
                --   * `in_progress` from a terminal ok state → 1 (a fresh pass)
                --   * reaching `indexed`/`empty`/`deleted`   → 1 (it recovered; history cleared)
                --   * reaching `failed`                → unchanged (the +1 already happened on the
                --                                        `in_progress` that began this pass)
                -- so a healthy document always reads `attempts=1`, and only a genuinely stuck one
                -- climbs. Without this, a manual re-ingest of a healthy doc would inflate the number
                -- and be indistinguishable from a document Kafka has retried 5×.
                attempts      = CASE
                                    WHEN EXCLUDED.state = 'in_progress'
                                     AND ingestion_status.state = 'failed'
                                    THEN ingestion_status.attempts + 1
                                    WHEN EXCLUDED.state = 'in_progress'
                                     AND ingestion_status.state = 'in_progress'
                                    THEN ingestion_status.attempts
                                    WHEN EXCLUDED.state = 'in_progress'
                                    THEN 1
                                    WHEN EXCLUDED.state = 'failed'
                                    THEN ingestion_status.attempts
                                    ELSE 1 END,
                chunk_count   = COALESCE(EXCLUDED.chunk_count, ingestion_status.chunk_count),
                error_type    = CASE WHEN EXCLUDED.state = 'failed'
                                     THEN EXCLUDED.error_type ELSE NULL END,
                error_message = CASE WHEN EXCLUDED.state IN ('failed', 'skipped')
                                     THEN EXCLUDED.error_message ELSE NULL END,
                event_id      = COALESCE(EXCLUDED.event_id, ingestion_status.event_id),
                request_id    = COALESCE(EXCLUDED.request_id, ingestion_status.request_id),
                duration_ms   = COALESCE(EXCLUDED.duration_ms, ingestion_status.duration_ms),
                stage_ms      = EXCLUDED.stage_ms,
                updated_at    = now()
            """,
            source_type, source_id, space_id, issue_key, source_label, state, stage,
            chunk_count, error_type, error_message, event_id, request_id, duration_ms,
            stage_ms or {},
        )

    async def query_ingestion_status(
        self,
        *,
        space_ids: Sequence[int] | None = None,
        source_types: Sequence[str] | None = None,
        states: Sequence[str] | None = None,
        issue_keys: Sequence[str] | None = None,
        source_label_contains: str | None = None,
        stale_after_seconds: int | None = None,
        limit: int = 50,
    ) -> IngestionStatusQueryResult:
        """Status rows plus exact per-state counts for the same filter.

        Counts cover every match; `rows` is a newest-first sample. `stale_after_seconds` finds
        documents stuck in_progress too long — a worker that died mid-stage never reaches "failed".
        """
        conds: list[str] = []
        args: list = []

        def add(sql: str, value) -> None:
            args.append(value)
            conds.append(sql.format(len(args)))

        if space_ids:
            add("space_id = ANY(${}::bigint[])", list(space_ids))
        if source_types:
            add("source_type = ANY(${}::text[])", list(source_types))
        if states:
            add("state = ANY(${}::text[])", list(states))
        if issue_keys:
            add("issue_key = ANY(${}::text[])", list(issue_keys))
        if source_label_contains:
            add("source_label ILIKE '%' || ${}::text || '%'", source_label_contains)
        if stale_after_seconds is not None:
            add(
                "(state = 'in_progress' AND updated_at < now() - make_interval(secs => ${}::double precision))",
                float(stale_after_seconds),
            )
        where = f"WHERE {' AND '.join(conds)}" if conds else ""

        total = await self._pool.fetchval(
            f"SELECT count(*) FROM ingestion_status {where}", *args
        )
        breakdown = await self._pool.fetch(
            f"SELECT state, count(*) AS n FROM ingestion_status {where} GROUP BY state", *args
        )
        rows = await self._pool.fetch(
            f"""
            SELECT source_type, source_id, space_id, issue_key, source_label, state, stage,
                   attempts, chunk_count, error_type, error_message, event_id, request_id,
                   duration_ms, stage_ms, first_seen_at, updated_at
            FROM ingestion_status {where}
            ORDER BY updated_at DESC
            LIMIT ${len(args) + 1}
            """,
            *args, limit,
        )
        return IngestionStatusQueryResult(
            total_count=total or 0,
            counts_by_state={r["state"]: r["n"] for r in breakdown},
            rows=[_row_to_status(r) for r in rows],
        )

    async def index_watermark(
        self, space_ids: Sequence[int] | None = None
    ) -> tuple["datetime | None", int]:
        """Return `(last_change, tracked_document_count)` for these spaces — the freshness token the
        answer cache keys on.

        `content_changed_at` moves only when a source's chunks are replaced or deleted,
        so its max is "when did anything searchable here last change". Not `updated_at`: that moves on
        every status write, including a re-ingest that changed nothing (a status drag). The count
        catches a status row removed outright (the maintenance scripts do that), which the max misses
        unless it was the newest row. Reads ingestion_status, not chunks, because a source that ingested to zero chunks is still
        a real change (its new row is counted).
        """
        if space_ids:
            row = await self._pool.fetchrow(
                """
                SELECT max(content_changed_at) AS last_change, count(*) AS n
                FROM ingestion_status
                WHERE space_id = ANY($1::bigint[])
                """,
                list(space_ids),
            )
        else:
            row = await self._pool.fetchrow(
                "SELECT max(content_changed_at) AS last_change, count(*) AS n FROM ingestion_status"
            )
        # Returned raw; the route decides the wire format of the token.
        return (row["last_change"] if row else None, (row["n"] if row else 0) or 0)

    async def get_vlm_result(self, cache_key: str) -> str | None:
        """A stored VLM result by content hash, so a Kafka replay after a crash doesn't pay for the
        same call again. Stores only the hash and text, never the attachment bytes.
        """
        return await self._pool.fetchval(
            "SELECT content FROM vlm_result_cache WHERE cache_key = $1", cache_key
        )

    async def put_vlm_result(self, cache_key: str, namespace: str, mime_type: str, content: str) -> None:
        """Persist the first completed VLM result; duplicates are equivalent and safely ignored."""
        await self._pool.execute(
            """
            INSERT INTO vlm_result_cache (cache_key, namespace, mime_type, content)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (cache_key) DO NOTHING
            """,
            cache_key,
            namespace,
            mime_type,
            content,
        )

    # --- VLM retry queue (app/ingest/vlm_retry.py) ---
    async def queue_vlm_retry(
        self, *, attachment_id: int, issue_id: int, space_id: int | None, event_id: str,
        message: dict, failed_calls: int, last_error: str, retry_after_seconds: float,
    ) -> None:
        """Queue (or re-queue) an attachment whose VLM calls failed. A new event_id resets the attempt
        count; the same one (a retry that failed again) keeps it, so the worker's cap holds."""
        await self._pool.execute(
            """
            INSERT INTO vlm_retry_queue
                (attachment_id, issue_id, space_id, event_id, message, failed_calls, last_error,
                 attempts, next_attempt_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, 0, now() + make_interval(secs => $8::double precision))
            ON CONFLICT (attachment_id) DO UPDATE SET
                issue_id     = EXCLUDED.issue_id,
                space_id     = EXCLUDED.space_id,
                message      = EXCLUDED.message,
                failed_calls = EXCLUDED.failed_calls,
                last_error   = EXCLUDED.last_error,
                attempts     = CASE WHEN vlm_retry_queue.event_id = EXCLUDED.event_id
                                    THEN vlm_retry_queue.attempts ELSE 0 END,
                next_attempt_at = CASE WHEN vlm_retry_queue.event_id = EXCLUDED.event_id
                                       THEN vlm_retry_queue.next_attempt_at
                                       ELSE EXCLUDED.next_attempt_at END,
                event_id     = EXCLUDED.event_id,
                updated_at   = now()
            """,
            attachment_id, issue_id, space_id, event_id, message, failed_calls, last_error[:2000],
            float(retry_after_seconds),
        )

    async def clear_vlm_retry(self, attachment_id: int) -> None:
        await self._pool.execute("DELETE FROM vlm_retry_queue WHERE attachment_id = $1", attachment_id)

    async def get_vlm_retry(self, attachment_id: int) -> VlmRetryItem | None:
        row = await self._pool.fetchrow(
            "SELECT attachment_id, event_id, message, attempts FROM vlm_retry_queue WHERE attachment_id = $1",
            attachment_id,
        )
        return _row_to_vlm_retry(row) if row else None

    async def due_vlm_retries(self, *, max_attempts: int, limit: int) -> list[VlmRetryItem]:
        rows = await self._pool.fetch(
            """
            SELECT attachment_id, event_id, message, attempts FROM vlm_retry_queue
            WHERE next_attempt_at <= now() AND attempts < $1
            ORDER BY next_attempt_at
            LIMIT $2
            """,
            max_attempts, limit,
        )
        return [_row_to_vlm_retry(row) for row in rows]

    async def record_vlm_retry_attempt(self, attachment_id: int, next_attempt_after_seconds: float) -> None:
        """Count the attempt before running it, so a crash mid-retry can't loop on the same row."""
        await self._pool.execute(
            """
            UPDATE vlm_retry_queue
            SET attempts = attempts + 1,
                next_attempt_at = now() + make_interval(secs => $2::double precision),
                updated_at = now()
            WHERE attachment_id = $1
            """,
            attachment_id, float(next_attempt_after_seconds),
        )

    async def count_vlm_retries(self) -> int:
        return await self._pool.fetchval("SELECT count(*) FROM vlm_retry_queue") or 0

    async def delete_issue(self, issue_id: int) -> int:
        """Delete derived chunks and pending retries; authoritative business rows stay upstream."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute("DELETE FROM chunks WHERE issue_id = $1 AND chunk_type <> 'sprint'", issue_id)
                await conn.execute(
                    "DELETE FROM chunk_parents WHERE issue_id = $1 AND chunk_type <> 'sprint'", issue_id
                )
                await _mark_content_changed(conn, "issue", issue_id)
                # Or the retry worker would re-ingest a deleted issue's attachments.
                await conn.execute("DELETE FROM vlm_retry_queue WHERE issue_id = $1", issue_id)
        if self._search_index is not None:
            await self._search_index.delete_issue(issue_id)
            # Same reason as replace_source.
            await _mark_content_changed(self._pool, "issue", issue_id)
        return _rows_affected(result)

    async def delete_sprint(self, sprint_id: int) -> int:
        """Delete only the sprint's derived goal chunks from both retrieval engines."""
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                result = await conn.execute(
                    "DELETE FROM chunks WHERE chunk_type = 'sprint' AND source_id = $1", sprint_id
                )
                await _mark_content_changed(conn, "sprint", sprint_id)
        if self._search_index is not None:
            await self._search_index.delete_source("sprint", sprint_id)
            await _mark_content_changed(self._pool, "sprint", sprint_id)
        return _rows_affected(result)

    async def get_issue_comments(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueCommentsQueryResult:
        """Every comment chunk for the given issues, complete — not a search top-K.

        A plain SELECT with no ranking, so a detail one ranked search could bury (a reopen reason in
        one of many comments) can't be missed once the issue key is known. See crag_loop's
        get_issue_comments tool.
        """
        total = await self._pool.fetchval(
            "SELECT count(*) FROM chunks WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'comment' "
            "AND issue_key = ANY($2::text[])",
            list(space_ids), list(issue_keys),
        )
        # Chronological: created_at if present, else source_id (ids grow with insertion). The old
        # `ORDER BY chunk_index` left order to chance — every single-chunk comment has index 0.
        rows = await self._pool.fetch(
            """
            SELECT id, issue_id, issue_key, source_id, content, chunk_index, provenance
            FROM chunks
            WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'comment' AND issue_key = ANY($2::text[])
            ORDER BY issue_key, provenance->>'created_at' NULLS LAST, source_id, chunk_index
            LIMIT $3
            """,
            list(space_ids), list(issue_keys), limit,
        )
        comments = [
            IssueComment(
                id=r["id"], issue_id=r["issue_id"], issue_key=r["issue_key"],
                source_id=r["source_id"], content=r["content"], chunk_index=r["chunk_index"],
                provenance=r.get("provenance") or {},
            )
            for r in rows
        ]
        return IssueCommentsQueryResult(total_count=total or 0, comments=thread_order(comments))

    async def get_issue_details(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueDetailsQueryResult:
        """Every issue title+description chunk for the given issues, complete — not a search top-K.

        A lookup by known key shouldn't depend on ranking: an issue's own chunk could rank below its
        comments for a query that was just its key.
        """
        total = await self._pool.fetchval(
            "SELECT count(*) FROM chunks WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'issue' "
            "AND issue_key = ANY($2::text[])",
            list(space_ids), list(issue_keys),
        )
        rows = await self._pool.fetch(
            """
            SELECT id, issue_id, issue_key, source_id, content, chunk_index
            FROM chunks
            WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'issue' AND issue_key = ANY($2::text[])
            ORDER BY issue_key, chunk_index
            LIMIT $3
            """,
            list(space_ids), list(issue_keys), limit,
        )
        return IssueDetailsQueryResult(
            total_count=total or 0,
            details=[
                IssueDetail(
                    id=r["id"], issue_id=r["issue_id"], issue_key=r["issue_key"],
                    source_id=r["source_id"], content=r["content"], chunk_index=r["chunk_index"],
                )
                for r in rows
            ],
        )

    async def get_issue_attachments(
        self, space_ids: Sequence[int], issue_keys: Sequence[str], limit: int = 200
    ) -> IssueAttachmentsQueryResult:
        """Every attachment chunk for the given issues, complete — not a search top-K.

        Facts buried in an attachment (a SKU in a PDF table) are hard to phrase as a query; "what SKU
        does ATC-46's attachment use" flipped between answering and abstaining across phrasings.
        """
        total = await self._pool.fetchval(
            "SELECT count(*) FROM chunks WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'attachment' "
            "AND issue_key = ANY($2::text[])",
            list(space_ids), list(issue_keys),
        )
        rows = await self._pool.fetch(
            """
            SELECT id, issue_id, issue_key, source_id, content, chunk_index, page_number, provenance
            FROM chunks
            WHERE space_id = ANY($1::bigint[]) AND chunk_type = 'attachment' AND issue_key = ANY($2::text[])
            ORDER BY issue_key, page_number NULLS FIRST, chunk_index
            LIMIT $3
            """,
            list(space_ids), list(issue_keys), limit,
        )
        return IssueAttachmentsQueryResult(
            total_count=total or 0,
            attachments=[
                IssueAttachment(
                    id=r["id"], issue_id=r["issue_id"], issue_key=r["issue_key"],
                    source_id=r["source_id"], content=r["content"], chunk_index=r["chunk_index"],
                    page_number=r["page_number"],
                    provenance=_coerce_provenance(r["provenance"]),
                )
                for r in rows
            ],
        )

    async def get_sections(
        self, space_ids: Sequence[int], section_ids: Sequence[str]
    ) -> List[ChunkParent]:
        """Parent sections by id (from a hit's provenance.section_ids), limited to the caller's
        spaces. An id outside those spaces is simply not returned.
        """
        rows = await self._pool.fetch(
            """
            SELECT id, chunk_type, issue_id, issue_key, space_id, source_id, content,
                   page_number, page_end
            FROM chunk_parents
            WHERE space_id = ANY($1::bigint[]) AND id = ANY($2::text[])
            ORDER BY issue_key, source_id, page_number NULLS FIRST, id
            """,
            list(space_ids), list(section_ids),
        )
        return [ChunkParent(**dict(r)) for r in rows]

    async def count(self, space_ids: Sequence[int] | None = None) -> int:
        """Total indexed chunks, or only those in `space_ids`. The scoped form exists because an
        unscoped total after ingesting one space says nothing about that space.
        """
        if space_ids is None:
            return await self._pool.fetchval("SELECT count(*) FROM chunks")
        return await self._pool.fetchval(
            "SELECT count(*) FROM chunks WHERE space_id = ANY($1)", list(space_ids)
        )

    async def search_vector(
        self, embedding: List[float], space_ids: Sequence[int], limit: int
    ) -> List[SearchHit]:
        """Dense retrieval: nearest neighbours by cosine distance, limited to the caller's spaces."""
        rows = await self._pool.fetch(
            """
            SELECT id, chunk_type, issue_id, issue_key, space_id, source_id, content, page_number, provenance,
                   1 - (embedding <=> $1) AS score
            FROM chunks
            WHERE space_id = ANY($2::bigint[])
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            np.asarray(embedding, dtype=np.float32),
            list(space_ids),
            limit,
        )
        return [_row_to_hit(r, retrievers=["vector"]) for r in rows]

    async def search_bm25(
        self, query: str, space_ids: Sequence[int], limit: int
    ) -> List[SearchHit]:
        """Lexical retrieval via OpenSearch BM25 (app/search/chunk_index.py).

        Replaced `ts_rank_cd` (no IDF, no length normalization) and then a pure-Python BM25 that
        re-tokenized the whole space on every query.
        """
        return await self.search_index.search(query, space_ids, limit)

    async def search_hybrid(
        self,
        embedding: List[float],
        query: str,
        space_ids: Sequence[int],
        limit: int,
        candidate_pool: int = 20,
    ) -> List[SearchHit]:
        """Fuse dense (pgvector) and lexical (OpenSearch) results with Reciprocal Rank Fusion.

        Both run concurrently; RRF uses ranks, so scores never need a common scale. A chunk missing
        from one store just drops out of that leg — /healthz reports the count drift for that reason.
        """
        vector_hits, lexical_hits = await asyncio.gather(
            self.search_vector(embedding, space_ids, candidate_pool),
            self.search_bm25(query, space_ids, candidate_pool),
        )
        return fuse_rrf(vector_hits, lexical_hits, limit=limit)

    async def iter_chunks_for_reindex(self, batch_size: int = 500):
        """Stream every chunk from Postgres, oldest id first, for POST /search/reindex.

        Keyset pagination (OFFSET gets quadratically slow on big corpora); yields batches for bulk
        indexing.
        """
        last_id = ""
        while True:
            rows = await self._pool.fetch(
                """
                SELECT id, chunk_type, issue_id, issue_key, space_id, source_id, chunk_index,
                       content, page_number, provenance
                FROM chunks
                WHERE id > $1
                ORDER BY id
                LIMIT $2
                """,
                last_id,
                batch_size,
            )
            if not rows:
                return
            yield [
                Chunk(
                    id=r["id"],
                    chunk_type=r["chunk_type"],
                    issue_id=r["issue_id"],
                    issue_key=r["issue_key"],
                    space_id=r["space_id"],
                    source_id=r["source_id"],
                    chunk_index=r["chunk_index"],
                    content=r["content"],
                    page_number=r["page_number"],
                    provenance=r.get("provenance") or {},
                )
                for r in rows
            ]
            last_id = rows[-1]["id"]


def thread_order(comments: List[IssueComment]) -> List[IssueComment]:
    """Regroups chronologically ordered comments so each reply follows its top-level comment.

    Top-level comments keep their order; a thread's replies keep theirs. A reply whose parent isn't
    in the list (deleted, or cut by the limit) stays where it was rather than being dropped.
    """
    present = {c.source_id for c in comments}
    replies: dict = {}
    for c in comments:
        parent = c.provenance.get("parent_id")
        if parent is not None and parent in present:
            replies.setdefault(parent, []).append(c)
    ordered = []
    for c in comments:
        parent = c.provenance.get("parent_id")
        if parent is not None and parent in present:
            continue
        ordered.append(c)
        ordered.extend(replies.get(c.source_id, []))
    return ordered


async def _mark_content_changed(conn, source_type: str, source_id: int) -> None:
    """Move the source's `content_changed_at` (the watermark's input). `conn` may be a pool.

    A source with no status row yet gets one later from the tracker, whose insert sets it too.
    """
    await conn.execute(
        "UPDATE ingestion_status SET content_changed_at = now() "
        "WHERE source_type = $1 AND source_id = $2",
        source_type,
        source_id,
    )


def _rows_affected(command_tag: str) -> int:
    # asyncpg returns tags like "DELETE 3"; the trailing integer is the row count.
    try:
        return int(command_tag.split()[-1])
    except (ValueError, IndexError):
        return 0


def _row_to_hit(row: asyncpg.Record, retrievers: List[str]) -> SearchHit:
    """Build a hit from a pgvector row (lexical hits are built by ChunkSearchIndex)."""
    return SearchHit(
        id=row["id"],
        chunk_type=row["chunk_type"],
        issue_id=row["issue_id"],
        issue_key=row["issue_key"],
        space_id=row["space_id"],
        source_id=row["source_id"],
        content=row["content"],
        score=float(row["score"]),
        retrievers=retrievers,
        page_number=row["page_number"],
        provenance=_coerce_provenance(row["provenance"]),
    )


def _row_to_status(row: asyncpg.Record) -> IngestionStatusRow:
    return IngestionStatusRow(
        source_type=row["source_type"],
        source_id=row["source_id"],
        space_id=row["space_id"],
        issue_key=row["issue_key"],
        source_label=row["source_label"],
        state=row["state"],
        stage=row["stage"],
        attempts=row["attempts"],
        chunk_count=row["chunk_count"],
        error_type=row["error_type"],
        error_message=row["error_message"],
        event_id=row["event_id"],
        request_id=row["request_id"],
        duration_ms=row["duration_ms"],
        # Same pool-codec guard `provenance` already needs: normal reads come back as a dict.
        stage_ms=_coerce_provenance(row["stage_ms"]),
        first_seen_at=row["first_seen_at"],
        updated_at=row["updated_at"],
    )


def _row_to_vlm_retry(row: asyncpg.Record) -> VlmRetryItem:
    return VlmRetryItem(
        attachment_id=row["attachment_id"],
        event_id=row["event_id"],
        message=_coerce_provenance(row["message"]),
        attempts=row["attempts"],
    )


def _coerce_provenance(value) -> dict:
    """Guard for JSONB values that arrive as strings (an old pool or test double without the codec),
    so one such connection can't cause a service-wide 500 again.
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("invalid JSON provenance; using empty locator")
            return {}
        if isinstance(decoded, dict):
            return decoded
    if value not in (None, ""):
        logger.warning("non-object provenance; using empty locator", extra={"type": type(value).__name__})
    return {}
