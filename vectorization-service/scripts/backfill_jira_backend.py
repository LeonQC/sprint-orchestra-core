"""One-time backfill: ingest jira-backend's EXISTING issues/comments into this service's index.

Why this needs to exist at all: Kafka events (jira.content.ingestion) only fire going forward, from
the moment ContentIngestionKafkaPublisher is enabled
(jira-backend's APP_KAFKA_CONTENT_INGESTION_ENABLED=true) -- they are not retroactive. Any issue or
comment created before that flag was ever turned on (which, in this project's own dev environment,
was *everything* -- confirmed by checking the running jira-backend process's env: only
APP_KAFKA_ATTACHMENT_INGESTION_ENABLED was set) has no corresponding Kafka message and never reaches
this service on its own.

This script closes that gap by reading directly from jira-backend's own Postgres and replaying each
row through the *exact same* `IngestPipeline.handle_issue` / `handle_comment` methods a real Kafka
message would have driven (see app/ingest/pipeline.py) -- so backfilled data is shaped identically to
data that arrives the normal way, with no separate/duplicated ingestion logic to drift out of sync.

Usage:
    JIRA_BACKEND_PG_DSN=postgresql://poc:poc123@localhost:5432/pocdb \
    VEC_EMBEDDING_PROVIDER=openai VEC_OPENAI_BASE_URL=http://localhost:4000 \
    VEC_EMBEDDING_MODEL=openai-embedding-3-small VEC_EMBEDDING_DIM=1024 VEC_EMBEDDING_TRUNCATE_DIM=1024 \
    python -m scripts.backfill_jira_backend --space-ids 1,3 --truncate-first
"""
from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import asyncpg

from app.config import Settings
from app.db.pool import create_pool
from app.db.vector_store import VectorStore
from app.search.factory import build_chunk_search_index
from app.ingest.contextualizer import build_context_generator
from app.ingest.vlm_cache import CachedVlmImageDescriber
from app.ingest.vlm_describer import build_vlm_describer, vlm_cache_namespace
from app.ingest.embedder import build_embedder
from app.ingest.pipeline import IngestPipeline, comment_provenance
from app.ingest.status import build_ingestion_tracker
from app.models import (
    AttachmentIngestionMessage,
    CommentIngestionMessage,
    IssueIngestionMessage,
    SprintIngestionMessage,
)


async def _fetch_sprints(jira_pool: asyncpg.Pool, space_ids: Optional[List[int]]):
    query = """
        SELECT id, name, space_id, goal, updated_at FROM sprints
    """
    if space_ids:
        query += " WHERE space_id = ANY($1::bigint[])"
        return await jira_pool.fetch(query, space_ids)
    return await jira_pool.fetch(query)


async def _fetch_issues(jira_pool: asyncpg.Pool, space_ids: Optional[List[int]]):
    query = """
        SELECT i.id, i.issue_key, i.space_id, i.title, i.description, i.updated_at,
               p.issue_key AS parent_key, p.title AS parent_title
        FROM issues i
        LEFT JOIN issues p ON p.id = i.parent_id
    """
    if space_ids:
        query += " WHERE i.space_id = ANY($1::bigint[])"
        return await jira_pool.fetch(query, space_ids)
    return await jira_pool.fetch(query)


async def _fetch_comments(jira_pool: asyncpg.Pool, space_ids: Optional[List[int]]):
    query = """
        SELECT c.id, c.issue_id, i.issue_key, i.space_id, c.content, c.updated_at, c.created_at,
               u.name AS author_name, c.parent_id, c.reply_to_id
        FROM comments c JOIN issues i ON i.id = c.issue_id
        LEFT JOIN users u ON u.id = c.author_id
    """
    if space_ids:
        query += " WHERE i.space_id = ANY($1::bigint[])"
        return await jira_pool.fetch(query, space_ids)
    return await jira_pool.fetch(query)


async def _fetch_attachments(jira_pool: asyncpg.Pool, space_ids: Optional[List[int]]):
    query = """
        SELECT a.id, a.issue_id, i.issue_key, i.space_id, a.original_filename,
               a.storage_filename, a.content_type, a.size_bytes, a.created_at
        FROM issue_attachments a JOIN issues i ON i.id = a.issue_id
    """
    if space_ids:
        query += " WHERE i.space_id = ANY($1::bigint[])"
        return await jira_pool.fetch(query, space_ids)
    return await jira_pool.fetch(query)


def _backfilled_chunk_types(sprints_only: bool, include_attachments: bool) -> List[str]:
    chunk_types = ["sprint"]
    if not sprints_only:
        chunk_types += ["issue", "comment"]
        if include_attachments:
            chunk_types.append("attachment")
    return chunk_types


async def _delete_chunks(vec_pool, chunk_types: List[str], space_ids: Optional[List[int]]) -> None:
    """Delete only what this run writes back. A plain `DELETE FROM chunks` ignored --space-ids and
    took every attachment with it, which only --include-attachments refills.
    """
    for table in ("chunks", "chunk_parents"):
        if space_ids:
            await vec_pool.execute(
                f"DELETE FROM {table} WHERE chunk_type = ANY($1::text[]) AND space_id = ANY($2::bigint[])",
                chunk_types, space_ids,
            )
        else:
            await vec_pool.execute(f"DELETE FROM {table} WHERE chunk_type = ANY($1::text[])", chunk_types)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--space-ids", type=str, default="",
        help="Comma-separated jira-backend space ids to backfill (default: every space)",
    )
    parser.add_argument(
        "--truncate-first", action="store_true",
        help="First delete the chunks this run writes back: the chosen spaces (default all) and only "
             "the chunk types being backfilled (attachments only with --include-attachments)",
    )
    parser.add_argument(
        "--include-attachments", action="store_true",
        help="Also backfill attachments: fetches binaries from S3/MinIO and parses via Docling "
             "(much slower than issues/comments — OCR model load alone takes several seconds, "
             "then real per-file conversion time on top).",
    )
    parser.add_argument(
        "--sprints-only", action="store_true",
        help="Backfill sprint goals only. Useful when sprint events were enabled after the "
             "existing sprint rows were created; avoids re-embedding every issue and comment.",
    )
    parser.add_argument(
        "--comment-provenance-only", action="store_true",
        help="Fast, embed-free fill of comment chunks' author/created_at/parent_id/reply_to_id "
             "provenance (what get_issue_comments orders and labels by). Use when comments were "
             "ingested before CommentIngestionMessage carried those fields; re-embeds nothing. Run "
             "POST /search/reindex afterwards so OpenSearch's stored copy matches.",
    )
    args = parser.parse_args()
    space_ids = [int(s) for s in args.space_ids.split(",") if s.strip()] or None

    jira_dsn = os.environ.get("JIRA_BACKEND_PG_DSN", "postgresql://poc:poc123@localhost:5432/pocdb")

    settings = Settings()
    print(f"embedder: provider={settings.embedding_provider} model={settings.embedding_model} "
          f"dim={settings.embedding_dim}")
    print(f"jira-backend source: {jira_dsn}  space_ids filter: {space_ids or '(all)'}")

    vec_pool = await create_pool(settings)
    jira_pool = await asyncpg.create_pool(jira_dsn, min_size=1, max_size=4)
    embedder = build_embedder(settings)
    # With the OpenSearch index, so BM25 sees the same chunks. Without it these scripts wrote Postgres
    # only, and OpenSearch kept serving the old text until someone rebuilt it.
    search_index = build_chunk_search_index(settings)
    await search_index.ensure_index()
    store = VectorStore(vec_pool, search_index)
    # Was omitted here (defaulting IngestPipeline to a no-op context generator) even when
    # VEC_CONTEXTUAL_RETRIEVAL_ENABLED=true — the live Kafka path (app/main.py) already wires this up
    # correctly; the backfill script just never matched it, so every backfill-produced chunk silently
    # skipped contextualization regardless of the setting. Found live: re-ran the attachment backfill
    # expecting contextualized chunks and found byte-identical content to the pre-fix run.
    context_generator = build_context_generator(settings)
    # Same class of bug as the context_generator comment above, caught this time before it shipped:
    # this script builds its own IngestPipeline separately from app/main.py's, so any dependency added
    # there has to be matched here too or it silently no-ops for every backfill run.
    raw_vlm_describer = build_vlm_describer(settings)
    vlm_describer = (
        CachedVlmImageDescriber(raw_vlm_describer, store, vlm_cache_namespace(settings))
        if settings.vlm_ocr_enabled
        else raw_vlm_describer
    )
    # Also records ingestion_status, like the live consumer. Without it a backfill left the status
    # table describing whatever happened before (rows said "indexed" for chunks that were gone).
    tracker = build_ingestion_tracker(settings, store)
    pipeline = IngestPipeline(settings, embedder, store, context_generator, vlm_describer, tracker)

    try:
        if args.comment_provenance_only:
            rows = await _fetch_comments(jira_pool, space_ids)
            matched = 0
            for row in rows:
                provenance = comment_provenance(CommentIngestionMessage(
                    event_id="provenance-only", event_type="comment_upserted",
                    emitted_at=datetime.now(timezone.utc), comment_id=row["id"], issue_id=row["issue_id"],
                    issue_key=row["issue_key"], space_id=row["space_id"], author_name=row["author_name"],
                    created_at=row["created_at"], parent_id=row["parent_id"], reply_to_id=row["reply_to_id"],
                ))
                res = await vec_pool.execute(
                    "UPDATE chunks SET provenance = $2 WHERE chunk_type = 'comment' AND source_id = $1",
                    row["id"], provenance,
                )
                if res.rsplit(" ", 1)[-1] != "0":
                    matched += 1
            print(f"comment-provenance-only: updated {matched}/{len(rows)} comments in the vec store.")
            return

        if args.truncate_first:
            chunk_types = _backfilled_chunk_types(args.sprints_only, args.include_attachments)
            await _delete_chunks(vec_pool, chunk_types, space_ids)
            print(f"cleared {', '.join(chunk_types)} chunks for {space_ids or 'every space'}")
            print("(OpenSearch still holds deleted sources: delete the index and POST /search/reindex)\n")

        sprints = await _fetch_sprints(jira_pool, space_ids)
        print(f"backfilling {len(sprints)} sprints...")
        for row in sprints:
            msg = SprintIngestionMessage(
                event_id=str(uuid.uuid4()),
                event_type="sprint_upserted",
                emitted_at=row["updated_at"] or datetime.now(timezone.utc),
                sprint_id=row["id"],
                sprint_name=row["name"],
                space_id=row["space_id"],
                goal=row["goal"],
            )
            await pipeline.handle_sprint(msg)

        if not args.sprints_only:
            issues = await _fetch_issues(jira_pool, space_ids)
            print(f"backfilling {len(issues)} issues...")
            for row in issues:
                msg = IssueIngestionMessage(
                    event_id=str(uuid.uuid4()),
                    event_type="issue_upserted",
                    emitted_at=row["updated_at"] or datetime.now(timezone.utc),
                    issue_id=row["id"],
                    issue_key=row["issue_key"],
                    space_id=row["space_id"],
                    title=row["title"],
                    description=row["description"],
                    parent_key=row["parent_key"],
                    parent_title=row["parent_title"],
                )
                await pipeline.handle_issue(msg)

            comments = await _fetch_comments(jira_pool, space_ids)
            print(f"backfilling {len(comments)} comments...")
            for row in comments:
                msg = CommentIngestionMessage(
                    event_id=str(uuid.uuid4()),
                    event_type="comment_upserted",
                    emitted_at=row["updated_at"] or datetime.now(timezone.utc),
                    comment_id=row["id"],
                    issue_id=row["issue_id"],
                    issue_key=row["issue_key"],
                    space_id=row["space_id"],
                    content=row["content"],
                    author_name=row["author_name"],
                    created_at=row["created_at"],
                    parent_id=row["parent_id"],
                    reply_to_id=row["reply_to_id"],
                )
                await pipeline.handle_comment(msg)

        if args.include_attachments and not args.sprints_only:
            attachment_bucket = os.environ.get("JIRA_ATTACHMENT_BUCKET", "jira-attachments")
            attachments = await _fetch_attachments(jira_pool, space_ids)
            print(f"backfilling {len(attachments)} attachments (bucket={attachment_bucket})...")
            for i, row in enumerate(attachments, 1):
                msg = AttachmentIngestionMessage(
                    event_id=str(uuid.uuid4()),
                    event_type="attachment_uploaded",
                    emitted_at=row["created_at"] or datetime.now(timezone.utc),
                    attachment_id=row["id"],
                    issue_id=row["issue_id"],
                    issue_key=row["issue_key"],
                    space_id=row["space_id"],
                    filename=row["original_filename"],
                    mime_type=row["content_type"],
                    byte_size=row["size_bytes"],
                    storage_backend="s3",
                    bucket=attachment_bucket,
                    storage_key=row["storage_filename"],
                )
                await pipeline.handle_attachment(msg)
                if i % 10 == 0 or i == len(attachments):
                    print(f"  ...{i}/{len(attachments)}")

        print(f"\ndone -> {await store.count()} chunks now in the vector store")
    finally:
        await embedder.aclose()
        await context_generator.aclose()
        await vlm_describer.aclose()
        await search_index.aclose()
        await vec_pool.close()
        await jira_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
