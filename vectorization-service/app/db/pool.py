"""asyncpg connection pool lifecycle, registered with pgvector's type codec."""
from __future__ import annotations

import json

import asyncpg
from pgvector.asyncpg import register_vector

from app.config import Settings


async def _init_connection(conn: asyncpg.Connection) -> None:
    # asyncpg intentionally leaves json/jsonb as strings unless an application codec is registered.
    # That violated the service boundary once `chunks.provenance JSONB` was added: the domain model
    # and FastAPI response both require a dictionary, but a real DB read returned the string "{}" and
    # response validation crashed every semantic search. Register both directions once, on every
    # pooled connection, so callers and write paths consistently deal in native Python values.
    #
    # Found live via an attachment (image) chunk, whose provenance is never empty (carries bbox/
    # source_type) — that's what actually triggers this, not text sources like issue/comment chunks
    # whose provenance is `{}`. The failure surfaces two layers away, as `/search` returning 502
    # (see app/api/routes.py), which ai-service then reports as "nothing in the knowledge base" —
    # easy to misdiagnose as an OCR/docling parsing gap when ingestion itself was never the problem.
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(
            type_name,
            schema="pg_catalog",
            encoder=json.dumps,
            decoder=json.loads,
            format="text",
        )
    # Teach asyncpg how to encode/decode the pgvector `vector` type on every pooled connection.
    await register_vector(conn)


async def create_pool(settings: Settings) -> asyncpg.Pool:
    return await asyncpg.create_pool(
        dsn=settings.pg_dsn,
        min_size=settings.pg_pool_min_size,
        max_size=settings.pg_pool_max_size,
        init=_init_connection,
        # The extension lives in public; all derived tables resolve inside rag. The runtime role
        # has no privileges on public business tables, so a bad query fails rather than reading Jira.
        server_settings={"search_path": "rag, public", "application_name": "vectorization-service"},
    )
