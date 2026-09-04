"""Construction of the OpenSearch client, kept out of chunk_index.py so tests can build a
ChunkSearchIndex against any client (a real one, a stub) without going through settings.
"""
from __future__ import annotations

from opensearchpy import AsyncOpenSearch

from app.config import Settings
from app.search.chunk_index import ChunkSearchIndex


def build_opensearch_client(settings: Settings) -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        timeout=settings.opensearch_request_timeout_seconds,
        # One retry, on a different node when there is one. Retrying a search is safe (idempotent);
        # retrying an indexing call is safe too because every document is written under a
        # deterministic _id.
        max_retries=1,
        retry_on_timeout=True,
    )


def build_chunk_search_index(settings: Settings) -> ChunkSearchIndex:
    return ChunkSearchIndex(
        build_opensearch_client(settings), settings.opensearch_chunk_index
    )
