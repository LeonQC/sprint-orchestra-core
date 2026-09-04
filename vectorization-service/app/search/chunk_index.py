"""OpenSearch index over chunks — the keyword (BM25) half of hybrid retrieval.

Postgres has no real BM25: `ts_rank_cd` lacks IDF and length normalization, and the pure-Python BM25
that followed re-tokenized the whole space on every query (O(n)). OpenSearch gives real BM25 over an
inverted index. OpenSearch rather than Elasticsearch because we deploy to Amazon OpenSearch Service
(IAM, CloudWatch, VPC); the queries and mappings here work on both, only the client differs.

One document per chunk (not per issue), since RAG retrieves the passage that goes into the context.
jira-backend keeps separate issue-level indices for the search bar.

Consistency: Postgres is the source of truth; this index is written right after the Postgres commit,
best effort. If OpenSearch is down it falls behind. Instead of an outbox, POST /search/reindex rebuilds
it, and /healthz reports the document-count drift.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from opensearchpy import AsyncOpenSearch, NotFoundError
from opensearchpy.helpers import async_bulk

from app.models import Chunk, SearchHit
from app.search.analysis import INDEX_SETTINGS

logger = logging.getLogger(__name__)

CHUNK_MAPPING: Dict[str, Any] = {
    "properties": {
        "chunk_type": {"type": "keyword"},
        "issue_id": {"type": "long"},
        # Indexed twice: `issue_key` (keyword) for exact filters, `issue_key.text` searchable, so a
        # query for a bare "ATLAS-6" matches even when the chunk text never mentions the key.
        "issue_key": {
            "type": "keyword",
            "fields": {"text": {"type": "text", "analyzer": "orchestra_identifier"}},
        },
        "space_id": {"type": "long"},
        "source_id": {"type": "long"},
        "chunk_index": {"type": "integer"},
        "content": {"type": "text", "analyzer": "orchestra_text"},
        "page_number": {"type": "integer"},
        # Format-specific locator (slide, sheet range, bbox): stored so hits render from OpenSearch,
        # but `enabled: false` — nothing searches it, and its varying keys would bloat the mapping.
        "provenance": {"type": "object", "enabled": False},
    }
}


class ChunkSearchIndex:
    """Write path (kept in step with VectorStore) plus the BM25 read path."""

    def __init__(self, client: AsyncOpenSearch, index: str):
        self._client = client
        self._index = index

    @property
    def index_name(self) -> str:
        return self._index

    async def ensure_index(self) -> None:
        """Create the index with its mapping if it does not exist. Safe to call on every boot."""
        if await self._client.indices.exists(index=self._index):
            return
        await self._client.indices.create(
            index=self._index,
            body={"settings": INDEX_SETTINGS, "mappings": CHUNK_MAPPING},
        )
        logger.info("created opensearch index %s", self._index)

    # --- write path -------------------------------------------------------------------------

    async def replace_source(self, chunk_type: str, source_id: int, chunks: Sequence[Chunk]) -> None:
        """Same as VectorStore.replace_source: delete this source's documents, then index the new ones.
        Not a plain upsert, because a re-parse can produce fewer chunks and the extras would linger.
        """
        await self.delete_source(chunk_type, source_id, refresh=False)
        await self.index_chunks(chunks)
        # One refresh at the end, so old and new documents swap together and the result is searchable
        # right away (ingestion smoke evals rely on that).
        await self.refresh()

    async def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Bulk-index chunks under their deterministic ids (the Postgres primary key), so redelivery or
        a repeated reindex overwrites instead of duplicating. That's what makes /search/reindex safe to
        run anytime.
        """
        if not chunks:
            return
        await async_bulk(
            self._client,
            [
                {
                    "_op_type": "index",
                    "_index": self._index,
                    "_id": chunk.id,
                    "_source": _chunk_to_doc(chunk),
                }
                for chunk in chunks
            ],
        )

    async def refresh(self) -> None:
        """Make recent writes searchable now (otherwise up to ~1s delay)."""
        await self._client.indices.refresh(index=self._index, ignore_unavailable=True)

    async def delete_source(self, chunk_type: str, source_id: int, refresh: bool = True) -> int:
        return await self._delete_by_query(
            {
                "bool": {
                    "filter": [
                        {"term": {"chunk_type": chunk_type}},
                        {"term": {"source_id": source_id}},
                    ]
                }
            },
            refresh=refresh,
        )

    async def delete_issue(self, issue_id: int) -> int:
        """Drop every chunk belonging to an issue — its own text, its comments, its attachments."""
        return await self._delete_by_query({"bool": {
            "filter": [{"term": {"issue_id": issue_id}}],
            "must_not": [{"term": {"chunk_type": "sprint"}}],
        }})

    async def _delete_by_query(self, query: Dict[str, Any], refresh: bool = True) -> int:
        try:
            result = await self._client.delete_by_query(
                index=self._index, body={"query": query}, refresh=refresh, conflicts="proceed"
            )
        except NotFoundError:
            # Index not created yet — nothing indexed means nothing to delete.
            return 0
        return int(result.get("deleted", 0))

    # --- read path --------------------------------------------------------------------------

    async def search(
        self, query: str, space_ids: Sequence[int], limit: int
    ) -> List[SearchHit]:
        """BM25 search limited to the caller's spaces.

        `space_ids` is a filter, not part of the query: it's an authorization boundary, so it must not
        affect scores, and it has to apply before top-k (post-filtering top-k returns fewer results,
        not the best visible ones).
        """
        if not query.strip() or not space_ids:
            return []
        response = await self._client.search(
            index=self._index,
            body={
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            # Prose: OR semantics, looser than the old AND query. This is the first
                            # stage before RRF and a reranker, so recall matters more than precision.
                            {"match": {"content": {"query": query}}},
                            # Identifier: AND semantics. "ATLAS-6" tokenizes to ["atlas", "6"], so OR
                            # would match every issue key in the space.
                            {"match": {"issue_key.text": {"query": query, "operator": "and"}}},
                        ],
                        "minimum_should_match": 1,
                        "filter": [{"terms": {"space_id": list(space_ids)}}],
                    }
                },
            },
        )
        return [_doc_to_hit(hit) for hit in response["hits"]["hits"]]

    async def count(self, space_ids: Optional[Sequence[int]] = None) -> int:
        query = (
            {"terms": {"space_id": list(space_ids)}} if space_ids is not None else {"match_all": {}}
        )
        try:
            result = await self._client.count(index=self._index, body={"query": query})
        except NotFoundError:
            return 0
        return int(result["count"])

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def aclose(self) -> None:
        await self._client.close()


def _chunk_to_doc(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_type": chunk.chunk_type,
        "issue_id": chunk.issue_id,
        "issue_key": chunk.issue_key,
        "space_id": chunk.space_id,
        "source_id": chunk.source_id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "page_number": chunk.page_number,
        "provenance": chunk.provenance or {},
    }


def _doc_to_hit(hit: Dict[str, Any]) -> SearchHit:
    src = hit["_source"]
    return SearchHit(
        id=hit["_id"],
        chunk_type=src["chunk_type"],
        issue_id=src["issue_id"],
        issue_key=src["issue_key"],
        space_id=src["space_id"],
        source_id=src["source_id"],
        content=src["content"],
        score=float(hit["_score"]),
        retrievers=["bm25"],
        page_number=src.get("page_number"),
        provenance=src.get("provenance") or {},
    )
