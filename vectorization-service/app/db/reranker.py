"""Second-stage reranking: a cross-encoder re-scores the RRF-fused candidate pool.

RRF combines two rankings that never looked at query and chunk together. A cross-encoder scores the
(query, chunk) pair jointly — more accurate, but too slow for the whole index, hence the standard
two-stage setup: retrieve broadly and cheaply, then rerank a small pool.

On by default (`VEC_RERANK_ENABLED`): load tests showed only ~10ms P50 / ~50ms P95 overhead. Needs
sentence-transformers (a large torch download). See README "Reranking" for numbers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Protocol

from app.models import SearchHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    async def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]: ...


class NoopReranker:
    """Default: pass the first-stage ranking through unchanged, just truncated to top_n."""

    async def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        return hits[:top_n]


class CrossEncoderReranker:
    """A local `sentence_transformers.CrossEncoder`, loaded once at startup (~1-2s). `predict` is
    blocking and CPU-bound, so it runs in a worker thread to keep the event loop free.
    """

    def __init__(self, model_name: str, score_threshold: float | None = None):
        from sentence_transformers import CrossEncoder  # imported lazily: heavy, torch-backed

        logger.info(
            "loading cross-encoder reranker model=%s score_threshold=%s", model_name, score_threshold
        )
        self._model = CrossEncoder(model_name)
        self._score_threshold = score_threshold

    async def rerank(self, query: str, hits: List[SearchHit], top_n: int) -> List[SearchHit]:
        if not hits:
            return hits
        # Score "issue key + content": the key isn't part of `content`, so for a query like "ATLAS-6"
        # the cross-encoder couldn't tell the chunk was relevant — lexical-query MRR fell from 0.70 to
        # 0.25 with content alone.
        pairs = [(query, f"{hit.issue_key} {hit.content}") for hit in hits]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        if self._score_threshold is not None:
            # Drop candidates the cross-encoder judged irrelevant, even if that leaves nothing: the CRAG
            # loop treats "No results found." as a cue to search again, which beats a padded top-1 guess.
            ranked = [pair for pair in ranked if pair[1] >= self._score_threshold]
        return [
            SearchHit(
                id=hit.id,
                chunk_type=hit.chunk_type,
                issue_id=hit.issue_id,
                issue_key=hit.issue_key,
                space_id=hit.space_id,
                source_id=hit.source_id,
                content=hit.content,
                score=float(score),
                retrievers=hit.retrievers,
                page_number=hit.page_number,
                provenance=hit.provenance,
            )
            for hit, score in ranked[:top_n]
        ]


def build_reranker(settings) -> Reranker:
    if not settings.rerank_enabled:
        return NoopReranker()
    return CrossEncoderReranker(settings.rerank_model, settings.rerank_score_threshold)
