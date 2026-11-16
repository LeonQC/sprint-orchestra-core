"""Reciprocal Rank Fusion — pure function, no I/O, so it's unit-testable without a database.

RRF combines *ranks* rather than raw scores because a pgvector cosine distance and an OpenSearch
BM25 score live on unrelated scales — and, since the swap, in different processes entirely, so there
is not even a shared corpus to calibrate against (same reasoning as the tiered ranking in
jira-backend's SearchService: never blend incomparable numbers directly). A chunk that appears in both candidate lists gets the sum of
both reciprocal-rank contributions, so it naturally outranks a chunk only one retriever found —
without any hand-tuned weight between the two signals.
"""
from __future__ import annotations

from typing import Dict, List, Set

from app.models import SearchHit

# Standard choice from the original RRF paper (Cormack et al., 2009) — large enough that a rank-1
# hit doesn't completely dominate a rank-2 hit from the other retriever, small enough that rank
# still matters more than how many retrievers found a result.
RRF_K = 60


def fuse_rrf(*ranked_lists: List[SearchHit], limit: int, k: int = RRF_K) -> List[SearchHit]:
    """Merge any number of already-ranked hit lists into one fused ranking."""
    fused_scores: Dict[str, float] = {}
    latest_hit_by_id: Dict[str, SearchHit] = {}
    retrievers_by_id: Dict[str, Set[str]] = {}

    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            fused_scores[hit.id] = fused_scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            latest_hit_by_id[hit.id] = hit
            if hit.id not in retrievers_by_id:
                retrievers_by_id[hit.id] = set()
            retrievers_by_id[hit.id].update(hit.retrievers)

    ranked_ids = sorted(fused_scores, key=lambda hit_id: fused_scores[hit_id], reverse=True)[:limit]
    results = []
    for hit_id in ranked_ids:
        hit = latest_hit_by_id[hit_id]
        results.append(SearchHit(
            id=hit_id,
            chunk_type=hit.chunk_type,
            issue_id=hit.issue_id,
            issue_key=hit.issue_key,
            space_id=hit.space_id,
            source_id=hit.source_id,
            content=hit.content,
            score=fused_scores[hit_id],
            retrievers=sorted(retrievers_by_id[hit_id]),
            page_number=hit.page_number,
            provenance=hit.provenance,
        ))
    return results
