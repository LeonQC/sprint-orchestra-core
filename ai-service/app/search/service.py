"""Collapse vectorization-service's chunk hits into one result per issue (its best chunk).

Used by "find related issues" and duplicate detection, which ask "is there already an issue about
this?", not "which passage answers this".

Excluded chunk types:
  * attachment — OCR'd screenshots share boilerplate ("SYNTHETIC STAGING SCREENSHOT"), so they
    cluster together regardless of topic (avg similarity 0.54 to each other vs 0.47 for issues).
    /ask citations are unaffected.
  * sprint — a sprint chunk's issue_id/issue_key are the sprint's own id/name and can collide with a
    real issue id.

Two incompatible score scales — don't mix them:
  * mode="vector", no rerank: cosine similarity, 0-1. `DEFAULT_MIN_SCORE` = 0.7 (related issues
    scored ~0.75-0.85, same-domain unrelated ~0.64-0.66; a heuristic, not a calibrated value).
  * reranked (VEC_RERANK_ENABLED, server-side): an unbounded cross-encoder logit. Seen live: real
    matches 5.39 / 1.71, unrelated -4.6, no-match queries ~-11.4. `RERANKED_MIN_SCORE` = 0.0 ("the
    model judged it relevant").
Hybrid (RRF) scores have no absolute meaning (~0.03 max), so callers must use mode="vector".
`min_score` exists so a query with no real match returns nothing instead of its best weak hit.
"""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional

from app.agent.retrieval_tool import RetrievedChunk

DEFAULT_EXCLUDED_CHUNK_TYPES: FrozenSet[str] = frozenset({"attachment", "sprint"})
DEFAULT_MIN_SCORE = 0.7
RERANKED_MIN_SCORE = 0.0


def relevance_confidence(score: float, reranked: bool) -> float:
    """Map a raw `score` to a real 0-1 confidence, whichever scale it's on.

    Callers can't interpret the raw score (they can't see whether the server reranked), and the
    duplicates UI rendered `score * 100` — showing "640% match" for a 6.395 logit, and a misleading
    "20%" for a barely-relevant 0.2.

      * reranked: sigmoid(logit). The cross-encoder is trained with binary cross-entropy, so this is
        its own P(relevant); logit 0 (RERANKED_MIN_SCORE) maps to exactly 0.5.
      * not reranked: cosine is already 0-1; just clamp.

    The raw `score` is still returned for debugging and custom thresholds.
    """
    if not reranked:
        return min(max(score, 0.0), 1.0)
    # Overflow-safe sigmoid — a deeply negative logit (live no-match observations sat around -11.4,
    # and nothing bounds it) would overflow the naive exp(-score) form.
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1.0 + exp_score)


def dedupe_by_issue(
    chunks: List[RetrievedChunk],
    limit: int,
    exclude_chunk_types: Optional[FrozenSet[str]] = DEFAULT_EXCLUDED_CHUNK_TYPES,
    min_score: Optional[float] = DEFAULT_MIN_SCORE,
) -> List[RetrievedChunk]:
    best_by_issue: Dict[int, RetrievedChunk] = {}
    for chunk in chunks:
        if exclude_chunk_types and chunk.chunk_type in exclude_chunk_types:
            continue
        if min_score is not None and chunk.score < min_score:
            continue
        current = best_by_issue.get(chunk.issue_id)
        if current is None or chunk.score > current.score:
            best_by_issue[chunk.issue_id] = chunk
    ranked = sorted(best_by_issue.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]
