"""Precedent search: "has someone here already solved this?"

  1. RETRIEVE: semantic search on title + description only (comment chunks excluded). On its own this
     just finds issues that sound alike.
  2. JUDGE: one structured call that keeps only issues worth opening and says why. "none_relevant" is
     a valid result.

Comment threads are never read. An earlier version read them to find buried lessons, but truncating
a long thread could drop the key comment, and the model then wrote an empty "takeaway" from filler
("deployed to staging") — worse than saying nothing. Title + description already capture what makes
a precedent (same component, config, subsystem). `takeaway` is now filled only when the description
itself states an approach, so it's often null.

Finished work is ranked first (`_rank`): a closed issue is something to learn from; an open lookalike
is something to coordinate with. Any failure returns an empty report with `degraded=True`, never a 500.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.precedent.prompts import (
    Candidate,
    plain_text,
    render_precedent_message,
    render_precedent_system_prompt,
)
from app.precedent.schemas import PrecedentReport
from app.search.service import dedupe_by_issue, relevance_confidence
from app.tracing import llm_step

logger = logging.getLogger(__name__)

_MAX_VALIDATION_RETRIES = 2
_MAX_OUTPUT_TOKENS = 1500
# Chunks to pull before dedup — several can belong to one issue, so the raw pool must be deeper than
# the candidate count. Capped at 50: vectorization-service's own search limit is `le=50`.
_RAW_FETCH = 50
# Candidates handed to the model. Deliberately more than the 3 it may return: the extras are what
# give it something to reject, which is what makes `none_relevant` meaningful rather than reflexive.
_MAX_CANDIDATES = 8
# Floor on `relevance_confidence`, not the raw score: the raw scale depends on whether the server
# reranked, so a raw threshold means different things in different deployments (the "640% match"
# bug). 0.5 = "more likely relevant than not" (logit 0 for reranked hits, 0.5 cosine otherwise).
# Loose on purpose; the judging call decides real relevance.
_MIN_CONFIDENCE = 0.5

# Match on the issue's own title + description only. Comments carry incidental words ("carried into
# Sprint 7"), and since dedupe keeps each issue's best chunk, one passing remark could make an
# unrelated ticket the top match.
_MATCH_EXCLUDED_CHUNK_TYPES = frozenset({"attachment", "sprint", "comment"})

# Cap on description text in the query, so a long ticket's important opening isn't cut off by the
# embedding model's limit.
_MAX_QUERY_DESC_CHARS = 2000


@dataclass(slots=True)
class PrecedentResult:
    report: PrecedentReport
    candidates_considered: int = 0
    dropped_precedents: int = 0
    degraded: bool = False
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None
    #: Retrieval found nothing above the floor — a young or empty space, not a failure.
    no_candidates: bool = False
    candidate_keys: List[str] = field(default_factory=list)
    #: issue_key -> real title from query_issues, never from the model (one less field for it to get
    #: wrong).
    candidate_titles: Dict[str, str] = field(default_factory=dict)


def _empty_report() -> PrecedentReport:
    return PrecedentReport(verdict="none_relevant", precedents=[])


def _rank(candidates: List[Candidate]) -> List[Candidate]:
    """Finished work first; stable, so retrieval order still breaks ties within each group."""
    return sorted(candidates, key=lambda c: not c.is_done)


async def find_precedents(
    client,
    model: str,
    retrieval,
    space_ids: Sequence[int],
    issue_key: str,
    title: str,
    description: str,
) -> PrecedentResult:
    start = time.perf_counter()
    # Plain text, not stored HTML: with tags left in, the same issue found 0 candidates instead of 4.
    query = f"{title}\n\n{plain_text(description, _MAX_QUERY_DESC_CHARS)}".strip()

    try:
        search = await retrieval.search_with_meta(query, space_ids, limit=_RAW_FETCH, mode="vector")
        above_floor = [
            h for h in search.hits
            if h.chunk_type not in _MATCH_EXCLUDED_CHUNK_TYPES
            and relevance_confidence(h.score, search.reranked) >= _MIN_CONFIDENCE
        ]
        # Already filtered above (normalized scale, chunk type), so dedupe only collapses to one chunk
        # per issue.
        hits = dedupe_by_issue(
            above_floor, _MAX_CANDIDATES + 1,
            exclude_chunk_types=_MATCH_EXCLUDED_CHUNK_TYPES, min_score=float("-inf"),
        )
        # The current issue is always its own best semantic match; it is not a precedent for itself.
        keys = [h.issue_key for h in hits if h.issue_key.upper() != issue_key.upper()][:_MAX_CANDIDATES]
        if not keys:
            return PrecedentResult(
                report=_empty_report(), no_candidates=True,
                latency_seconds=time.perf_counter() - start,
            )

        # Two fetches: query_issues has status/type but no body text, and /issues/details has the
        # body. No comment fetch — see the module docstring for why that was cut.
        rows = await retrieval.query_issues(space_ids, {"issue_keys": keys, "limit": len(keys)})
        details = await retrieval.get_issue_details(space_ids, keys)
    except Exception as exc:  # noqa: BLE001 - retrieval failure degrades, never 500
        logger.warning("precedent retrieval failed; returning an empty report", exc_info=True)
        return PrecedentResult(
            report=_empty_report(), degraded=True,
            latency_seconds=time.perf_counter() - start, error=str(exc),
        )

    by_key = {str(r.get("issue_key")): r for r in rows.issues}
    body_by_key: dict = {}
    for detail in details.details:
        # An issue has one `chunk_type='issue'` body chunk; join defensively in case it was split.
        body_by_key.setdefault(str(detail.get("issue_key")), []).append(detail.get("content") or "")

    candidates = [
        Candidate(
            issue_key=key,
            title=str((by_key.get(key) or {}).get("title") or ""),
            status=(by_key.get(key) or {}).get("status"),
            issue_type=(by_key.get(key) or {}).get("issue_type"),
            description="\n".join(body_by_key.get(key, [])),
        )
        for key in keys
    ]
    candidates = _rank(candidates)
    candidate_titles = {c.issue_key: c.title for c in candidates}

    system_prompt = render_precedent_system_prompt()
    user_message = render_precedent_message(issue_key, title, description, candidates)
    try:
        with llm_step("link_similar_work_items.find_precedents"):
            report, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=PrecedentReport,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a judging failure degrades, never 500
        logger.warning("precedent judging failed; returning an empty report", exc_info=True)
        return PrecedentResult(
            report=_empty_report(), candidates_considered=len(candidates), degraded=True,
            latency_seconds=time.perf_counter() - start, error=str(exc),
            candidate_keys=[c.issue_key for c in candidates], candidate_titles=candidate_titles,
        )

    valid = {c.issue_key.upper() for c in candidates}
    kept = [p for p in report.precedents if p.issue_key.upper() in valid]
    dropped = len(report.precedents) - len(kept)
    if dropped:
        logger.warning("precedent(s) dropped: cited an issue that was never a candidate (%d)", dropped)

    usage = usage_from_completion(completion)
    return PrecedentResult(
        # Normalized rather than trusted: a "found" verdict whose citations all failed verification
        # has, in effect, found nothing, and the UI would otherwise claim a result it cannot show.
        report=PrecedentReport(verdict="found" if kept else "none_relevant", precedents=kept),
        candidates_considered=len(candidates),
        dropped_precedents=dropped,
        latency_seconds=time.perf_counter() - start,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
        candidate_keys=[c.issue_key for c in candidates],
        candidate_titles=candidate_titles,
    )
