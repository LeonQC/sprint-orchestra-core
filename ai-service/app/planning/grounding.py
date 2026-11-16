"""Retrieval-grounded planning: show the planner this team's finished work before it writes a plan.

On its own, `plan_epic` only knows the space's label names, so it writes a generic Jira plan. This
module retrieves similar past issues from the space and hands them to the planner as examples.

Two separate jobs:
  1. STYLE — how work is written here. Any issue with a body works, finished or not.
  2. ESTIMATE ANCHORING — how big a number is here. Only finished issues with story points count
     (`Exemplar.is_estimate_anchor`); points on unfinished work are just another guess.

Retrieval follows `app/precedent/service.py`: match only on issue title+description chunks (comments
etc. pull in unrelated tickets), strip HTML before embedding, and filter on `relevance_confidence`,
not the raw score.

Searching once for the whole epic embeds the average of several pieces of work, which resembles none
of them — 40% of estimated issues ended up citing nothing. So `gather_requirement_exemplars` searches
once per stated requirement. With no requirements given, it falls back to one whole-proposal search.

Similarity alone left too few examples (a real board gave 2, neither sized), so there are two tiers,
labelled all the way to the API response (`Exemplar.source`):
  * "similar" — passed the relevance floor; good enough to size against.
  * "recent" — the board's latest finished, sized work, used to fill the remaining slots. Not
    topically matched, never presented as if it were.

Safety net: every estimate must name the exemplar it compared against (`estimate_reference`),
checked by `apply_estimate_grounding`, and all exemplars are returned so a reviewer can check.

Evidence: total-points error got 17% worse, but that metric also punishes splitting work into fewer
cards. Story-point distribution distance to the real board: 0.683 -> 0.477, better on 7 of 8 epics —
that's why the flag is on. Style is unproven (blind test 13-11, p=0.84).

Cost: the grounded prompt differs per call, so provider prefix caching stops working (fresh input
went from ~100 to ~4,500 tokens per call).

Any failure falls back to ungrounded planning, with `status` saying why.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Literal, Optional, Sequence, Set

from app.observability import PLANNING_GROUNDING_EXEMPLARS, PLANNING_GROUNDING_TOTAL
# Reused, not copied: embedding raw HTML found 0 candidates where plain text found 4.
from app.planning.house_format import HouseFormat, detect_house_format
from app.precedent.prompts import plain_text
from app.search.service import dedupe_by_issue, relevance_confidence

logger = logging.getLogger(__name__)

# Same done-set as app/sprint_recovery/graph.py.
_DONE_STATUSES = frozenset({"done", "completed"})

# Chunks fetched before collapsing to one per issue; 50 is vectorization-service's /search cap.
_RAW_FETCH = 50

# "More likely relevant than not", on the normalized confidence (same as precedent search).
_MIN_CONFIDENCE = 0.5

# Lower floor for per-requirement search: that's a "closest thing, if any" question, and later steps
# (pattern extraction, the planner's own rule) can still reject a weak match. Measured on this board:
# a real analogy scored 0.355, non-matches 0.000-0.05. Short queries also score lower in general.
_MIN_REQUIREMENT_CONFIDENCE = 0.1

# Per-exemplar description budget, so eight examples can't crowd out the proposal (same as precedent).
_MAX_DESC_CHARS = 700

# Cap on the proposal text used as the search query.
_MAX_QUERY_CHARS = 2000

# Issue bodies sampled to detect the board's layout. More than the exemplar count because layout
# statistics (e.g. how many acceptance criteria) are noisy on a few samples.
_FORMAT_SAMPLE = 40

# Comments are read only for already-chosen exemplars, never matched on. Pattern extraction looks for
# what recurs across several issues, so one truncated thread is just one weak signal (and a pattern
# backed by a single issue is dropped anyway). Caps keep one long thread from crowding the rest out.
_MAX_COMMENTS_PER_ISSUE = 4
_MAX_COMMENT_CHARS = 260

# The planner writes stories, tasks and bugs, so show those. Epics are never sized here and read like
# outcome statements; subtasks are a smaller granularity than a plan produces.
_EXCLUDED_ISSUE_TYPES = frozenset({"epic", "subtask"})

# Match only on issue bodies: comments/attachments/sprints share incidental words ("carried into
# Sprint 7") and can make an unrelated ticket look like the closest match.
_MATCH_EXCLUDED_CHUNK_TYPES = frozenset({"attachment", "sprint", "comment"})

# Max past keys one estimate may cite — a list of six is a shrug, not a comparison.
MAX_ESTIMATE_REFERENCES = 3

GroundingStatus = Literal["grounded", "style_only", "no_history", "retrieval_failed", "disabled"]
ExemplarSource = Literal["similar", "recent"]


@dataclass(slots=True)
class Exemplar:
    """One past issue from this space, as the planner will see it."""

    issue_key: str
    title: str
    issue_type: Optional[str]
    status: Optional[str]
    story_points: Optional[int]
    description: str
    #: Normalized 0-1 relevance (raw scores depend on whether the server reranked).
    #: 0.0 for a "recent" exemplar, which was never scored.
    confidence: float
    #: "similar" = passed the relevance floor. "recent" = backfilled from latest finished work.
    source: ExemplarSource = "similar"
    #: Only when `include_comments`; read by pattern extraction, never shown to the planner.
    comments: List[str] = field(default_factory=list)

    @property
    def is_estimate_anchor(self) -> bool:
        """Finished and sized — unfinished points are a guess, not evidence."""
        return self.story_points is not None and (self.status or "").lower() in _DONE_STATUSES


@dataclass(slots=True)
class RequirementEvidence:
    """One stated requirement and the past work found for it specifically.

    `index` is 1-based because the model sees it and cites it back.
    """

    index: int
    text: str
    exemplars: List[Exemplar] = field(default_factory=list)

    @property
    def anchors(self) -> List[Exemplar]:
        return [exemplar for exemplar in self.exemplars if exemplar.is_estimate_anchor]


@dataclass(slots=True)
class GroundingContext:
    """What was retrieved for one planning call, plus a status saying what it amounts to.

    "Grounded in your history" and "nothing to ground in" must never look the same to the caller.
    """

    status: GroundingStatus = "disabled"
    exemplars: List[Exemplar] = field(default_factory=list)
    latency_seconds: float = 0.0
    error: Optional[str] = None
    #: Per-requirement evidence; empty on the whole-proposal path.
    requirements: List[RequirementEvidence] = field(default_factory=list)
    #: The board's description layout. Kept even if pattern extraction rejects the history as
    #: irrelevant — formatting conventions apply to any epic. See `house_format.py`.
    house_format: HouseFormat = field(default_factory=HouseFormat)

    @property
    def anchors(self) -> List[Exemplar]:
        return [e for e in self.exemplars if e.is_estimate_anchor]

    @property
    def estimates_grounded(self) -> bool:
        return bool(self.anchors)

    def valid_keys(self) -> set:
        return {e.issue_key.upper() for e in self.exemplars}


def with_stated_requirements(context: GroundingContext, requirements: Sequence[str]) -> GroundingContext:
    """Keep user scope even when history is unavailable; lack of evidence never removes a requirement."""
    if context.requirements:
        return context
    stated = [text.strip() for text in requirements if text and text.strip()]
    return replace(context, requirements=[
        RequirementEvidence(index=index, text=text) for index, text in enumerate(stated, start=1)
    ])


def _rank(exemplars: List[Exemplar]) -> List[Exemplar]:
    """Anchors first, then "similar" before "recent"; stable, so retrieval order is kept within groups.

    Matters because the list is cut to `limit`: otherwise a board whose closest matches are unfinished
    could end up with zero anchors while finished work sat just below the cut.
    """
    return sorted(exemplars, key=lambda e: (not e.is_estimate_anchor, e.source != "similar"))


def _usable(row: dict, body: str) -> bool:
    """Needs a body (a bare title teaches nothing) and an allowed issue type."""
    return bool(body) and (row.get("issue_type") or "").lower() not in _EXCLUDED_ISSUE_TYPES


async def gather_exemplars(
    retrieval,
    space_ids: Sequence[int],
    query_text: str,
    *,
    limit: int,
    exclude_issue_keys: Optional[Set[str]] = None,
    include_comments: bool = False,
    backfill: bool = True,
    min_confidence: float = _MIN_CONFIDENCE,
) -> GroundingContext:
    """Retrieve up to `limit` past issues from `space_ids` that resemble `query_text`.

    `backfill=False` is for per-requirement search. With backfill on, every requirement got the same
    three "recent" cards (anchors rank first), which pushed out its own matches, and the planner
    cited none of them.

    `space_ids` is the caller's authorized scope from the route, never from the model.

    `exclude_issue_keys` is only for the eval: it hides the real epic's own children so the eval
    measures grounding, not copying. Empty in production.
    """
    start = time.perf_counter()
    query = plain_text(query_text, _MAX_QUERY_CHARS)
    if not query:
        return _recorded(GroundingContext(status="no_history", latency_seconds=time.perf_counter() - start))

    try:
        # Vector mode: a proposal shares words with unrelated tickets more often than meaning.
        search = await retrieval.search_with_meta(query, space_ids, limit=_RAW_FETCH, mode="vector")
        above_floor = [
            hit for hit in search.hits
            if hit.chunk_type not in _MATCH_EXCLUDED_CHUNK_TYPES
            and relevance_confidence(hit.score, search.reranked) >= min_confidence
        ]
        # Filtering already happened above on the normalized scale; dedupe only collapses to one chunk
        # per issue, so its own score filter is turned off.
        hits = dedupe_by_issue(
            above_floor, _RAW_FETCH,
            exclude_chunk_types=_MATCH_EXCLUDED_CHUNK_TYPES, min_score=float("-inf"),
        )
        # No early return when nothing is similar: novel work is exactly what the backfill is for.

        # Fetch metadata for more keys than `limit` so `_rank` can promote anchors from below the cut.
        excluded = {key.upper() for key in (exclude_issue_keys or set())}
        similar_keys = [hit.issue_key for hit in hits if hit.issue_key.upper() not in excluded]
        confidence_by_key = {
            hit.issue_key: relevance_confidence(hit.score, search.reranked) for hit in hits
        }
        # Similar-issue metadata and the recency backfill don't depend on each other, so run them
        # together.
        recent_query = None if not backfill else retrieval.query_issues(space_ids, {
            "statuses": sorted(_DONE_STATUSES),
            "issue_types": ["story", "task", "bug"],
            "order_by": "updated_at", "order": "desc",
            # Extra rows: unsized and body-less ones are filtered out below.
            "limit": min(limit * 4, 200),
        })
        if similar_keys and recent_query is not None:
            similar_rows, recent_rows = await asyncio.gather(
                retrieval.query_issues(
                    space_ids, {"issue_keys": similar_keys, "limit": len(similar_keys)},
                ),
                recent_query,
            )
            similar_issues = similar_rows.issues
        elif similar_keys:
            similar_rows = await retrieval.query_issues(
                space_ids, {"issue_keys": similar_keys, "limit": len(similar_keys)},
            )
            similar_issues = similar_rows.issues
            recent_rows = None
        elif recent_query is None:
            return _recorded(GroundingContext(
                status="no_history", latency_seconds=time.perf_counter() - start,
            ))
        else:
            # Don't query `{"issue_keys": []}`: an empty list means "no filter" there and returns
            # arbitrary rows, which would then wrongly block real backfill rows below.
            recent_rows = await recent_query
            similar_issues = []
        by_key = {str(row.get("issue_key")): row for row in similar_issues}
        ordered_keys: List[str] = [key for key in similar_keys if key in by_key]
        source_by_key: Dict[str, str] = {key: "similar" for key in ordered_keys}
        # Each key here costs a body fetch next, and limit * 2 already covers the worst case.
        backfill_budget = limit * 2
        for row in (recent_rows.issues if recent_rows is not None else []):
            if backfill_budget <= 0:
                break
            key = str(row.get("issue_key"))
            if key.upper() in excluded or key in by_key or row.get("story_points") is None:
                continue
            by_key[key] = row
            ordered_keys.append(key)
            source_by_key[key] = "recent"
            backfill_budget -= 1
        if not ordered_keys:
            return _recorded(GroundingContext(
                status="no_history", latency_seconds=time.perf_counter() - start,
            ))
        # Also fetch bodies of everything the recency query returned, only to detect the board's
        # layout. 8 samples read a board that is 49% one / 37% two / 14% three criteria as "usually 1",
        # and the planner then wrote one criterion per card. Same call, just more keys.
        format_keys = list(dict.fromkeys(
            ordered_keys + [
                str(row.get("issue_key")) for row in (recent_rows.issues if recent_rows else [])
            ]
        ))[:_FORMAT_SAMPLE]
        # No limit=len(...): that limit counts chunk rows, and a two-chunk body would get cut off.
        details = await retrieval.get_issue_details(space_ids, format_keys)
        # Only when asked: comments feed pattern extraction, not the planning prompt.
        comment_rows = (
            (await retrieval.get_issue_comments(space_ids, ordered_keys)).comments
            if include_comments else []
        )
    except Exception as exc:  # noqa: BLE001 - grounding is an enhancement; losing it must not fail the plan
        logger.warning("planning grounding retrieval failed; planning ungrounded", exc_info=True)
        return _recorded(GroundingContext(
            status="retrieval_failed", latency_seconds=time.perf_counter() - start, error=str(exc),
        ))

    comments_by_key: Dict[str, List[str]] = {}
    for comment in comment_rows:
        bucket = comments_by_key.setdefault(str(comment.get("issue_key")), [])
        if len(bucket) >= _MAX_COMMENTS_PER_ISSUE:
            continue
        text = plain_text(comment.get("content"), _MAX_COMMENT_CHARS)
        if text:
            bucket.append(text)

    body_by_key: Dict[str, List[str]] = {}
    for detail in details.details:
        # Usually one body chunk per issue; joined in case a long one was split.
        body_by_key.setdefault(str(detail.get("issue_key")), []).append(detail.get("content") or "")

    exemplars: List[Exemplar] = []
    for key in ordered_keys:
        body = plain_text("\n".join(body_by_key.get(key, [])), _MAX_DESC_CHARS)
        row = by_key.get(key) or {}
        if not _usable(row, body):
            continue
        exemplars.append(Exemplar(
            issue_key=key,
            title=str(row.get("title") or ""),
            issue_type=row.get("issue_type"),
            status=row.get("status"),
            story_points=row.get("story_points"),
            description=body,
            # "recent" exemplars were never scored, so 0.0 rather than a made-up number.
            confidence=(
                confidence_by_key.get(key, 0.0) if source_by_key[key] == "similar" else 0.0
            ),
            source=source_by_key[key],
            comments=comments_by_key.get(key, []),
        ))

    exemplars = _rank(exemplars)[:limit]
    # Layout comes from every retrieved body, not just the survivors: more samples, better counts.
    house = detect_house_format([
        plain_text("\n".join(body_by_key.get(key, [])), 4000) for key in format_keys
    ])
    if not exemplars:
        status: GroundingStatus = "no_history"
    elif any(e.is_estimate_anchor for e in exemplars):
        status = "grounded"
    else:
        status = "style_only"
    return _recorded(GroundingContext(
        status=status, exemplars=exemplars, latency_seconds=time.perf_counter() - start,
        house_format=house,
    ))


async def gather_requirement_exemplars(
    retrieval,
    space_ids: Sequence[int],
    proposal: str,
    requirements: Sequence[str],
    *,
    limit: int,
    per_requirement: int = 3,
    exclude_issue_keys: Optional[Set[str]] = None,
    include_comments: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
) -> GroundingContext:
    """One search per requirement, plus the whole-proposal search for style and layout.

    The requirement searches find comparables for sizing; the whole-proposal search still shows the
    board's general habits, so its measured behaviour is unchanged. `on_progress` lets the UI show
    which requirement is being searched (this part is otherwise slow and silent). `per_requirement`
    stays small so the prompt holds a few comparables, not a corpus.
    """
    start = time.perf_counter()
    stated = [text.strip() for text in requirements if text and text.strip()]
    base = await gather_exemplars(
        retrieval, space_ids, proposal, limit=limit,
        exclude_issue_keys=exclude_issue_keys, include_comments=include_comments,
    )
    if not stated or base.status in ("retrieval_failed",):
        # No requirements, or the index is down: the whole-proposal result already says which.
        return with_stated_requirements(base, stated)

    # One at a time: ten concurrent embedding calls once stalled a single-instance embedding server
    #. Hosted embeddings could take these concurrently.
    evidence: List[RequirementEvidence] = []
    for index, text in enumerate(stated, start=1):
        if on_progress:
            on_progress(f"looking for past work like requirement {index} of {len(stated)}")
        found = await gather_exemplars(
            retrieval, space_ids, text, limit=per_requirement,
            exclude_issue_keys=exclude_issue_keys, include_comments=include_comments,
            backfill=False, min_confidence=_MIN_REQUIREMENT_CONFIDENCE,
        )
        evidence.append(RequirementEvidence(index=index, text=text, exemplars=found.exemplars))

    # The flat list is the union, so citation checks, the estimate baseline and layout detection keep
    # working without knowing about requirements.
    merged = {exemplar.issue_key: exemplar for exemplar in base.exemplars}
    for item in evidence:
        for exemplar in item.exemplars:
            merged.setdefault(exemplar.issue_key, exemplar)
    return GroundingContext(
        status=base.status if base.exemplars else (
            "grounded" if any(item.anchors for item in evidence)
            else ("style_only" if any(item.exemplars for item in evidence) else base.status)
        ),
        exemplars=list(merged.values()),
        requirements=evidence,
        latency_seconds=time.perf_counter() - start,
        error=base.error,
        house_format=base.house_format,
    )


def _recorded(context: GroundingContext) -> GroundingContext:
    """Record metrics for every outcome. "disabled" is never counted: the code simply didn't run."""
    PLANNING_GROUNDING_TOTAL.labels(context.status).inc()
    PLANNING_GROUNDING_EXEMPLARS.observe(len(context.exemplars))
    return context


def apply_estimate_grounding(
    issues: List,
    context: Optional[GroundingContext],
    already_verified: Optional[Set[str]] = None,
) -> tuple:
    """Drop every `estimate_reference` the model didn't earn, and return how many were dropped.

    Dropped when: the key wasn't an exemplar (invented, or copied from prose), the issue has no
    estimate, it's a duplicate, or there are more than `MAX_ESTIMATE_REFERENCES`.

    `already_verified` (from `refine_plan`) keeps citations that were checked in an earlier call, so a
    refine round with slightly different retrieval doesn't delete them.

    Only the citation is dropped, never the estimate: an unanchored estimate is still useful, and the
    reviewer just sees no "compared to" badge.
    """
    valid = (context.valid_keys() if context else set()) | (already_verified or set())
    cleaned = []
    dropped = 0
    for issue in issues:
        references = list(getattr(issue, "estimate_reference", []) or [])
        if issue.estimate_story_points is None:
            kept: List[str] = []
        else:
            kept = []
            for reference in references:
                if reference.upper() in valid and reference not in kept:
                    kept.append(reference)
                if len(kept) >= MAX_ESTIMATE_REFERENCES:
                    break
        dropped += len(references) - len(kept)
        cleaned.append(issue if kept == references else issue.model_copy(update={"estimate_reference": kept}))
    if dropped:
        logger.info("dropped %d unverifiable estimate reference(s) from the plan", dropped)
    return cleaned, dropped
