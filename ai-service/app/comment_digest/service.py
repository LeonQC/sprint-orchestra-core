"""Comment-thread digest: one structured call -> context / discussion / outcome, plus open questions and
action items, each citing the comment(s) it came from.

On model failure it returns an empty digest with `degraded=True` (the UI still shows the raw
comments), never a 500. `_ground` drops any point whose comment ids aren't in the thread that was sent.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, TypeVar

from app.comment_digest.prompts import render_comment_digest_system_prompt, render_comment_thread_message
from app.comment_digest.schemas import AttributedPoint, CommentDigest, CommentIn, Outcome
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.tracing import llm_step

logger = logging.getLogger(__name__)

_MAX_VALIDATION_RETRIES = 2
_MAX_OUTPUT_TOKENS = 2000
# Below this, a thread is just too short to have anything worth extracting — skip the model call
# entirely rather than pay for (and wait on) a summary of two sentences.
_MIN_COMMENTS = 2
# Above this, keep only the most recent comments rather than silently truncating the model's context
# window from underneath it. Chosen well above any real AtlasCart thread observed so far; the
# response is honest about it either way via `truncated_count`.
_MAX_COMMENTS = 60

_P = TypeVar("_P", bound=AttributedPoint)


@dataclass(slots=True)
class CommentDigestResult:
    digest: CommentDigest
    degraded: bool
    skipped: bool  # too few comments to be worth summarizing — not a failure
    truncated_count: int
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None


def _empty_digest() -> CommentDigest:
    return CommentDigest(
        context="", discussion="", outcome=Outcome(text="", comment_ids=[]),
        open_questions=[], action_items=[],
    )


def _filter_grounded(points: List[_P], valid_ids: set[int]) -> List[_P]:
    kept: List[_P] = []
    for point in points:
        real_ids = [i for i in point.comment_ids if i in valid_ids]
        if not real_ids:
            logger.warning(
                "comment digest point dropped: no comment_ids resolved against the real thread (%r)",
                point.text[:60],
            )
            continue
        kept.append(point.model_copy(update={"comment_ids": real_ids}))
    return kept


def _ground(digest: CommentDigest, valid_ids: set[int]) -> CommentDigest:
    # `outcome` is required, so unlike list items it's kept with its bad ids removed instead of dropped.
    real_outcome_ids = [i for i in digest.outcome.comment_ids if i in valid_ids]
    return CommentDigest(
        context=digest.context,
        discussion=digest.discussion,
        outcome=Outcome(text=digest.outcome.text, comment_ids=real_outcome_ids),
        open_questions=_filter_grounded(digest.open_questions, valid_ids),
        action_items=_filter_grounded(digest.action_items, valid_ids),
    )


async def summarize_comments(client, model: str, comments: List[CommentIn]) -> CommentDigestResult:
    if len(comments) < _MIN_COMMENTS:
        return CommentDigestResult(
            digest=_empty_digest(), degraded=False, skipped=True, truncated_count=0, latency_seconds=0.0,
        )

    truncated_count = max(0, len(comments) - _MAX_COMMENTS)
    used_comments = comments[-_MAX_COMMENTS:] if truncated_count else comments
    valid_ids = {c.id for c in used_comments}

    system_prompt = render_comment_digest_system_prompt()
    user_message = render_comment_thread_message(used_comments, truncated_count)
    start = time.perf_counter()
    try:
        with llm_step("summarize_comments.summarize"):
            digest, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=CommentDigest,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a digest failure must degrade, never 500
        logger.warning("comment digest generation failed; degrading to an empty digest", exc_info=True)
        return CommentDigestResult(
            digest=_empty_digest(), degraded=True, skipped=False, truncated_count=truncated_count,
            latency_seconds=time.perf_counter() - start, error=str(exc),
        )

    usage = usage_from_completion(completion)
    return CommentDigestResult(
        digest=_ground(digest, valid_ids),
        degraded=False,
        skipped=False,
        truncated_count=truncated_count,
        latency_seconds=time.perf_counter() - start,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
    )
