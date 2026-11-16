"""Description review: one structured call returns copy-edit findings, applied in code.

The model never writes the new description. Each finding's `quote` is checked against the real
description and replaced here, so the result is computed from the original (see schemas.py); a quote
that isn't found is dropped.

Replacements only touch text outside HTML tags, so tag attributes (a quote "type" hitting
`type="button"`) and attachment chips can't be damaged. A model failure returns the description
unchanged with `degraded=True`. Nothing is saved; it only proposes.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.description_review.prompts import (
    render_description_message,
    render_description_review_system_prompt,
    strip_description_html,
)
from app.description_review.schemas import DescriptionReview, Finding
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.tracing import llm_step

logger = logging.getLogger(__name__)

_MAX_VALIDATION_RETRIES = 2
_MAX_OUTPUT_TOKENS = 2000
# Below this there isn't enough prose to copy-edit, and the model starts reaching for nitpicks to
# justify the call. Measured in stripped text, not raw HTML, so markup doesn't inflate it.
_MIN_TEXT_CHARS = 40
# A description far larger than any real one is a cost and latency risk, not a correctness one — the
# review still works on the leading portion, and `truncated` tells the caller the tail wasn't read.
_MAX_TEXT_CHARS = 12000

# re.split with one capturing group yields [text, tag, text, tag, ..., text]: even indices are the
# text segments, odd indices are the tags themselves.
_TAG_SPLIT_RE = re.compile(r"(<[^>]+>)")


@dataclass(slots=True)
class DescriptionReviewResult:
    review: DescriptionReview
    #: Original description with every applied finding's replacement made. None when nothing was
    #: applied — there is no "improved" version to offer, which is the expected case.
    improved_description: Optional[str]
    applied_findings: List[Finding] = field(default_factory=list)
    #: Reported by the model but not applied, because the quote didn't match the real description.
    dropped_findings: List[Finding] = field(default_factory=list)
    skipped: bool = False  # too little prose to review — not a failure
    truncated: bool = False
    degraded: bool = False
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None


def _empty_review() -> DescriptionReview:
    return DescriptionReview(verdict="fine_as_is", findings=[])


def _replace_outside_tags(html: str, quote: str, replacement: str) -> tuple[str, bool]:
    """Replace the first occurrence of `quote` that lies in a text segment, never inside a tag."""
    parts = _TAG_SPLIT_RE.split(html)
    for i in range(0, len(parts), 2):
        if quote in parts[i]:
            parts[i] = parts[i].replace(quote, replacement, 1)
            return "".join(parts), True
    return html, False


def _apply(description: str, findings: List[Finding]) -> tuple[str, List[Finding], List[Finding]]:
    """Apply findings in order, keeping only ones whose quote is found. Each is matched against the
    already-edited text, so one overlapping an earlier edit just fails to match and is dropped.
    """
    working = description
    applied: List[Finding] = []
    dropped: List[Finding] = []
    for finding in findings:
        if finding.quote == finding.suggestion:
            # A no-op "change" is noise in the reviewer's list, not an edit.
            dropped.append(finding)
            continue
        working, ok = _replace_outside_tags(working, finding.quote, finding.suggestion)
        if ok:
            applied.append(finding)
        else:
            logger.info(
                "description review finding dropped: quote not found in the description text (%r)",
                finding.quote[:60],
            )
            dropped.append(finding)
    return working, applied, dropped


async def review_description(
    client, model: str, description: str, title: Optional[str] = None
) -> DescriptionReviewResult:
    text = strip_description_html(description)
    if len(text) < _MIN_TEXT_CHARS:
        return DescriptionReviewResult(review=_empty_review(), improved_description=None, skipped=True)

    truncated = len(text) > _MAX_TEXT_CHARS
    system_prompt = render_description_review_system_prompt()
    user_message = render_description_message(text[:_MAX_TEXT_CHARS], title)
    start = time.perf_counter()
    try:
        with llm_step("improve_description.review"):
            review, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=DescriptionReview,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a review failure must degrade, never 500
        logger.warning("description review failed; returning the description unchanged", exc_info=True)
        return DescriptionReviewResult(
            review=_empty_review(), improved_description=None, truncated=truncated, degraded=True,
            latency_seconds=time.perf_counter() - start, error=str(exc),
        )

    improved, applied, dropped = _apply(description, review.findings)
    usage = usage_from_completion(completion)
    return DescriptionReviewResult(
        # Normalized, not taken on trust: a model that says "needs_changes" but whose findings all
        # failed verification has, in effect, found nothing actionable — reporting that as "needs
        # changes" would leave the UI claiming a problem it cannot show.
        review=DescriptionReview(
            verdict="needs_changes" if applied else "fine_as_is",
            findings=applied,
        ),
        improved_description=improved if applied else None,
        applied_findings=applied,
        dropped_findings=dropped,
        truncated=truncated,
        latency_seconds=time.perf_counter() - start,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
    )
