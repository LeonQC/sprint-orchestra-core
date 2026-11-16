"""Judge the retrieved history and extract reusable patterns, before planning from it.

Similar past issues (a customer import, a CSV upload, a batch account creation) are different
features; what's worth reusing is the shared shape — define schema, validate input, process
records, handle partial failures, track status. This module reads the retrieved issues once and
returns (a) whether the history is worth planning from and (b) the patterns that recur in it.

One LLM call, since judging relevance and extracting patterns are the same reading task.

`strength` is reported by the model (relevance is a semantic question a threshold gets wrong), but
`verify_assessment` checks it: drops citations never shown, drops patterns with too little evidence,
and downgrades `strength` if what survives no longer supports it.

Measured: the deterministic estimate (`derive_estimate`) didn't beat the model on accuracy (3 of 8)
but cut run-to-run noise by 40% (sd 0.123 -> 0.074). Kept for stability and auditability.

Any failure falls back to planning from the proposal alone — never a 500, never a fake history claim.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.llm.circuit_breaker import ModelUnavailableError, is_model_unavailable
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.planning.grounding import Exemplar, GroundingContext
from app.planning.pattern_schemas import (
    MIN_PATTERN_EVIDENCE,
    EvidenceAssessment,
    EvidenceStrength,
    HistoricalPattern,
)
from app.planning.prompts import render_pattern_extraction_prompt
from app.tracing import llm_step

logger = logging.getLogger(__name__)

_MAX_VALIDATION_RETRIES = 2
# Smaller than the planner's 8000: this stage emits a handful of short patterns and a sentence of
# reasoning, not a whole decomposition. Same budget `app/sprint_pace/service.py` uses.
_MAX_OUTPUT_TOKENS = 2000

# Fibonacci story-point values for `derive_estimate`'s one-step moves (shared with service.py).
_LADDER = (1, 2, 3, 5, 8, 13)


@dataclass(slots=True)
class AssessmentResult:
    assessment: EvidenceAssessment
    degraded: bool = False
    error: Optional[str] = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    #: Patterns removed by `verify_assessment` — cited an unknown key, or had too little support.
    dropped_patterns: int = 0
    #: True when verification had to lower the model's own `strength` label.
    downgraded: bool = False


def empty_assessment(reasoning: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        relevant_issue_keys=[], patterns=[], strength="none", reasoning=reasoning,
    )


def render_history_for_extraction(exemplars: Sequence[Exemplar]) -> str:
    """The past issues as the extraction call sees them. Rendered in code because only this prompt
    includes comments, which need their own per-issue framing.
    """
    lines: List[str] = []
    for exemplar in exemplars:
        points = exemplar.story_points if exemplar.story_points is not None else "unestimated"
        lines.append("")
        lines.append(f"--- {exemplar.issue_key} ---")
        lines.append(f"  type: {exemplar.issue_type or 'unknown'} | status: "
                     f"{exemplar.status or 'unknown'} | points: {points}")
        lines.append(f"  title: {exemplar.title}")
        lines.append(f"  description: {exemplar.description}")
        for comment in exemplar.comments:
            # Labelled as said-during-the-work, because that is what makes a comment worth reading
            # here: it reports what actually happened, where the description reports what was planned.
            lines.append(f"  comment (said while doing the work): {comment}")
    return "\n".join(lines)


def verify_assessment(
    assessment: EvidenceAssessment, context: GroundingContext
) -> tuple[EvidenceAssessment, int, bool]:
    """Keep only what the model can support, and lower `strength` if it's now too strong.

    1. Drop evidence keys for issues that weren't shown (invented or copied from prose).
    2. Drop patterns with fewer than `MIN_PATTERN_EVIDENCE` keys — a pattern must recur; one issue
       (maybe one truncated thread) can't support it.
    3. Lower `strength` to match: nothing left means "none"; no reusable shape can't be "strong".

    Returns the cleaned assessment, the number of dropped patterns, and whether strength was lowered.
    """
    shown = {exemplar.issue_key.upper() for exemplar in context.exemplars}
    relevant = [key for key in assessment.relevant_issue_keys if key.upper() in shown]

    kept: List[HistoricalPattern] = []
    dropped = 0
    for pattern in assessment.patterns:
        evidence = []
        for key in pattern.evidence_issue_keys:
            if key.upper() in shown and key not in evidence:
                evidence.append(key)
        if len(evidence) < MIN_PATTERN_EVIDENCE:
            dropped += 1
            logger.info(
                "dropping pattern %r: %d verifiable supporting issue(s), need %d",
                pattern.name, len(evidence), MIN_PATTERN_EVIDENCE,
            )
            continue
        kept.append(pattern.model_copy(update={"evidence_issue_keys": evidence}))

    strength: EvidenceStrength = assessment.strength
    if not kept and not relevant:
        strength = "none"
    elif not kept and strength == "strong":
        # Relevant issues but no surviving pattern is the definition of partial: pieces of this epic
        # resemble past work, and nothing shows a whole reusable shape.
        strength = "partial"
    downgraded = strength != assessment.strength
    if downgraded:
        logger.info(
            "downgrading evidence strength %s -> %s after verification",
            assessment.strength, strength,
        )
    return (
        assessment.model_copy(update={
            "relevant_issue_keys": relevant, "patterns": kept, "strength": strength,
        }),
        dropped,
        downgraded,
    )


async def assess_history(
    client, model: str, proposal: str, context: Optional[GroundingContext]
) -> AssessmentResult:
    """Read the retrieved history once: is it useful, and what does it teach? With nothing to read,
    returns "none" without calling the model.
    """
    start = time.perf_counter()
    if context is None or not context.exemplars:
        return AssessmentResult(
            assessment=empty_assessment("No past issues were retrieved for this proposal."),
            latency_seconds=time.perf_counter() - start,
        )

    user_message = (
        f"NEW EPIC PROPOSAL:\n{proposal}\n\n"
        f"PAST ISSUES FROM THIS BOARD ({len(context.exemplars)}):\n"
        f"{render_history_for_extraction(context.exemplars)}\n\n--- end of past issues ---"
    )
    try:
        with llm_step("plan_epic.assess_history"):
            assessment, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=EvidenceAssessment,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": render_pattern_extraction_prompt()},
                    {"role": "user", "content": user_message},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a failed reading of history must not fail the plan
        if is_model_unavailable(exc):
            # The planning call right after this would fail the same way.
            raise ModelUnavailableError() from exc
        logger.warning("historical pattern extraction failed; planning without patterns", exc_info=True)
        return AssessmentResult(
            assessment=empty_assessment(
                "Historical patterns could not be extracted; the plan was written from the "
                "proposal alone."
            ),
            degraded=True, error=str(exc), latency_seconds=time.perf_counter() - start,
        )

    cleaned, dropped, downgraded = verify_assessment(assessment, context)
    usage = usage_from_completion(completion)
    return AssessmentResult(
        assessment=cleaned,
        latency_seconds=time.perf_counter() - start,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
        dropped_patterns=dropped,
        downgraded=downgraded,
    )


def historical_baseline(reference_keys: Sequence[str], context: Optional[GroundingContext]) -> Optional[int]:
    """What the cited work actually cost: the median story points of the cited issues.

    Median, not mean: points are ordinal (the mean of 3 and 8 is 5.5, not a real size). With two
    citations it takes the lower one, the safe direction for a planner measured to over-estimate.
    Only finished, sized issues count.
    """
    if not context or not reference_keys:
        return None
    by_key = {
        exemplar.issue_key.upper(): exemplar
        for exemplar in context.exemplars
        if exemplar.is_estimate_anchor
    }
    points = sorted(
        by_key[key.upper()].story_points for key in reference_keys if key.upper() in by_key
    )
    if not points:
        return None
    return points[(len(points) - 1) // 2]


def derive_estimate(baseline: Optional[int], complexity: Optional[str]) -> Optional[int]:
    """Move one step on the Fibonacci ladder from what the comparable work cost.

    Code owns the number (from real history); the model only says smaller / same / larger. Never more
    than one step — a three-way judgement can't justify a bigger jump; for that, the planner's own
    estimate stays.
    """
    if baseline is None or baseline not in _LADDER:
        return None
    index = _LADDER.index(baseline)
    if complexity == "simpler":
        return _LADDER[max(0, index - 1)]
    if complexity == "more_complex":
        return _LADDER[min(len(_LADDER) - 1, index + 1)]
    return baseline


def estimate_confidence(
    reference_keys: Sequence[str], baseline: Optional[int], strength: EvidenceStrength,
) -> str:
    """How much this estimate rests on history, computed (not claimed). Per issue, because within one
    plan some issues have a real comparable and some don't.
    """
    if baseline is None or not reference_keys:
        return "low"
    if strength == "strong" and len(reference_keys) >= 2:
        return "high"
    return "medium"
