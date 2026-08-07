"""Structured output for the history-assessment / pattern-extraction step of epic planning.

Separate from schemas.py: that describes the plan; this describes what was learned from history
first, and can be tested on its own. Uses StructuredOutput so every field is required on the wire.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from app.llm.structured_output import StructuredOutput

EvidenceStrength = Literal["strong", "partial", "none"]

# One issue isn't a pattern; two is the minimum that shows something recurs. Also keeps a single
# truncated comment thread from inventing a lesson.
MIN_PATTERN_EVIDENCE = 2


class HistoricalPattern(StructuredOutput):
    """One reusable way this team has done a kind of work before.

    The point of the whole stage: "bulk customer import", "CSV product upload" and "batch account
    creation" are three different features, and the useful thing is not any of them — it is that all
    three needed schema definition, validation, partial-failure handling and status tracking. That
    shared shape transfers to a new bulk-import epic in a way the individual issues do not.
    """

    name: str = Field(
        ..., min_length=1, max_length=80,
        description="Short name for the recurring shape, e.g. 'bulk import with partial failure "
                    "handling'. Name the pattern, never one of the issues it came from.",
    )
    steps: List[str] = Field(
        default_factory=list,
        description="The recurring pieces of work, in the order this team has actually done them. "
                    "Each one a short phrase ('validate the input file before processing'), not a "
                    "restatement of a past issue's title.",
    )
    evidence_issue_keys: List[str] = Field(
        default_factory=list,
        description=f"Past issue keys that support this pattern — at least {MIN_PATTERN_EVIDENCE}, "
                    "and only keys from the historical issues you were shown. A shape you can only "
                    "point at one issue for is not a pattern and must not be reported.",
    )
    why_relevant: str = Field(
        ..., min_length=1,
        description="One sentence on what this pattern implies for the NEW epic specifically. If you "
                    "cannot say what it implies here, leave the pattern out.",
    )
    pitfalls: List[str] = Field(
        default_factory=list,
        description="Concrete trouble this team actually hit doing this kind of work before — a "
                    "retry nobody planned for, a validation discovered late, an integration that "
                    "broke. Usually found in comments rather than descriptions. Empty is correct "
                    "when the history shows none; never invent a plausible-sounding risk.",
    )


class EvidenceAssessment(StructuredOutput):
    """Whether the retrieved history is worth planning from, and what it teaches.

    `strength` is reported by the model rather than computed here, unlike this codebase's usual
    "LLM lists evidence, code computes the verdict" split (`PlanCritique`/`plan_needs_revision`).
    The reason is that the judgement is genuinely semantic — "is a CSV product upload relevant to
    employee onboarding" is not a function of counts or scores, which is exactly the thing a single
    similarity threshold gets wrong. What code still does rather than trust: verify every cited key
    was really shown, drop patterns below `MIN_PATTERN_EVIDENCE`, and downgrade `strength` when the
    surviving evidence does not support it (see `app.planning.patterns.verify_assessment`), so an
    over-confident label cannot survive contact with what is actually left.
    """

    relevant_issue_keys: List[str] = Field(
        default_factory=list,
        description="Of the issues you were shown, the ones that genuinely inform planning this "
                    "epic. Vector search returns nearest neighbours whether or not anything is "
                    "close, so an empty list is a normal and correct answer.",
    )
    patterns: List[HistoricalPattern] = Field(
        default_factory=list,
        description="Reusable shapes found across the relevant issues. Empty when the history shows "
                    "no shape worth reusing.",
    )
    strength: EvidenceStrength = Field(
        "none",
        description="'strong' = several relevant issues showing a reusable pattern that covers much "
                    "of this epic. 'partial' = no equivalent feature, but parts of this epic "
                    "resemble past work (validation, async processing, status tracking) and those "
                    "sub-patterns transfer. 'none' = nothing retrieved genuinely informs this epic. "
                    "'none' is an expected outcome, not a failure — say it rather than stretching.",
    )
    reasoning: str = Field(
        ..., min_length=1,
        description="One or two sentences on why the history does or does not apply, in terms a "
                    "reviewer can check against the issues listed.",
    )
