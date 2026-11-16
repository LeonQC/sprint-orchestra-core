"""Structured output schema for AI task drafting (instructor validates against it).

`labels` and `dependencies` default to empty, so nothing is invented. `estimate_story_points` may be
null, and `estimate_rationale` goes with it (null exactly when the estimate is), so an estimate can be
questioned. An earlier "never guess a number" wording made the model return null for everything;
asking for the number and putting the caveats in the rationale fixed that, together with the
StructuredOutput base (without it the field could be skipped).
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field, model_validator

from app.llm.structured_output import StructuredOutput


class TaskDraft(StructuredOutput):
    title: str = Field(
        ..., min_length=1, max_length=120,
        description="Concise, actionable issue title in imperative mood (e.g. 'Add rate limiting to "
                    "login endpoint', not 'We should probably add some rate limiting maybe').",
    )
    issue_type: Literal["bug", "task", "story"] = Field(
        "task", description="Best guess at the kind of work this is."
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Relevant labels, lowercase, only if clearly implied by the description — an "
                    "empty list is correct when nothing is clearly implied, not a failure to fill it in.",
    )
    estimate_story_points: Optional[int] = Field(
        None, ge=1, le=13,
        description="A rough Fibonacci-like estimate (1, 2, 3, 5, 8, 13). Give one whenever the "
                    "description names concrete work, stating your assumptions in "
                    "estimate_rationale rather than withholding the number. Null only when the "
                    "description is too vague to tell what would be built at all.",
    )
    estimate_rationale: Optional[str] = Field(
        None, max_length=280,
        description="One short sentence explaining what makes this estimate that size (scope, "
                    "unknowns, dependencies, etc). Null if and only if estimate_story_points is "
                    "null.",
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="Other work this explicitly depends on (issue keys or short phrases), only if "
                    "the description actually mentions something blocking — never invented.",
    )

    @model_validator(mode="after")
    def drop_rationale_without_an_estimate(self) -> "TaskDraft":
        """Keep "a rationale only with a number". Fixed up rather than raised, so a mismatch doesn't
        cost an instructor retry.
        """
        rationale = (self.estimate_rationale or "").strip()
        self.estimate_rationale = rationale if rationale and self.estimate_story_points is not None else None
        return self
