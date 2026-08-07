"""Structured output schema for AI epic planning.

One LLM call produces the epic and its issues together (issues come from the epic's goals in the same
pass). Tested: splitting estimation into its own node gave no improvement (delta 0.00), so it was
removed.

`depends_on` uses temp_ids from this response, never real issue keys — nothing exists in Jira yet.

Dependencies are suggestions, not things the human must justify: an unexplained edge is dropped
(`service.py::drop_unexplained_dependencies`), and only "blocks" edges become real Jira links
(`rollout_graph.py::_dependency_pairs`); "after" just affects ordering and sprint packing.

All models extend StructuredOutput so every field is required on the wire (app/llm/structured_output.py)
— otherwise an IssueDraft could come back without points, labels or dependencies.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from app.llm.structured_output import StructuredOutput

DependencyKind = Literal["blocks", "after"]
DEFAULT_DEPENDENCY_KIND: DependencyKind = "after"


class EpicDraft(StructuredOutput):
    title: str = Field(..., min_length=1, max_length=120)
    description: str = Field(
        ..., description="A few sentences summarizing the epic's scope and intent."
    )
    goals: List[str] = Field(
        default_factory=list,
        description="Concrete outcomes this epic should achieve, one per bullet.",
    )


class IssueDraft(StructuredOutput):
    temp_id: str = Field(
        ..., description="A small identifier unique within this response only (e.g. '1', '2'), used "
                          "to express dependencies via `depends_on` — never a real issue key."
    )
    title: str = Field(
        ..., min_length=1, max_length=120,
        description="Concise, actionable title in natural workplace language that product, design, "
                    "QA, operations, and engineering can understand.",
    )
    description: str = Field(
        ..., min_length=1,
        description="1-3 plain-language sentences describing what needs to be done, who or what it "
                    "helps, and why — technical detail only where it is essential.",
    )
    issue_type: Literal["story", "task", "bug"] = Field(
        "task", description="story = user-facing, value-delivering work a stakeholder would recognize "
                             "as 'a feature landed'. task = internal/technical/enabling work with no "
                             "direct user-visible outcome by itself (infra, wiring, config, refactors). "
                             "bug = fixing a defect in existing behavior — rare in a net-new epic unless "
                             "the proposal explicitly describes fixing something broken.",
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Relevant labels, lowercase, only if clearly implied — an empty list is correct "
                    "when nothing is clearly implied.",
    )
    estimate_story_points: Optional[int] = Field(
        None, ge=1, le=13,
        description="A best-effort rough Fibonacci-like planning estimate (1, 2, 3, 5, 8, 13). "
                    "Prefer a reviewed estimate with uncertainty explained in the rationale; use null "
                    "only when there is genuinely no basis for sizing this issue.",
    )
    estimate_rationale: Optional[str] = Field(
        None, description="One short sentence explaining what makes this estimate that size (scope, "
                           "unknowns, dependencies, etc). Null if and only if estimate_story_points is "
                           "null.",
    )
    estimate_reference: List[str] = Field(
        default_factory=list,
        description="Issue keys of the past, finished issues this estimate was sized against — only "
                    "keys from the historical examples supplied in the prompt, never invented and "
                    "never an issue from this plan (those have temp_ids, not keys). Empty is correct "
                    "and expected when no past issue was comparable, or when none were supplied.",
    )
    complexity_vs_reference: Optional[Literal["simpler", "similar", "more_complex"]] = Field(
        None,
        description="Compared with the issues in `estimate_reference`, is this work smaller, about "
                    "the same, or larger? Answer the comparison only — the number itself is derived "
                    "from what those issues actually cost. Null exactly when estimate_reference is "
                    "empty.",
    )
    requirement_index: Optional[int] = Field(
        None, ge=1,
        description="Which of the stated requirements this issue delivers, by its number. Null when "
                    "no requirements were stated, or when this issue serves none of them directly "
                    "(supporting work the epic still needs). Its estimate is compared against the "
                    "past issues retrieved for that requirement specifically.",
    )
    origin: Literal["epic_requirement", "historical_pattern", "planning_necessity"] = Field(
        "epic_requirement",
        description="Why this issue exists at all. 'epic_requirement' = the proposal explicitly asks "
                    "for it. 'historical_pattern' = a reusable pattern from this team's past work "
                    "says this kind of epic needs it. 'planning_necessity' = neither states it, but "
                    "the work cannot be delivered without it. Never 'historical_pattern' unless a "
                    "pattern you were actually given implies it.",
    )
    pattern_basis: List[str] = Field(
        default_factory=list,
        description="Names of the historical patterns that led to this issue — only names from the "
                    "patterns supplied in the prompt. Empty is correct and common: most issues come "
                    "straight from the proposal.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="temp_ids of other issues in this same response that this issue should follow — "
                    "only where the decomposition genuinely implies an ordering, never invented just "
                    "to show structure. Most issues in a good plan have none.",
    )
    dependency_kinds: Dict[str, DependencyKind] = Field(
        default_factory=dict,
        description="How strong each depends_on edge is. 'blocks' = the later work genuinely cannot "
                    "be done at all until the earlier one lands, and is published as a real blocking "
                    "link. 'after' = it is merely better sequenced later and the team could work in "
                    "parallel; this only affects sprint ordering. Use 'after' whenever unsure — an "
                    "edge with no entry is treated as 'after'.",
    )
    dependency_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Everyday-language reason for each depends_on temp_id: what specific decision, "
                    "information, or finished work is needed, how it will be used, and the consequence "
                    "of starting too early. If you cannot explain an edge this way, leave the edge out "
                    "of depends_on entirely — an unexplained edge is dropped, not shown to the human.",
    )


class EpicPlanDraft(StructuredOutput):
    epic: EpicDraft
    issues: List[IssueDraft] = Field(
        ..., min_length=1,
        description="A decomposition of the epic into concrete, independently workable issues — "
                    "typically 3 to 12 depending on the proposal's scope.",
    )

    @model_validator(mode="after")
    def validate_issue_temp_ids(self) -> "EpicPlanDraft":
        """Never let duplicate model identifiers disappear in the later dict-based graph pass."""
        temp_ids = [issue.temp_id.strip() for issue in self.issues]
        if any(not temp_id for temp_id in temp_ids):
            raise ValueError("issue temp ids cannot be blank")
        if any(issue.temp_id != issue.temp_id.strip() for issue in self.issues):
            raise ValueError("issue temp ids cannot have leading or trailing whitespace")
        if len(temp_ids) != len(set(temp_ids)):
            raise ValueError("issue temp ids must be unique")
        return self


# --- Multi-agent planning (app/planning/graph.py) ---
# The planner reuses EpicPlanDraft; only the critic needs its own output shape.


class UnrealisticEstimate(StructuredOutput):
    temp_id: str = Field(..., description="The issue whose estimate looks off.")
    reason: str = Field(..., description="Why this estimate looks too small or too large for the work described.")


class MissingDependency(StructuredOutput):
    temp_id: str = Field(..., description="The issue that is likely missing a depends_on edge.")
    likely_depends_on: str = Field(..., description="The other issue's temp_id it probably needs to wait on.")
    reason: str = Field(..., description="Why this ordering is likely required.")


class PlanCritique(StructuredOutput):
    """Evidence only — deliberately has no self-reported verdict field (e.g. `approved: bool`). A
    model asked to both list problems AND declare a verdict can produce an internally inconsistent
    answer (a real gap listed, verdict still "approved"). Whether the plan needs another revision
    round is instead computed from these lists by `app.planning.service.plan_needs_revision`, a plain,
    independently unit-tested function — the same "LLM lists evidence, code computes the verdict"
    split `app.sprint_pace.service`'s risk-level handling already uses.

    Every list field follows this codebase's existing Optional/List convention throughout
    `IssueDraft`: an empty list is a correct, positive statement that nothing of that kind was found —
    never invent a problem just to appear thorough.
    """

    coverage_gaps: List[str] = Field(
        default_factory=list,
        description="Concrete, missing pieces of work the proposal implies but the plan doesn't cover. "
                    "Empty is correct when the decomposition is genuinely complete.",
    )
    redundant_issue_groups: List[List[str]] = Field(
        default_factory=list,
        description="Groups (by temp_id) of issues that substantially overlap and should likely be "
                    "merged. Empty is correct when there's no real overlap.",
    )
    unrealistic_estimates: List[UnrealisticEstimate] = Field(
        default_factory=list,
        description="Estimates that look clearly too small or too large for the described scope. "
                    "Empty is correct when every estimate looks reasonable (or is null).",
    )
    missing_dependencies: List[MissingDependency] = Field(
        default_factory=list,
        description="Issue pairs that are very likely to need a depends_on edge the plan doesn't "
                    "have. Empty is correct when the ordering already given looks right.",
    )
