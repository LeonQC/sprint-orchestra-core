"""AI epic planning: proposal -> epic + issues + sprint allocation.

Only the first step uses the LLM:
  1. `plan_epic` — one instructor call -> validated EpicPlanDraft, or an error the user can act on.
  2. `validate_and_order` — breaks dependency cycles and returns issues in topological order.
  3. `allocate_sprints` — packs ordered issues into sprints using real historical velocity.

Nothing is written to a database; the caller gets a preview to edit and commit ("AI drafts, human
approves"). `critique_plan` / `plan_needs_revision` / `critique_to_instruction` are pieces of the
optional multi-agent path (graph.py); they live here as plain functions so the graph only does
orchestration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from app.llm.circuit_breaker import ModelUnavailableError, is_model_unavailable
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.planning.grounding import GroundingContext, apply_estimate_grounding
from app.planning.pattern_schemas import EvidenceAssessment
from app.planning.patterns import derive_estimate, historical_baseline
from app.planning.prompts import (
    render_critic_prompt,
    render_epic_plan_prompt,
    render_epic_plan_refine_prompt,
)
from app.planning.schemas import (
    DEFAULT_DEPENDENCY_KIND,
    EpicDraft,
    EpicPlanDraft,
    IssueDraft,
    PlanCritique,
)
from app.tracing import llm_step

logger = logging.getLogger(__name__)

STORY_POINT_VALUES = frozenset({1, 2, 3, 5, 8, 13})

def is_valid_story_point(value: Optional[int]) -> bool:
    """Use the same reviewed Fibonacci set as the Plan Epic UI."""
    return isinstance(value, int) and not isinstance(value, bool) and value in STORY_POINT_VALUES


def drop_unexplained_dependencies(issues: List[IssueDraft]) -> List[IssueDraft]:
    """Drop every AI-proposed dependency the model didn't explain.

    Replaces a regex gate that made the reviewer supply a "concrete enough" reason before approval —
    it accepted "aaaa... rather than bbbb" and rejected real reasons, so it measured vocabulary, not
    reasoning. An edge with no explanation at all is likely not real, so it's dropped. Only applied to
    model output; a human-added dependency needs no explanation (as in Jira, Linear, Asana).
    """
    cleaned: List[IssueDraft] = []
    for issue in issues:
        kept = [
            dependency for dependency in issue.depends_on
            if issue.dependency_reasons.get(dependency, "").strip()
        ]
        if len(kept) == len(issue.depends_on):
            cleaned.append(issue)
            continue
        logger.info(
            "dropping %d unexplained dependency edge(s) from issue %s",
            len(issue.depends_on) - len(kept), issue.temp_id,
        )
        cleaned.append(issue.model_copy(update={
            "depends_on": kept,
            "dependency_reasons": {d: issue.dependency_reasons[d] for d in kept},
            "dependency_kinds": {
                d: issue.dependency_kinds[d] for d in kept if d in issue.dependency_kinds
            },
        }))
    return cleaned


def drop_invalid_requirement_indexes(
    issues: List[IssueDraft], requirement_count: int,
) -> tuple:
    """Clear any `requirement_index` that doesn't match a stated requirement.

    Models get 1-based list references off by one, and a wrong number is worse than none (wrong
    claim, wrong history for estimates). Cleared, not guessed.
    """
    cleaned: List[IssueDraft] = []
    dropped = 0
    for issue in issues:
        index = issue.requirement_index
        if index is not None and not 1 <= index <= requirement_count:
            logger.info(
                "clearing requirement_index=%s on issue %s: only %d requirement(s) were stated",
                index, issue.temp_id, requirement_count,
            )
            dropped += 1
            cleaned.append(issue.model_copy(update={"requirement_index": None}))
        else:
            cleaned.append(issue)
    return cleaned, dropped


def restore_untouched_provenance(
    refined: List[IssueDraft], original: List[IssueDraft],
) -> List[IssueDraft]:
    """Restore the provenance an unchanged issue came in with, when the model dropped it.

    The refine prompt asks to keep it and shows every field, yet an "add a QA task" round returned six
    untouched issues with all their `estimate_reference` values gone. So it's guaranteed in code.

    Per field, only when still valid:
      * estimate_reference / complexity_vs_reference (justify a size): only if the size is unchanged.
      * origin / pattern_basis (why the issue exists): only if the title is unchanged.
    Never invents provenance, and never overrides a value the model actually changed.
    """
    prior = {issue.temp_id: issue for issue in original}
    restored: List[IssueDraft] = []
    for issue in refined:
        before = prior.get(issue.temp_id)
        if before is None:
            restored.append(issue)
            continue
        update = {}
        if (
            not issue.estimate_reference
            and before.estimate_reference
            and issue.estimate_story_points == before.estimate_story_points
        ):
            update["estimate_reference"] = list(before.estimate_reference)
            update["complexity_vs_reference"] = before.complexity_vs_reference
        if issue.title.strip() == before.title.strip():
            if issue.origin == "epic_requirement" and before.origin != "epic_requirement":
                # "epic_requirement" is the default, so "omitted" and "chosen" look the same. Only
                # restore a non-default prior; a stale value left on an unchanged issue is the more
                # visible, easier-to-fix mistake.
                update["origin"] = before.origin
            if not issue.pattern_basis and before.pattern_basis:
                update["pattern_basis"] = list(before.pattern_basis)
        restored.append(issue.model_copy(update=update) if update else issue)
    return restored


def apply_historical_estimates(
    issues: List[IssueDraft], grounding: Optional[GroundingContext],
) -> tuple:
    """Replace the model's story points with what the cited work actually cost.

    Code owns the number, the model owns only the comparison (`complexity_vs_reference`): "how big
    was ATC-29" is a fact; "is this bigger than ATC-29" is a judgement. `patterns.derive_estimate`
    then moves one Fibonacci step from the real number.

    Runs after `apply_estimate_grounding`, so only verified citations can change a number. With no
    usable citation (none, or unfinished/unsized issues), the model's own estimate stays — an honest
    judgement call.
    """
    adjusted: List[IssueDraft] = []
    derived_count = 0
    for issue in issues:
        baseline = historical_baseline(issue.estimate_reference, grounding)
        derived = derive_estimate(baseline, issue.complexity_vs_reference)
        if derived is None or issue.estimate_story_points is None:
            adjusted.append(issue)
            continue
        if derived != issue.estimate_story_points:
            logger.info(
                "issue %s: %s pts from the model -> %s pts derived from %s (%s)",
                issue.temp_id, issue.estimate_story_points, derived,
                ", ".join(issue.estimate_reference), issue.complexity_vs_reference or "similar",
            )
        derived_count += 1
        adjusted.append(issue.model_copy(update={"estimate_story_points": derived}))
    return adjusted, derived_count


# Small retry budget (user is waiting), generous tokens (a whole plan is a big output).
_MAX_VALIDATION_RETRIES = 2
# Raised from 8000 when descriptions started following the board's layout (context + criteria +
# estimate reasoning in HTML, for every issue). At 8000 a plan sometimes hit IncompleteOutputException.
_MAX_OUTPUT_TOKENS = 16000
# The critic returns four short lists, so it gets the smaller budget.
_MAX_REVIEW_OUTPUT_TOKENS = 2000


@dataclass(slots=True)
class PlanResult:
    plan: EpicPlanDraft
    degraded: bool
    latency_seconds: float
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0
    #: Citations removed for naming an issue the model was never shown. Reported so the catch rate is
    #: visible.
    dropped_estimate_references: int = 0
    #: Estimates whose number came from real cited work rather than the model.
    historically_derived_estimates: int = 0


@dataclass(slots=True)
class SprintBucket:
    sprint_index: int
    issue_temp_ids: List[str] = field(default_factory=list)
    total_points: int = 0


PLAN_FAILED_MESSAGE = "The AI couldn't produce a usable plan this time. Please try again."
REFINE_FAILED_MESSAGE = "The AI couldn't apply this change. Your plan is unchanged; please try again."


class PlanGenerationError(RuntimeError):
    """The model answered, but no valid plan came out of it after instructor's re-asks."""


def _render_current_plan(epic: EpicDraft, issues: List[IssueDraft]) -> str:
    """Render every field the model is asked to preserve.

    `labels` and `estimate_rationale` used to be missing here while the prompt said to keep them, so
    one refine round wiped them from every existing issue. Anything not rendered here is lost on the
    next refine — add new IssueDraft fields here too.
    """
    lines = [f"Epic: {epic.title}", f"Description: {epic.description}"]
    if epic.goals:
        lines.append("Goals: " + "; ".join(epic.goals))
    lines.append("")
    lines.append("Issues:")
    for issue in issues:
        points = issue.estimate_story_points if issue.estimate_story_points is not None else "unestimated"
        dep = f" | depends on: {', '.join(issue.depends_on)}" if issue.depends_on else ""
        labels = f" | labels: {', '.join(issue.labels)}" if issue.labels else " | labels: none"
        lines.append(
            f"- [{issue.temp_id}] ({issue.issue_type}, {points} pts){labels}{dep}: {issue.title}"
        )
        if issue.description:
            lines.append(f"  {issue.description}")
        if issue.estimate_rationale:
            lines.append(f"  Estimate rationale: {issue.estimate_rationale}")
        if issue.estimate_reference:
            comparison = f" ({issue.complexity_vs_reference})" if issue.complexity_vs_reference else ""
            lines.append(f"  Estimate sized against: {', '.join(issue.estimate_reference)}{comparison}")
        if issue.requirement_index is not None:
            lines.append(f"  Delivers requirement: {issue.requirement_index}")
        lines.append(f"  Origin: {issue.origin}"
                     + (f" via pattern(s): {', '.join(issue.pattern_basis)}" if issue.pattern_basis else ""))
        for dependency in issue.depends_on:
            kind = issue.dependency_kinds.get(dependency, DEFAULT_DEPENDENCY_KIND)
            reason = issue.dependency_reasons.get(dependency)
            if reason:
                lines.append(f"  Dependency {dependency} ({kind}): {reason}")
    return "\n".join(lines)


def _render_proposal(proposal: str, grounding: Optional[GroundingContext]) -> str:
    """The proposal plus the requirements as a numbered list.

    They must be in the instruction itself: when they only appeared as headers in the evidence
    section, every issue in a five-requirement plan came back as requirement 1. Same 1-based numbering
    as `requirement_index`.
    """
    if grounding is None or not grounding.requirements:
        return proposal
    lines = [proposal.strip(), "", "REQUIREMENTS — what this epic must deliver:"]
    lines.extend(
        f"  {requirement.index}. {requirement.text}" for requirement in grounding.requirements
    )
    return "\n".join(lines)


async def plan_epic(
    client,
    model: str,
    proposal: str,
    existing_labels: Optional[List[str]] = None,
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> PlanResult:
    """`grounding=None` gives the original ungrounded behaviour. Retrieval is the caller's job (it owns
    the authorized space scope).
    """
    system_prompt = render_epic_plan_prompt(existing_labels, grounding, assessment)
    start = time.perf_counter()
    try:
        with llm_step("plan_epic.generate"):
            plan, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=EpicPlanDraft,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _render_proposal(proposal, grounding)},
                ],
            )
        usage = usage_from_completion(completion)
        issues, dropped = apply_estimate_grounding(plan.issues, grounding)
        issues, _ = drop_invalid_requirement_indexes(
            issues, len(grounding.requirements) if grounding else 0,
        )
        issues, derived = apply_historical_estimates(issues, grounding)
        plan = plan.model_copy(update={"issues": drop_unexplained_dependencies(issues)})
        return PlanResult(
            plan=plan, degraded=False, latency_seconds=time.perf_counter() - start,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens,
            estimated_cost_usd=estimate_cost_usd(
                served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
                usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
            ),
            dropped_estimate_references=dropped,
            historically_derived_estimates=derived,
        )
    except Exception as exc:  # noqa: BLE001 - every failure becomes one of two messages for the user
        # No stand-in plan: a one-issue copy of the proposal reads as the AI's work and plans nothing.
        if is_model_unavailable(exc):
            raise ModelUnavailableError() from exc
        logger.warning("epic plan generation failed after retries", extra={"error": str(exc)})
        raise PlanGenerationError(PLAN_FAILED_MESSAGE) from exc


async def refine_plan(
    client,
    model: str,
    epic: EpicDraft,
    issues: List[IssueDraft],
    instruction: str,
    existing_labels: Optional[List[str]] = None,
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> PlanResult:
    """Apply a free-text edit ("add a QA task", "combine 2 and 3") to an existing plan. Raises on
    failure; the plan the human was reviewing stays with the caller.
    """
    system_prompt = render_epic_plan_refine_prompt(existing_labels, grounding, assessment)
    user_message = _render_current_plan(epic, issues) + f"\n\nInstruction: {instruction}"
    # Citations already on the plan were verified when it was generated; this call's retrieval may
    # differ, so don't re-judge them (see `apply_estimate_grounding`).
    carried_references = {
        reference.upper() for issue in issues for reference in issue.estimate_reference
    }
    start = time.perf_counter()
    try:
        with llm_step("plan_epic.refine"):
            plan, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=EpicPlanDraft,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        usage = usage_from_completion(completion)
        refined = restore_untouched_provenance(plan.issues, issues)
        refined, dropped = apply_estimate_grounding(refined, grounding, carried_references)
        refined, _ = drop_invalid_requirement_indexes(
            refined, len(grounding.requirements) if grounding else 0,
        )
        refined, derived = apply_historical_estimates(refined, grounding)
        plan = plan.model_copy(update={"issues": drop_unexplained_dependencies(refined)})
        return PlanResult(
            plan=plan, degraded=False, latency_seconds=time.perf_counter() - start,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens,
            estimated_cost_usd=estimate_cost_usd(
                served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
                usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
            ),
            dropped_estimate_references=dropped,
            historically_derived_estimates=derived,
        )
    except Exception as exc:  # noqa: BLE001 - every failure becomes one of two messages for the user
        # The caller still holds the plan; only the edit is refused.
        if is_model_unavailable(exc):
            raise ModelUnavailableError() from exc
        logger.warning("plan refinement failed after retries", extra={"error": str(exc)})
        raise PlanGenerationError(REFINE_FAILED_MESSAGE) from exc


@dataclass(slots=True)
class CritiqueResult:
    critique: PlanCritique
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0


async def critique_plan(
    client,
    model: str,
    proposal: str,
    epic: EpicDraft,
    issues: List[IssueDraft],
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> CritiqueResult:
    """Review a plan and report findings (never edit). On failure, return an empty critique with
    `error` set — a broken reviewer must not force revisions or block a likely-fine plan.
    """
    user_message = f"Original proposal:\n{proposal}\n\nCurrent plan:\n{_render_current_plan(epic, issues)}"
    try:
        with llm_step("plan_epic.critique"):
            critique, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=PlanCritique,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_REVIEW_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": render_critic_prompt(grounding, assessment)},
                    {"role": "user", "content": user_message},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a failed review must not block an otherwise fine plan
        logger.warning(
            "plan critique failed after retries; treating the plan as approved",
            extra={"error": str(exc)},
        )
        return CritiqueResult(critique=PlanCritique(), error=str(exc))
    usage = usage_from_completion(completion)
    return CritiqueResult(
        critique=critique,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
    )


def plan_needs_revision(critique: PlanCritique) -> bool:
    """Loop-back condition, computed in code (not from a self-reported `approved` flag): any concrete
    finding is worth one revision pass.
    """
    return bool(
        critique.coverage_gaps
        or critique.redundant_issue_groups
        or critique.unrealistic_estimates
        or critique.missing_dependencies
    )


def critique_to_instruction(critique: PlanCritique) -> str:
    """Turn a critique into the instruction format `refine_plan` already takes, so no third prompt is
    needed.
    """
    lines: List[str] = []
    for gap in critique.coverage_gaps:
        lines.append(f"- Add work covering a gap in the plan: {gap}")
    for group in critique.redundant_issue_groups:
        if len(group) > 1:
            lines.append(f"- Merge these overlapping issues into one: {', '.join(group)}")
    for item in critique.unrealistic_estimates:
        lines.append(f"- Re-estimate issue {item.temp_id}: {item.reason}")
    for dep in critique.missing_dependencies:
        lines.append(f"- Make issue {dep.temp_id} depend on {dep.likely_depends_on}: {dep.reason}")
    return "A plan review flagged the following. Apply each one:\n" + "\n".join(lines)


def validate_and_order(issues: List[IssueDraft]) -> List[IssueDraft]:
    """Drop dependencies that form a cycle or name an unknown temp_id, and return issues in
    topological order. Never raises — a messy graph should still give some valid order.
    """
    by_id = {issue.temp_id: issue for issue in issues}
    pending_deps = {
        issue.temp_id: {d for d in issue.depends_on if d in by_id and d != issue.temp_id}
        for issue in issues
    }

    ordered: List[IssueDraft] = []
    remaining = set(by_id.keys())

    while remaining:
        ready = sorted(tid for tid in remaining if not (pending_deps[tid] & remaining))
        if not ready:
            # Cycle: deterministically pick the lowest remaining temp_id and drop its unresolved
            # edges into the remaining set, rather than failing the whole plan over a bad LLM output.
            tid = sorted(remaining)[0]
            dropped = pending_deps[tid] & remaining
            logger.warning("dependency cycle detected; dropping edges from %s to %s", tid, dropped)
            pending_deps[tid] -= dropped
            ready = [tid]
        for tid in ready:
            issue = by_id[tid]
            dependencies = sorted(pending_deps[tid])
            # Keep reasons/kinds only for surviving edges. Human-added edges have none, which is fine.
            reasons = {
                dependency: issue.dependency_reasons[dependency].strip()
                for dependency in dependencies
                if issue.dependency_reasons.get(dependency, "").strip()
            }
            kinds = {
                dependency: issue.dependency_kinds[dependency]
                for dependency in dependencies
                if dependency in issue.dependency_kinds
            }
            ordered.append(issue.model_copy(update={
                "depends_on": dependencies,
                "dependency_reasons": reasons,
                "dependency_kinds": kinds,
            }))
            remaining.discard(tid)

    return ordered


def allocate_sprints(
    ordered_issues: List[IssueDraft],
    sprint_capacity_points: Optional[float],
    target_sprint_count: Optional[int] = None,
) -> List[SprintBucket]:
    """Pack issues into sprints in topological order; capacity is the hard limit when given.

    `target_sprint_count` is a wish, not permission to overload (the old code ignored capacity and
    put 31 points into an 8-point sprint). A single issue bigger than capacity sits alone, for the
    human to split. Unestimated issues count as 0.
    """
    if not ordered_issues:
        return []

    capacity = sprint_capacity_points if sprint_capacity_points and sprint_capacity_points > 0 else None
    if capacity is not None:
        buckets = [SprintBucket(sprint_index=0)]
        bucket_by_issue: dict[str, int] = {}
        for issue in ordered_issues:
            points = issue.estimate_story_points or 0
            # First-fit that respects dependencies: a dependent can share its prerequisite's sprint or
            # come later, never earlier. (Next-fit made [5, 5, 3, 3] at capacity 8 into 3 sprints.)
            earliest_bucket = max(
                (bucket_by_issue[dependency] for dependency in issue.depends_on
                 if dependency in bucket_by_issue),
                default=0,
            )
            destination = next(
                (
                    bucket
                    for bucket in buckets[earliest_bucket:]
                    if not bucket.issue_temp_ids or bucket.total_points + points <= capacity
                ),
                None,
            )
            if destination is None:
                buckets.append(SprintBucket(sprint_index=len(buckets)))
                destination = buckets[-1]
            destination.issue_temp_ids.append(issue.temp_id)
            destination.total_points += points
            bucket_by_issue[issue.temp_id] = destination.sprint_index

        # If packing used fewer sprints than asked, split buckets until the count is met. Splitting
        # never breaks capacity or order.
        desired = min(target_sprint_count or 0, len(ordered_issues))
        while desired > len(buckets):
            split_index = max(
                (i for i, bucket in enumerate(buckets) if len(bucket.issue_temp_ids) > 1),
                key=lambda i: (len(buckets[i].issue_temp_ids), buckets[i].total_points),
                default=-1,
            )
            if split_index < 0:
                break
            bucket = buckets[split_index]
            midpoint = len(bucket.issue_temp_ids) // 2
            left_ids = bucket.issue_temp_ids[:midpoint]
            right_ids = bucket.issue_temp_ids[midpoint:]
            points_by_id = {
                issue.temp_id: issue.estimate_story_points or 0 for issue in ordered_issues
            }
            buckets[split_index:split_index + 1] = [
                SprintBucket(
                    sprint_index=0,
                    issue_temp_ids=left_ids,
                    total_points=sum(points_by_id[temp_id] for temp_id in left_ids),
                ),
                SprintBucket(
                    sprint_index=0,
                    issue_temp_ids=right_ids,
                    total_points=sum(points_by_id[temp_id] for temp_id in right_ids),
                ),
            ]
        for index, bucket in enumerate(buckets):
            bucket.sprint_index = index
        return buckets

    if target_sprint_count and target_sprint_count > 0:
        buckets = [SprintBucket(sprint_index=i) for i in range(target_sprint_count)]
        total_points = sum(issue.estimate_story_points or 0 for issue in ordered_issues)
        target_per_bucket = total_points / target_sprint_count if total_points else 0
        idx = 0
        for issue in ordered_issues:
            points = issue.estimate_story_points or 0
            current = buckets[idx]
            if (
                idx < target_sprint_count - 1
                and current.issue_temp_ids
                and current.total_points >= target_per_bucket
            ):
                idx += 1
                current = buckets[idx]
            current.issue_temp_ids.append(issue.temp_id)
            current.total_points += points
        return [b for b in buckets if b.issue_temp_ids]

    buckets = [SprintBucket(sprint_index=0)]
    for issue in ordered_issues:
        points = issue.estimate_story_points or 0
        current = buckets[-1]
        if capacity is not None and current.issue_temp_ids and current.total_points + points > capacity:
            buckets.append(SprintBucket(sprint_index=len(buckets)))
            current = buckets[-1]
        current.issue_temp_ids.append(issue.temp_id)
        current.total_points += points
    return buckets
