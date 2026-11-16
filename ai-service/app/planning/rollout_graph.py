"""Durable Plan Epic workflow: generate -> human review/edit -> publish the whole plan to Jira.

graph.py covers how a plan is generated; this covers what happens after: an approval that may take
hours, and publishing that must survive a process restart. It wraps the existing plan-generation call.

- Every Jira write (new sprint, child issue, dependency link) is its own graph step, checkpointed by
  AsyncPostgresSaver, so a restart resumes from the last recorded step.
- Space membership is re-checked when the decision arrives (`review_node`), not trusted from start.
- A saved checkpoint doesn't resume by itself: reading status (`aget_state`) never runs the graph
  (verified: a crashed thread stayed at "committing"). `retry_rollout` is the explicit trigger.
- A crash and a clean failure differ: after a crash a task is still pending, so `ainvoke(None)`
  continues; after a clean failure (jira-backend down, node returned "failed") the graph already
  ended, so `retry_rollout` re-arms the right node. The idempotency ledger (`committed`,
  `epic_issue_id`) means finished writes are never resent.

This replaced the old browser-side publish loop entirely.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from hashlib import sha256
from typing import Callable, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.auth.space_membership import SpaceMembershipChecker, SpaceMembershipError
from app.llm.circuit_breaker import ModelUnavailableError
from app.planning.jira_commit_client import JiraCommitClient, JiraCommitError
from app.planning.grounding import GroundingContext, with_stated_requirements
from app.planning.patterns import AssessmentResult
from app.planning.rollout_schemas import (
    EvidenceState,
    GroundingState,
    HouseFormatState,
    RolloutState,
    SprintBucketState,
    SprintTargetState,
)
from app.planning.schemas import EpicDraft, IssueDraft
from app.planning.service import (
    PlanGenerationError,
    allocate_sprints,
    is_valid_story_point,
    plan_epic,
    refine_plan,
    validate_and_order,
)

logger = logging.getLogger(__name__)

# Optional SSE progress callback, via ContextVar because the compiled graph is shared (same idea as
# sprint_recovery's ON_STAGE_VAR, kept separate on purpose). Only plan_node uses it — the other
# nodes are a pause or fast Jira writes.
ON_STAGE_VAR: ContextVar[Optional[Callable[[str], None]]] = ContextVar("rollout_on_stage", default=None)

_MAX_SPRINT_NAME_LENGTH = 100


def _idempotency_key(thread_id: str, mutation: str, identity: str = "") -> str:
    """Stable, header-safe idempotency key for one Jira write in one rollout thread. The changeable
    part (temp id, sprint index) is hashed so odd characters can't break the HTTP header; retries of
    the same node get the same key.
    """
    digest = sha256(f"{mutation}:{identity}".encode("utf-8")).hexdigest()[:24]
    return f"plan-epic:{thread_id}:{mutation}:{digest}"


def _thread_id_from_config(config) -> str:
    thread_id = str((config or {}).get("configurable", {}).get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("durable Plan Epic publishing requires a workflow thread_id")
    return thread_id


def _buckets_to_state(buckets) -> List[SprintBucketState]:
    return [
        SprintBucketState(
            sprint_index=b.sprint_index, issue_temp_ids=b.issue_temp_ids, total_points=b.total_points
        )
        for b in buckets
    ]


def _grounding_to_state(context: GroundingContext, dropped_references: int) -> GroundingState:
    return GroundingState(
        status=context.status,
        exemplars=[
            {
                "issue_key": exemplar.issue_key,
                "title": exemplar.title,
                "issue_type": exemplar.issue_type,
                "status": exemplar.status,
                "story_points": exemplar.story_points,
                "confidence": round(exemplar.confidence, 4),
                "estimate_anchor": exemplar.is_estimate_anchor,
                "source": exemplar.source,
            }
            for exemplar in context.exemplars
        ],
        house_format=HouseFormatState(
            sections=list(context.house_format.sections),
            item_marker=context.house_format.item_marker,
            item_count_low=context.house_format.item_count_low,
            item_count_high=context.house_format.item_count_high,
            item_count_max=context.house_format.item_count_max,
            coverage=dict(context.house_format.coverage),
            sample_size=context.house_format.sample_size,
        ),
        estimate_anchor_count=len(context.anchors),
        dropped_estimate_references=dropped_references,
    )


def _evidence_to_state(result: Optional[AssessmentResult]) -> EvidenceState:
    if result is None:
        return EvidenceState(
            strength="unassessed", reasoning="", relevant_issue_keys=[], patterns=[],
            dropped_patterns=0, downgraded=False,
        )
    assessment = result.assessment
    return EvidenceState(
        strength=assessment.strength,
        reasoning=assessment.reasoning,
        relevant_issue_keys=list(assessment.relevant_issue_keys),
        patterns=[pattern.model_dump() for pattern in assessment.patterns],
        dropped_patterns=result.dropped_patterns,
        downgraded=result.downgraded,
    )


async def _assess_and_plan(
    state: RolloutState, client, model: str, planning_graph_getter, grounding: GroundingContext,
    history_assessor, on_stage,
):
    # Injected the same way `grounding_gatherer` is, and for the same reason: the node stays a pure
    # function of its arguments, so a test can exercise the pattern stage without a live model.
    assessment_result = None
    if history_assessor is not None and grounding.exemplars:
        if on_stage:
            on_stage("working out which past work actually applies here")
        assessment_result = await history_assessor(state["proposal"], grounding)
    assessment = assessment_result.assessment if assessment_result else None
    if on_stage:
        # The longest step (~30s of planning); it used to show no label at all.
        found = len(assessment.patterns) if assessment else 0
        on_stage(
            f"writing the plan from {len(grounding.exemplars)} past issues"
            + (f" and {found} reusable pattern{'s' if found != 1 else ''}" if found else "")
        )
    planning_graph = planning_graph_getter()
    if planning_graph is not None:
        from app.planning.graph import plan_epic_multiagent

        result = await plan_epic_multiagent(
            planning_graph, state["proposal"], state["existing_labels"], grounding, assessment,
        )
    else:
        result = await plan_epic(
            client, model, state["proposal"], state["existing_labels"], grounding, assessment,
        )

    return assessment_result, result


async def plan_node(
    state: RolloutState, client, model: str, planning_graph_getter, grounding_gatherer=None,
    history_assessor=None,
) -> dict:
    """Round 0: generate the initial plan with whichever path is configured (same functions as POST
    /plan-epic).

    `grounding_gatherer` is an optional async `(space_id, proposal) -> GroundingContext` passed in
    from main.py, so the node stays testable without a live service. No membership check needed: the
    start endpoint already authorized `space_id` in this same request.
    """
    on_stage = ON_STAGE_VAR.get()
    if on_stage:
        on_stage("generating the rollout plan")
    grounding = GroundingContext(status="disabled")
    if grounding_gatherer is not None:
        if on_stage:
            on_stage("reading how this team has planned similar work")
        grounding = await grounding_gatherer(
            state["space_id"], state["proposal"], state.get("requirements") or [], on_stage,
        )
    grounding = with_stated_requirements(grounding, state.get("requirements") or [])
    try:
        assessment_result, result = await _assess_and_plan(
            state, client, model, planning_graph_getter, grounding, history_assessor, on_stage,
        )
    except (ModelUnavailableError, PlanGenerationError) as exc:
        # Ends the run here (see `_after_plan`); the user starts a new one.
        return {"status": "failed", "error": str(exc), "failed_step": "plan"}

    ordered = validate_and_order(result.plan.issues)
    buckets = allocate_sprints(ordered, state["sprint_capacity_points"], state["target_sprint_count"])
    return {
        "epic": result.plan.epic,
        "issues": ordered,
        "sprint_buckets": _buckets_to_state(buckets),
        "grounding": _grounding_to_state(grounding, result.dropped_estimate_references),
        "evidence": _evidence_to_state(assessment_result),
        "degraded": result.degraded,
        "error": result.error,
        "status": "pending_approval",
        # The extra reading call is part of what this plan cost; billing it separately from the
        # planning call would understate the feature by exactly the stage that defines it.
        "input_tokens": result.input_tokens + (assessment_result.input_tokens if assessment_result else 0),
        "output_tokens": result.output_tokens + (assessment_result.output_tokens if assessment_result else 0),
        "cached_input_tokens": result.cached_input_tokens
            + (assessment_result.cached_input_tokens if assessment_result else 0),
        "estimated_cost_usd": result.estimated_cost_usd
            + (assessment_result.estimated_cost_usd if assessment_result else 0.0),
        "planning_latency_seconds": result.latency_seconds
            + (assessment_result.latency_seconds if assessment_result else 0.0),
    }


def _drop_stale_estimate_references(
    edited: List[IssueDraft], original: List[IssueDraft],
) -> List[IssueDraft]:
    """If a human changes an issue's story points, drop the AI's citation for them.

    `estimate_reference` is published as "Sized against: …", which is false next to a number the
    reviewer typed. (The UI already clears `estimate_rationale` on manual edits.) Enforced here so it
    holds for any caller. Newly added issues keep what they came with — people don't have to justify
    their own edits.
    """
    original_points = {issue.temp_id: issue.estimate_story_points for issue in original}
    cleaned: List[IssueDraft] = []
    for issue in edited:
        changed = (
            issue.temp_id in original_points
            and issue.estimate_story_points != original_points[issue.temp_id]
        )
        if changed and issue.estimate_reference:
            logger.info(
                "dropping the AI's estimate citation from %s: a human changed the estimate from "
                "%s to %s", issue.temp_id, original_points[issue.temp_id], issue.estimate_story_points,
            )
            cleaned.append(issue.model_copy(update={"estimate_reference": []}))
        else:
            cleaned.append(issue)
    return cleaned


def _plan_preview(state: RolloutState) -> dict:
    return {
        "epic": state["epic"].model_dump() if state["epic"] else None,
        "issues": [i.model_dump() for i in state["issues"]],
        "sprint_buckets": state["sprint_buckets"],
        "grounding": state.get("grounding"),
        "evidence": state.get("evidence"),
    }


def _validate_final_plan(
    epic: Optional[EpicDraft], issues: List[IssueDraft], *, require_estimates: bool,
) -> None:
    """Reject an unsafe final plan before the first Jira write: blank text, duplicate temp ids,
    inconsistent dependency maps, circular boilerplate. Checked here, not just in the UI.
    """
    if epic is None:
        raise ValueError("final plan is missing its epic")
    if not epic.title.strip():
        raise ValueError("epic title cannot be blank")
    if not epic.description.strip():
        raise ValueError("epic description cannot be blank")
    if not issues:
        raise ValueError("final plan must contain at least one issue")

    temp_ids = [issue.temp_id.strip() for issue in issues]
    if any(not temp_id for temp_id in temp_ids):
        raise ValueError("issue temp ids cannot be blank")
    if len(temp_ids) != len(set(temp_ids)):
        raise ValueError("issue temp ids must be unique")
    if any(issue.temp_id != issue.temp_id.strip() for issue in issues):
        raise ValueError("issue temp ids cannot have leading or trailing whitespace")

    known_ids = set(temp_ids)
    dependencies_by_id: dict[str, set[str]] = {}
    for issue in issues:
        if not issue.title.strip():
            raise ValueError(f"issue {issue.temp_id} title cannot be blank")
        if not issue.description.strip():
            raise ValueError(f"issue {issue.temp_id} description cannot be blank")
        if require_estimates and not is_valid_story_point(issue.estimate_story_points):
            raise ValueError(
                f"issue {issue.temp_id} requires a valid story-point estimate "
                "(1, 2, 3, 5, 8, or 13) before sprint publishing"
            )

        dependencies = list(issue.depends_on)
        if len(dependencies) != len(set(dependencies)):
            raise ValueError(f"issue {issue.temp_id} repeats a dependency")
        unknown = set(dependencies) - known_ids
        if unknown:
            raise ValueError(
                f"issue {issue.temp_id} references unknown dependency temp ids: {sorted(unknown)}"
            )
        if issue.temp_id in dependencies:
            raise ValueError(f"issue {issue.temp_id} cannot depend on itself")
        dependencies_by_id[issue.temp_id] = set(dependencies)

    # A cycle has no valid first issue and would otherwise be silently modified by validate_and_order.
    remaining = set(known_ids)
    while remaining:
        ready = {temp_id for temp_id in remaining if not (dependencies_by_id[temp_id] & remaining)}
        if not ready:
            raise ValueError(f"final plan contains a dependency cycle among: {sorted(remaining)}")
        remaining -= ready


async def _preflight_existing_sprints(
    state: RolloutState, sprint_targets: List[SprintTargetState], jira: JiraCommitClient,
) -> None:
    """Re-read existing destinations at approval time, after a potentially long human pause."""
    seen: set[int] = set()
    for target in sprint_targets:
        if target["mode"] != "existing":
            continue
        sprint_id = int(target["sprint_id"])
        if sprint_id in seen:
            continue
        seen.add(sprint_id)
        try:
            await jira.validate_future_sprint(
                state["space_id"], sprint_id, state["user_id"], state["username"],
            )
        except JiraCommitError as exc:
            raise ValueError(str(exc)) from exc


async def review_node(
    state: RolloutState, space_membership: SpaceMembershipChecker, jira: JiraCommitClient,
) -> dict:
    """Pause (`interrupt`) with the current plan, then apply the decision. Everything after the
    interrupt, including the RBAC re-check, runs only when a decision arrives — maybe much later.
    """
    decision_payload = interrupt(_plan_preview(state))
    decision = decision_payload.get("decision")

    try:
        await space_membership.validate(state["user_id"], state["username"], [state["space_id"]])
    except SpaceMembershipError as exc:
        logger.warning("rollout RBAC recheck failed at approval time", extra={"error": str(exc)})
        return {"decision": "reject", "status": "rejected", "error": str(exc)}

    if decision == "reject":
        return {"decision": "reject", "status": "rejected"}

    if decision == "edit":
        edited_epic = EpicDraft.model_validate(decision_payload["epic"])
        edited_issues = _drop_stale_estimate_references(
            [IssueDraft.model_validate(i) for i in decision_payload["issues"]], state["issues"],
        )
        supplied_targets = decision_payload.get("sprint_targets")
        _validate_final_plan(
            edited_epic, edited_issues, require_estimates=supplied_targets is not None,
        )
        ordered = validate_and_order(edited_issues)
        if supplied_targets is None:
            buckets = allocate_sprints(ordered, state["sprint_capacity_points"], state["target_sprint_count"])
            sprint_buckets = _buckets_to_state(buckets)
            sprint_targets: List[SprintTargetState] = []
        else:
            sprint_targets = _validate_sprint_targets(supplied_targets, ordered)
            await _preflight_existing_sprints(state, sprint_targets, jira)
            sprint_buckets = [
                SprintBucketState(
                    sprint_index=t["sprint_index"],
                    issue_temp_ids=t["issue_temp_ids"],
                    total_points=sum(
                        i.estimate_story_points or 0
                        for i in ordered
                        if i.temp_id in t["issue_temp_ids"]
                    ),
                )
                for t in sprint_targets
            ]
        return {
            "decision": "edit",
            "epic": edited_epic,
            "issues": ordered,
            "sprint_buckets": sprint_buckets,
            "sprint_targets": sprint_targets,
            "status": "committing",
            "error": None,
            "failed_step": None,
        }

    _validate_final_plan(
        state["epic"], state["issues"], require_estimates=bool(state.get("sprint_targets")),
    )
    await _preflight_existing_sprints(state, state.get("sprint_targets") or [], jira)
    return {"decision": "approve", "status": "committing", "error": None, "failed_step": None}


def _validate_sprint_targets(raw_targets: list, issues: List[IssueDraft]) -> List[SprintTargetState]:
    """Validate the browser's final bucket destinations before any Jira write occurs."""
    known_ids = {i.temp_id for i in issues}
    seen_issue_ids: set[str] = set()
    seen_indexes: set[int] = set()
    seen_new_names: set[str] = set()
    targets: List[SprintTargetState] = []

    for raw in raw_targets:
        sprint_index = int(raw["sprint_index"])
        if sprint_index < 0:
            raise ValueError("sprint target index cannot be negative")
        if sprint_index in seen_indexes:
            raise ValueError(f"duplicate sprint target index: {sprint_index}")
        seen_indexes.add(sprint_index)
        issue_temp_ids = list(raw.get("issue_temp_ids") or [])
        if not issue_temp_ids:
            raise ValueError(f"sprint target {sprint_index} must contain at least one issue")
        if len(issue_temp_ids) != len(set(issue_temp_ids)):
            raise ValueError(f"sprint target {sprint_index} repeats an issue temp id")
        unknown = set(issue_temp_ids) - known_ids
        duplicate_assignments = set(issue_temp_ids) & seen_issue_ids
        if unknown:
            raise ValueError(f"sprint target references unknown issue temp ids: {sorted(unknown)}")
        if duplicate_assignments:
            raise ValueError(f"issues assigned to more than one sprint: {sorted(duplicate_assignments)}")
        seen_issue_ids.update(issue_temp_ids)

        mode = raw.get("mode")
        sprint_id = raw.get("sprint_id")
        sprint_name = (raw.get("sprint_name") or "").strip() or None
        if mode == "existing":
            if sprint_id is None or int(sprint_id) <= 0:
                raise ValueError("existing sprint target requires a positive sprint_id")
            sprint_id = int(sprint_id)
            sprint_name = None
        elif mode == "new":
            if not sprint_name:
                raise ValueError("new sprint target requires sprint_name")
            if "\n" in sprint_name or "\r" in sprint_name:
                raise ValueError("new sprint target name must be a single line")
            if len(sprint_name) > _MAX_SPRINT_NAME_LENGTH:
                raise ValueError(
                    f"new sprint target name cannot exceed {_MAX_SPRINT_NAME_LENGTH} characters"
                )
            normalized_name = sprint_name.casefold()
            if normalized_name in seen_new_names:
                raise ValueError(f"duplicate new sprint target name: {sprint_name}")
            seen_new_names.add(normalized_name)
            sprint_id = None
        else:
            raise ValueError(f"unknown sprint target mode: {mode}")

        if issue_temp_ids:
            targets.append(SprintTargetState(
                sprint_index=sprint_index,
                issue_temp_ids=issue_temp_ids,
                mode=mode,
                sprint_id=sprint_id,
                sprint_name=sprint_name,
            ))

    unassigned = known_ids - seen_issue_ids
    if unassigned:
        raise ValueError(f"issues missing a sprint target: {sorted(unassigned)}")
    return targets


def _after_review(state: RolloutState) -> str:
    return END if state["status"] == "rejected" else "create_epic"


async def create_epic_node(
    state: RolloutState, jira: JiraCommitClient, thread_id: str,
) -> dict:
    """Create the epic issue once. If `epic_issue_id` is already set (a resume after a crash), skip —
    never create a second epic.
    """
    if state["epic_issue_id"] is not None:
        return {}
    try:
        issue_id, issue_key = await jira.create_epic_issue(
            state["space_id"], state["epic"], state["user_id"], state["username"],
            idempotency_key=_idempotency_key(thread_id, "epic"),
        )
    except JiraCommitError as exc:
        logger.error("rollout epic creation failed", extra={"error": str(exc)})
        return {"status": "failed", "error": str(exc), "failed_step": "create_epic"}
    return {"epic_issue_id": issue_id, "epic_issue_key": issue_key, "failed_step": None}


def _after_create_epic(state: RolloutState) -> str:
    return END if state["status"] == "failed" else "prepare_sprint"


async def prepare_sprint_node(
    state: RolloutState, jira: JiraCommitClient, thread_id: str,
) -> dict:
    """Resolve one selected bucket to a real sprint id, checkpointing after each new sprint."""
    resolved = dict(state.get("created_sprints") or {})
    remaining = [
        t for t in state.get("sprint_targets") or []
        if str(t["sprint_index"]) not in resolved
    ]
    if not remaining:
        return {"failed_step": None}

    target = remaining[0]
    try:
        sprint_id = (
            int(target["sprint_id"])
            if target["mode"] == "existing"
            else await jira.create_sprint(
                state["space_id"], target["sprint_name"], state["user_id"], state["username"],
                idempotency_key=_idempotency_key(
                    thread_id, "sprint", str(target["sprint_index"]),
                ),
            )
        )
    except JiraCommitError as exc:
        logger.error("rollout sprint creation failed", extra={"error": str(exc)})
        return {"status": "failed", "error": str(exc), "failed_step": "prepare_sprint"}

    resolved[str(target["sprint_index"])] = sprint_id
    return {"created_sprints": resolved, "failed_step": None}


def _after_prepare_sprint(state: RolloutState) -> str:
    if state["status"] == "failed":
        return END
    if len(state.get("created_sprints") or {}) < len(state.get("sprint_targets") or []):
        return "prepare_sprint"
    return "commit_one"


def _sprint_id_for_issue(state: RolloutState, temp_id: str) -> Optional[int]:
    for target in state.get("sprint_targets") or []:
        if temp_id in target["issue_temp_ids"]:
            return (state.get("created_sprints") or {}).get(str(target["sprint_index"]))
    return None


async def commit_one_node(
    state: RolloutState, jira: JiraCommitClient, thread_id: str,
) -> dict:
    """Commit exactly one not-yet-committed issue under the epic. Checks the idempotency ledger first
    every time, so an issue already in `committed` is never sent again.
    """
    remaining = [i for i in state["issues"] if i.temp_id not in state["committed"]]
    if not remaining:
        return {"status": "committing", "failed_step": None}

    issue = remaining[0]
    try:
        issue_id, issue_key = await jira.create_issue(
            state["space_id"], issue, state["epic_issue_id"],
            _sprint_id_for_issue(state, issue.temp_id), state["user_id"], state["username"],
            idempotency_key=_idempotency_key(thread_id, "issue", issue.temp_id),
        )
    except JiraCommitError as exc:
        logger.error("rollout commit failed on temp_id=%s", issue.temp_id, extra={"error": str(exc)})
        return {"status": "failed", "error": str(exc), "failed_step": "commit_one"}

    committed = dict(state["committed"])
    committed[issue.temp_id] = issue_key
    committed_ids = dict(state.get("committed_issue_ids") or {})
    committed_ids[issue.temp_id] = issue_id
    return {
        "committed": committed,
        "committed_issue_ids": committed_ids,
        "status": "committing",
        "failed_step": None,
    }


def _after_commit_one(state: RolloutState) -> str:
    if state["status"] == "failed":
        return END
    return "link_one" if len(state["committed"]) == len(state["issues"]) else "commit_one"


def _dependency_pairs(state: RolloutState) -> list[tuple[str, str]]:
    # Links are published regardless of sprint placement (the old path skipped them all without
    # sprint targets). Old checkpoints may lack source ids for already-created children; only those
    # edges are skipped.
    # Only "blocks" edges become Jira links: `is blocked by` flags the card as blocked, too strong
    # for "better done later" edges. "after" edges are already honoured by ordering and sprint packing.
    known = {i.temp_id for i in state["issues"]}
    source_ids = set((state.get("committed_issue_ids") or {}).keys())
    return [
        (issue.temp_id, dependency_id)
        for issue in state["issues"]
        for dependency_id in issue.depends_on
        if dependency_id in known
        and issue.temp_id in source_ids
        and issue.dependency_kinds.get(dependency_id) == "blocks"
    ]


async def link_one_node(
    state: RolloutState, jira: JiraCommitClient, thread_id: str,
) -> dict:
    """Create one dependency link after every issue exists, then checkpoint its link id."""
    committed_links = dict(state.get("committed_links") or {})
    remaining = [
        pair for pair in _dependency_pairs(state)
        if f"{pair[0]}>{pair[1]}" not in committed_links
    ]
    if not remaining:
        return {"status": "committed", "failed_step": None}

    source_temp_id, dependency_temp_id = remaining[0]
    try:
        link_id = await jira.create_issue_link(
            state["committed_issue_ids"][source_temp_id],
            state["committed"][dependency_temp_id],
            state["user_id"],
            state["username"],
            idempotency_key=_idempotency_key(
                thread_id, "link", f"{source_temp_id}>{dependency_temp_id}",
            ),
        )
    except JiraCommitError as exc:
        logger.error(
            "rollout dependency link failed on %s>%s",
            source_temp_id,
            dependency_temp_id,
            extra={"error": str(exc)},
        )
        return {"status": "failed", "error": str(exc), "failed_step": "link_one"}

    committed_links[f"{source_temp_id}>{dependency_temp_id}"] = link_id
    return {
        "committed_links": committed_links,
        "status": "committing",
        "failed_step": None,
    }


def _after_link_one(state: RolloutState) -> str:
    if state["status"] in ("committed", "failed"):
        return END
    return "link_one"


def _after_plan(state: RolloutState) -> str:
    return END if state["status"] == "failed" else "review"


def build_rollout_graph(
    client,
    model: str,
    space_membership: SpaceMembershipChecker,
    jira: JiraCommitClient,
    planning_graph_getter,
    grounding_gatherer=None,
    history_assessor=None,
):
    """`planning_graph_getter` is a callable so app.state.planning_graph is read at run time, never
    captured stale at startup.
    """

    async def _plan(state: RolloutState) -> dict:
        return await plan_node(
            state, client, model, planning_graph_getter, grounding_gatherer, history_assessor,
        )

    async def _review(state: RolloutState) -> dict:
        return await review_node(state, space_membership, jira)

    async def _create_epic(state: RolloutState, config=None) -> dict:
        return await create_epic_node(state, jira, _thread_id_from_config(config))

    async def _commit_one(state: RolloutState, config=None) -> dict:
        return await commit_one_node(state, jira, _thread_id_from_config(config))

    async def _prepare_sprint(state: RolloutState, config=None) -> dict:
        return await prepare_sprint_node(state, jira, _thread_id_from_config(config))

    async def _link_one(state: RolloutState, config=None) -> dict:
        return await link_one_node(state, jira, _thread_id_from_config(config))

    graph = StateGraph(RolloutState)
    graph.add_node("plan", _plan)
    graph.add_node("review", _review)
    graph.add_node("create_epic", _create_epic)
    graph.add_node("prepare_sprint", _prepare_sprint)
    graph.add_node("commit_one", _commit_one)
    graph.add_node("link_one", _link_one)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _after_plan, {"review": "review", END: END})
    graph.add_conditional_edges("review", _after_review, {"create_epic": "create_epic", END: END})
    graph.add_conditional_edges(
        "create_epic", _after_create_epic, {"prepare_sprint": "prepare_sprint", END: END}
    )
    graph.add_conditional_edges(
        "prepare_sprint", _after_prepare_sprint,
        {"prepare_sprint": "prepare_sprint", "commit_one": "commit_one", END: END},
    )
    graph.add_conditional_edges(
        "commit_one", _after_commit_one, {"commit_one": "commit_one", "link_one": "link_one", END: END}
    )
    graph.add_conditional_edges("link_one", _after_link_one, {"link_one": "link_one", END: END})
    return graph


async def retry_rollout(compiled_graph, thread_id: str) -> dict:
    """Un-stick a rollout at "committing" or "failed".

    - "committing" (likely a crash): a task is still pending, so `ainvoke(None)` continues it.
      Reading status never does this (verified: a crashed thread stayed put across reads).
    - "failed" (clean failure; the graph ended, nothing pending): `ainvoke(None)` would do nothing.
      Re-arm with `aupdate_state(..., as_node=...)` — `as_node` is the node whose OUTGOING edge
      fires (verified on a 2-node test). So as "review" re-arms create_epic, and as "create_epic"
      re-arms commit_one.

    Callers must check the status first; other statuses aren't meaningful here.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = await compiled_graph.aget_state(config)
    if snap.values.get("status") == "committing":
        return await compiled_graph.ainvoke(None, config=config)

    failed_step = snap.values.get("failed_step")
    if failed_step is None:
        failed_step = "create_epic" if snap.values.get("epic_issue_id") is None else "commit_one"
    predecessor = {
        "create_epic": "review",
        "prepare_sprint": "create_epic",
        "commit_one": "prepare_sprint",
        "link_one": "commit_one",
    }[failed_step]
    await compiled_graph.aupdate_state(
        config, {"status": "committing", "error": None, "failed_step": None}, as_node=predecessor
    )
    return await compiled_graph.ainvoke(None, config=config)
