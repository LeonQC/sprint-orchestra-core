"""Multi-agent epic planning: planner -> critic with a bounded revision loop, as a LangGraph
StateGraph. Alternative to service.py's single-call `plan_epic`; off by default
(`epic_planning_multiagent_enabled`).

LangGraph isn't required for this control flow (crag_loop does a similar loop in plain Python). It's
used because routing rules become data (`add_conditional_edges`), per-node events come free via
`astream_events`, and it's just orchestration over the same instructor client.

Not included on purpose: no checkpointer (each request builds a plan from scratch; the frontend
sends the whole plan on refine), and `refine_plan` stays outside the graph (a small edit doesn't need
a review loop).

The compiled graph is shared across requests; per-request data flows through state, and
client/model are bound into the node closures.

There used to be a separate estimator node. Measured: no gain at all (4.12 vs 4.12 on 8 proposals)
for a third of the cost, so it was removed and the planner estimates again
.
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.circuit_breaker import ModelUnavailableError
from app.planning.grounding import GroundingContext
from app.planning.pattern_schemas import EvidenceAssessment
from app.planning.schemas import EpicDraft, EpicPlanDraft, IssueDraft, PlanCritique
from app.planning.service import (
    PlanGenerationError,
    PlanResult,
    critique_plan,
    critique_to_instruction,
    plan_epic,
    plan_needs_revision,
    refine_plan,
)
from app.tracing import llm_step

logger = logging.getLogger(__name__)

# One revision pass (each costs two more LLM calls). On 8 proposals the critic asked for zero
# revisions, so a higher cap would only budget for a path never seen to run.
MAX_REVISION_ROUNDS = 1


class PlanningState(TypedDict):
    """Plain TypedDict, not a Pydantic model: nothing outside this module's own nodes ever writes it,
    so there's no untrusted boundary to validate. This codebase reserves Pydantic for exactly those
    boundaries (LLM output in `schemas.py`, HTTP bodies in `api/routes.py`) and uses plain
    dataclasses/dicts for internal bookkeeping (`PlanResult`, `SprintBucket`, `AgentAnswer`).
    """

    proposal: str
    existing_labels: List[str]
    # Retrieved once, read by both nodes. A critic seeing a different sample of history than the
    # planner would flag well-anchored estimates and waste the revision on its own priors.
    grounding: Optional[GroundingContext]
    #: The history verdict and patterns, extracted once and shown to both nodes — otherwise the critic
    #: flags pattern-driven issues as unmotivated.
    assessment: Optional[EvidenceAssessment]
    epic: Optional[EpicDraft]
    issues: List[IssueDraft]
    critique: Optional[PlanCritique]
    needs_revision: bool
    revision_round: int
    #: Summed across rounds, same as the token counters — a revision that invents a citation is
    #: exactly as worth surfacing as a first draft that does.
    dropped_estimate_references: int
    historically_derived_estimates: int
    degraded: bool
    # "planner_failed" (a revision round failed; the previous plan is kept) | "critic_failed" | None.
    # Running out of revision rounds isn't listed: it's not a failure, and `degraded` drives the
    # frontend's "this plan wasn't fully reviewed" warning.
    degraded_reason: Optional[str]
    error: Optional[str]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float


def _initial_state(
    proposal: str,
    existing_labels: Optional[List[str]],
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> PlanningState:
    return PlanningState(
        proposal=proposal,
        existing_labels=existing_labels or [],
        grounding=grounding,
        assessment=assessment,
        epic=None,
        issues=[],
        critique=None,
        needs_revision=False,
        revision_round=0,
        dropped_estimate_references=0,
        historically_derived_estimates=0,
        degraded=False,
        degraded_reason=None,
        error=None,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        estimated_cost_usd=0.0,
    )


async def planner_node(state: PlanningState, client, model: str) -> dict:
    """Round 0 makes a fresh plan with estimates (same as the single-call path). Later rounds reuse
    `refine_plan`, with the critique turned into the same kind of instruction a human would type, so
    there's no second planning prompt to keep in sync.
    """
    critique = state.get("critique")
    grounding = state.get("grounding")
    assessment = state.get("assessment")
    if critique is None:
        result = await plan_epic(
            client, model, state["proposal"], state["existing_labels"], grounding, assessment,
        )
        revision_round = state["revision_round"]
    else:
        revision_round = state["revision_round"] + 1
        try:
            with llm_step(f"plan_epic.revise_round_{revision_round}"):
                result = await refine_plan(
                    client,
                    model,
                    state["epic"],
                    state["issues"],
                    critique_to_instruction(critique),
                    state["existing_labels"],
                    grounding,
                    assessment,
                )
        except (ModelUnavailableError, PlanGenerationError) as exc:
            # Round 0's plan is real; a failed revision shouldn't throw it away.
            result = PlanResult(
                plan=EpicPlanDraft(epic=state["epic"], issues=state["issues"]),
                degraded=True, latency_seconds=0.0, error=str(exc),
            )

    update = {
        "epic": result.plan.epic,
        "issues": result.plan.issues,
        "revision_round": revision_round,
        "dropped_estimate_references":
            state["dropped_estimate_references"] + result.dropped_estimate_references,
        "historically_derived_estimates":
            state["historically_derived_estimates"] + result.historically_derived_estimates,
        # Consumed — a stale critique here would make the next planner_node call think it's revising
        # again, and would leak into the loop-back condition.
        "critique": None,
        "needs_revision": False,
        "input_tokens": state["input_tokens"] + result.input_tokens,
        "output_tokens": state["output_tokens"] + result.output_tokens,
        "cached_input_tokens": state["cached_input_tokens"] + result.cached_input_tokens,
        "estimated_cost_usd": state["estimated_cost_usd"] + result.estimated_cost_usd,
    }
    if result.degraded:
        update.update(degraded=True, degraded_reason="planner_failed", error=result.error)
    return update


async def critic_node(state: PlanningState, client, model: str) -> dict:
    # Round N's critique feeds revise_round_N.
    with llm_step(f"plan_epic.critique_round_{state['revision_round'] + 1}"):
        result = await critique_plan(
            client, model, state["proposal"], state["epic"], state["issues"], state.get("grounding"),
            state.get("assessment"),
        )
    update = {
        "critique": result.critique,
        "needs_revision": plan_needs_revision(result.critique),
        "input_tokens": state["input_tokens"] + result.input_tokens,
        "output_tokens": state["output_tokens"] + result.output_tokens,
        "cached_input_tokens": state["cached_input_tokens"] + result.cached_input_tokens,
        "estimated_cost_usd": state["estimated_cost_usd"] + result.estimated_cost_usd,
    }
    if result.error is not None:
        update.update(degraded=True, degraded_reason="critic_failed", error=result.error)
    return update


def _after_planner(state: PlanningState) -> str:
    if state["degraded_reason"] == "planner_failed":
        return END  # the kept plan was already reviewed once
    return "critic"


def _after_critic(state: PlanningState) -> str:
    if state["degraded_reason"] == "critic_failed" or not state["needs_revision"]:
        return END
    if state["revision_round"] >= MAX_REVISION_ROUNDS:
        # Deliberately not `degraded=True` — see PlanningState.degraded_reason's comment.
        logger.info(
            "plan critique still had findings at the revision cap; returning the current plan",
            extra={"revision_round": state["revision_round"]},
        )
        return END
    return "planner"


def build_planning_graph(client, model: str):
    """Compile once at startup (app/main.py) and reuse."""

    async def _planner(state: PlanningState) -> dict:
        return await planner_node(state, client, model)

    async def _critic(state: PlanningState) -> dict:
        return await critic_node(state, client, model)

    graph = StateGraph(PlanningState)
    graph.add_node("planner", _planner)
    graph.add_node("critic", _critic)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", _after_planner, {"critic": "critic", END: END})
    graph.add_conditional_edges("critic", _after_critic, {"planner": "planner", END: END})
    return graph.compile()


async def plan_epic_multiagent(
    compiled_graph,
    proposal: str,
    existing_labels: Optional[List[str]] = None,
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> PlanResult:
    """Return the same PlanResult shape as `plan_epic`, so everything downstream and the frontend
    contract are identical either way.
    """
    from app.observability import PLANNING_REVISION_ROUNDS

    start = time.perf_counter()
    final = await compiled_graph.ainvoke(
        _initial_state(proposal, existing_labels, grounding, assessment)
    )
    PLANNING_REVISION_ROUNDS.observe(final["revision_round"])
    return PlanResult(
        plan=EpicPlanDraft(epic=final["epic"], issues=final["issues"]),
        degraded=final["degraded"],
        latency_seconds=time.perf_counter() - start,
        error=final["error"],
        input_tokens=final["input_tokens"],
        output_tokens=final["output_tokens"],
        cached_input_tokens=final["cached_input_tokens"],
        estimated_cost_usd=final["estimated_cost_usd"],
        dropped_estimate_references=final["dropped_estimate_references"],
        historically_derived_estimates=final["historically_derived_estimates"],
    )
