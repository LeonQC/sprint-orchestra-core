"""State for the durable epic-rollout workflow (app/planning/rollout_graph.py).

A plain TypedDict, not Pydantic: only this workflow's nodes write it (API input is validated by the
route's own models first), and AsyncPostgresSaver checkpoints it on every step — primitives plus
EpicDraft/IssueDraft, which it already knows how to serialize.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from app.planning.schemas import EpicDraft, IssueDraft

RolloutStatus = Literal[
    "pending_approval",  # paused at the interrupt; nothing committed yet
    "committing",         # decision received, commit node is writing to Jira
    "committed",          # every issue in the (possibly edited) plan now has a real issue_key
    "rejected",           # human rejected; nothing was ever written
    "failed",             # a step failed (`failed_step`; "plan" = no plan was produced, see `error`)
]

RolloutDecision = Literal["approve", "edit", "reject"]


class SprintBucketState(TypedDict):
    sprint_index: int
    issue_temp_ids: List[str]
    total_points: int


class SprintTargetState(TypedDict):
    """The human's final destination for one non-empty planning bucket.

    Kept separate from ``SprintBucketState`` because bin-packing answers *which issues belong
    together*, while this answers *where that bucket is published*: an existing sprint or a new
    sprint created as part of the durable workflow.
    """

    sprint_index: int
    issue_temp_ids: List[str]
    mode: Literal["existing", "new"]
    sprint_id: Optional[int]
    sprint_name: Optional[str]


class GroundingExemplarState(TypedDict):
    """One past issue the planner was shown, flattened to primitives.

    Deliberately not the `app.planning.grounding.Exemplar` dataclass itself: this dict is what
    `AsyncPostgresSaver` writes into the checkpoint row, and the module docstring's rule is
    primitives plus the Pydantic models the checkpointer already handles. It also has to survive a
    process restart and still render — a rollout paused for approval overnight must still be able to
    show the reviewer which issues the estimates were compared against, and re-retrieving them at
    resume time would show a *different* set than the plan was actually written from.
    """

    issue_key: str
    title: str
    issue_type: Optional[str]
    status: Optional[str]
    story_points: Optional[int]
    confidence: float
    estimate_anchor: bool
    source: str


class HouseFormatState(TypedDict):
    """The detected description layout, flattened for the checkpoint."""

    sections: List[str]
    item_marker: Optional[str]
    item_count_low: Optional[int]
    item_count_high: Optional[int]
    item_count_max: Optional[int]
    coverage: Dict[str, float]
    sample_size: int


class GroundingState(TypedDict):
    """Field-for-field the shape `app/api/routes.py::PlanGroundingOut` renders, so the durable path
    and the plain `POST /plan-epic` path show a reviewer exactly the same receipts.

    "Field-for-field" is load-bearing rather than descriptive: `_rollout_status_response` splats this
    dict straight into that model, so a field added to the model and not here makes every rollout
    status call fail Pydantic validation. That happened — `house_format` was added to the response and
    not to the checkpoint, and the SSE stream returned its progress events and then died before the
    result frame, which reaches the browser as a bare "network error".
    """

    status: str
    exemplars: List[GroundingExemplarState]
    house_format: HouseFormatState
    estimate_anchor_count: int
    dropped_estimate_references: int


class EvidenceState(TypedDict):
    """The history verdict and the patterns it produced, flattened to primitives for the checkpoint.

    Same reasoning as `GroundingExemplarState`: this is what `AsyncPostgresSaver` writes into the
    row, and a rollout paused for approval overnight must still be able to tell the reviewer WHY the
    plan looks like it does — re-deriving it at resume time would show a different reading than the
    plan was actually written from.
    """

    strength: str
    reasoning: str
    relevant_issue_keys: List[str]
    patterns: List[dict]
    dropped_patterns: int
    downgraded: bool


class RolloutState(TypedDict):
    # --- Set once, at the start node, never mutated afterward ---
    proposal: str
    #: The requirement bullets the plan was generated from, kept so a resume re-plans against the same
    #: stated scope rather than whatever the proposal text alone implies.
    requirements: List[str]
    existing_labels: List[str]
    sprint_capacity_points: Optional[float]
    target_sprint_count: Optional[int]
    space_id: int
    user_id: str
    username: str

    # --- Written by plan_node; may be re-written by an "edit" decision at resume time ---
    epic: Optional[EpicDraft]
    issues: List[IssueDraft]
    sprint_buckets: List[SprintBucketState]
    sprint_targets: List[SprintTargetState]
    # What the plan was grounded in, recorded at generation time. Not re-derived later: see
    # GroundingExemplarState's docstring on why a resume must show the original set.
    grounding: GroundingState
    #: What reading that history concluded, and the patterns extracted from it.
    evidence: EvidenceState

    # --- Written by the interrupt's resume payload ---
    decision: Optional[RolloutDecision]

    # --- Written once by create_epic_node; checked before every commit_one_node entry (including a
    # post-crash resume) so the epic issue itself is never created twice. ---
    epic_issue_id: Optional[int]
    epic_issue_key: Optional[str]

    # One entry is checkpointed after each sprint destination is resolved. For an existing sprint
    # this records the selected id; for a new sprint it records the id returned by jira-backend.
    created_sprints: Dict[str, int]  # sprint_index -> real sprint id

    # --- Idempotency ledger, written by commit_one_node, keyed by temp_id. Checked before every
    # create (including after a crash), so nothing is created twice. ---
    committed: Dict[str, str]  # temp_id -> real issue_key
    committed_issue_ids: Dict[str, int]  # temp_id -> jira-backend issue id

    # Dependency links are side effects too, so they get their own per-link checkpoint ledger.
    committed_links: Dict[str, int]  # "source_temp_id>dependency_temp_id" -> issue-link id

    # --- Status/error bookkeeping ---
    status: RolloutStatus
    degraded: bool
    error: Optional[str]
    failed_step: Optional[Literal["plan", "create_epic", "prepare_sprint", "commit_one", "link_one"]]
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float
    planning_latency_seconds: float


def initial_rollout_state(
    proposal: str,
    existing_labels: List[str],
    sprint_capacity_points: Optional[float],
    target_sprint_count: Optional[int],
    space_id: int,
    user_id: str,
    username: str,
    # Last and defaulted so every existing call site keeps working unchanged; a rollout started
    # without stated requirements plans from the proposal exactly as it did before.
    requirements: Optional[List[str]] = None,
) -> RolloutState:
    return RolloutState(
        proposal=proposal,
        requirements=list(requirements or []),
        existing_labels=existing_labels,
        sprint_capacity_points=sprint_capacity_points,
        target_sprint_count=target_sprint_count,
        space_id=space_id,
        user_id=user_id,
        username=username,
        epic=None,
        issues=[],
        sprint_buckets=[],
        sprint_targets=[],
        grounding=GroundingState(
            status="disabled", exemplars=[],
            house_format=HouseFormatState(
                sections=[], item_marker=None, item_count_low=None, item_count_high=None,
                item_count_max=None, coverage={}, sample_size=0,
            ),
            estimate_anchor_count=0, dropped_estimate_references=0,
        ),
        evidence=EvidenceState(
            strength="unassessed", reasoning="", relevant_issue_keys=[], patterns=[],
            dropped_patterns=0, downgraded=False,
        ),
        decision=None,
        epic_issue_id=None,
        epic_issue_key=None,
        created_sprints={},
        committed={},
        committed_issue_ids={},
        committed_links={},
        status="pending_approval",
        degraded=False,
        error=None,
        failed_step=None,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        estimated_cost_usd=0.0,
        planning_latency_seconds=0.0,
    )
