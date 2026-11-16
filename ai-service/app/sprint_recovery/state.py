"""State for the sprint-recovery workflow (app/sprint_recovery/graph.py). A plain TypedDict, like
RolloutState: only this workflow's nodes write it, and AsyncPostgresSaver must be able to checkpoint it.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict

from app.sprint_recovery.schemas import (
    EvidenceItem,
    RecoveryPlan,
    RiskSignal,
    RootCauseHypothesis,
    SprintSnapshot,
)

RecoveryStatus = Literal[
    "diagnosing",          # gathering evidence / generating hypotheses
    "no_risk_found",       # zero deterministic risk signals — healthy sprint, terminal, no LLM call made
    "awaiting_clarification",
    "awaiting_plan_approval",
    "revising",            # plan rejected with feedback, looping back into plan_node
    "committing",
    "committed",
    "waiting_reevaluation",  # actions committed, waiting on the automatic or a manual re-check
    "recovered",
    "escalated",
    "rejected",
    "failed",
]


class SprintRecoveryState(TypedDict):
    # --- Set once ---
    space_id: int
    sprint_id: int
    sprint_name: str
    user_id: str
    username: str
    max_clarification_rounds: int
    max_escalation_rounds: int
    max_token_budget: int

    # --- Diagnosis ---
    risk_signals: List[RiskSignal]
    evidence: List[EvidenceItem]
    hypotheses: List[RootCauseHypothesis]
    clarification_rounds: int
    clarification_question: Optional[str]
    clarification_answer: Optional[str]

    # --- Sprint-wide context (see SprintSnapshot: previously computed then discarded) ---
    sprint_snapshot: Optional[SprintSnapshot]

    # --- Plans / approval ---
    plans: List[RecoveryPlan]
    decision: Optional[Literal["approve", "edit", "reject", "revise"]]
    approved_plan: Optional[RecoveryPlan]
    # A human rejecting all plans with feedback loops back to plan_node instead of ending the run;
    # capped by max_plan_revision_rounds.
    human_plan_feedback: Optional[str]
    plan_revision_round: int
    max_plan_revision_rounds: int
    # What a human just said (clarification answer, rewind note, or revise feedback), kept only until
    # plan_node answers it. The persistent fields can't be used for this: later rounds would keep
    # answering the same old note.
    unanswered_human_note: Optional[str]
    # plan_node's reply to that note — what changed, or why it couldn't. Without it, a note that
    # couldn't change anything looked ignored. Reset on every plan_node run.
    note_response: Optional[str]

    # --- Execution (idempotency ledger) ---
    # New uuid4 from approval_node on every approval. The idempotency key used to be built from round
    # counters, which a time-travel rewind doesn't reset — so a different plan approved at the same
    # rounds reused the old keys, jira-backend replayed the old responses, and the new actions silently
    # never ran. None until the first approval.
    commit_attempt_id: Optional[str]
    committed_actions: Dict[str, str]  # action_index (str) -> result summary
    # issue_key -> `updatedAt` read back from jira-backend after our write. reevaluate waits for the
    # index to reach these instead of sleeping (see `_await_index_catch_up`).
    index_watermarks: Dict[str, str]
    # issue_key -> `updatedAt` before our write. Our write resets the "untouched for N days" clock, so
    # a priority bump alone used to produce "recovered". This baseline lets risk detection ignore our
    # own writes (see `own_writes` in `_detect_risk_signals`).
    pre_write_updated_at: Dict[str, str]
    # Set on every reevaluate: True if the index catch-up timed out. A stale read can only make the
    # sprint look more at risk, never less, so it's a caveat on at-risk results only.
    index_catch_up_timed_out: bool
    # Real provider-reported tokens (via `_call_model`). The pre-call budget check still uses an
    # estimate, since a call's size isn't known in advance.
    token_usage: int
    # Real $ for this thread (app/llm/pricing.py). It used to be a flat 4000 tokens per call.
    cost_usd: float

    # --- Escalation loop ---
    escalation_round: int
    completion_forecast_pct: Optional[float]
    # Plain-language list of actions committed in earlier rounds (never reset, unlike
    # committed_actions), so plan_node doesn't repeat them — round 2 once re-raised the same three
    # priorities to "highest".
    prior_committed_actions: List[str]

    status: RecoveryStatus
    error: Optional[str]
    # Username of whoever ended the check with stop_recovery (graph.py); shown to the next person
    # who opens the sprint.
    stopped_by: Optional[str]


def initial_recovery_state(
    space_id: int, sprint_id: int, sprint_name: str, user_id: str, username: str,
    max_clarification_rounds: int = 2, max_escalation_rounds: int = 2, max_token_budget: int = 200_000,
    max_plan_revision_rounds: int = 2,
) -> SprintRecoveryState:
    return SprintRecoveryState(
        space_id=space_id, sprint_id=sprint_id, sprint_name=sprint_name,
        user_id=user_id, username=username,
        max_clarification_rounds=max_clarification_rounds,
        max_escalation_rounds=max_escalation_rounds,
        max_token_budget=max_token_budget,
        risk_signals=[], evidence=[], hypotheses=[],
        clarification_rounds=0, clarification_question=None, clarification_answer=None,
        sprint_snapshot=None,
        plans=[], decision=None, approved_plan=None,
        human_plan_feedback=None, plan_revision_round=0, max_plan_revision_rounds=max_plan_revision_rounds,
        unanswered_human_note=None, note_response=None,
        commit_attempt_id=None,
        committed_actions={}, index_watermarks={}, pre_write_updated_at={},
        index_catch_up_timed_out=False, token_usage=0, cost_usd=0.0,
        escalation_round=0, completion_forecast_pct=None, prior_committed_actions=[],
        status="diagnosing", error=None, stopped_by=None,
    )
