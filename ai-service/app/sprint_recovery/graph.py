"""Sprint recovery workflow: diagnose -> gather evidence -> grounded hypotheses -> clarify if unsure
-> propose plans -> durable human approval -> idempotent execution -> wait and re-evaluate ->
escalate or close.

LangGraph isn't strictly required, but it earns more here than in the rollout workflow:
  1. Several differently-shaped pauses (clarify, approve plan, wait for re-evaluation), all using the
     same `interrupt()` protocol instead of a status value and resume handler per shape.
  2. A real conditional replan loop ("new plans given what happened"), not just a retry.

Grounding: each hypothesis must cite evidence ids gathered this run (like crag_loop's
`_ground_answer`), but `_apply_grounding` drops only the ungrounded hypothesis, not everything.

Loop limits: `max_clarification_rounds`, `max_escalation_rounds`, and `max_token_budget` (checked
before every LLM call).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from contextvars import ContextVar
from datetime import date, datetime, timezone
from typing import Callable, Dict, List, Optional

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openinference.instrumentation import using_tags
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# One span per node run, nested under the route's span. thread_id is on every span because one
# workflow spans several HTTP calls (sometimes days apart), so no single trace covers it.
_tracer = trace.get_tracer(__name__)

# Optional SSE progress callback, per request. A ContextVar because the compiled graph is shared;
# the streaming endpoint sets it before create_task (which copies the context). Unset = no-op.
ON_STAGE_VAR: ContextVar[Optional[Callable[[str], None]]] = ContextVar("sprint_recovery_on_stage", default=None)

# Run identity for `commit_one_action_node`'s idempotency key (see state.py `commit_attempt_id`),
# without threading thread_id through every node closure.
THREAD_ID_VAR: ContextVar[str] = ContextVar("sprint_recovery_thread_id", default="unknown")

# Labels for the only two nodes that call an LLM; the rest are pauses or fast Jira calls.
_STAGE_LABELS = {
    "diagnose": "detecting risk signals and analyzing evidence",
    "plan": "generating recovery plans",
}

# Plain-English names for `RiskSignal.signal_type`, shown in the UI's evidence panel and tooltips
# (the raw enum once showed up there as "low_completion_forecast: 0/3 issues done").
_SIGNAL_LABELS = {
    "blocked_no_flag": "Marked as blocked",
    "long_in_progress": "Stuck in progress",
    "owner_overloaded": "Assignee overloaded",
    "no_dependency_recorded": "Undocumented dependency",
    "scope_added_late": "Scope added late in the sprint",
    "low_completion_forecast": "Behind on completion",
}

# Issue keys mentioned in free text (e.g. "waiting on PAY-97"), same pattern as crag_loop. Needed
# because an undocumented dependency points at an issue that was never flagged itself.
_ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]*-\d+\b", re.IGNORECASE)

from app.auth.space_membership import SpaceMembershipChecker, SpaceMembershipError
from app.llm.circuit_breaker import ModelUnavailableError, is_model_unavailable
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.llm.types import Usage
from app.llm.usage_ledger import FEATURE_SPRINT_RECOVERY, Caller, UsageLedger, charge_for
from app.observability import get_request_id
from app.sprint_recovery.jira_actions_client import JiraActionError, JiraActionsClient
from app.sprint_recovery.prompts import render_diagnose_prompt, render_plan_prompt
from app.sprint_recovery.schemas import (
    DiagnosisResult,
    EvidenceItem,
    RecoveryAction,
    RecoveryPlan,
    RecoveryPlanSet,
    RiskSignal,
    RootCauseHypothesis,
    SprintIssueSummary,
    SprintSnapshot,
)
from app.sprint_recovery.state import SprintRecoveryState
from app.tracing import llm_step

logger = logging.getLogger(__name__)

# Loop counts come from the state (max_clarification_rounds / max_escalation_rounds), not here.
_TOKENS_PER_LLM_CALL_ESTIMATE = 4000  # coarse, pre-call budget check — see _check_budget's docstring

# `(usage, cost_usd) -> None`, wired to ServiceStats.record so this workflow's spend shows up in
# GET /stats and the Prometheus cost counter. A plain argument (not a ContextVar): it's process-wide.
UsageRecorder = Callable[[Usage, float], None]


def _check_budget(state: SprintRecoveryState) -> bool:
    """True if there's room for one more LLM call.

    Uses an estimate because real usage is only known after the call, and the point is to stop
    before spending. Recorded usage is the real number (see `_call_model`).
    """
    return state["token_usage"] + _TOKENS_PER_LLM_CALL_ESTIMATE <= state["max_token_budget"]


# Moved to app/llm/instructor_usage.py (a second copy elsewhere ignored OpenAI cached tokens and
# overstated cost). Aliased so existing call sites and tests are unchanged.
_usage_from_completion = usage_from_completion


def _charge_recorder(state: "SprintRecoveryState", model: str, runtime: tuple, step: str):
    """Record this workflow's spend with a feature, user and session, like any HTTP feature.

    Without it, recovery spend reached /stats but had no feature or user, so a per-feature or per-user
    cost breakdown couldn't see it. The session is the workflow (one thread_id = one recovery).
    """
    provider, deployment = runtime

    def _record(usage: Usage) -> None:
        UsageLedger(
            feature=FEATURE_SPRINT_RECOVERY,
            caller=Caller(
                user_id=state.get("user_id") or "",
                username=state.get("username") or "",
                session_id=f"recovery-{THREAD_ID_VAR.get()}",
            ),
            request_id=get_request_id() or "",
            charges=[charge_for(step, provider, model, deployment, usage)],
            detail={
                "step": step,
                "space_id": state.get("space_id"),
                "sprint_id": state.get("sprint_id"),
                "sprint_name": state.get("sprint_name") or "",
                "escalation_round": state.get("escalation_round"),
            },
        ).attach()

    return _record


async def _call_model(
    client, model: str, response_model, prompt: str, on_usage: Optional[UsageRecorder] = None,
    on_charge: Optional[Callable[[Usage], None]] = None, step: str = "call",
):
    """One structured LLM call, returning `(parsed, tokens, cost_usd)` with real token counts.

    Nodes used to add a flat estimate per call, so reported usage was a guess. Known gap: instructor
    may retry a bad parse (max_retries=2) and only the last response's usage is counted, so retries
    are slightly undercounted.
    """
    # Tag the SDK's own LLM span too (same reason as routes._tagged).
    with using_tags([FEATURE_SPRINT_RECOVERY]):
        with llm_step(f"sprint_recovery.{step}"):
            try:
                parsed, completion = await client.chat.completions.create_with_completion(
                    model=model, response_model=response_model, max_retries=2,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                if is_model_unavailable(exc):
                    raise ModelUnavailableError() from exc
                raise
    usage = _usage_from_completion(completion)
    cost = estimate_cost_usd(
        served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
        usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
    )
    if on_usage is not None:
        try:
            on_usage(usage, cost)
        except Exception:  # noqa: BLE001 - cost bookkeeping must never break a real workflow
            logger.exception("failed recording sprint-recovery LLM usage")
    if on_charge is not None:
        try:
            on_charge(usage)
        except Exception:  # noqa: BLE001 - same rule: attribution never breaks the workflow
            logger.exception("failed attributing sprint-recovery LLM usage")
    tokens = usage.input_tokens + usage.output_tokens + usage.cache_read_input_tokens
    # A provider reporting 0 tokens would make the budget unenforceable, so fall back to the estimate.
    return parsed, tokens or _TOKENS_PER_LLM_CALL_ESTIMATE, cost


async def _gather_evidence(
    retrieval, space_id: int, sprint_id: int, flagged_issue_keys: List[str],
) -> List[EvidenceItem]:
    """Fetch comments, descriptions, attachments and history for the flagged issues — no ranking.
    There's no model in the loop yet to decide whether to search more.
    """
    evidence: List[EvidenceItem] = []
    n = 0
    if not flagged_issue_keys:
        return evidence
    comments = await retrieval.get_issue_comments([space_id], flagged_issue_keys)
    for c in comments.comments:
        n += 1
        evidence.append(EvidenceItem(
            citation_id=f"ev{n}", issue_key=c["issue_key"], source_type="comment", content=c["content"],
        ))
    details = await retrieval.get_issue_details([space_id], flagged_issue_keys)
    for d in details.details:
        n += 1
        evidence.append(EvidenceItem(
            citation_id=f"ev{n}", issue_key=d["issue_key"], source_type="description", content=d["content"],
        ))
    attachments = await retrieval.get_issue_attachments([space_id], flagged_issue_keys)
    for a in attachments.attachments:
        n += 1
        evidence.append(EvidenceItem(
            citation_id=f"ev{n}", issue_key=a["issue_key"], source_type="attachment", content=a["content"],
        ))
    history = await retrieval.query_issue_history([space_id], {"issue_keys": flagged_issue_keys, "limit": 50})
    for h in history.changes:
        # Non-field events (issue_created, code_link_added, ...) have null field/from/to and their text
        # in `description`; rendering them with the field template produced "None: None -> None".
        # issueOrder changes are backlog drag-and-drop noise.
        field_name = h.get("field_name")
        if field_name == "issueOrder":
            continue
        if field_name:
            content = f"{field_name}: {h['from_value']} -> {h['to_value']} ({h['changed_at']})"
        elif h.get("description"):
            content = f"{h.get('event_type', 'change')}: {h['description']} ({h['changed_at']})"
        else:
            continue  # neither a field change nor a description — nothing to say, don't manufacture evidence
        n += 1
        evidence.append(EvidenceItem(
            citation_id=f"ev{n}", issue_key=h["issue_key"], source_type="history", content=content,
        ))
    return evidence


_STALE_AFTER_DAYS = 2  # in_progress/blocked with no update for longer than this — see docstring below
# `owner_overloaded` thresholds — product judgements, named so they're easy to tune.
# Points-based when anything is estimated: one person holding this share of the sprint's open points
# is overloaded, however many tickets that is (one 13-pointer can block more than five 1-pointers).
_OVERLOADED_POINTS_SHARE = 0.4
# Needs at least this many open issues, so "the last person with work left" isn't flagged.
_OVERLOADED_MIN_ISSUES = 2
# Fallback when nothing in the sprint is estimated.
_OVERLOADED_OPEN_ISSUES = 3
# A move into this sprint after this fraction of it has elapsed counts as scope added late.
_SCOPE_ADDED_LATE_ELAPSED_FRAC = 0.5
# Behind-schedule threshold: more than this many points behind the time-based expectation (e.g. 30%
# of the sprint gone -> ~30% done expected). Pace Check uses gentler bands; this is a stronger signal.
_COMPLETION_LAG_PERCENTAGE_POINTS = 20


def _days_since(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400


def _parse_iso_date(iso_ts: Optional[str]) -> Optional[date]:
    if not iso_ts:
        return None
    try:
        return datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def _sprint_context(retrieval, space_id: int, sprint_id: int) -> dict:
    """Best-effort lookup of the sprint's own record: start/end dates and the sprint goal.

    Never raises — every consumer falls back to a date/goal-independent rule. The goal matters: plans
    have an `impact_on_goal` field, but the model was never shown the goal and was guessing it from
    ticket titles.
    """
    try:
        result = await retrieval.query_sprints([space_id], {"limit": 200})
        for sprint in result.sprints:
            if sprint.get("sprint_id") == sprint_id:
                # Dates arrive as JSON strings, not `date` objects.
                return {
                    "start_date": date.fromisoformat(str(sprint["start_date"])[:10]) if sprint.get("start_date") else None,
                    "end_date": date.fromisoformat(str(sprint["end_date"])[:10]) if sprint.get("end_date") else None,
                    "goal": sprint.get("goal"),
                    "sprint_name": sprint.get("sprint_name"),
                }
    except Exception:  # noqa: BLE001 - supplementary context, not load-bearing
        return {}
    return {}


# Ceiling for `_await_index_catch_up` (normally satisfied on the first poll), not a guessed delay.
_INDEX_CATCH_UP_TIMEOUT_SECONDS = 10.0
_INDEX_CATCH_UP_POLL_SECONDS = 0.25


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


async def _await_index_catch_up(retrieval, space_id: int, sprint_id: int, watermarks: Dict[str, str]) -> bool:
    """Wait until vectorization-service's index reflects the writes this workflow just made, or until
    the timeout. Returns True if it caught up.

    jira-backend publishes to Kafka and the index updates asynchronously, so a re-check right after our
    writes can read a stale sprint. (A flat 2s sleep used to cover this, and was a guess.)

    Signal: for each issue we wrote, `commit_one_action_node` saved jira-backend's `updatedAt`. Done
    when every such issue's indexed `updated_at` is at least that new, or the issue has left the sprint
    (what a committed `move_out_of_sprint` looks like). On timeout, read anyway.
    """
    if not watermarks:
        return True
    deadline = time.monotonic() + _INDEX_CATCH_UP_TIMEOUT_SECONDS
    while True:
        try:
            result = await retrieval.query_issues([space_id], {"sprint_ids": [sprint_id], "limit": 200})
            indexed = {row["issue_key"]: row.get("updated_at") for row in result.issues}
            caught_up = all(
                key not in indexed or _is_at_least(_parse_ts(indexed.get(key)), _parse_ts(expected))
                for key, expected in watermarks.items()
            )
        except Exception:  # noqa: BLE001 - one flaky poll shouldn't end the wait; the timeout still
            # bounds it, and a real outage surfaces in the next _detect_risk_signals call.
            logger.warning(
                "sprint-recovery index catch-up poll failed, retrying within the timeout budget",
                exc_info=True, extra={"space_id": space_id, "sprint_id": sprint_id},
            )
            caught_up = False
        if caught_up:
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "sprint-recovery read model did not catch up before the timeout; reading anyway",
                extra={"space_id": space_id, "sprint_id": sprint_id, "watermarks": watermarks},
            )
            return False
        await asyncio.sleep(_INDEX_CATCH_UP_POLL_SECONDS)


def _is_at_least(indexed: Optional[datetime], expected: Optional[datetime]) -> bool:
    """A missing/unparseable timestamp can't prove staleness, so treat it as caught up."""
    if indexed is None or expected is None:
        return True
    return indexed >= expected


def _last_update_not_by_this_workflow(issue: dict, own_writes: Optional[Dict[str, tuple]]) -> Optional[str]:
    """The issue's `updated_at`, unless the last update was this workflow's own — then the time before
    our write.

    jira-backend stamps `updatedAt = now()` on every write, so committing a priority change on a
    6-day-stale ticket made it look fresh, both staleness signals vanished, and the sprint reported
    "recovered" without any real progress. `own_writes` maps issue_key -> (before, after) from
    `commit_one_action_node`; if the index is newer than our write, a human touched it since.
    """
    indexed = issue.get("updated_at")
    if not own_writes:
        return indexed
    entry = own_writes.get(issue.get("issue_key"))
    if not entry:
        return indexed
    before, after = entry
    if not before:
        return indexed
    indexed_ts, after_ts = _parse_ts(indexed), _parse_ts(after)
    if indexed_ts is not None and after_ts is not None and indexed_ts > after_ts:
        return indexed  # somebody real updated it after we did
    return before


async def _detect_risk_signals(
    retrieval, space_id: int, sprint_id: int, own_writes: Optional[Dict[str, tuple]] = None,
) -> List[RiskSignal]:
    """Compute risk signals in code — the LLM only explains them.

    - long_in_progress / blocked_no_flag: only when the issue hasn't been updated for
      `_STALE_AFTER_DAYS` (a ticket moved five minutes ago isn't a risk). This is also how "the owner
      isn't updating the ticket" is detected.
    - owner_overloaded: see the `_OVERLOADED_*` thresholds.
    - low_completion_forecast: compared with elapsed time, not a flat cutoff (40% done on day 3 of 10
      is ahead, not behind). Fires when > 20 points behind schedule after the first fifth of the
      sprint, or < 90% done with 2 days left.
    - scope_added_late: an issue moved into this sprint after `_SCOPE_ADDED_LATE_ELAPSED_FRAC` of it.
    """
    result = await retrieval.query_issues([space_id], {"sprint_ids": [sprint_id], "limit": 200})
    sprint_ctx = await _sprint_context(retrieval, space_id, sprint_id)
    signals: List[RiskSignal] = []
    # issue_key -> owner, so later prompts can name a person. Unassigned is stated, not omitted.
    unfinished_by_owner: Dict[str, List[str]] = {}
    unfinished_points_by_owner: Dict[str, int] = {}
    total_unfinished_points = 0

    for issue in result.issues:
        status = (issue.get("status") or "").lower()
        owner = issue.get("assignee_name")
        owner_note = f" Owner: {owner}." if owner else " No one is assigned to it."
        if status not in ("done", "completed"):
            points = issue.get("story_points") or 0
            total_unfinished_points += points
            if owner:
                unfinished_by_owner.setdefault(owner, []).append(issue["issue_key"])
                unfinished_points_by_owner[owner] = unfinished_points_by_owner.get(owner, 0) + points

        days_stale = _days_since(_last_update_not_by_this_workflow(issue, own_writes))
        if days_stale is None or days_stale < _STALE_AFTER_DAYS:
            continue  # too fresh to call a problem yet, regardless of status
        if status in ("in_progress", "in progress"):
            signals.append(RiskSignal(
                issue_key=issue["issue_key"], signal_type="long_in_progress",
                description=(
                    f"{issue['issue_key']} has been in_progress for {days_stale:.0f} day(s) with no "
                    f"update — the owner may not be reporting status.{owner_note}"
                ),
            ))
        elif status == "blocked":
            # `blocked` is a real board status, but nothing used to set this signal for it.
            signals.append(RiskSignal(
                issue_key=issue["issue_key"], signal_type="blocked_no_flag",
                description=(
                    f"{issue['issue_key']} has been blocked for {days_stale:.0f} day(s) with no "
                    f"update.{owner_note}"
                ),
            ))

    # One person holding most of the open work is a common risk, and the actionable form names who.
    use_points = total_unfinished_points > 0
    for owner, keys in sorted(unfinished_by_owner.items()):
        overloaded = False
        detail = ""
        if use_points:
            share = unfinished_points_by_owner[owner] / total_unfinished_points
            if share >= _OVERLOADED_POINTS_SHARE and len(keys) >= _OVERLOADED_MIN_ISSUES:
                overloaded = True
                detail = (
                    f"{owner} is carrying {unfinished_points_by_owner[owner]}/{total_unfinished_points} "
                    f"story points ({share:.0%}) of this sprint's unfinished work, across "
                    f"{len(keys)} issues ({', '.join(sorted(keys))}) — more than one person can "
                    "realistically land."
                )
        elif len(keys) >= _OVERLOADED_OPEN_ISSUES:
            overloaded = True
            detail = (
                f"{owner} is carrying {len(keys)} unfinished issues in this sprint "
                f"({', '.join(sorted(keys))}, no story points estimated) — more than one person can "
                "realistically land."
            )
        if overloaded:
            signals.append(RiskSignal(
                # Attach to one of their tickets so plan_node has something to act on.
                issue_key=sorted(keys)[0], signal_type="owner_overloaded", description=detail,
            ))

    start_date, end_date = sprint_ctx.get("start_date"), sprint_ctx.get("end_date")

    total = result.total_count
    done = result.counts_by_status.get("done", 0) + result.counts_by_status.get("completed", 0)
    if total > 0:
        # Story points, not issue count: 8 trivial tickets done out of 10 isn't 80% of the work. Falls
        # back to count when nothing is estimated.
        total_points = sum(i.get("story_points") or 0 for i in result.issues)
        done_points = sum(
            i.get("story_points") or 0 for i in result.issues
            if (i.get("status") or "").lower() in ("done", "completed")
        )
        if total_points > 0:
            forecast = round(100 * done_points / total_points, 1)
            basis = f"{done_points}/{total_points} story points done ({forecast}%)"
        else:
            forecast = round(100 * done / total, 1)
            basis = f"{done}/{total} issues done ({forecast}%, no story points estimated)"

        days_left = (end_date - date.today()).days if end_date else None
        duration_days = (end_date - start_date).days if start_date and end_date else None
        elapsed_frac = (
            max(0.0, min(1.0, (date.today() - start_date).days / duration_days))
            if duration_days else None
        )
        expected_pct = round(100 * elapsed_frac, 1) if elapsed_frac is not None else None
        schedule_gap = round(expected_pct - forecast, 1) if expected_pct is not None else None
        behind_schedule = (
            elapsed_frac is not None
            and elapsed_frac >= 0.2
            and schedule_gap is not None
            and schedule_gap > _COMPLETION_LAG_PERCENTAGE_POINTS
        )
        near_deadline_and_short = days_left is not None and days_left <= 2 and forecast < 90
        if behind_schedule or near_deadline_and_short:
            schedule_note = (
                f" — about {expected_pct:.0f}% would be expected by now, so this is "
                f"{schedule_gap:.0f} percentage points behind schedule"
                if behind_schedule and expected_pct is not None and schedule_gap is not None else ""
            )
            deadline_note = (
                f" — only {max(days_left, 0)} day(s) left and less than 90% complete"
                if near_deadline_and_short else ""
            )
            signals.append(RiskSignal(
                signal_type="low_completion_forecast",
                description=f"{basis}{schedule_note}{deadline_note}.",
            ))

    # scope_added_late: an issue moved into this sprint late (e.g. two "quick" tickets dragged in on
    # day 7 of 10). The other signals would blame the team; this one names the real cause.
    sprint_name = sprint_ctx.get("sprint_name")
    duration_days = (end_date - start_date).days if start_date and end_date else None
    if duration_days and sprint_name and result.issues:
        try:
            history = await retrieval.query_issue_history(
                [space_id], {"issue_keys": [i["issue_key"] for i in result.issues], "fields": ["sprint"], "limit": 200},
            )
            latest_add_by_issue: Dict[str, date] = {}
            for change in history.changes:
                if change.get("to_value") != sprint_name:
                    continue
                changed_at = _parse_iso_date(change.get("changed_at"))
                if changed_at is None:
                    continue
                issue_key = change.get("issue_key")
                if issue_key and (issue_key not in latest_add_by_issue or changed_at > latest_add_by_issue[issue_key]):
                    latest_add_by_issue[issue_key] = changed_at
            for issue_key, added_on in sorted(latest_add_by_issue.items()):
                elapsed_at_add = max(0.0, min(1.0, (added_on - start_date).days / duration_days))
                if elapsed_at_add >= _SCOPE_ADDED_LATE_ELAPSED_FRAC:
                    signals.append(RiskSignal(
                        issue_key=issue_key, signal_type="scope_added_late",
                        description=(
                            f"{issue_key} was added to {sprint_name} on {added_on.isoformat()}, "
                            f"{elapsed_at_add:.0%} of the way through the sprint — scope added this "
                            "late is a common reason a sprint misses its goal independent of how the "
                            "originally committed work is going."
                        ),
                    ))
        except Exception:  # noqa: BLE001 - supplementary signal, must not break diagnosis outright
            pass

    return signals


def _apply_grounding(hypotheses: List[RootCauseHypothesis], evidence: List[EvidenceItem]) -> List[RootCauseHypothesis]:
    """Drop any hypothesis citing an evidence id that wasn't gathered. Per hypothesis (unlike
    crag_loop's whole-answer reject), so one bad claim doesn't discard the good ones.
    """
    valid_ids = {e.citation_id for e in evidence}
    kept = []
    for h in hypotheses:
        if not h.supporting_evidence_ids:
            logger.warning("dropping hypothesis with no supporting_evidence_ids", extra={"statement": h.statement})
            continue
        if not set(h.supporting_evidence_ids).issubset(valid_ids):
            logger.warning(
                "dropping ungrounded hypothesis citing unknown evidence ids",
                extra={"statement": h.statement, "cited": h.supporting_evidence_ids},
            )
            continue
        kept.append(h)
    return kept


async def _build_sprint_snapshot(retrieval, space_id: int, sprint_id: int) -> SprintSnapshot:
    """The whole sprint: goal, timeline position and every issue.

    `plan_node` only knew issues that tripped a signal, so a sprint-level problem had nothing to act
    on, and strategic options ("move the two lowest-priority tickets out") were impossible. One extra
    structured query (no LLM).
    """
    result = await retrieval.query_issues([space_id], {"sprint_ids": [sprint_id], "limit": 200})
    ctx = await _sprint_context(retrieval, space_id, sprint_id)
    start_date, end_date = ctx.get("start_date"), ctx.get("end_date")
    duration = (end_date - start_date).days if start_date and end_date else None
    return SprintSnapshot(
        sprint_name=ctx.get("sprint_name"),
        goal=ctx.get("goal"),
        days_remaining=(end_date - date.today()).days if end_date else None,
        elapsed_percent=(
            int(round(100 * max(0.0, min(1.0, (date.today() - start_date).days / duration))))
            if duration else None
        ),
        issues=[
            SprintIssueSummary(
                issue_key=i["issue_key"], title=i.get("title"), status=i.get("status"),
                priority=i.get("priority"), assignee_name=i.get("assignee_name"),
                story_points=i.get("story_points"),
            )
            for i in result.issues
        ],
    )


async def diagnose_node(
    state: SprintRecoveryState, client, model: str, retrieval,
    on_usage: Optional[UsageRecorder] = None,
    llm_runtime: tuple = ("Unknown", "hosted"),
) -> dict:
    if not _check_budget(state):
        return {"status": "failed", "error": "token budget exhausted before diagnosis could complete"}

    signals = await _detect_risk_signals(retrieval, state["space_id"], state["sprint_id"])

    # No risk signals = healthy sprint: stop without an LLM call. Otherwise the model got an empty
    # prompt and asked the user what evidence was gathered (happened on all 7 AtlasCart sprints).
    if not signals:
        return {
            "status": "no_risk_found", "risk_signals": [], "evidence": [], "hypotheses": [],
            "clarification_question": None,
        }

    flagged_keys = sorted({s.issue_key for s in signals if s.issue_key})
    evidence = await _gather_evidence(retrieval, state["space_id"], state["sprint_id"], flagged_keys)

    # Turn each code-computed signal into citable evidence. Before, the model couldn't cite them
    # (grounding only accepts evidence ids), so a sprint-level-only signal left zero valid hypotheses.
    signal_evidence = [
        EvidenceItem(
            citation_id=f"risk{i}", issue_key=s.issue_key or "", source_type="risk_signal",
            content=f"{_SIGNAL_LABELS.get(s.signal_type, s.signal_type)}: {s.description}",
        )
        for i, s in enumerate(signals, start=1)
    ]
    evidence = signal_evidence + evidence

    if state["clarification_answer"]:
        evidence = evidence + [EvidenceItem(
            citation_id=f"human-clarification-{state['clarification_rounds']}",
            issue_key="", source_type="comment", content=state["clarification_answer"],
        )]

    evidence_text = "\n".join(f"[{e.citation_id}] ({e.issue_key} / {e.source_type}) {e.content}" for e in evidence)
    signals_text = "\n".join(f"- {s.signal_type}: {s.description}" for s in signals)
    # The sprint goal: "will we hit the commitment?" is what a PM actually asks.
    sprint_goal = (await _sprint_context(retrieval, state["space_id"], state["sprint_id"])).get("goal")
    goal_text = f"Sprint goal (what this sprint committed to deliver): {sprint_goal}\n" if sprint_goal else ""
    prompt = render_diagnose_prompt(
        sprint_name=state["sprint_name"], goal_text=goal_text,
        signals_text=signals_text, evidence_text=evidence_text,
    )
    result, tokens_used, cost_used = await _call_model(
        client, model, DiagnosisResult, prompt, on_usage,
        on_charge=_charge_recorder(state, model, llm_runtime, "diagnose"), step="diagnose",
    )
    grounded = _apply_grounding(result.hypotheses, evidence)
    update = {
        "risk_signals": signals, "evidence": evidence, "hypotheses": grounded,
        # Kept so plan_node can act on issues that didn't trip a signal of their own.
        "sprint_snapshot": await _build_sprint_snapshot(retrieval, state["space_id"], state["sprint_id"]),
        "token_usage": state["token_usage"] + tokens_used,
        "cost_usd": state.get("cost_usd", 0.0) + cost_used,
    }
    if result.overall_confidence == "insufficient" and state["clarification_rounds"] < state["max_clarification_rounds"]:
        update["status"] = "diagnosing"  # will route to clarify
        # Without a question this skips the human round; a smaller model did that 8/8 times until the
        # schema was made strict (see schemas.py).
        update["clarification_question"] = result.clarifying_question
    else:
        update["status"] = "diagnosing"  # will route to plan
        update["clarification_question"] = None
    return update


def _after_diagnose(state: SprintRecoveryState) -> str:
    if state["status"] in ("failed", "no_risk_found"):
        return END
    if state["clarification_question"] and state["clarification_rounds"] < state["max_clarification_rounds"]:
        return "clarify"
    return "plan"


async def clarify_node(state: SprintRecoveryState) -> dict:
    """Pause for one specific question, then loop back to diagnose with the answer as evidence.
    (A free-text answer, unlike the plan-approval pause.)
    """
    answer = interrupt({"question": state["clarification_question"], "round": state["clarification_rounds"] + 1})
    return {
        "clarification_answer": answer.get("answer", ""),
        # Separate from clarification_answer so plan_node knows this note still needs a reply.
        "unanswered_human_note": answer.get("answer", ""),
        "clarification_rounds": state["clarification_rounds"] + 1,
        "status": "diagnosing",
    }


def _render_snapshot(snapshot: Optional[SprintSnapshot]) -> str:
    """The full sprint roster as prompt text, so strategic options ("protect the goal, drop the rest")
    can see what "the rest" is.
    """
    if snapshot is None or not snapshot.issues:
        return ""
    lines = [
        f"- {i.issue_key} [{i.status or 'unknown status'}] "
        f"{'(' + str(i.story_points) + ' pts) ' if i.story_points is not None else ''}"
        f"{'priority=' + i.priority + ' ' if i.priority else ''}"
        f"{'owner=' + i.assignee_name + ' ' if i.assignee_name else 'unassigned '}"
        f"— {i.title or ''}".rstrip()
        for i in snapshot.issues
    ]
    header = "Every issue in this sprint right now:"
    timing = []
    if snapshot.elapsed_percent is not None:
        timing.append(f"{snapshot.elapsed_percent}% of the sprint has elapsed")
    if snapshot.days_remaining is not None:
        timing.append(f"{snapshot.days_remaining} day(s) remain")
    timing_text = f"({', '.join(timing)})\n" if timing else ""
    return f"\n{timing_text}{header}\n" + "\n".join(lines) + "\n"


async def plan_node(
    state: SprintRecoveryState, client, model: str, retrieval,
    on_usage: Optional[UsageRecorder] = None,
    llm_runtime: tuple = ("Unknown", "hosted"),
) -> dict:
    """Propose recovery plans.

    The model once used evidence ids ("ev15") as issue keys. So the prompt lists real issue keys
    separately, and `_validate_plan_issue_keys` drops any action naming an unknown key.
    """
    if not _check_budget(state):
        # Clear the pending human note on every exit, failures included, so it isn't raised again on
        # the next retry or escalation as if just said.
        return {
            "status": "failed", "error": "token budget exhausted before plan generation could complete",
            "unanswered_human_note": None,
        }
    snapshot = state.get("sprint_snapshot")
    flagged_issue_keys = sorted(
        {e.issue_key for e in state["evidence"] if e.issue_key}
        | {s.issue_key for s in state["risk_signals"] if s.issue_key}
        | {k.upper() for e in state["evidence"] for k in _ISSUE_KEY_RE.findall(e.content)}
    )
    # Plans may act on any issue in the sprint, not just flagged ones — "move the two lowest-priority
    # tickets out" acts on issues that aren't themselves the problem.
    known_issue_keys = sorted(set(flagged_issue_keys) | {i.issue_key for i in (snapshot.issues if snapshot else [])})

    # Thanks to the roster, even a sprint-level-only risk (no issue key) is plannable.
    if not known_issue_keys:
        # Nothing at all to reason about — not even a sprint roster. Genuinely nothing to offer.
        return {
            "status": "escalated",
            "error": "Risk was detected at the sprint level, but it isn't tied to any specific issue "
                     "this workflow can act on — a human needs to decide what to do.",
            "unanswered_human_note": None,  # consumed either way — see the budget-guard return above
        }
    hyps_text ="\n".join(f"- ({h.confidence}) {h.statement} [cites: {', '.join(h.supporting_evidence_ids)}]" for h in state["hypotheses"])
    prior_note = ""
    if state["escalation_round"] > 0:
        prior_actions = state.get("prior_committed_actions") or []
        done_text = "\n".join(f"  - {a}" for a in prior_actions) or "  (nothing recorded)"
        prior_note = (
            f"\n\nThis is escalation round {state['escalation_round']}: the risk is still present after "
            f"{len(prior_actions)} action(s) already committed in earlier rounds, listed below. Do not "
            "repeat any of these — in particular, do not raise a priority that a prior action already "
            "raised (it has no further effect once at the ceiling), and do not re-comment the same "
            "request already made. Propose genuinely different actions, or escalate visibility/ownership "
            "in a way the list below hasn't already tried:\n" + done_text
        )
    if state.get("human_plan_feedback"):
        prior_note += (
            f"\n\nA human reviewed the previous plans and rejected all of them with this feedback: "
            f"\"{state['human_plan_feedback']}\". Address this feedback directly in the new plans — do "
            "not repeat what was rejected."
        )
    # Show the real goal so `impact_on_goal` means what it says.
    sprint_goal = (await _sprint_context(retrieval, state["space_id"], state["sprint_id"])).get("goal")
    goal_text = f"Sprint goal (judge every plan's impact_on_goal against THIS): {sprint_goal}\n" if sprint_goal else ""
    # A rewind note ("we'll add 2 engineers") used to be considered only inside hypotheses, so the
    # plans never mentioned it. The template asks for a direct reply only when a note is pending.
    human_note = state.get("unanswered_human_note")
    prompt = render_plan_prompt(
        sprint_name=state["sprint_name"], goal_text=goal_text, snapshot_text=_render_snapshot(snapshot),
        hypotheses_text=hyps_text, prior_note=prior_note, known_issue_keys=known_issue_keys,
        human_note=human_note,
    )
    result, tokens_used, cost_used = await _call_model(
        client, model, RecoveryPlanSet, prompt, on_usage,
        on_charge=_charge_recorder(state, model, llm_runtime, "plan"), step="plan",
    )
    plans = _validate_plan_issue_keys(result.plans, known_issue_keys)
    if not plans:
        return {
            "status": "failed", "error": "every proposed plan cited only invalid/unknown issue keys",
            # Charged anyway: the call was paid for.
            "token_usage": state["token_usage"] + tokens_used,
            "cost_usd": state.get("cost_usd", 0.0) + cost_used,
            "unanswered_human_note": None,  # consumed either way — see the budget-guard return above
            # Still return the reply to the human's note, so they learn why it couldn't be honored.
            "note_response": result.response_to_human_note if human_note else None,
        }
    return {
        "plans": plans, "status": "awaiting_plan_approval",
        "token_usage": state["token_usage"] + tokens_used,
        "cost_usd": state.get("cost_usd", 0.0) + cost_used,
        # Reset: the note is answered, and a reply should only sit next to the plans it describes.
        "note_response": result.response_to_human_note if human_note else None,
        "unanswered_human_note": None,
        # Clear revise feedback once used, or it gets re-applied to every later round.
        "human_plan_feedback": None,
    }


def _validate_plan_issue_keys(plans: List[RecoveryPlan], known_issue_keys: List[str]) -> List[RecoveryPlan]:
    """Drop actions naming an issue key outside the known set (the code backstop for the prompt). A
    plan left with no actions is dropped rather than shown empty.
    """
    known = set(known_issue_keys)
    kept_plans = []
    for plan in plans:
        kept_actions = []
        for a in plan.actions:
            bad = {a.target_issue_key} | ({a.depends_on_issue_key} if a.depends_on_issue_key else set())
            bad -= known
            if bad:
                logger.warning("dropping action citing unknown issue key(s)", extra={"action_type": a.action_type, "bad_keys": list(bad)})
                continue
            kept_actions.append(a)
        if kept_actions:
            kept_plans.append(RecoveryPlan(
                plan_id=plan.plan_id, name=plan.name, rationale=plan.rationale,
                impact_on_goal=plan.impact_on_goal, actions=kept_actions,
            ))
        else:
            logger.warning("dropping plan with zero valid actions after issue-key validation", extra={"plan_id": plan.plan_id})
    return kept_plans


def _after_plan(state: SprintRecoveryState) -> str:
    return END if state["status"] in ("failed", "escalated") else "approval"


async def approval_node(
    state: SprintRecoveryState, space_membership: SpaceMembershipChecker, jira: JiraActionsClient,
) -> dict:
    """Plan-approval pause: approve, edit, reject — or `revise`, which loops back to plan_node with the
    human's feedback (capped by max_plan_revision_rounds, then treated as a reject).
    """
    payload = interrupt({"plans": [p.model_dump() for p in state["plans"]]})
    decision = payload.get("decision")

    try:
        await space_membership.validate(state["user_id"], state["username"], [state["space_id"]])
    except SpaceMembershipError as exc:
        return {"decision": "reject", "status": "rejected", "error": str(exc)}

    if decision == "revise":
        feedback = (payload.get("feedback") or "").strip()
        if not feedback:
            return {"decision": "reject", "status": "rejected", "error": "revise requires feedback text"}
        if state["plan_revision_round"] >= state["max_plan_revision_rounds"]:
            return {
                "decision": "reject", "status": "rejected",
                "error": f"reached the {state['max_plan_revision_rounds']}-revision limit without an "
                         "approved plan — rejecting rather than looping forever",
            }
        return {
            "decision": "revise", "status": "revising",
            "human_plan_feedback": feedback, "plan_revision_round": state["plan_revision_round"] + 1,
            # Gets a direct reply like a clarification answer (see unanswered_human_note in state.py).
            "unanswered_human_note": feedback,
        }

    if decision == "reject":
        return {"decision": "reject", "status": "rejected"}

    plan_id = payload.get("planId") or payload.get("plan_id")
    chosen = next((p for p in state["plans"] if p.plan_id == plan_id), None)
    if chosen is None:
        return {"decision": "reject", "status": "failed", "error": f"unknown plan_id {plan_id!r}"}

    if decision == "edit" and payload.get("actions") is not None:
        chosen = RecoveryPlan(
            plan_id=chosen.plan_id, name=chosen.name, rationale=chosen.rationale,
            impact_on_goal=chosen.impact_on_goal,
            actions=[RecoveryAction.model_validate(a) for a in payload["actions"]],
        )

    # Capture each target issue's pre-write `updatedAt` here, before any write this round. This return
    # is checkpointed, so a crash mid-commit can't make the resumed attempt read a post-write time as the
    # baseline (reproduced with a two-process crash test; it brought back the false "recovered").
    # Target keys only: no watermark is ever written for depends_on keys, and a baseline without one
    # would freeze that issue's staleness forever.
    pre_writes = dict(state.get("pre_write_updated_at") or {})
    keys_needing_baseline = {
        a.target_issue_key for a in chosen.actions if a.target_issue_key not in pre_writes
    }
    for key in sorted(keys_needing_baseline):
        before = await jira.issue_updated_at(state["space_id"], key, state["user_id"], state["username"])
        if before:
            pre_writes[key] = before

    return {
        "decision": decision or "approve", "approved_plan": chosen, "status": "committing",
        # Fresh per approval; see state.py `commit_attempt_id`.
        "commit_attempt_id": str(uuid.uuid4()),
        "committed_actions": {}, "index_watermarks": {}, "pre_write_updated_at": pre_writes,
    }


def _after_approval(state: SprintRecoveryState) -> str:
    if state["status"] == "revising":
        return "plan"
    return END if state["status"] in ("rejected", "failed") else "commit_one_action"


async def commit_one_action_node(state: SprintRecoveryState, jira: JiraActionsClient) -> dict:
    """One action per graph step (checkpoints happen between steps), so a crash can't lose track of
    which actions already ran. Same as rollout_graph's commit_one_node.
    """
    plan = state["approved_plan"]
    remaining = [
        (i, a) for i, a in enumerate(plan.actions) if str(i) not in state["committed_actions"]
    ]
    if not remaining:
        return {"status": "waiting_reevaluation"}

    idx, action = remaining[0]
    # Key = thread_id + commit_attempt_id + action index. The old key used round counters that a rewind
    # doesn't reset, so a different plan approved at the same rounds replayed jira-backend's cached
    # response and silently did nothing. A fresh id per approval avoids that; thread_id also keeps two
    # threads for the same sprint apart.
    idempotency_key = f"sprint-recovery:{THREAD_ID_VAR.get()}:{state.get('commit_attempt_id')}:{idx}"
    # Capture before writing: our own write bumps `updatedAt` and would look like real progress (see
    # `_last_update_not_by_this_workflow`). First write wins, so the human baseline is kept.
    pre_writes = dict(state.get("pre_write_updated_at") or {})
    if action.target_issue_key not in pre_writes:
        before = await jira.issue_updated_at(
            state["space_id"], action.target_issue_key, state["user_id"], state["username"],
        )
        if before:
            pre_writes[action.target_issue_key] = before

    try:
        result_summary = await jira.execute(
            state["space_id"], action, state["user_id"], state["username"], idempotency_key=idempotency_key,
        )
    except JiraActionError as exc:
        # Keep the baseline even on failure: if the write actually succeeded and only the response
        # timed out, a retry's baseline read would come after our write.
        return {"status": "failed", "error": str(exc), "pre_write_updated_at": pre_writes}

    committed = dict(state["committed_actions"])
    committed[str(idx)] = result_summary
    done = len(committed) == len(plan.actions)
    # Read our write's `updatedAt` back so reevaluate can wait for the index to catch up
    # (see `_await_index_catch_up`).
    watermarks = dict(state.get("index_watermarks") or {})
    updated_at = await jira.issue_updated_at(
        state["space_id"], action.target_issue_key, state["user_id"], state["username"],
    )
    if updated_at:
        watermarks[action.target_issue_key] = updated_at
    return {
        "committed_actions": committed, "index_watermarks": watermarks, "pre_write_updated_at": pre_writes,
        "status": "waiting_reevaluation" if done else "committing",
    }


def _after_commit_one_action(state: SprintRecoveryState) -> str:
    if state["status"] == "failed":
        return END
    if state["status"] == "waiting_reevaluation":
        return "wait_for_reevaluation"
    return "commit_one_action"


async def wait_for_reevaluation_node(state: SprintRecoveryState) -> dict:
    """Pause until re-evaluation is requested: automatically right after approval, or by a human
    ("Re-check now").
    """
    interrupt({"waiting_since_committed_actions": len(state["committed_actions"]), "sprint_id": state["sprint_id"]})
    return {"status": "diagnosing"}


async def reevaluate_node(state: SprintRecoveryState, retrieval) -> dict:
    """Re-run the same code-computed risk detection, so "recovered" is a fact, not the LLM's opinion.

    Waits for the index to reflect our writes first; it could otherwise read a stale sprint.
    """
    watermarks = state.get("index_watermarks") or {}
    caught_up = await _await_index_catch_up(
        retrieval, state["space_id"], state["sprint_id"], watermarks,
    )
    # Pair each written issue as (before our write, after) so staleness ignores our own touch.
    pre_writes = state.get("pre_write_updated_at") or {}
    own_writes = {key: (before, watermarks.get(key)) for key, before in pre_writes.items()}
    signals = await _detect_risk_signals(
        retrieval, state["space_id"], state["sprint_id"], own_writes=own_writes,
    )
    # Any signal means still at risk. (Checking specific signal names once missed blocked issues
    # and reported "recovered".)
    still_at_risk = bool(signals)
    if not still_at_risk:
        # Recorded on success too, as the honest answer to "did the index catch up?" (a stale read
        # can't produce a false "recovered" anyway — see state.py).
        return {"status": "recovered", "risk_signals": signals, "index_catch_up_timed_out": not caught_up}

    if state["escalation_round"] >= state["max_escalation_rounds"]:
        return {"status": "escalated", "risk_signals": signals, "index_catch_up_timed_out": not caught_up}
    plan = state.get("approved_plan")
    just_done = [
        _describe_action(a) for i, a in enumerate(plan.actions) if str(i) in (state.get("committed_actions") or {})
    ] if plan else []
    return {
        "status": "diagnosing", "risk_signals": signals, "escalation_round": state["escalation_round"] + 1,
        "committed_actions": {}, "index_watermarks": {}, "index_catch_up_timed_out": not caught_up,
        # Reset alongside index_watermarks: they only make sense as a pair. A leftover baseline with no
        # watermark would hide a real human update on that issue forever. An issue acted on again gets
        # a fresh baseline from commit_one_action_node.
        "pre_write_updated_at": {},
        # Cleared with the plan; approval_node mints a new one on the next approval.
        "commit_attempt_id": None,
        # Refresh the roster (cheap, no LLM) so the next round doesn't plan against the sprint as it was
        # before round 1's changes. Evidence and hypotheses are kept to avoid a second diagnose call.
        "sprint_snapshot": await _build_sprint_snapshot(retrieval, state["space_id"], state["sprint_id"]),
        "approved_plan": None, "decision": None,
        "prior_committed_actions": state.get("prior_committed_actions", []) + just_done,
    }


def _after_reevaluate(state: SprintRecoveryState) -> str:
    if state["status"] == "diagnosing":
        return "plan"  # re-plan directly — hypotheses already exist, no need to re-diagnose from scratch
    return END


def _traced(node_name: str, fn):
    """Wrap a node in one span per run (and set SSE stage labels and THREAD_ID_VAR). thread_id comes from
    LangGraph's RunnableConfig, since it's run identity, not workflow data.
    """
    async def _wrapped(state: SprintRecoveryState, config=None) -> dict:
        thread_id = (config or {}).get("configurable", {}).get("thread_id", "unknown")
        on_stage = ON_STAGE_VAR.get()
        label = _STAGE_LABELS.get(node_name)
        if on_stage and label:
            on_stage(label)
        token = THREAD_ID_VAR.set(thread_id)
        try:
            # Exceptions are recorded by hand: interrupt() pauses for a human by raising
            # GraphInterrupt, and the default handling marked every approval wait as an error.
            with _tracer.start_as_current_span(
                f"sprint_recovery.{node_name}",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    "sprint_recovery.thread_id": thread_id,
                    "sprint_recovery.status_before": state.get("status", ""),
                },
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    result = await fn(state)
                except GraphBubbleUp:
                    span.set_attribute("sprint_recovery.paused", True)
                    raise
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
                    raise
                if "status" in result:
                    span.set_attribute("sprint_recovery.status_after", result["status"])
                return result
        finally:
            THREAD_ID_VAR.reset(token)
    return _wrapped


def build_sprint_recovery_graph(
    client, model: str, space_membership: SpaceMembershipChecker, retrieval, jira: JiraActionsClient,
    on_usage: Optional[UsageRecorder] = None,
    # (provider, deployment) for pricing, same as the agent. Passed in because the model name can't
    # tell local (free) from hosted.
    llm_runtime: tuple = ("Unknown", "hosted"),
):
    async def _diagnose(state: SprintRecoveryState) -> dict:
        return await diagnose_node(state, client, model, retrieval, on_usage, llm_runtime)

    async def _plan(state: SprintRecoveryState) -> dict:
        return await plan_node(state, client, model, retrieval, on_usage, llm_runtime)

    async def _approval(state: SprintRecoveryState) -> dict:
        return await approval_node(state, space_membership, jira)

    async def _commit_one(state: SprintRecoveryState) -> dict:
        return await commit_one_action_node(state, jira)

    async def _reevaluate(state: SprintRecoveryState) -> dict:
        return await reevaluate_node(state, retrieval)

    graph = StateGraph(SprintRecoveryState)
    graph.add_node("diagnose", _traced("diagnose", _diagnose))
    graph.add_node("clarify", _traced("clarify", clarify_node))
    graph.add_node("plan", _traced("plan", _plan))
    graph.add_node("approval", _traced("approval", _approval))
    graph.add_node("commit_one_action", _traced("commit_one_action", _commit_one))
    graph.add_node("wait_for_reevaluation", _traced("wait_for_reevaluation", wait_for_reevaluation_node))
    graph.add_node("reevaluate", _traced("reevaluate", _reevaluate))

    graph.add_edge(START, "diagnose")
    graph.add_conditional_edges("diagnose", _after_diagnose, {"clarify": "clarify", "plan": "plan", END: END})
    graph.add_edge("clarify", "diagnose")
    graph.add_conditional_edges("plan", _after_plan, {"approval": "approval", END: END})
    graph.add_conditional_edges(
        "approval", _after_approval, {"commit_one_action": "commit_one_action", "plan": "plan", END: END},
    )
    graph.add_conditional_edges(
        "commit_one_action", _after_commit_one_action,
        {"commit_one_action": "commit_one_action", "wait_for_reevaluation": "wait_for_reevaluation", END: END},
    )
    graph.add_edge("wait_for_reevaluation", "reevaluate")
    graph.add_conditional_edges("reevaluate", _after_reevaluate, {"plan": "plan", END: END})
    return graph


async def trigger_reevaluation(compiled_graph, thread_id: str, source: str, detail: str = "") -> dict:
    """Resume the `wait_for_reevaluation` pause. `source` is only recorded."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": thread_id}}
    return await compiled_graph.ainvoke(Command(resume={"source": source, "detail": detail}), config=config)


def _has_uninterrupted_pending_task(snap) -> bool:
    """True when LangGraph has a pending task and no human is being waited on — i.e. a crash.

    Status alone can't tell a crash at "diagnosing" from a real pause in `clarify` (which always has an
    interrupt). A real pause must never be resumed through this path.
    """
    return bool(snap.next) and not any(t.interrupts for t in snap.tasks)


def is_resumable_after_a_crash(snap) -> bool:
    """Whether `retry_recovery_commit` can do anything here: a "failed" thread, or a real crash. Used
    by the /retry route so a thread waiting on a human (clarify/approval/re-check) isn't retried
    through this generic path.
    """
    return snap.values.get("status") == "failed" or _has_uninterrupted_pending_task(snap)


async def retry_recovery_commit(compiled_graph, thread_id: str) -> dict:
    """Un-stick a thread that `is_resumable_after_a_crash` allows. Three cases, checked in this order:

    1. A crash left a pending task: `ainvoke(None)` resumes where the checkpoint stopped.
    2. "failed" after a plan was approved (Jira error mid-commit): re-arm commit_one_action via
       `as_node="approval"`.
    3. "failed" before any approval (budget, or every action had unknown keys): `approved_plan` is
       None, so case 2 would crash. Re-enter `plan` via `as_node="diagnose"` (diagnosis is kept).

    Case 1 goes first: after a crash elsewhere, `approved_plan` may be a stale value from an earlier
    round, and case 2 would re-send actions already posted to Jira.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snap = await compiled_graph.aget_state(config)
    if _has_uninterrupted_pending_task(snap):
        return await compiled_graph.ainvoke(None, config=config)
    if snap.values.get("approved_plan") is not None:
        await compiled_graph.aupdate_state(config, {"status": "committing", "error": None}, as_node="approval")
        return await compiled_graph.ainvoke(None, config=config)
    await compiled_graph.aupdate_state(config, {"status": "diagnosing", "error": None}, as_node="diagnose")
    return await compiled_graph.ainvoke(None, config=config)


async def stop_recovery(compiled_graph, thread_id: str, stopped_by: str) -> dict:
    """End an unfinished thread wherever it is paused: clarify, approval, re-check, or after a failure.

    Rejecting a plan was the only way out, and only at approval, so a thread left waiting on a
    clarification or a re-check held its sprint until its owner came back. Written as approval's
    "rejected" outcome because that edge already leads to END; nothing is sent to Jira.
    """
    config = {"configurable": {"thread_id": thread_id}}
    await compiled_graph.aupdate_state(
        config,
        {"status": "rejected", "decision": "reject", "error": f"Stopped by {stopped_by}", "stopped_by": stopped_by},
        as_node="approval",
    )
    return (await compiled_graph.aget_state(config)).values


async def list_checkpoint_history(compiled_graph, thread_id: str) -> list:
    """All checkpoints for this thread, newest first (the UI's time-travel picker). `next` shows where
    a rewind would resume, e.g. `('plan',)` = right after diagnosis.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return [s async for s in compiled_graph.aget_state_history(config)]


def _describe_action(action: RecoveryAction) -> str:
    """One executed action as a plain sentence (including the full comment text), so a human can audit
    what was done in Jira on their behalf.
    """
    key = action.target_issue_key
    if action.action_type == "add_comment":
        return f'Commented on {key}: "{action.comment_body}"'
    if action.action_type == "change_priority":
        return f"Raised {key} priority to {action.new_priority}"
    if action.action_type == "move_out_of_sprint":
        return f"Moved {key} out of the sprint"
    if action.action_type == "link_dependency":
        return f"Marked {key} as blocked by {action.depends_on_issue_key}"
    return f"{action.action_type} on {key}"


async def build_escalation_summary(compiled_graph, thread_id: str) -> dict:
    """What was tried across every escalation round, for the human taking over.

    Returns structured data (rounds, still_at_risk_reasons, unaddressed_issue_keys) — one long
    paragraph was unreadable in the UI. `flatten_escalation_summary` makes the string version for
    notifications and logs.

    `unaddressed_issue_keys` separates two kinds of escalation: every flagged issue was acted on (the
    automation ran out of options; real work is needed) vs. some flagged issue was never touched (a
    gap). Sprint-level signals are excluded: no action type can complete story points, so they're
    always still firing.

    Reads the checkpoint history (one "waiting_reevaluation" snapshot per round) instead of tracking a
    new field. Call it from the code that resumed the graph and saw "escalated" — a node can't read
    its own workflow's history.
    """
    config = {"configurable": {"thread_id": thread_id}}
    history = [s async for s in compiled_graph.aget_state_history(config)]
    history.reverse()  # aget_state_history yields newest-first; want chronological order here

    rounds: list = []
    seen_rounds = set()
    actioned_issue_keys: set = set()
    for snap in history:
        if snap.values.get("status") != "waiting_reevaluation":
            continue
        plan = snap.values.get("approved_plan")
        if plan is None:
            continue
        round_num = snap.values.get("escalation_round", 0)
        if round_num in seen_rounds:
            continue  # aget_state_history can repeat a snapshot around branch points (e.g. time-travel)
        seen_rounds.add(round_num)
        # Describe actions from the plan (full comment text etc.), not from committed_actions' terse
        # results; committed_actions only says which ones ran.
        committed = snap.values.get("committed_actions") or {}
        actions = []
        for i, a in enumerate(plan.actions):
            if str(i) not in committed:
                continue
            actions.append(_describe_action(a))
            actioned_issue_keys.add(a.target_issue_key)
        rounds.append({
            "round": round_num + 1, "plan_name": plan.name, "rationale": plan.rationale, "actions": actions,
        })

    final_signals = (history[-1].values.get("risk_signals") if history else None) or []
    flagged_issue_keys = {s.issue_key for s in final_signals if s.issue_key}
    return {
        "rounds": rounds,
        "still_at_risk_reasons": [s.description for s in final_signals],
        "unaddressed_issue_keys": sorted(flagged_issue_keys - actioned_issue_keys),
    }


def flatten_escalation_summary(summary: dict) -> str:
    """`build_escalation_summary` as one string, for notifications and logs (no UI to lay out cards)."""
    rounds = summary.get("rounds") or []
    if not rounds:
        return "No recovery plan was ever committed before this workflow reached its escalation cap."
    lines = [
        f"Round {r['round']}: approved \"{r['plan_name']}\" ({r['rationale']}). "
        f"Actions taken: {'; '.join(r['actions']) if r['actions'] else 'none'}."
        for r in rounds
    ]
    reasons = summary.get("still_at_risk_reasons") or []
    if reasons:
        lines.append("Still at risk because: " + " ".join(reasons))
    unaddressed = summary.get("unaddressed_issue_keys") or []
    if unaddressed:
        lines.append(f"Never got attempted: {', '.join(unaddressed)}.")
    return " ".join(lines)


async def time_travel_resume(compiled_graph, thread_id: str, target_checkpoint_id: str, note: str) -> dict:
    """Rewind to an earlier checkpoint and continue from there, with `note` treated as a clarification
    answer given at that point (`as_node="clarify"` always re-enters diagnose). Rewrites this thread's
    history rather than forking; the old future stays visible in `aget_state_history`.
    """
    target_config = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": target_checkpoint_id}
    }
    forked_config = await compiled_graph.aupdate_state(
        target_config,
        # Set everything clarify_node would set, since it doesn't run on this path — including
        # unanswered_human_note, or plan_node wouldn't reply to the note.
        {"clarification_answer": note, "unanswered_human_note": note, "clarification_rounds": 0},
        as_node="clarify",
    )
    return await compiled_graph.ainvoke(None, config=forked_config)
