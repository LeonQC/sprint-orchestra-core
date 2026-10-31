"""12 hand-written sprint-recovery scenarios, ground truth defined before the first run.

Each scenario seeds one known root cause (an undocumented dependency, an overloaded owner, a duplicate
ticket, ...) and checks two things: does `diagnose_node` name the right cause, and does `plan_node`
pick a sensible action. Two scenarios put two different problems in one sprint, to check that the
diagnosis keeps them apart instead of merging them.

A fake retrieval client returns the hand-crafted content; the model, the graph and the
grounding/validation code are real. Grading is deterministic (substring / issue-key matching), not an
LLM judge, so a correct answer in other words can occasionally fail: every scenario prints the full
hypothesis and plan so a "FAIL" can be checked by eye. 12 scenarios is a regression check, not a
benchmark.

Usage: python -m eval.sprint_recovery_scenarios_eval
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.auth.space_membership import NoopSpaceMembershipChecker
from app.config import Settings
from app.drafting.instructor_client import build_instructor_client
from app.sprint_recovery.graph import build_sprint_recovery_graph
from app.sprint_recovery.state import initial_recovery_state
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Only labels: retrieval, Jira and space membership are all fakes in this eval, so no real space or
# sprint is read. Fixed values keep it runnable without a database (CI runs it; see ci.yml).
SPACE_ID = 1
SPRINT_ID = 7

# 8 filler "done" issues regardless of how many target issues a scenario has (1 or 2) keeps the
# completion forecast safely above the 70% checkpoint (80-88.9% done) — keeps `low_completion_forecast`
# from firing as a second, unwanted signal in every scenario, so each one isolates the phenomenon (or,
# for the 2 multi-signal scenarios, the *two* phenomena) it's designed to test.
_FILLER_DONE = [
    {"issue_key": f"PAY-{900 + i}", "status": "done", "updated_at": "2026-07-30T00:00:00Z"} for i in range(8)
]


class ScenarioRetrieval:
    """Generalizes `PoisonedRetrieval`/tests' `FakeRetrieval` for this script: 1-2 in_progress target
    issues (drives the deterministic `long_in_progress` signal and `_gather_evidence`'s flagged-key
    fetch for each), plus whatever comments/history the scenario plants as evidence.
    """

    def __init__(self, target_keys: List[str], comments: List[dict], history: Optional[List[dict]] = None):
        self._target_keys = target_keys
        self._comments = comments
        self._history = history or []

    async def query_issues(self, space_ids, filters):
        target_rows = [
            {"issue_key": k, "status": "in_progress", "updated_at": "2026-07-15T00:00:00Z"} for k in self._target_keys
        ]
        rows = list(_FILLER_DONE) + target_rows
        counts = {"done": 8, "in_progress": len(target_rows)}
        return type("R", (), {"total_count": len(rows), "counts_by_status": counts, "counts_by_type": {}, "issues": rows})()

    async def query_issue_history(self, space_ids, filters):
        return type("R", (), {"total_count": len(self._history), "changes": self._history})()

    async def get_issue_comments(self, space_ids, issue_keys, limit=200):
        return type("R", (), {"total_count": len(self._comments), "comments": self._comments})()

    async def get_issue_details(self, space_ids, issue_keys, limit=200):
        return type("R", (), {"total_count": 0, "details": []})()

    async def get_issue_attachments(self, space_ids, issue_keys, limit=200):
        return type("R", (), {"total_count": 0, "attachments": []})()


class NoopJiraActions:
    """Every scenario stops at the plan-approval interrupt without a decision ever being submitted —
    this eval grades the *proposed* plan, not execution, so `execute` should never be called."""

    async def execute(self, space_id, action, user_id, username):
        raise AssertionError("sprint_recovery_scenarios_eval should never reach action execution")


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(k.lower() in lowered for k in keywords)


def _all_hypotheses_text(hyps) -> str:
    return " ".join(h.statement for h in hyps)


def _all_plan_actions(plans):
    return [a for p in plans for a in p.actions]


# diagnosis_check(combined hypothesis text, clarifying_question or None, raw hypotheses list) -> (ok, note)
# The raw hypotheses list is only needed by the 2 multi-signal scenarios (to check each issue got its
# *own* distinct hypothesis, not just that both keys appear somewhere in the combined text) — every
# single-issue scenario's check ignores that third argument.
@dataclass
class Scenario:
    id: str
    ground_truth: str
    target_keys: List[str]
    comments: List[dict]
    history: List[dict] = field(default_factory=list)
    clarify_answer: str = "Confirmed — that's accurate, no further information beyond what's documented."
    diagnosis_check: Callable[[str, Optional[str], list], Tuple[bool, str]] = None
    plan_check: Optional[Callable[[list], Tuple[bool, str]]] = None


SCENARIOS: List[Scenario] = [
    Scenario(
        id="undocumented_dependency",
        ground_truth="PAY-142 is blocked on PAY-97, but no formal Jira link exists — only mentioned in a comment.",
        target_keys=["PAY-142"],
        comments=[{
            "issue_key": "PAY-142",
            "content": "Still blocked — waiting on PAY-97 to ship the shared auth module before we can "
                       "proceed. No formal link exists yet in Jira, just tracking it here.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-97" in text.upper(),
            "hypothesis must name PAY-97 as the blocking issue",
        ),
        plan_check=lambda actions: (
            any(a.target_issue_key == "PAY-97" or a.depends_on_issue_key == "PAY-97" for a in actions),
            "plan must include an action that references PAY-97 (link_dependency or reprioritize it)",
        ),
    ),
    Scenario(
        id="false_alarm_intentional_slow_burn",
        ground_truth="PAY-210 has been in_progress a while by design (a deliberately slow research spike) — "
                     "not actually at risk. Correct behavior is to NOT recommend an urgent/severe action.",
        target_keys=["PAY-210"],
        comments=[{
            "issue_key": "PAY-210",
            "content": "This ticket is an exploratory research spike, intentionally paced slowly per team "
                       "agreement. Not urgent, no sprint-goal impact — please don't escalate priority or "
                       "pull it from the sprint over this.",
        }],
        diagnosis_check=lambda text, q, hyps: (True, "no specific claim required — graded on the plan instead"),
        plan_check=lambda actions: (
            not any(
                a.target_issue_key == "PAY-210" and (
                    a.action_type == "move_out_of_sprint"
                    or (a.action_type == "change_priority" and (a.new_priority or "").lower() in ("highest", "high"))
                )
                for a in actions
            ),
            "must NOT propose move_out_of_sprint or an escalated priority on PAY-210 given the explicit "
            "'this is intentional, don't escalate' comment",
        ),
    ),
    Scenario(
        id="scope_creep_late_addition",
        ground_truth="PAY-305 was added to the sprint mid-sprint (2026-07-28, after the sprint's normal "
                     "planning), not part of original scope.",
        target_keys=["PAY-305"],
        comments=[{
            "issue_key": "PAY-305",
            "content": "This was added mid-sprint after planning was already done — wasn't part of the "
                       "original committed scope.",
        }],
        history=[{
            "issue_key": "PAY-305", "field_name": "sprint", "from_value": "None", "to_value": "Sprint 7",
            "changed_at": "2026-07-28T00:00:00Z",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-305" in text.upper() and _contains_any(text, ["scope", "added", "mid-sprint", "late"]),
            "hypothesis must name PAY-305 and reference late-addition/scope-creep",
        ),
    ),
    Scenario(
        id="conflicting_status_reports",
        ground_truth="Two comments on PAY-410 disagree: an older one claims it's done, a newer one (from "
                     "QA) says it's still failing tests. Correct behavior is to notice the conflict / trust "
                     "the more recent contradicting report, not silently accept 'it's done'.",
        target_keys=["PAY-410"],
        comments=[
            {"issue_key": "PAY-410", "content": "[2026-07-18] Engineer: PAY-410 is actually done, just forgot to close it out."},
            {"issue_key": "PAY-410", "content": "[2026-07-29] QA: PAY-410 still failing regression tests, not ready — please don't close."},
        ],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-410" in text.upper() and (
                _contains_any(text, ["conflict", "unclear", "unresolved", "discrepan", "test", "fail"])
                or (q is not None and "PAY-410" in q.upper())
            ),
            "must surface the conflicting reports (via hypothesis wording or a clarifying question) "
            "rather than silently trusting the older 'it's done' comment over the newer QA failure report",
        ),
        plan_check=None,  # deliberately ungraded — see module docstring on not forcing binary grades everywhere
    ),
    Scenario(
        id="external_team_blocker",
        ground_truth="PAY-520 is blocked on the Payments Infra team provisioning a Kafka topic — an "
                     "external dependency with no corresponding Jira issue at all.",
        target_keys=["PAY-520"],
        comments=[{
            "issue_key": "PAY-520",
            "content": "Blocked on the Payments Infra team to provision the new Kafka topic — not tracked "
                       "as a Jira issue on our side, they said ETA is next week.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-520" in text.upper() and _contains_any(text, ["infra", "external", "kafka", "payments infra"]),
            "hypothesis must name PAY-520 and identify the blocker as an external/untracked team dependency",
        ),
        plan_check=lambda actions: (
            all(a.action_type != "link_dependency" or a.depends_on_issue_key not in (None, "") for a in actions),
            "structural only: any link_dependency action must still name a real depends_on_issue_key "
            "(there is no real issue for the external team, so a well-behaved plan should avoid inventing "
            "one — `_validate_plan_issue_keys` would strip it either way)",
        ),
    ),
    Scenario(
        id="stale_status_already_done",
        ground_truth="PAY-630 shows status=in_progress but the most recent comment says engineering work "
                     "is actually complete and deployed — a stale status, not a real blocker.",
        target_keys=["PAY-630"],
        comments=[{
            "issue_key": "PAY-630",
            "content": "Merged and deployed last week, just need someone to flip the Jira status — "
                       "engineering side is fully complete.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-630" in text.upper() and _contains_any(text, ["stale", "already done", "deployed", "complete", "merged"]),
            "hypothesis must name PAY-630 and recognize the status looks stale (work is actually done)",
        ),
        plan_check=lambda actions: (
            not any(a.target_issue_key == "PAY-630" and a.action_type == "move_out_of_sprint" for a in actions),
            "must NOT propose move_out_of_sprint on PAY-630 — the work is done, not at risk",
        ),
    ),
    Scenario(
        id="incorrect_estimate",
        ground_truth="PAY-710 was estimated at 2 points but a comment reveals it actually touches 4 "
                     "services plus a data migration — a significant underestimate, not a blocker.",
        target_keys=["PAY-710"],
        comments=[{
            "issue_key": "PAY-710",
            "content": "Started digging into PAY-710 and it's way bigger than the 2-point estimate — it "
                       "actually touches four different services, needs a coordinated data migration, and "
                       "a phased rollout. We significantly underestimated the complexity here.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-710" in text.upper() and _contains_any(text, ["underestimat", "estimate", "complexity"]),
            "hypothesis must name PAY-710 and identify the root cause as an estimate/complexity mismatch, "
            "not a dependency or blocker",
        ),
        plan_check=lambda actions: (
            not any(a.action_type == "link_dependency" for a in actions),
            "must NOT fabricate a link_dependency action — no real dependency exists in the evidence, "
            "this is an estimation problem",
        ),
    ),
    Scenario(
        id="owner_overloaded",
        ground_truth="PAY-810 stalled because its sole owner is juggling 3 other critical items in "
                     "parallel — a capacity/bandwidth problem, not a technical blocker.",
        target_keys=["PAY-810"],
        comments=[{
            "issue_key": "PAY-810",
            "content": "PAY-810 hasn't moved much this week — I'm currently the sole owner on three other "
                       "critical production incidents in parallel, so this has stalled purely from "
                       "bandwidth, not technical difficulty.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-810" in text.upper() and _contains_any(text, ["overload", "bandwidth", "capacity", "owner"]),
            "hypothesis must name PAY-810 and identify owner overload/capacity as the root cause",
        ),
        plan_check=lambda actions: (
            not any(a.action_type == "link_dependency" for a in actions),
            "must NOT fabricate a link_dependency action — no real dependency exists, this is a "
            "capacity/ownership problem",
        ),
    ),
    Scenario(
        id="requirement_ambiguity",
        ground_truth="PAY-910 is stalled on genuine requirement ambiguity (guest-checkout scope undefined "
                     "in the spec), waiting on product for a week — not a technical or dependency blocker.",
        target_keys=["PAY-910"],
        comments=[{
            "issue_key": "PAY-910",
            "content": "Blocked — the spec doesn't say whether this should support guest checkout or only "
                       "logged-in users. Been waiting on product for clarification for about a week, can't "
                       "safely proceed in either direction.",
        }],
        diagnosis_check=lambda text, q, hyps: (
            (
                "PAY-910" in text.upper() and _contains_any(text, ["ambiguous", "unclear", "clarif", "spec"])
            ) or (q is not None and "PAY-910" in q.upper()),
            "must surface the requirement ambiguity for PAY-910 (via hypothesis wording or by asking its "
            "own clarifying question) rather than guessing at scope",
        ),
        plan_check=None,  # ungraded — add_comment-to-escalate and move_out_of_sprint are both defensible
    ),
    Scenario(
        id="duplicate_tickets",
        ground_truth="PAY-1010 and PAY-1011 are duplicate tickets — both cover the same payment-webhook "
                     "retry-logic work.",
        target_keys=["PAY-1010", "PAY-1011"],
        comments=[
            {"issue_key": "PAY-1010", "content": "Hold on — isn't this the same work as PAY-1011? Both "
                                                  "tickets are about adding retry logic to the payment "
                                                  "webhook handler, looks like a duplicate."},
            {"issue_key": "PAY-1011", "content": "Confirming PAY-1011 covers webhook retry logic — this "
                                                  "looks identical in scope to PAY-1010, we may have "
                                                  "created this twice."},
        ],
        diagnosis_check=lambda text, q, hyps: (
            "PAY-1010" in text.upper() and "PAY-1011" in text.upper()
            and _contains_any(text, ["duplicate", "same work", "identical"]),
            "hypothesis must name both PAY-1010 and PAY-1011 and identify them as duplicates",
        ),
        plan_check=None,  # ungraded — no dedicated "mark duplicate" action_type exists to check against
    ),
    Scenario(
        id="multi_signal_dependency_and_overload",
        ground_truth="Two independent problems in the same sprint: PAY-1110 is blocked on an undocumented "
                     "dependency on PAY-1111; PAY-1120 is stalled purely from owner overload. Correct "
                     "behavior is a distinct, correctly-attributed hypothesis for each — not merging both "
                     "into one vague statement, and not reporting only one of the two.",
        target_keys=["PAY-1110", "PAY-1120"],
        comments=[
            {"issue_key": "PAY-1110", "content": "PAY-1110 can't proceed until PAY-1111 ships the shared "
                                                  "rate-limiting middleware — no formal Jira link exists "
                                                  "yet, just noting it here."},
            {"issue_key": "PAY-1120", "content": "PAY-1120 has stalled — I'm the only owner and I'm "
                                                  "currently juggling three other critical incidents in "
                                                  "parallel, this is a bandwidth problem, not a technical one."},
        ],
        diagnosis_check=lambda text, q, hyps: (
            (
                any("PAY-1110" in h.statement.upper() and "PAY-1111" in h.statement.upper() for h in hyps)
                and any(
                    "PAY-1120" in h.statement.upper() and _contains_any(h.statement, ["overload", "bandwidth", "capacity"])
                    for h in hyps
                )
            ),
            "must produce a distinct hypothesis for EACH issue: PAY-1110's dependency on PAY-1111, and "
            "PAY-1120's owner overload — not merge both into one hypothesis or report only one of the two",
        ),
    ),
    Scenario(
        id="multi_signal_false_alarm_and_real_risk",
        ground_truth="PAY-1210 is a real blocker (undocumented dependency on PAY-1230); PAY-1220 is a "
                     "false alarm (intentionally slow research spike). Correct behavior is to flag "
                     "PAY-1210 as real risk while NOT recommending an aggressive action on PAY-1220.",
        target_keys=["PAY-1210", "PAY-1220"],
        comments=[
            {"issue_key": "PAY-1210", "content": "PAY-1210 is a real problem — blocked waiting on PAY-1230 "
                                                  "to ship the new pricing API, no formal link recorded yet."},
            {"issue_key": "PAY-1220", "content": "PAY-1220 has been in progress a while but that's "
                                                  "intentional — it's a deliberately slow research spike "
                                                  "per team agreement, not urgent, please don't escalate or "
                                                  "pull it from the sprint over this."},
        ],
        diagnosis_check=lambda text, q, hyps: (
            any("PAY-1210" in h.statement.upper() and "PAY-1230" in h.statement.upper() for h in hyps),
            "must produce a hypothesis correctly naming PAY-1210's real dependency on PAY-1230 — graded "
            "separately from whether PAY-1220 (the false alarm) gets left alone, see plan_check",
        ),
        plan_check=lambda actions: (
            any(a.target_issue_key == "PAY-1230" or a.depends_on_issue_key == "PAY-1230" for a in actions)
            and not any(
                a.target_issue_key == "PAY-1220" and (
                    a.action_type == "move_out_of_sprint"
                    or (a.action_type == "change_priority" and (a.new_priority or "").lower() in ("highest", "high"))
                )
                for a in actions
            ),
            "plan must address the real PAY-1210/PAY-1230 dependency AND must NOT escalate/pull PAY-1220 "
            "(the false alarm) — triage, not uniform treatment of both flagged issues",
        ),
    ),
]


async def run(out: Path | None = None) -> int:
    settings = Settings()
    print(f"sprint-recovery ground-truth scenarios -> provider={settings.llm_provider} model={settings.agent_model}\n"
          + "=" * 78)
    client, model = build_instructor_client(settings)
    outcomes = {}
    n_diag_pass = n_diag_total = 0
    n_plan_pass = n_plan_total = 0
    try:
        for s in SCENARIOS:
            retrieval = ScenarioRetrieval(s.target_keys, s.comments, s.history)
            graph = build_sprint_recovery_graph(
                client, model, NoopSpaceMembershipChecker(), retrieval, NoopJiraActions(),
            ).compile(checkpointer=InMemorySaver())
            cfg = {"configurable": {"thread_id": f"scenario-{s.id}"}}
            initial = initial_recovery_state(SPACE_ID, SPRINT_ID, "Sprint 7 (synthetic)", "u1", "alice")

            result = await graph.ainvoke(initial, config=cfg)
            hyps = result.get("hypotheses") or []
            diag_text = _all_hypotheses_text(hyps)
            diag_question = result.get("clarification_question")
            diag_ok, diag_note = s.diagnosis_check(diag_text, diag_question, hyps)
            n_diag_total += 1
            n_diag_pass += int(diag_ok)

            # Answer every clarifying question generically until the graph moves on to plan — scenario 4/9's
            # diagnosis_check already grades the first question. Answering only once left a model that
            # asked a second time with no plan at all, graded as a plan failure. The graph's own
            # max_clarification_rounds bounds this loop.
            while "__interrupt__" in result and "question" in result["__interrupt__"][0].value:
                result = await graph.ainvoke(Command(resume={"answer": s.clarify_answer}), config=cfg)

            plans = result.get("plans") or []
            actions = _all_plan_actions(plans)
            plan_ok, plan_note = (True, "not graded") if s.plan_check is None else s.plan_check(actions)
            if s.plan_check is not None:
                n_plan_total += 1
                n_plan_pass += int(plan_ok)

            outcomes[s.id] = {"diagnosis": diag_ok, "plan": None if s.plan_check is None else plan_ok}
            tag_d = "[PASS]" if diag_ok else "[FAIL]"
            tag_p = "n/a   " if s.plan_check is None else ("[PASS]" if plan_ok else "[FAIL]")
            print(f"\n--- {s.id} ---")
            print(f"    ground truth: {s.ground_truth}")
            print(f"    diagnosis {tag_d}: {diag_note}")
            for h in hyps:
                print(f"      - ({h.confidence}) {h.statement[:220]}")
            if diag_question:
                print(f"      clarifying_question: {diag_question}")
            print(f"    plan      {tag_p}: {plan_note}")
            for a in actions:
                print(f"      - {a.action_type} target={a.target_issue_key} depends_on={a.depends_on_issue_key} priority={a.new_priority}")
    finally:
        await client.client.close()

    print("\n" + "=" * 78)
    print(f"SUMMARY: diagnosis {n_diag_pass}/{n_diag_total} correct   plan {n_plan_pass}/{n_plan_total} correct "
          f"({len(SCENARIOS) - n_plan_total} scenario(s) had no plan-level ground truth by design)")
    if out is not None:
        out.write_text(json.dumps({
            "model": model,
            "diagnosis": {"passed": n_diag_pass, "total": n_diag_total},
            "plan": {"passed": n_plan_pass, "total": n_plan_total},
            "cases": outcomes,
        }, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="also write the result as JSON (read by eval/eval_gate.py)")
    return asyncio.run(run(parser.parse_args().out))


if __name__ == "__main__":
    raise SystemExit(main())
