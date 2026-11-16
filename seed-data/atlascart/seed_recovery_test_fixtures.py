#!/usr/bin/env python3
"""Seeds the sprints needed to demo//test every branch of `app/sprint_recovery/graph.py`.

**Why this exists.** Verified live: all 7 real AtlasCart sprints
are historical and fully `done`, so every one of them correctly reports "no risk found". That is the
honest result for that data -- but it means the real corpus alone cannot demonstrate the parts of
this workflow that actually matter (plan -> approve -> execute -> re-check -> recovered/escalated).
These fixtures fill exactly that gap, and nothing else: they are additive, clearly marked, and
deleted+recreated on every run so a demo always starts from a known state.

**Idempotent by construction.** Every sprint this creates carries `[recovery-demo]` in its goal and
every issue carries the `recovery-demo` label. On each run, everything bearing those markers is
deleted first. Re-run it as many times as you like -- including right after a demo that moved an
issue out of its sprint -- and you get the same starting state back. Nothing outside those markers
is ever touched.

**What each fixture is engineered to do.** The one deterministic per-issue signal that exists
(`_detect_risk_signals`'s `long_in_progress`) fires on any `in_progress` issue; each fixture's
description/comment then steers -- does NOT force, plan generation is a real LLM call on real
retrieved data -- the diagnosis toward a specific outcome:

  1. "Sprint 8 - Checkout Hardening"  -> RECOVERED. Evidence says the issue was pulled into the
     sprint by mistake and must come out. The plan proposes `move_out_of_sprint`; once committed the
     issue leaves the sprint-scoped risk query entirely, so the next re-check finds nothing at risk.
  2. "Sprint 9 - Fraud Provider Rollout" -> ESCALATED. Evidence says the blocker is external (vendor
     legal, no ETA) and the issue must stay in the sprint. None of the four available action types
     can clear an external blocker, so `long_in_progress` keeps firing until `max_escalation_rounds`
     is spent and the workflow hands off to a human.
  3. "Sprint 10 - Search Relevance" -> NO RISK FOUND. All issues `done`, zero signals; the workflow
     short-circuits to a clean "All clear" without spending an LLM call at all.

Steers, not guarantees: the model decides, so a run can go differently.

Talks to jira-backend with the internal service token + `X-User-Id`/`X-Username` headers
`InternalServiceAuthFilter` accepts (dev defaults, matching jira-backend's application.yml and
`ai-service/app/config.py`) -- no login needed, same pattern as every other direct-to-service
script here. Sprint-membership and content changes now reach vectorization-service over Kafka on
their own, so no backfill step is needed; Redis is flushed at the end only because a stale cached
`/ask` answer is indistinguishable from a real regression.

Usage:
    python seed-data/atlascart/seed_recovery_test_fixtures.py --space-id 5000074
    python seed-data/atlascart/seed_recovery_test_fixtures.py --space-id 5000074 --cleanup-only
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
CLEAR_ANSWER_CACHE = (
    'for p in "aicache:*" "litellm.response_cache:*"; do '
    'redis-cli --scan --pattern "$p" | grep -E "^aicache:|:[0-9a-f]{64}\$" | xargs -r redis-cli DEL >/dev/null; done'
)
JIRA_BACKEND_URL = "http://localhost:8081"
INTERNAL_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "dev-internal-service-token-min-32-chars-long!!")
USER_ID = "5000154"
USERNAME = "atlascart.maya"

# Markers that make this script idempotent — see the module docstring.
SPRINT_MARKER = "[recovery-demo]"
ISSUE_LABEL = "recovery-demo"

HEADERS = {
    "X-Internal-Token": INTERNAL_TOKEN,
    "X-User-Id": USER_ID,
    "X-Username": USERNAME,
}


def _cleanup(client: httpx.Client, space_id: int) -> None:
    """Deletes every issue and sprint this script previously created, identified purely by marker.

    Issues go first: a sprint delete would otherwise orphan them into the backlog, where they'd keep
    firing `long_in_progress` on the *next* run's diagnosis and quietly corrupt the fixture. Issues
    are found by label rather than by sprint membership on purpose — a completed "recovered" demo has
    already moved its issue out of the sprint, so sprint membership can't find it any more.
    """
    issues = client.get(f"/api/spaces/{space_id}/issues", headers=HEADERS)
    issues.raise_for_status()
    stale_issues = [
        i for i in issues.json()
        if ISSUE_LABEL in [str(lbl).lower() for lbl in (i.get("labels") or [])]
    ]
    for issue in stale_issues:
        client.delete(f"/api/spaces/{space_id}/issues/{issue['issueKey']}", headers=HEADERS)

    sprints = client.get(f"/api/spaces/{space_id}/sprints", headers=HEADERS)
    sprints.raise_for_status()
    stale_sprints = [s for s in sprints.json() if SPRINT_MARKER in (s.get("goal") or "")]
    for sprint in stale_sprints:
        client.delete(f"/api/spaces/{space_id}/sprints/{sprint['id']}", headers=HEADERS)

    if stale_issues or stale_sprints:
        print(f"  cleaned up {len(stale_issues)} issue(s) and {len(stale_sprints)} sprint(s) from a previous run")


def _create_sprint(client: httpx.Client, space_id: int, name: str, goal: str) -> int:
    """`status="future"`, not `"active"` -- **found live**: jira-backend enforces one active sprint
    per space and the real AtlasCart Sprint 7 holds that slot, so `"active"` 400s here. The workflow
    itself never reads sprint status, so this changes nothing about how the demo behaves. The UI's
    "AI recovery" button is gated to non-`completed` sprints (a recovery check on a finished sprint
    is meaningless), which `future` satisfies.

    No `startDate`/`endDate` sent at all -- **found live, the second date-related bug in this
    function**: `_detect_risk_signals` now reads sprint dates too (a "too early in the sprint to
    judge completion" grace period, added after direct product feedback), and there is no
    non-overlapping date range these fixtures could occupy that also puts "today" inside it: the real
    AtlasCart Sprint 7 already spans today by design (see `refresh_demo_data.sh`), the only free gap
    is entirely in the future (after Sprint 7 ends), and jira-backend rejects overlapping ranges
    outright. A future-dated range makes "today" read as day-zero-or-earlier, which is exactly what
    the new grace period is designed to treat as "too early to have an opinion" -- silently emptying
    signals that used to fire. Sending no dates at all sidesteps the false choice: null dates can't
    overlap anything, and every date-aware check in `_detect_risk_signals` already falls back to its
    plain, date-independent rule when it has no dates to reason about -- exactly this case.
    """
    resp = client.post(
        f"/api/spaces/{space_id}/sprints",
        json={"name": name, "goal": f"{SPRINT_MARKER} {goal}", "status": "future"},
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _create_issue(
    client: httpx.Client, space_id: int, sprint_id: int, title: str, description: str,
    status: str = "in_progress", assignee_id: int = None,
) -> dict:
    resp = client.post(
        f"/api/spaces/{space_id}/issues",
        json={
            "title": title, "description": description, "issueType": "task", "status": status,
            "priority": "high", "assigneeId": int(assignee_id if assignee_id is not None else USER_ID),
            "reporterId": int(USER_ID),
            "sprintId": sprint_id, "storyPoints": 5, "labels": [ISSUE_LABEL],
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def _add_comment(client: httpx.Client, issue_id: int, content: str) -> None:
    resp = client.post(
        f"/api/issues/{issue_id}/comments",
        json={"authorId": int(USER_ID), "content": content},
        headers=HEADERS,
    )
    resp.raise_for_status()


def _backdate_issue(issue_id: int, days_ago: int) -> None:
    """Pushes `updated_at` into the past directly in jira-backend's database. Not exposed through the
    REST API at all -- `Issue.java`'s `@PreUpdate` always stamps `Instant.now()`, so there's no request
    body field that could do this.

    **Why this exists**: `_detect_risk_signals`'s `long_in_progress`/`blocked_no_flag` now require real
    staleness (`_STALE_AFTER_DAYS`), not just the status value -- found live, this is also the fix for
    "can the system tell if an owner isn't updating their ticket." A freshly-created fixture issue has
    `updated_at = now()` regardless of what its description claims, so without this every "still
    waiting", "stalled for days" fixture would read as perfectly fresh and never trip a signal at all.
    """
    subprocess.run(
        ["docker", "exec", "orchestra-postgres", "psql", "-U", "poc", "-d", "pocdb", "-c",
         f"UPDATE issues SET updated_at = now() - interval '{int(days_ago)} days' WHERE id = {int(issue_id)}"],
        check=True, capture_output=True,
    )


def _wait_for_kafka_lag_to_drain(timeout_seconds: float = 60.0) -> None:
    """**Found live, not theorized**: `jira.content.ingestion` is a single-partition topic (the
    partition key is `issueKey`, so per-issue ordering matters more than throughput here) — everything
    this script's own issue/comment creation just published, plus every other real write anyone else
    has made recently, is strictly serialized through vectorization-service's one consumer thread. This
    script's own backfill step makes the *data* correct immediately (a direct write, bypassing Kafka),
    but it does nothing about that backlog still draining in the background. A demo started right after
    "ready" can approve a plan whose own commit actions queue in *after* that backlog — reproduced
    live: a `change_priority` action's watermark took several minutes to catch up because it was queued
    behind ~1800 backlogged messages from this exact reseed, well past `reevaluate_node`'s 10s
    `_await_index_catch_up` timeout, surfacing as an honest but confidence-shaking "unconfirmed sync"
    badge on the very first demo action. Waiting here for the consumer group's lag to hit zero before
    ever printing "ready" means the backlog this script itself created is gone before a human — or the
    workflow's own first commit — could ever queue in behind it.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = subprocess.run(
            ["docker", "exec", "orchestra-kafka", "kafka-consumer-groups", "--bootstrap-server", "localhost:9092",
             "--describe", "--group", "vectorization-service"],
            capture_output=True, text=True,
        )
        # Columns: GROUP TOPIC PARTITION CURRENT-OFFSET LOG-END-OFFSET LAG CONSUMER-ID ... (0-indexed).
        lag = None
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 6 and parts[1] == "jira.content.ingestion":
                lag = int(parts[5])
                break
        if lag == 0:
            return
        if time.monotonic() >= deadline:
            print(
                f"  (gave up waiting for the Kafka consumer to drain after {timeout_seconds:.0f}s, "
                f"lag={lag} — a demo run right away may see an honest 'unconfirmed sync' badge)"
            )
            return
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", type=int, required=True)
    parser.add_argument("--jira-url", default=JIRA_BACKEND_URL)
    # Fixtures add issues to the space, which changes its counts for evals; this removes them again.
    parser.add_argument("--cleanup-only", action="store_true",
                        help="delete the recovery-demo sprints and issues, create nothing")
    args = parser.parse_args()

    created: List[tuple] = []
    with httpx.Client(base_url=args.jira_url, timeout=15.0) as client:
        print(f"resetting recovery-demo fixtures in space {args.space_id} ...")
        _cleanup(client, args.space_id)
        if args.cleanup_only:
            subprocess.run(["docker", "exec", "orchestra-redis", "sh", "-c", CLEAR_ANSWER_CACHE],
                           check=False, capture_output=True)
            print("done: recovery-demo fixtures removed")
            return 0

        # 1. RECOVERED
        s1 = _create_sprint(
            client, args.space_id, "Sprint 8 - Checkout Hardening",
            "Harden the checkout flow ahead of the public beta.",
        )
        i1 = _create_issue(
            client, args.space_id, s1, "Legacy CSV export button (deprecated)",
            "This ticket is actively consuming committed sprint capacity even though it was never "
            "part of the sprint's committed scope -- it was mistakenly pulled in during backlog "
            "grooming and is now at real risk of causing the team to miss the sprint's committed "
            "goal. Corrective action required now: remove it from the current sprint immediately to "
            "protect the committed capacity for the rest of the team's work. This is an active scope "
            "risk, not a resolved or false-alarm situation.",
        )
        _add_comment(
            client, i1["id"],
            "PM confirmed this is actively at risk of blowing our committed capacity -- pull it out "
            "of the current sprint now, do not leave it in place.",
        )
        _backdate_issue(i1["id"], days_ago=3)
        created.append(("RECOVERED", s1, "Sprint 8 - Checkout Hardening", i1["issueKey"]))

        # 2. ESCALATED after the full replan loop.
        # Three issues, not one, and each one's evidence forbids removal in the strongest terms.
        # **Found live**: with a single removable issue, the model complied for two rounds and then
        # de-scoped it on round three (`move_out_of_sprint`), which emptied the sprint and ended in
        # `recovered` — a reasonable call on its part, but it meant the escalation branch never fired.
        # Three issues make one de-scope insufficient to clear the risk, and the compliance framing
        # gives the model a concrete reason not to reach for removal at all.
        s2 = _create_sprint(
            client, args.space_id, "Sprint 9 - Fraud Provider Rollout",
            "Integrate the third-party fraud-scoring provider.",
        )
        blocked_specs = [
            ("Enable third-party fraud-scoring provider",
             "the vendor's legal team completing their data-processing agreement review"),
            ("Sign off fraud-scoring data residency review",
             "the external data-protection auditor returning their residency assessment"),
            ("Obtain PCI attestation for the fraud provider",
             "the payment processor issuing the updated PCI attestation letter"),
        ]
        first_key = None
        for title, blocker in blocked_specs:
            issue = _create_issue(
                client, args.space_id, s2, title,
                f"Blocked on {blocker} -- there is no engineering task that can move this forward and "
                "no ETA has been given. This is a committed compliance deliverable: it must NOT be "
                "removed from the sprint, must NOT be de-scoped, and must NOT be reassigned to "
                "another team, because doing so would breach the audit commitment the compliance "
                "stakeholder signed off on. Leave it in the sprint and escalate to a human decision "
                "maker instead.",
                # `blocked`, not `in_progress` -- **found live**: the evidence text has always said
                # these are blocked, but the status field disagreed, and a real reviewer would notice
                # that mismatch immediately. Now that `_detect_risk_signals` has a real `blocked_no_flag`
                # check (see graph.py), the status field can just tell the truth.
                status="blocked",
            )
            _add_comment(
                client, issue["id"],
                "Checked again today, still waiting on the external party. Nothing engineering can do "
                "here, and per compliance this cannot be pulled out of the sprint.",
            )
            _backdate_issue(issue["id"], days_ago=5)
            first_key = first_key or issue["issueKey"]
        created.append(("ESCALATED", s2, "Sprint 9 - Fraud Provider Rollout", f"{first_key} +2 more"))

        # 3. NO RISK FOUND — everything done, so zero signals fire and no LLM call is made at all.
        s3 = _create_sprint(
            client, args.space_id, "Sprint 10 - Search Relevance",
            "Ship the improved search ranking model.",
        )
        i3 = _create_issue(
            client, args.space_id, s3, "Tune search ranking weights",
            "Reweighted the lexical/vector blend and shipped it behind a flag. Verified in staging, "
            "rolled out to 100%, no regressions reported.",
            status="done",
        )
        created.append(("NO RISK", s3, "Sprint 10 - Search Relevance", i3["issueKey"]))

        # 4. Sprint-level risk with no single ticket to blame — the only signal is the sprint-wide
        # completion forecast, which carries no issue_key. `plan_node` still plans directly against
        # the sprint roster (see `_build_sprint_snapshot`/`known_issue_keys` in graph.py), so this
        # demonstrates a plan whose target issue never independently tripped any signal of its own.
        # Every issue is `planned` (so `long_in_progress`/`blocked_no_flag` never fire) while the
        # sprint is 0% complete (so `low_completion_forecast` does).
        #
        # **Found live, and a real bug in this exact fixture**: every issue this script creates
        # defaults to the same assignee (`_create_issue`'s `assignee_id` param). That was harmless
        # before `owner_overloaded` existed, but once it did, 3 planned issues all owned by one person
        # deterministically tripped `owner_overloaded` — which carries a real issue_key (see that
        # signal's implementation) — collapsing this fixture straight back into the single-ticket plan
        # path it exists to NOT demonstrate. Spreading these across 3 different people fixes it, and
        # is more honest anyway: "the whole team was pulled onto an incident" is a team-wide capacity
        # story, not one person's overload.
        s4 = _create_sprint(
            client, args.space_id, "Sprint 11 - Mobile Beta Prep",
            "Prepare the mobile client for the public beta.",
        )
        team = [int(USER_ID), 5000155, 5000156]  # atlascart.maya, .liam, .emma
        for title, owner in zip(
            ("Draft mobile beta rollout plan", "Prepare app store release notes",
             "Set up mobile crash reporting"),
            team,
        ):
            _create_issue(
                client, args.space_id, s4, title,
                "Scheduled for this sprint. Not started yet — the team has not picked this up.",
                status="planned", assignee_id=owner,
            )
        created.append(("SPRINT-LEVEL", s4, "Sprint 11 - Mobile Beta Prep", "3 planned issues, 3 owners"))

    # **Found live**: every write above except `_backdate_issue` goes through jira-backend's REST API,
    # which now fires a real Kafka event vectorization-service picks up on its own within a couple of
    # seconds -- normally no explicit re-sync is needed at all. `_backdate_issue` is the one exception:
    # it's a raw SQL UPDATE against jira-backend's own database, deliberately bypassing the app (see
    # its docstring for why), so it publishes nothing. Without this backfill, vectorization-service's
    # copy of every backdated issue would keep showing the original creation-time `updated_at` forever,
    # and every "still waiting for days" fixture would look perfectly fresh and never trip a signal.
    print("backfilling vectorization-service (picks up the backdated timestamps the Kafka events "
          "above never carried) ...")
    vec_dir = REPO_ROOT / "vectorization-service"
    vec_python = vec_dir / ".venv" / "bin" / "python3"
    env = {**os.environ, "PYTHONPATH": str(vec_dir)}
    subprocess.run(
        [str(vec_python) if vec_python.exists() else sys.executable,
         "scripts/backfill_jira_backend.py", "--space-ids", str(args.space_id)],
        cwd=str(vec_dir), env=env, check=True,
    )

    # Cached answers only: the same Redis holds login sessions, per-user AI quotas and LiteLLM spend.
    print("clearing cached AI answers in Redis ...")
    subprocess.run(["docker", "exec", "orchestra-redis", "sh", "-c", CLEAR_ANSWER_CACHE],
                   check=False, capture_output=True)

    print("waiting for vectorization-service's Kafka consumer to drain the backlog this script's own "
          "issue/comment creation just published (single-partition topic, strictly serialized) ...")
    _wait_for_kafka_lag_to_drain()

    print("\nready. In the UI: Backlog page -> the sprint -> \"AI recovery\" button.\n")
    for outcome, sprint_id, name, issue_key in created:
        print(f"  [{outcome:17s}] sprint_id={sprint_id}  {name}  ({issue_key})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
