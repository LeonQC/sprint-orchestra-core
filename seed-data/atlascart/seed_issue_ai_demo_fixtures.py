#!/usr/bin/env python3
"""Seeds the issues needed to demo the two issue-level AI actions in the ticket detail view:
"Improve description" (`app/description_review/`) and "Link similar work items", built as a
precedent finder (`app/precedent/`).

**Why this exists.** The real AtlasCart corpus can partly demo both, but not their interesting
states: Improve description correctly abstains on it (those descriptions were authored carefully —
87% come back "fine_as_is"), and precedent search's candidate
pool skews toward one subsystem (catalog/search) without a case that intentionally spans two
different subsystems, or one that's a live collision rather than a finished precedent. This adds
exactly the missing states.

**Matching is title+description only — direct product decision.** An earlier version of this fixture
set relied on a candidate's COMMENT THREAD to carry the reusable lesson, because a description alone
rarely states one. That mirrored a real capability the feature briefly had and then dropped: comment
threads turned out to hide a structural failure mode (the model's comment budget could drop the one
comment with the actual substance and keep the administrative filler around it, producing a
takeaway that looked present but was hollow) for a feature meant to stay a fast, low-friction check.
Comment-reading was cut entirely — see app/precedent/service.py's module docstring — so every
fixture below states its reusable lesson IN THE DESCRIPTION ITSELF, matching what the shipped feature
can actually see.

**Four scenarios, each showing a different way "similar" can mean something (or nothing):**

  1. SAME SUBSYSTEM, DONE vs IN-FLIGHT — "Importer retry loop" (done) / "Retry failed supplier
     catalog syncs" (planned). The straightforward case: same component, one finished, one about to
     start.
  2. SAME TECHNICAL PATTERN, DIFFERENT SUBSYSTEM — "Retry order-confirmation emails" (done, retry
     logic for the notification path) / "Retry failed shipping-label print jobs" (planned, retry
     logic for a completely different integration). Demonstrates that matching by title+description
     finds a genuine precedent across subsystems when the underlying technique is the same, not just
     when the feature area is the same.
  3. SAME COMPONENT NAMED EXPLICITLY — "Fix pagination bug in ProductListView" (done) / "Add a
     sort-by-price control to ProductListView" (planned). The case you originally asked about: two
     issues that share a literal component name, with no other vocabulary in common.
  4. OPEN COLLISION, NOT A PRECEDENT — "Add multi-currency price display to checkout" (in_progress)
     / "Show prices in the shopper's local currency at checkout" (planned). Two near-duplicate asks,
     one already being worked. The prompt asks the model to flag this as a collision to coordinate
     on rather than a solved example — a live ticket is not something to "learn from" the way a done
     one is.
  5. NOTHING RELEVANT — "Update the office wifi password rotation doc" (planned, genuinely
     unrelated). The negative control: `none_relevant` is a correct answer, not a missing feature.

Plus one fixture unrelated to precedent search: a description with real, checkable copy-edit defects
(typos, one broken sentence, one genuinely ambiguous line — nothing that would require inventing
content) for the Improve Description "needs_changes" + diff-and-apply flow.

Additive, clearly marked, and deleted+recreated on every run so a demo always starts from a known
state — same shape as `seed_recovery_test_fixtures.py`, which exists for the same reason for the
sprint-recovery workflow.

**Space id is resolved by key ('ATC'), never hardcoded.** `m3_seed.py inject` fully replaces the
AtlasCart space on every run and jira-backend assigns it a fresh id each time (confirmed: it has
changed across real reseeds). A script that hardcodes a space id silently breaks — or silently
targets the WRONG, stale space — the next time anyone reseeds. This one asks jira-backend for the
current id of whichever space has key "ATC" every time it runs, so it can be invoked as its own step
right after `m3_seed.py inject` without hand-editing anything, which is what `refresh_demo_data.sh`
now does.

Steers, not guarantees — every precedent-search fixture is judged by a real LLM call over real
retrieval. If a run goes differently, that is the honest result, not a broken script.

Talks to jira-backend with the internal service token + `X-User-Id`/`X-Username` headers
`InternalServiceAuthFilter` accepts (dev defaults), no login needed — same pattern as every
other direct-to-service script here. Content reaches vectorization-service over Kafka on its own, so
no backfill step is needed; the script waits for the new issues to become searchable before exiting,
since a demo run before indexing finishes would find nothing and look broken.

Usage:
    python seed-data/atlascart/seed_issue_ai_demo_fixtures.py
    python seed-data/atlascart/seed_issue_ai_demo_fixtures.py --space-id 5000018   # skip the lookup
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

import httpx

JIRA_BACKEND_URL = "http://localhost:8081"
VECTORIZATION_URL = "http://localhost:8100"
INTERNAL_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "dev-internal-service-token-min-32-chars-long!!")
USER_ID = "5000154"
USERNAME = "atlascart.maya"
SPACE_KEY = "ATC"

# The marker that makes this script idempotent — see the module docstring.
ISSUE_LABEL = "ai-demo"

HEADERS = {
    "X-Internal-Token": INTERNAL_TOKEN,
    "X-User-Id": USER_ID,
    "X-Username": USERNAME,
}


def _resolve_space_id(client: httpx.Client, explicit: Optional[int]) -> int:
    if explicit is not None:
        return explicit
    resp = client.get("/api/spaces", headers=HEADERS)
    resp.raise_for_status()
    for space in resp.json():
        if space.get("key") == SPACE_KEY:
            return space["id"]
    raise SystemExit(
        f"No space with key {SPACE_KEY!r} found — is jira-backend up and has "
        f"`m3_seed.py inject` been run? (pass --space-id to override this lookup)"
    )


# --- Fixture: a description with real, checkable copy-edit defects ---------------------------------
# Deliberately only defects the reviewer is actually in scope to fix (see
# app/description_review/templates/description_review_system.jinja): misspellings, one broken
# sentence, and one line that genuinely supports two readings. No missing sections, because "add the
# acceptance criteria" is explicitly NOT something this feature is allowed to propose.
MESSY_DESCRIPTION = """<h3>Context</h3>
<p>The nightly supplier import silently drops rows when a price feild is empty. Operations only notice when a prodcut disappears from the storefront, which is usually a day later.</p>
<p>The importer logs the row count it recieved but not the count it actually wrote, so there is nothing to compare against and because of that no alert fires.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — The importer reports skipped rows separately from written rows.</li>
<li>AC2 — Operations are notified when the skipped count is above zero after the run completes.</li>
</ul>
<h3>Notes</h3>
<p>The importer should not stop the run when it finds a bad row and log it.</p>
"""

# --- Scenario 1: same subsystem, done vs planned. Lesson lives in the DESCRIPTION now. -------------
PRECEDENT_DESCRIPTION = """<h3>Context</h3>
<p>Supplier catalog pulls fail intermittently when the upstream API times out, leaving that supplier's products stale until someone re-runs the job by hand.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — A timed-out supplier pull is retried automatically without operator action.</li>
<li>AC2 — A supplier that keeps failing stops being retried and is reported instead.</li>
</ul>
<h3>What we tried and what worked</h3>
<p>A first pass retried inline inside the importer worker on a timer. That made a degraded (not fully down) supplier API worse: every worker retried the same slow endpoint at once and saturated the connection pool, stalling healthy suppliers too.</p>
<p>Replaced that with a per-supplier queue and exponential backoff, so one slow supplier's retries never affect any other supplier's schedule. Added a circuit breaker that stops retrying after three consecutive failures and reports the supplier instead of retrying forever.</p>
"""

CURRENT_DESCRIPTION = """<h3>Context</h3>
<p>When a supplier catalog sync fails we currently leave it alone until an operator notices and re-runs it. We want failed syncs to recover on their own.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — A failed supplier sync is retried automatically.</li>
<li>AC2 — Repeated failures surface to the team instead of retrying forever.</li>
</ul>
"""

# --- Scenario 2: same technical pattern, different subsystem (email vs. label printing). -----------
CROSS_DOMAIN_PRECEDENT_DESCRIPTION = """<h3>Context</h3>
<p>Order confirmation emails occasionally fail to send when the outbound mail relay has a transient error. Customers miss their receipt and support gets the ticket.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — A transient send failure is retried automatically.</li>
<li>AC2 — One slow mail provider never delays sending to a different provider.</li>
</ul>
<h3>What we tried and what worked</h3>
<p>Retrying right where the send call happens caused a burst of simultaneous retries during a provider slowdown, which briefly saturated the outbound connection pool for every provider, not just the slow one.</p>
<p>Fixed by queuing failed sends with exponential backoff per recipient domain instead of retrying inline. Each domain's backoff runs independently, so a flaky provider no longer slows down mail to everyone else.</p>
"""

CROSS_DOMAIN_CURRENT_DESCRIPTION = """<h3>Context</h3>
<p>Shipping label print jobs occasionally fail when the print vendor's API has a transient error, and right now the order just sits without a label until someone notices.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — A failed label print job is retried automatically.</li>
<li>AC2 — One vendor outage should not delay label printing that doesn't use that vendor.</li>
</ul>
"""

# --- Scenario 3: same component named explicitly, otherwise unrelated work. ------------------------
COMPONENT_PRECEDENT_DESCRIPTION = """<h3>Context</h3>
<p>ProductListView shows the wrong products on page 2 and later whenever the last page has fewer items than a full page.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — Every page after the first shows the correct products for that page.</li>
<li>AC2 — A last page with fewer than a full page of items still shows the right products, not a repeat of an earlier page.</li>
</ul>
<h3>What we tried and what worked</h3>
<p>The page offset was computed from the page SIZE configured in the UI instead of the page size actually returned by the API, so a short last page threw off every offset after it. Fixed by computing the next page's offset from the API's own returned count, not the requested one.</p>
"""

COMPONENT_CURRENT_DESCRIPTION = """<h3>Context</h3>
<p>Sellers want to sort ProductListView by price instead of only the default relevance order.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — ProductListView offers a price low-to-high and high-to-low sort control.</li>
<li>AC2 — Changing the sort order keeps the current page's filters applied.</li>
</ul>
"""

# --- Scenario 4: an open collision, not a precedent to learn from. ---------------------------------
COLLISION_IN_PROGRESS_DESCRIPTION = """<h3>Context</h3>
<p>Beta shoppers outside the US see totals in USD even when their account locale implies a different currency, which is confusing at checkout.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — Checkout displays the order total converted to the shopper's account-locale currency.</li>
<li>AC2 — The underlying charge still runs in USD; only the displayed total changes.</li>
</ul>
"""

COLLISION_CURRENT_DESCRIPTION = """<h3>Context</h3>
<p>Support keeps getting confused-shopper tickets about checkout showing USD regardless of where the shopper is. We want checkout to show prices in the shopper's own local currency.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — Checkout shows the total in the shopper's local currency.</li>
<li>AC2 — The receipt shows both the local-currency total and the USD amount actually charged.</li>
</ul>
"""

# --- Scenario 5: nothing in the space is relevant. --------------------------------------------------
UNRELATED_DESCRIPTION = """<h3>Context</h3>
<p>The office wifi password rotation instructions on the internal wiki still reference the old building and the previous access point vendor. Update the page so new starters can follow it.</p>
<h3>Acceptance criteria</h3>
<ul>
<li>AC1 — The page names the current building and vendor.</li>
<li>AC2 — A new starter can follow it without asking anyone.</li>
</ul>
"""


def _cleanup(client: httpx.Client, space_id: int) -> None:
    resp = client.get(f"/api/spaces/{space_id}/issues", headers=HEADERS)
    resp.raise_for_status()
    stale = [
        i for i in resp.json()
        if ISSUE_LABEL in [str(lbl).lower() for lbl in (i.get("labels") or [])]
    ]
    for issue in stale:
        client.delete(f"/api/spaces/{space_id}/issues/{issue['issueKey']}", headers=HEADERS)
    if stale:
        print(f"  cleaned up {len(stale)} issue(s) from a previous run: "
              f"{', '.join(i['issueKey'] for i in stale)}")


def _create_issue(
    client: httpx.Client, space_id: int, title: str, description: str,
    status: str = "planned", issue_type: str = "task", points: int = 3,
) -> dict:
    resp = client.post(
        f"/api/spaces/{space_id}/issues",
        json={
            "title": title, "description": description, "issueType": issue_type, "status": status,
            "priority": "medium", "assigneeId": int(USER_ID), "reporterId": int(USER_ID),
            "storyPoints": points, "labels": [ISSUE_LABEL],
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def _wait_until_searchable(space_id: int, issue_key: str, timeout_s: int = 90) -> bool:
    """Blocks until the new issue is retrievable, so a demo run right after seeding isn't empty.

    Indexing is asynchronous (jira-backend -> Kafka -> vectorization-service), so there is a real
    window where the issue exists in Postgres, renders fine in the UI, and is invisible to precedent
    search. Polling the structured query is enough: the row and its chunks are written by the same
    consumer, so once the key is queryable its text is indexed too.
    """
    deadline = time.time() + timeout_s
    with httpx.Client(base_url=VECTORIZATION_URL, timeout=10.0) as vec:
        while time.time() < deadline:
            try:
                resp = vec.post("/issues/query", json={"space_ids": [space_id],
                                                       "issue_keys": [issue_key], "limit": 1})
                if resp.status_code == 200 and resp.json().get("total_count", 0) > 0:
                    return True
            except httpx.HTTPError:
                pass  # vectorization-service still starting, or a transient blip; keep waiting
            time.sleep(3)
    return False


def main(explicit_space_id: Optional[int]) -> int:
    with httpx.Client(base_url=JIRA_BACKEND_URL, timeout=30.0) as client:
        space_id = _resolve_space_id(client, explicit_space_id)
        print(f"seeding issue-AI demo fixtures into space {space_id} (key={SPACE_KEY})...")
        _cleanup(client, space_id)

        messy = _create_issue(
            client, space_id, "[ai-demo] Bulk price import silently drops rows",
            MESSY_DESCRIPTION, status="planned", issue_type="bug", points=3,
        )
        print(f"  {messy['issueKey']}  Improve description -> expect needs_changes (typos + one ambiguous line)")

        precedent = _create_issue(
            client, space_id, "[ai-demo] Importer retry loop for transient supplier timeouts",
            PRECEDENT_DESCRIPTION, status="done", issue_type="story", points=5,
        )
        current = _create_issue(
            client, space_id, "[ai-demo] Retry failed supplier catalog syncs automatically",
            CURRENT_DESCRIPTION, status="planned", issue_type="story", points=5,
        )
        print(f"  {current['issueKey']}  Link similar work items -> expect {precedent['issueKey']} "
              f"(same subsystem, done, lesson in its own description)")

        cross_precedent = _create_issue(
            client, space_id, "[ai-demo] Retry order-confirmation email sends with backoff",
            CROSS_DOMAIN_PRECEDENT_DESCRIPTION, status="done", issue_type="story", points=5,
        )
        cross_current = _create_issue(
            client, space_id, "[ai-demo] Add automatic retry to shipping label print jobs",
            CROSS_DOMAIN_CURRENT_DESCRIPTION, status="planned", issue_type="story", points=3,
        )
        print(f"  {cross_current['issueKey']}  Link similar work items -> expect {cross_precedent['issueKey']} "
              f"(different subsystem, same retry pattern)")

        component_precedent = _create_issue(
            client, space_id, "[ai-demo] Fix pagination bug in ProductListView",
            COMPONENT_PRECEDENT_DESCRIPTION, status="done", issue_type="bug", points=3,
        )
        component_current = _create_issue(
            client, space_id, "[ai-demo] Add a sort-by-price control to ProductListView",
            COMPONENT_CURRENT_DESCRIPTION, status="planned", issue_type="story", points=3,
        )
        print(f"  {component_current['issueKey']}  Link similar work items -> expect "
              f"{component_precedent['issueKey']} (same component named explicitly, nothing else shared)")

        collision_in_progress = _create_issue(
            client, space_id, "[ai-demo] Add multi-currency price display to checkout",
            COLLISION_IN_PROGRESS_DESCRIPTION, status="in_progress", issue_type="story", points=5,
        )
        collision_current = _create_issue(
            client, space_id, "[ai-demo] Show prices in the shopper's local currency at checkout",
            COLLISION_CURRENT_DESCRIPTION, status="planned", issue_type="story", points=5,
        )
        print(f"  {collision_current['issueKey']}  Link similar work items -> expect "
              f"{collision_in_progress['issueKey']} flagged as an in-progress COLLISION, not a "
              f"finished precedent")

        unrelated = _create_issue(
            client, space_id, "[ai-demo] Update the office wifi password rotation doc",
            UNRELATED_DESCRIPTION, status="planned", issue_type="task", points=1,
        )
        print(f"  {unrelated['issueKey']}  Link similar work items -> expect none_relevant")

        new_keys = [
            precedent["issueKey"], current["issueKey"],
            cross_precedent["issueKey"], cross_current["issueKey"],
            component_precedent["issueKey"], component_current["issueKey"],
            collision_in_progress["issueKey"], collision_current["issueKey"],
            unrelated["issueKey"],
        ]
        print(f"\nwaiting for {len(new_keys)} new issues to become searchable "
              f"(Kafka -> vectorization-service)...")
        ok = all(_wait_until_searchable(space_id, key) for key in new_keys)
        if ok:
            print("  indexed — precedent search will see them.")
        else:
            print("  WARNING: still not searchable. The issues exist and the UI will show them, but "
                  "precedent search may return nothing until indexing catches up.", file=sys.stderr)

        print("\ndone. In the ticket detail view's 'Improve Story' menu, try:")
        print(f"  {messy['issueKey']:<9} -> Improve description")
        print(f"  {current['issueKey']:<9} -> Link similar work items -> {precedent['issueKey']}")
        print(f"  {cross_current['issueKey']:<9} -> Link similar work items -> {cross_precedent['issueKey']}")
        print(f"  {component_current['issueKey']:<9} -> Link similar work items -> {component_precedent['issueKey']}")
        print(f"  {collision_current['issueKey']:<9} -> Link similar work items -> {collision_in_progress['issueKey']} (open collision)")
        print(f"  {unrelated['issueKey']:<9} -> Link similar work items -> nothing relevant")
        return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", type=int, default=None,
                        help="Skip the by-key space lookup and use this space id directly.")
    args = parser.parse_args()
    raise SystemExit(main(args.space_id))
