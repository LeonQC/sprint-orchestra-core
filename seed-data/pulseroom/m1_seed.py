#!/usr/bin/env python3
"""Seed PRM — PulseRoom (Synthetic M1): 1-month telehealth clinic & prescription queue board.

Designed as the second engineering workspace next to ATC so cross-workspace semantic + reasoning
comparisons have real parallel stories (idempotency, retry double-apply, stale resend link,
missing index handler, fail-closed timeout, audited correction vs raw DB edit, gateway identity
header stripping, uniform 404 anti-enumeration, monolith-first decision, active-sprint blockers).

Idempotent: re-running replaces PRM rows in place. Grants access to every user/group on ATC plus
common local demo users (alice, john, charles, michael23fire) so the board is visible in the UI.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import psycopg

DB_URL = "postgresql://poc:poc123@localhost:5432/pocdb"
SPACE_KEY = "PRM"
SPACE_NAME = "PulseRoom (Synthetic M1)"
SPACE_COLOR = "#0D9488"

SPRINTS = [
    ("S1", "PulseRoom Sprint 1",
     "Stand up the clinical intake API and let a patient book a video visit in staging.",
     date(2026, 10, 5), date(2026, 10, 11), "completed", 1),
    ("S2", "PulseRoom Sprint 2",
     "Make telehealth check-in and prescription handoff safe under reconnects and retries.",
     date(2026, 10, 12), date(2026, 10, 18), "completed", 2),
    ("S3", "PulseRoom Sprint 3",
     "Give clinical operations auditable recovery tools and reliable patient notifications.",
     date(2026, 10, 19), date(2026, 10, 25), "completed", 3),
    ("S4", "PulseRoom Sprint 4",
     "Prove triage caching, signaling latency, and compliance gates before pilot clinics open.",
     date(2026, 10, 26), date(2026, 11, 8), "active", 4),
]

# (key, parent_key, sprint_code, issue_type, status, priority, points, title, description)
ISSUES = [
    (
        "PRM-1", None, None, "epic", "done", "high", None,
        "Launch core telehealth intake and visit scheduling",
        "Context:\nBuild the initial patient intake, chart lookup, and video-visit booking flow.\n\n"
        "Acceptance criteria:\n- Patients can book a slot and clinicians see the queue.",
    ),
    (
        "PRM-2", None, None, "epic", "done", "high", None,
        "Harden prescription queue and clinical safety controls",
        "Context:\nEnsure check-in, renewals, and pharmacy actions are repeat-safe and audited.\n\n"
        "Acceptance criteria:\n- No duplicate rooms or double-counted dosage on retry.",
    ),
    (
        "PRM-3", None, None, "epic", "in_progress", "high", None,
        "Prepare PulseRoom for the multi-clinic pilot",
        "Context:\nClose triage cache invalidation and cold-turn signaling latency before pilot launch.\n\n"
        "Acceptance criteria:\n- Triage view reflects allergy edits and signaling meets P95 target.",
    ),
    (
        "PRM-4", "PRM-1", "S1", "story", "done", "high", 5,
        "Keep a single clinical API deployment for the M1 pilot",
        "Context:\nEngineers debated splitting video signaling, intake, and prescription workflows into "
        "separate microservices before the pilot.\n\n"
        "Decision:\nNo service split for now. Keep one clinical API deployment during M1 and split "
        "signaling or prescription services only after live pilot telemetry shows a measured CPU or "
        "connection-pool bottleneck that vertical scaling cannot absorb.\n\n"
        "Acceptance criteria:\n- Single deployable artifact documented in architecture notes.",
    ),
    (
        "PRM-5", "PRM-1", "S1", "story", "done", "medium", 3,
        "Allow urgent-care intake before full insurance card upload",
        "Context:\nWalk-in telehealth patients often lacked their card photo on mobile.\n\n"
        "Decision:\nUrgent-care triage intake is not blocked on insurance card upload; coverage "
        "verification runs before prescription sign-off instead.",
    ),
    (
        "PRM-6", "PRM-1", "S1", "task", "done", "medium", 3,
        "Preserve partially filled intake form across mobile browser restarts",
        "Context:\nA valid draft intake questionnaire is restored after a page refresh or mobile "
        "browser restart. Expired or tampered draft tokens are wiped with one audit warning and the "
        "patient sees a clean blank intake form instead of a crash screen.",
    ),
    (
        "PRM-7", "PRM-1", "S1", "story", "done", "medium", 3,
        "Basic symptom keyword lookup in the clinician triage bar",
        "Context:\nThe triage search bar uses an exact case-insensitive substring match over chief "
        "complaint and medication names. Medical abbreviation expansion and fuzzy typo matching were "
        "explicitly left out of scope for Sprint 1.",
    ),
    (
        "PRM-8", "PRM-1", "S1", "bug", "done", "medium", 2,
        "Reject lab CSV import rows with missing or negative reference ranges",
        "Context:\nWhen a clinic uploads a lab-panel CSV with a blank reference cell or a negative "
        "threshold, the importer rejects the row with its original CSV line number. A true zero lower "
        "bound is kept distinct from a blank cell.",
    ),
    (
        "PRM-9", "PRM-1", "S1", "story", "done", "high", 3,
        "Return identical 404 for unknown and archived medical record numbers",
        "Context:\nBoth a never-issued MRN (medical record number) and a patient chart archived under "
        "a privacy hold return the exact same 404 response body and status. This is deliberate so an "
        "external caller cannot probe whether a specific person has ever been treated at the clinic.",
    ),
    (
        "PRM-10", "PRM-1", "S1", "story", "done", "high", 5,
        "Validate formulary CSV uploads and reject conflicting duplicate NDC rows",
        "Context:\nWhen a pharmacy formulary file lists the same drug NDC twice with different maximum "
        "daily doses, the importer rejects the upload, reports both line numbers, and writes nothing "
        "to PostgreSQL.",
    ),
    (
        "PRM-11", "PRM-2", "S2", "story", "done", "highest", 5,
        "Strip client-supplied clinician role headers at the API gateway",
        "Context:\nThe API gateway strips any client-supplied `X-Provider-Role` and `X-Clinician-Id` "
        "headers before attaching the role from the verified staff session token, stopping a patient "
        "browser from spoofing a prescribing doctor. Public clinic-hours and symptom-checker pages "
        "continue to work without a session.",
    ),
    (
        "PRM-12", "PRM-2", "S2", "task", "done", "medium", 3,
        "Agree pilot device matrix for video visits",
        "Context:\nDesktop Chrome, Safari on iOS, and Chrome on one mid-range Android handset were "
        "agreed for M1 pilot verification; legacy tablet browsers stay out of scope.",
    ),
    (
        "PRM-13", "PRM-2", "S2", "story", "done", "high", 5,
        "Distinguish waiting-room visits from dispensed prescriptions in cancellation rules",
        "Context:\nOperations cannot cancel a visit or prescription once the pharmacy has marked the "
        "prescription dispensed. Dispensed orders must go through the clinical reversal / recall "
        "workflow instead of simple cancellation.",
    ),
    (
        "PRM-14", "PRM-2", "S2", "bug", "done", "highest", 5,
        "Mobile reconnect during check-in creates two live video visit rooms",
        "Context:\nA patient on unstable clinic Wi-Fi tapped 'Join waiting room' twice across a brief "
        "reconnect and the system provisioned two video rooms (VISIT-2041 and VISIT-2042) and double-"
        "booked two 15-minute slots on the on-call physician's calendar, even though only one copay "
        "authorization succeeded.\n\n"
        "Root cause:\nThe check-in handler inserted the visit room before checking whether the "
        "appointment's check-in idempotency token had already been consumed. Gateway logs showed two "
        "distinct client request IDs reaching the API with 201 responses, ruling out an automatic "
        "load-balancer retry.",
    ),
    (
        "PRM-15", "PRM-2", "S2", "task", "done", "highest", 3,
        "Enforce check-in idempotency key before room and calendar allocation",
        "Context:\nPermanent API fix split out of PRM-14: a repeated check-in key returns the existing "
        "video room and calendar hold before any new room or clinician slot is allocated.",
    ),
    (
        "PRM-16", "PRM-2", "S2", "task", "done", "high", 2,
        "Disable Join waiting room button while check-in request is in flight",
        "Context:\nImmediate mobile UI mitigation for PRM-14 so a slow connection cannot fire a second "
        "tap while the first request is pending.",
    ),
    (
        "PRM-17", "PRM-2", "S2", "story", "done", "high", 5,
        "Block prescription creation when real-time insurance eligibility times out",
        "Context:\nIf the external insurance eligibility service times out or returns an unknown payer "
        "status during prescription sign-off, the API stops prescription creation and leaves the draft "
        "medication order untouched, rather than silently issuing an uncovered self-pay prescription.",
    ),
    (
        "PRM-18", "PRM-2", "S2", "story", "done", "high", 3,
        "Replace raw payer gateway error codes with plain patient guidance",
        "Context:\nPatients seeing raw `ELIG_GATEWAY_504` text thought their appointment was cancelled. "
        "Replaced with clear copy explaining that the visit slot is still held and telling them how to "
        "retry or continue with staff assistance.",
    ),
    (
        "PRM-19", "PRM-2", "S2", "bug", "done", "highest", 5,
        "Prescription renewal callback retry doubles the total daily dosage",
        "Context:\nA patient with 10mg on their prior chart and a 20mg renewal adjustment ended up with "
        "60mg recorded after a webhook retry.\n\n"
        "Root cause:\nThe renewal-merge callback ran twice on a timeout retry and re-applied the "
        "dosage delta onto the already-updated medication row instead of returning the first renewal "
        "snapshot. The fix persists and returns the first renewal result before acknowledging the "
        "webhook, so the example resolves to 30mg, not 60mg.",
    ),
    (
        "PRM-20", "PRM-2", "S3", "story", "done", "highest", 5,
        "Store pharmacy reversal key before calling external e-prescribe network",
        "Context:\nTo prevent a retried prescription reversal/refund from notifying the pharmacy network "
        "twice, the reversal key is written before the outbound provider call. Timeout recovery looks "
        "up the first reversal record and returns it instead of sending a second network request.",
    ),
    (
        "PRM-21", "PRM-2", "S3", "bug", "done", "highest", 5,
        "Require audited clinical correction action instead of direct PostgreSQL edits for slot/chart repair",
        "Context:\nAfter PRM-14 double-booked physician calendar slots, an engineer proposed fixing the "
        "schedule directly in PostgreSQL.\n\n"
        "Resolution:\nOperations required an authorized, repeat-safe correction action that records the "
        "visit reference, the clinical reason, the staff actor, and the slot/dosage delta in an "
        "immutable audit log compliance officers can inspect, rather than raw SQL updates.",
    ),
    (
        "PRM-22", "PRM-2", "S3", "bug", "done", "highest", 5,
        "Resent SMS visit reminder opens the pre-reschedule appointment room",
        "Context:\nIssue was closed once after initial SMS reminders were verified to open the right "
        "visit, then reopened when a patient requested a second reminder SMS after rescheduling: the "
        "first SMS used the new appointment token, but the resend path still read a cached legacy "
        "booking ID and opened VISIT-1902 instead of VISIT-1909.\n\n"
        "Fix:\nBoth initial send and resend now build the join link from the same canonical appointment "
        "version ID.",
    ),
    (
        "PRM-23", "PRM-2", "S3", "story", "done", "high", 3,
        "Give charge nurses a live waiting-room and prescription recovery console",
        "Context:\nPart of Sprint 3's recovery goal alongside PRM-14 follow-ups and PRM-21 audited "
        "corrections.",
    ),
    (
        "PRM-24", "PRM-3", "S3", "story", "done", "high", 5,
        "Index patient allergy and risk flags for the triage search bar",
        "Context:\nInitial indexing for triage search completed in Sprint 3.",
    ),
    (
        "PRM-25", "PRM-3", "S3", "bug", "done", "high", 5,
        "After a clinician updates a drug allergy flag, triage search still shows the old risk tier",
        "Context:\nThe search indexing event handler was wired for new chart intake but missing on the "
        "allergy-update path. Chart edit and intake events now share the same idempotent indexing "
        "handler with lag and dead-letter metrics, and an accepted allergy change becomes visible in "
        "triage search within fifteen seconds (tighter than storefront search because clinical triage "
        "is safety-critical).",
    ),
    (
        "PRM-26", "PRM-3", "S4", "story", "in_progress", "high", 5,
        "Cache clinician triage queue safely across pilot clinics",
        "Context:\nRead-through caching works for the clinic queue and chart summary routes, but cache "
        "invalidation during a bulk night-shift roster import is still being tested.",
    ),
    (
        "PRM-27", "PRM-3", "S4", "subtask", "in_progress", "high", 3,
        "Verify triage cache invalidation when bulk roster import updates on-call assignments",
        "Context:\nFocused subtask under PRM-26: found one stale waiting-room badge on the list view "
        "while single-chart invalidation stayed within the 2-second clinical target.",
    ),
    (
        "PRM-28", "PRM-3", "S4", "task", "in_progress", "high", 5,
        "Load-test video check-in through prescription sign-off",
        "Context:\nWaiting-room and chart-read workloads pass at pilot concurrency, but end-to-end "
        "visit sign-off misses its P95 latency goal while WebRTC TURN relay credentials and payer "
        "checks are cold.",
    ),
    (
        "PRM-29", "PRM-3", "S4", "subtask", "planned", "medium", 2,
        "Add cold TURN-credential and payer-eligibility step to the pilot load scenario",
        "Context:\nSubtask under PRM-28 to reproduce the cold-path P95 spike deterministically.",
    ),
    (
        "PRM-30", "PRM-3", "S4", "task", "planned", "high", 3,
        "Run the multi-clinic HIPAA and clinical safety readiness review",
        "Context:\nFinal checklist before opening the pilot to external patients.",
    ),
]

COMMENTS = [
    (
        "PRM-14",
        "Attached redacted gateway trace `duplicate-checkin-requests.log`: two separate request IDs "
        "reached the clinical API 180ms apart and both returned 201 Created; there was zero proxy or "
        "gateway retry. Follow-ups split into PRM-15 (idempotency key), PRM-16 (disable Join button), "
        "and PRM-21 (audited calendar slot repair).",
    ),
    (
        "PRM-22",
        "Reopened after patient reported the second SMS reminder still opened the old pre-reschedule "
        "room VISIT-1902. Regression test now covers both initial send and resend after reschedule.",
    ),
    (
        "PRM-25",
        "Clinical safety set a 15-second upper bound for allergy-flag search visibility after edit; "
        "shared idempotent handler now covers both intake and chart-edit events.",
    ),
    (
        "PRM-26",
        "PRM-27 caught one stale clinic-queue list entry during bulk roster import; keeping PRM-26 "
        "in_progress until that list invalidation race is closed.",
    ),
]


def seed() -> int:
    now = datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc)
    with psycopg.connect(DB_URL) as conn:
        owner = conn.execute(
            "SELECT id FROM users WHERE username='atlascart.maya' ORDER BY id LIMIT 1"
        ).fetchone()
        if not owner:
            owner = conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()
        owner_id = owner[0]

        existing = conn.execute(
            "SELECT id FROM spaces WHERE space_key=%s AND deleted_at IS NULL", (SPACE_KEY,),
        ).fetchone()
        if existing:
            space_id = existing[0]
            # Clean child rows so re-run is idempotent
            conn.execute(
                "DELETE FROM comments WHERE issue_id IN (SELECT id FROM issues WHERE space_id=%s)",
                (space_id,),
            )
            conn.execute(
                "DELETE FROM issue_history WHERE issue_id IN (SELECT id FROM issues WHERE space_id=%s)",
                (space_id,),
            )
            conn.execute("DELETE FROM issues WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM sprints WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM space_members WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM space_groups WHERE space_id=%s", (space_id,))
            conn.execute(
                "UPDATE spaces SET name=%s, color=%s, owner_id=%s, issue_counter=%s, updated_at=%s "
                "WHERE id=%s",
                (SPACE_NAME, SPACE_COLOR, owner_id, len(ISSUES), now, space_id),
            )
        else:
            space_id = conn.execute(
                "INSERT INTO spaces (name, space_key, color, owner_id, issue_counter, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (SPACE_NAME, SPACE_KEY, SPACE_COLOR, owner_id, len(ISSUES), now, now),
            ).fetchone()[0]

        # Copy all ATC space_members + space_groups, and also add local demo users (alice/john/charles/michael23fire)
        atc = conn.execute(
            "SELECT id FROM spaces WHERE space_key='ATC' AND deleted_at IS NULL LIMIT 1"
        ).fetchone()
        member_roles: dict[int, str] = {owner_id: "owner"}
        if atc:
            for uid, role in conn.execute(
                "SELECT user_id, role FROM space_members WHERE space_id=%s", (atc[0],),
            ).fetchall():
                member_roles[uid] = role
            for (gid,) in conn.execute(
                "SELECT group_id FROM space_groups WHERE space_id=%s", (atc[0],),
            ).fetchall():
                conn.execute(
                    "INSERT INTO space_groups (space_id, group_id, added_at) VALUES (%s, %s, %s)",
                    (space_id, gid, now),
                )
        for (uid,) in conn.execute(
            "SELECT id FROM users WHERE username IN ('alice','john','charles','michael23fire')"
        ).fetchall():
            member_roles.setdefault(uid, "admin")

        for uid, role in member_roles.items():
            conn.execute(
                "INSERT INTO space_members (space_id, user_id, role, joined_at) VALUES (%s, %s, %s, %s)",
                (space_id, uid, role, now),
            )

        sprint_ids: dict[str, int] = {}
        for code, name, goal, start_d, end_d, status, order in SPRINTS:
            sid = conn.execute(
                "INSERT INTO sprints (space_id, name, goal, start_date, end_date, status, "
                "created_at, updated_at, sprint_order) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (space_id, name, goal, start_d, end_d, status, now, now, order),
            ).fetchone()[0]
            sprint_ids[code] = sid

        issue_ids: dict[str, int] = {}
        for idx, (key, parent_key, scode, itype, status, prio, pts, title, desc) in enumerate(ISSUES, 1):
            parent_id = issue_ids.get(parent_key) if parent_key else None
            sp_id = sprint_ids.get(scode) if scode else None
            iid = conn.execute(
                "INSERT INTO issues (issue_key, space_id, sprint_id, parent_id, title, description, "
                "issue_type, status, priority, assignee_id, reporter_id, story_points, issue_order, "
                "created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (key, space_id, sp_id, parent_id, title, desc, itype, status, prio,
                 owner_id, owner_id, pts, idx, now, now),
            ).fetchone()[0]
            issue_ids[key] = iid

        for key, text in COMMENTS:
            conn.execute(
                "INSERT INTO comments (issue_id, author_id, content, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (issue_ids[key], owner_id, text, now, now),
            )

        # Reopen history on PRM-22 so get_issue_history also sees it
        conn.execute(
            "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, "
            "to_value, description, created_at) VALUES (%s, %s, 'status_changed', 'status', "
            "'done', 'in_progress', 'Reopened: resend SMS still used pre-reschedule room', %s)",
            (issue_ids["PRM-22"], owner_id, now),
        )
        conn.commit()
    print(f"Seeded {SPACE_KEY} (space_id={space_id}) with {len(ISSUES)} issues, {len(SPRINTS)} sprints, "
          f"and {len(member_roles)} members.")
    return space_id


if __name__ == "__main__":
    from m3_seed import seed as seed_m3
    seed_m3()

