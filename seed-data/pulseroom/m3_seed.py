#!/usr/bin/env python3
"""Seed PRM — PulseRoom (Synthetic M3): 3-month telehealth clinic & prescription platform.

7 two-week Sprints (S1..S6 completed, S7 active) + 2 future sprints, and 75 issues (PRM-1..PRM-75)
preserving all M1 facts while adding full Month-2 and Month-3 clinician identity, formulary import,
audit trail, session hardening, and multi-clinic pilot stories.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from html import escape as h

import psycopg

DB_URL = "postgresql://poc:poc123@localhost:5432/pocdb"
SPACE_KEY = "PRM"
SPACE_NAME = "PulseRoom (Synthetic M3)"
SPACE_COLOR = "#0D9488"

# (code, name, goal, start, end, status, order, init_pts, comp_pts, final_pts, init_cnt, comp_cnt, final_cnt)
SPRINTS = [
    ("F1", "Telehealth family proxy access — Sprint 1", "Future family caregiver proxy scheduling.", None, None, "future", 0, None, None, None, None, None, None),
    ("S1", "PulseRoom Sprint 1", "Stand up the clinical intake API and let a patient book a video visit in staging.", date(2026, 6, 8), date(2026, 6, 19), "completed", 0, 34, 34, 37, 10, 11, 11),
    ("F2", "Telehealth family proxy access — Sprint 2", "Future pediatric consent & proxy chart switch.", None, None, "future", 1, None, None, None, None, None, None),
    ("S2", "PulseRoom Sprint 2", "Make telehealth check-in and prescription handoff safe under reconnects and retries.", date(2026, 6, 22), date(2026, 7, 3), "completed", 1, 38, 35, 38, 10, 10, 11),
    ("S3", "PulseRoom Sprint 3", "Give clinical operations auditable recovery tools and reliable patient notifications.", date(2026, 7, 6), date(2026, 7, 17), "completed", 2, 32, 32, 35, 9, 10, 10),
    ("S4", "PulseRoom Sprint 4", "Harden patient and clinician sessions, sign-in rate limiting, and chart merge.", date(2026, 7, 20), date(2026, 7, 31), "completed", 3, 31, 28, 34, 8, 8, 9),
    ("S5", "PulseRoom Sprint 5", "Support pharmacy fulfilment lifecycle, prescription cancellation, and reversal safety.", date(2026, 8, 3), date(2026, 8, 14), "completed", 4, 33, 30, 36, 9, 8, 9),
    ("S6", "PulseRoom Sprint 6", "Give clinical pharmacists role-checked formulary editing, background bulk imports, and audit history.", date(2026, 8, 17), date(2026, 8, 28), "completed", 5, 36, 33, 39, 9, 9, 10),
    ("S7", "PulseRoom Sprint 7", "Prove triage caching, signaling latency, accessibility, and HIPAA gates before pilot clinics open.", date(2026, 8, 31), date(2026, 9, 11), "active", 6, None, None, None, None, None, None),
]

LABELS = ["intake", "video-room", "prescriptions", "security", "auth", "formulary", "triage-search", "performance", "hipaa-pilot", "accessibility", "incident-follow-up"]


def desc(ctx: str, acs: list[str], est: str, bug: tuple[str, str, str, str] | None = None) -> str:
    parts = ["<h3>Context</h3>", f"<p>{h(ctx)}</p>"]
    if bug:
        w, s, exp, act = bug
        parts += [
            "<h3>Bug report</h3><ul>",
            f"<li><strong>When / environment:</strong> {h(w)}</li>",
            f"<li><strong>Steps to reproduce:</strong> {h(s)}</li>",
            f"<li><strong>Expected behavior:</strong> {h(exp)}</li>",
            f"<li><strong>Actual behavior:</strong> {h(act)}</li></ul>",
        ]
    parts += ["<h3>Acceptance criteria</h3><ul>"] + [f"<li><strong>AC{i}</strong> — {h(a)}</li>" for i, a in enumerate(acs, 1)] + ["</ul>"]
    parts += ["<h3>Estimate reasoning</h3>", f"<p>{h(est)}</p>"]
    return "\n".join(parts)


# (key, parent, sprint, type, status, prio, pts, title, description, labels)
ISSUES_PART1 = [
    ("PRM-1", None, None, "epic", "done", "high", None, "Launch core telehealth intake and visit scheduling",
     desc("Build the initial patient intake, chart lookup, and video-visit booking flow.", ["Patients can book a slot and clinicians see the queue."], "No story points — epic tracked across Sprint 1 child stories."), ["intake", "video-room"]),
    ("PRM-2", None, None, "epic", "done", "high", None, "Harden prescription queue and clinical safety controls",
     desc("Ensure check-in, renewals, and pharmacy actions are repeat-safe and audited.", ["No duplicate rooms or double-counted dosage on retry."], "No story points — epic tracked across Sprint 2 and Sprint 3 stories."), ["prescriptions", "security"]),
    ("PRM-3", None, None, "epic", "in_progress", "high", None, "Prepare PulseRoom for the multi-clinic pilot",
     desc("Close triage cache invalidation and cold-turn signaling latency before pilot launch.", ["Triage view reflects allergy edits and signaling meets P95 target."], "No story points — pilot readiness epic spanning search, caching, and load tests."), ["hipaa-pilot", "performance"]),
    ("PRM-4", "PRM-1", "S1", "story", "done", "high", 5, "Keep a single clinical API deployment for the M1 pilot",
     desc("Engineers debated splitting video signaling, intake, and prescription workflows into separate microservices before the pilot. Decision: No service split for now. Keep one clinical API deployment during M1 and split signaling or prescription services only after live pilot telemetry shows a measured CPU or connection-pool bottleneck that vertical scaling cannot absorb.",
          ["Single deployable artifact documented in architecture notes.", "Split happens only after live pilot telemetry shows a measured CPU or connection-pool bottleneck that vertical scaling cannot absorb."],
          "5 — Covers architecture review, connection-pool benchmarks, and deployment documentation."), ["intake", "hipaa-pilot"]),
    ("PRM-5", "PRM-1", "S1", "story", "done", "medium", 3, "Allow urgent-care intake before full insurance card upload",
     desc("Walk-in telehealth patients often lacked their card photo on mobile. Decision: Urgent-care triage intake is not blocked on insurance card upload; coverage verification runs before prescription sign-off instead.",
          ["Urgent-care triage intake proceeds without insurance card photo; eligibility runs before prescription sign-off."],
          "3 — Adjusts intake state machine and defers payer check to prescription step."), ["intake"]),
    ("PRM-6", "PRM-1", "S1", "task", "done", "medium", 3, "Preserve partially filled intake form across mobile browser restarts",
     desc("A valid draft intake questionnaire is restored after a page refresh or mobile browser restart. Expired or tampered draft tokens are wiped with one audit warning and the patient sees a clean blank intake form instead of a crash screen.",
          ["Valid draft intake form restores on refresh; expired or tampered tokens wipe cleanly with one audit warning."],
          "3 — Encrypted session storage plus token signature verification and tests."), ["intake"]),
    ("PRM-7", "PRM-1", "S1", "story", "done", "medium", 3, "Basic symptom keyword lookup in the clinician triage bar",
     desc("The triage search bar uses an exact case-insensitive substring match over chief complaint and medication names. Medical abbreviation expansion and fuzzy typo matching were explicitly left out of scope for Sprint 1.",
          ["Case-insensitive substring match over chief complaint and medication names; abbreviations and typo matching excluded."],
          "3 — Indexed query and basic filter bar UI."), ["triage-search"]),
    ("PRM-8", "PRM-1", "S1", "bug", "done", "medium", 2, "Reject lab CSV import rows with missing or negative reference ranges",
     desc("When a clinic uploads a lab-panel CSV with a blank reference cell or a negative threshold, the importer rejects the row with its original CSV line number. A true zero lower bound is kept distinct from a blank cell.",
          ["Blank or negative reference thresholds are rejected with line number; zero lower bound is accepted."],
          "2 — Parser fix and unit tests for zero vs blank cells.",
          ("Staging clinic CSV upload", "Upload a lab panel with blank or negative reference range", "Row rejected with CSV line number while 0.0 lower bound passes", "Parser coerced blank cells to 0 and allowed negative upper bounds")), ["formulary"]),
    ("PRM-9", "PRM-1", "S1", "story", "done", "high", 3, "Return identical 404 for unknown and archived medical record numbers",
     desc("Both a never-issued MRN (medical record number) and a patient chart archived under a privacy hold return the exact same 404 response body and status. This is deliberate so an external caller cannot probe whether a specific person has ever been treated at the clinic.",
          ["Unknown MRNs and privacy-archived MRNs return identical 404 status and body to prevent patient enumeration."],
          "3 — Uniform error serializer and security regression test."), ["security", "intake"]),
    ("PRM-10", "PRM-1", "S1", "story", "done", "high", 5, "Validate formulary CSV uploads and reject conflicting duplicate NDC rows",
     desc("When a pharmacy formulary file lists the same drug NDC twice with different maximum daily doses, the importer rejects the upload, reports both line numbers, and writes nothing to PostgreSQL.",
          ["Conflicting duplicate NDC rows reject the entire upload with both line numbers and zero DB writes."],
          "5 — All-or-nothing transaction wrapper and duplicate-NDC detector."), ["formulary"]),
    ("PRM-11", "PRM-2", "S2", "story", "done", "highest", 5, "Strip client-supplied clinician role headers at the API gateway",
     desc("The API gateway strips any client-supplied `X-Provider-Role` and `X-Clinician-Id` headers before attaching the role from the verified staff session token, stopping a patient browser from spoofing a prescribing doctor. Public clinic-hours and symptom-checker pages continue to work without a session.",
          ["Gateway strips client-supplied X-Provider-Role and X-Clinician-Id headers and attaches verified staff session identity.", "Public clinic-hours and symptom-checker routes remain accessible without a session."],
          "5 — Gateway header filter, session token claim mapping, and security smoke suite."), ["auth", "security"]),
    ("PRM-12", "PRM-2", "S2", "task", "done", "medium", 3, "Agree pilot device matrix for video visits",
     desc("Desktop Chrome, Safari on iOS, and Chrome on one mid-range Android handset were agreed for M1 pilot verification; legacy tablet browsers stay out of scope.",
          ["Documented pilot browser matrix signed off by clinical ops and QA."],
          "3 — Device matrix verification across WebRTC waiting room."), ["video-room"]),
    ("PRM-13", "PRM-2", "S2", "story", "done", "high", 5, "Distinguish waiting-room visits from dispensed prescriptions in cancellation rules",
     desc("Operations cannot cancel a visit or prescription once the pharmacy has marked the prescription dispensed. Unfulfilled waiting-room visits and un-dispensed prescriptions can be cancelled with an actor and reason, releasing the clinician slot once; dispensed orders must go through the clinical reversal / recall workflow instead of simple cancellation.",
          ["Un-dispensed prescriptions and waiting-room visits can be cancelled with reason and release the slot once.", "Dispensed prescriptions reject cancellation with 409 Conflict and require the clinical reversal workflow."],
          "5 — Lifecycle state machine, slot release compensation, and 409 guard."), ["prescriptions"]),
    ("PRM-14", "PRM-2", "S2", "bug", "done", "highest", 5, "Mobile reconnect during check-in creates two live video visit rooms",
     desc("A patient on unstable clinic Wi-Fi tapped 'Join waiting room' twice across a brief reconnect and the system provisioned two video rooms (VISIT-2041 and VISIT-2042) and double-booked two 15-minute slots on the on-call physician's calendar, even though only one copay authorization succeeded. Root cause: The check-in handler inserted the visit room before checking whether the appointment's check-in idempotency token had already been consumed. Gateway logs showed two distinct client request IDs reaching the API with 201 responses, ruling out an automatic load-balancer retry.",
          ["Repeated check-in taps across mobile reconnect create only one video room and one physician calendar slot."],
          "5 — Incident investigation, trace capture, and coordination of PRM-15/16/21.",
          ("Pilot staging on unstable clinic Wi-Fi", "Tap Join waiting room twice across a 180ms mobile reconnect", "One video room provisioned and one physician slot booked", "Created two live rooms (VISIT-2041 and VISIT-2042) and double-booked two physician calendar slots")), ["video-room", "incident-follow-up"]),
    ("PRM-15", "PRM-2", "S2", "task", "done", "highest", 3, "Enforce check-in idempotency key before room and calendar allocation",
     desc("Permanent API fix split out of PRM-14: a repeated check-in key returns the existing video room and calendar hold before any new room or clinician slot is allocated.",
          ["Repeated check-in idempotency key returns the existing room and physician slot without allocating a second one."],
          "3 — Unique idempotency constraint checked before room/calendar transaction."), ["video-room", "incident-follow-up"]),
    ("PRM-16", "PRM-2", "S2", "task", "done", "high", 2, "Disable Join waiting room button while check-in request is in flight",
     desc("Immediate mobile UI mitigation for PRM-14 so a slow connection cannot fire a second tap while the first request is pending.",
          ["Join waiting room button locks immediately on tap and re-enables on error or timeout."],
          "2 — Mobile pending-state lock and browser back-navigation reset."), ["video-room"]),
    ("PRM-17", "PRM-2", "S2", "story", "done", "high", 5, "Block prescription creation when real-time insurance eligibility times out",
     desc("If the external insurance eligibility service times out or returns an unknown payer status during prescription sign-off, the API stops prescription creation and leaves the draft medication order untouched, rather than silently issuing an uncovered self-pay prescription.",
          ["Payer timeout or unknown status stops prescription creation and leaves the draft medication order untouched."],
          "5 — Payer adapter timeout handling, transaction rollback, and clinical tests."), ["prescriptions"]),
    ("PRM-18", "PRM-2", "S2", "story", "done", "high", 3, "Replace raw payer gateway error codes with plain patient guidance",
     desc("Patients seeing raw `ELIG_GATEWAY_504` text thought their appointment was cancelled. Replaced with clear copy explaining that the visit slot is still held and telling them how to retry or continue with staff assistance.",
          ["Timeout banner hides raw ELIG_GATEWAY_504 codes and confirms the visit slot remains held."],
          "3 — Patient-friendly error mapping and copy review."), ["prescriptions", "intake"]),
    ("PRM-19", "PRM-2", "S2", "bug", "done", "highest", 5, "Prescription renewal callback retry doubles the total daily dosage",
     desc("A patient with 10mg on their prior chart and a 20mg renewal adjustment ended up with 60mg recorded after a webhook retry. Root cause: The renewal-merge callback ran twice on a timeout retry and re-applied the dosage delta onto the already-updated medication row instead of returning the first renewal snapshot. The fix persists and returns the first renewal result before acknowledging the webhook, so the example resolves to 30mg, not 60mg.",
          ["10mg prior chart + 20mg renewal adjustment resolves to 30mg once, never 60mg on webhook retry.", "Repeated renewal callbacks return the stored first renewal snapshot without re-applying the delta."],
          "5 — Idempotent renewal snapshot storage and retry regression suite.",
          ("Staging e-prescribe renewal webhook retry", "Submit +20mg renewal on a 10mg active chart and replay the callback once", "Chart records 30mg total daily dosage", "Retry re-applied delta onto updated row and recorded 60mg")), ["prescriptions", "incident-follow-up"]),
    ("PRM-20", "PRM-2", "S3", "story", "done", "highest", 5, "Store pharmacy reversal key before calling external e-prescribe network",
     desc("To prevent a retried prescription reversal/refund from notifying the pharmacy network twice, the reversal key is written before the outbound provider call. Timeout recovery looks up the first reversal record and returns it instead of sending a second network request.",
          ["Eligible cancelled/recalled prescription writes the reversal key before calling the external e-prescribe network.", "Retrying a timed-out reversal returns the first reversal record and never sends a second pharmacy network request.", "Failed or timed-out reversals remain visible to clinical ops with a safe recovery state and correlation ID."],
          "5 — Pre-call reversal key persistence, provider adapter idempotency, and timeout recovery."), ["prescriptions", "security"]),
    ("PRM-21", "PRM-2", "S3", "bug", "done", "highest", 5, "Require audited clinical correction action instead of direct PostgreSQL edits for slot/chart repair",
     desc("After PRM-14 double-booked physician calendar slots, an engineer proposed fixing the schedule directly in PostgreSQL. Resolution: Operations required an authorized, repeat-safe correction action that records the visit reference, the clinical reason, the staff actor, and the slot/dosage delta in an immutable audit log compliance officers can inspect, rather than raw SQL updates.",
          ["Clinical schedule and dosage repairs run through an authorized, repeat-safe correction endpoint.", "Immutable audit log records visit reference, clinical reason, staff actor, and slot/dosage delta."],
          "5 — Audited repair endpoint, role check, and immutable compliance log table.",
          ("Post-PRM-14 duplicate slot cleanup", "Attempt to release double-booked physician slots via ad-hoc SQL", "Authorized correction action logs visit reference, clinical reason, staff actor, and slot/dosage delta", "Raw SQL update bypassed compliance audit log")), ["security", "incident-follow-up"]),
    ("PRM-22", "PRM-2", "S3", "bug", "done", "highest", 5, "Resent SMS visit reminder opens the pre-reschedule appointment room",
     desc("Issue was closed once after initial SMS reminders were verified to open the right visit, then reopened when a patient requested a second reminder SMS after rescheduling: the first SMS used the new appointment token, but the resend path still read a cached legacy booking ID and opened VISIT-1902 instead of VISIT-1909. Fix: Both initial send and resend now build the join link from the same canonical appointment version ID.",
          ["Initial and resent SMS reminders both open VISIT-1909 after a reschedule, never VISIT-1902.", "Patient session check prevents opening another patient's video visit link."],
          "5 — Canonical appointment version lookup across initial send and resend workers.",
          ("Patient rescheduled from VISIT-1902 to VISIT-1909 and requested SMS resend", "Click join link in resent SMS reminder", "Link opens rescheduled room VISIT-1909", "Resend path read cached legacy booking ID and opened VISIT-1902")), ["video-room", "incident-follow-up"]),
    ("PRM-23", "PRM-2", "S3", "story", "done", "high", 3, "Give charge nurses a live waiting-room and prescription recovery console",
     desc("Part of Sprint 3's recovery goal alongside PRM-14 follow-ups and PRM-21 audited corrections. Lets charge nurses inspect stuck check-ins and trigger audited slot recovery without raw DB access.",
          ["Charge nurses can view stuck waiting-room visits and run audited recovery actions."],
          "3 — Console table, status filters, and integration with PRM-21 repair API."), ["video-room", "prescriptions"]),
    ("PRM-24", "PRM-3", "S3", "story", "done", "high", 5, "Index patient allergy and risk flags for the triage search bar",
     desc("Initial indexing for triage search completed in Sprint 3 so clinicians can filter active patients by drug allergy, chief complaint, and MRN while keeping discharged/inactive charts hidden.",
          ["Triage search finds active charts by drug allergy, chief complaint, or MRN; inactive charts stay hidden."],
          "5 — Triage search index schema, initial backfill, and active-chart filter."), ["triage-search"]),
    ("PRM-25", "PRM-3", "S3", "bug", "done", "high", 5, "After a clinician updates a drug allergy flag, triage search still shows the old risk tier",
     desc("The search indexing event handler was wired for new chart intake but missing on the allergy-update path. Chart edit and intake events now share the same idempotent indexing handler with lag and dead-letter metrics, and an accepted allergy change becomes visible in triage search within fifteen seconds (tighter than storefront search because clinical triage is safety-critical).",
          ["Accepted drug-allergy edit becomes visible in triage search within 15 seconds.", "Chart edit and intake events share one idempotent indexing handler with lag and dead-letter metrics."],
          "5 — Shared event handler, 15-second SLA monitor, and dead-letter replay.",
          ("Clinician updated penicillin allergy to high-risk anaphylaxis on active chart", "Save allergy edit and search triage bar within 15 seconds", "Updated risk tier appears in triage search within 15 seconds", "Triage search showed old risk tier until nightly reindex because allergy-update handler was missing")), ["triage-search", "incident-follow-up"]),
    ("PRM-26", "PRM-3", "S7", "story", "in_progress", "high", 5, "Cache clinician triage queue safely across pilot clinics",
     desc("Read-through caching works for the clinic queue and chart summary routes, but cache invalidation during a bulk night-shift roster import is still being tested.",
          ["Repeated triage queue reads meet cache-hit target.", "Roster and allergy updates invalidate affected entries within the 2-second clinical target."],
          "5 — Read-through cache keys, invalidation hooks, and PostgreSQL fallback."), ["performance", "triage-search"]),
    ("PRM-27", "PRM-26", "S7", "subtask", "in_progress", "high", None, "Verify triage cache invalidation when bulk roster import updates on-call assignments",
     desc("Focused subtask under PRM-26: found one stale waiting-room badge on the list view while single-chart invalidation stayed within the 2-second clinical target.",
          ["Bulk roster import invalidates waiting-room list badges within 2 seconds."],
          "No story points — subtask under PRM-26 isolating the bulk roster list-invalidation race."), ["performance", "triage-search"]),
    ("PRM-28", "PRM-3", "S7", "task", "in_progress", "high", 5, "Load-test video check-in through prescription sign-off",
     desc("Waiting-room and chart-read workloads pass at pilot concurrency, but end-to-end visit sign-off misses its P95 latency goal while WebRTC TURN relay credentials and payer checks are cold.",
          ["Pilot workload reaches target concurrency with synthetic patients and stubbed payer.", "P95 latency stays within target across check-in, video room, and prescription sign-off."],
          "5 — Load generator, synthetic patient fixtures, and P95 telemetry."), ["performance", "hipaa-pilot"]),
    ("PRM-29", "PRM-28", "S7", "subtask", "planned", "medium", None, "Add cold TURN-credential and payer-eligibility step to the pilot load scenario",
     desc("Subtask under PRM-28 to reproduce the cold-path P95 spike deterministically without hitting live payer networks.",
          ["Load script isolates cold TURN-credential and cold payer-eligibility latency from warm check-in."],
          "No story points — subtask under PRM-28 for deterministic cold-dependency reproduction."), ["performance", "hipaa-pilot"]),
    ("PRM-30", "PRM-3", "S7", "task", "planned", "high", 3, "Run the multi-clinic HIPAA and clinical safety readiness review",
     desc("Final go/no-go checklist before opening the pilot to external patients. Any open highest-priority clinical safety blocker or unverified P95/accessibility gate makes the review a no-go.",
          ["Every HIPAA and clinical safety gate links to a verified test, dashboard, or named risk owner.", "Any open highest-priority blocker results in a documented no-go."],
          "3 — Cross-clinic evidence review, rollback plan, and sign-off record."), ["hipaa-pilot", "security"]),
]

ISSUES_PART2 = [
    ("PRM-31", None, None, "epic", "in_progress", "high", None, "Give patients and clinicians secure telehealth accounts and session controls",
     desc("Add patient portal accounts, clinician SSO session binding, HIPAA idle timeouts, and walk-in intake merge without blocking urgent-care check-in.",
          ["Patients and clinicians authenticate safely with strict session controls.", "Walk-in intake records merge cleanly into a verified patient chart once."],
          "No story points — Month-2 identity epic."), ["auth", "security"]),
    ("PRM-32", None, None, "epic", "in_progress", "high", None, "Support end-to-end pharmacy routing, cancellation, and clinical reversals",
     desc("Extend initial prescription handoff into a complete pharmacy fulfilment lifecycle from clinic selection through cancellation and reversal.",
          ["Patients can pick a participating pharmacy and track prescription status.", "Operations and pharmacists can cancel or reverse orders only when lifecycle rules allow."],
          "No story points — Month-2 pharmacy operations epic."), ["prescriptions"]),
    ("PRM-33", None, None, "epic", "in_progress", "medium", None, "Give clinical pharmacists safe formulary management and bulk imports",
     desc("Let authorized clinical pharmacists maintain drug formulary entries, run large background CSV imports, and inspect an append-only compliance audit trail.",
          ["Only clinical pharmacists can edit formulary entries.", "Large formulary imports validate all-or-nothing in the background and refresh search."],
          "No story points — Month-3 formulary operations epic."), ["formulary", "triage-search", "security"]),
    ("PRM-34", "PRM-1", "S1", "task", "done", "medium", 3, "Create PostgreSQL schema for clinics, appointments, and intake charts",
     desc("Establish core tables with foreign keys and strict UTC timestamps for clinics, patient charts, and video appointments.",
          ["Migrations apply cleanly and enforce foreign keys across clinics, charts, and appointments."],
          "3 — Base schema migration and rollback check."), ["intake"]),
    ("PRM-35", "PRM-1", "S1", "story", "done", "medium", 3, "Display real-time clinic waiting-room position to checked-in patients",
     desc("Show patients their current queue position and estimated wait window while waiting for the on-call clinician.",
          ["Checked-in patient sees live queue position without exposing other patients' names."],
          "3 — Polling status card and privacy-safe queue serializer."), ["video-room"]),
    ("PRM-36", "PRM-31", "S4", "story", "done", "high", 5, "Let returning patients create a portal account without blocking urgent-care walk-ins",
     desc("Allow patients to register with email, password, and date-of-birth verification after accepting HIPAA consent, while keeping urgent-care walk-in intake available without an account.",
          ["Valid patient registration creates one portal account and starts a signed-in session.", "Declining account creation still allows urgent-care walk-in intake."],
          "5 — Registration form, password hashing, HIPAA consent record, and walk-in fallback."), ["auth", "intake"]),
    ("PRM-37", "PRM-31", "S4", "story", "done", "high", 5, "Merge walk-in intake questionnaire into verified patient chart after sign-in",
     desc("When a walk-in patient signs into an existing portal account, attach their unsubmitted intake symptoms and allergies to the verified chart once without duplicating medications.",
          ["Walk-in intake answers merge once into the verified patient chart.", "Refreshing or retrying sign-in returns the same merged chart without duplicating entries."],
          "5 — One-time walk-in token consumption, allergy deduplication, and retry tests."), ["auth", "intake"]),
    ("PRM-38", "PRM-31", "S4", "task", "done", "high", 3, "Harden clinician and patient session cookies with 15-minute HIPAA idle timeout",
     desc("Configure session cookies with Secure, HttpOnly, SameSite=Strict, and a 15-minute clinical idle timeout that returns the user to sign-in without losing an unsubmitted server-side clinical note draft, while ensuring application, gateway, and audit logs never record raw session tokens.",
          ["Session cookies set Secure, HttpOnly, SameSite=Strict, and enforce a 15-minute idle expiration.", "Idle expiration requires re-authentication without losing a saved server-side clinical note draft.", "Gateway, application, and audit logs contain zero raw cookie or session token values."],
          "3 — Cookie policy middleware, 15-minute HIPAA idle timer, and log-redaction scan."), ["auth", "security"]),
    ("PRM-39", "PRM-31", "S4", "task", "done", "medium", 3, "Rate-limit repeated patient and clinician sign-in failures",
     desc("Throttle automated credential stuffing by returning a temporary 429 Too Many Requests after the configured failure threshold, keyed on source IP and a hashed account/MRN identifier. A valid sign-in resets the failure counter, and security metrics/logs never expose raw email addresses or MRNs.",
          ["Repeated sign-in failures from one source return a temporary 429 after the threshold.", "A successful sign-in clears the failure counter without exposing prior failed attempts.", "Metrics and logs key on source IP plus a hashed account/MRN identifier rather than raw email or MRN."],
          "3 — Gateway rate-limit counter, hash key derivation, and deterministic tests."), ["auth", "security"]),
    ("PRM-40", "PRM-31", "S4", "task", "done", "medium", 3, "Add gateway authentication contract tests for patient and clinician roles",
     desc("Verify gateway routing with anonymous requests, valid patient sessions, valid clinician sessions, expired tokens, and forged X-Provider-Role headers.",
          ["CI suite verifies anonymous, patient, clinician, expired, and forged-header requests."],
          "3 — Four contract test scenarios wired into CI."), ["auth", "security"]),
    ("PRM-41", "PRM-31", "S4", "story", "done", "medium", 3, "Show clear session-timeout warning two minutes before HIPAA idle lock",
     desc("Give clinicians an accessible modal warning two minutes before the 15-minute idle lock so they can extend their session during a long chart review.",
          ["Accessible countdown modal appears two minutes before idle expiration and supports keyboard extension."],
          "3 — Countdown timer, keyboard focus trap, and extend-session ping."), ["auth", "accessibility"]),
    ("PRM-42", "PRM-32", "S5", "story", "done", "high", 5, "Return participating local and mail-order pharmacy options during visit checkout",
     desc("Return nearby retail and mail-order pharmacy options with estimated pickup window and copay tier from a patient's ZIP code using stubbed pharmacy directory responses.",
          ["Supported ZIP codes return retail and mail-order pharmacy options with pickup window and copay tier.", "Unsupported ZIP codes show one plain guidance message without raw directory errors."],
          "5 — Pharmacy directory adapter, ZIP lookup, and timeout fallback."), ["prescriptions"]),
    ("PRM-43", "PRM-32", "S5", "story", "done", "medium", 5, "Let patients select a preferred pharmacy before visit sign-off",
     desc("Allow patients to pick or change their preferred retail or mail-order pharmacy in an accessible selector and keep the chosen pharmacy visible on the visit summary.",
          ["Keyboard-accessible pharmacy selector updates visit summary and resets stale selection on ZIP change."],
          "5 — Pharmacy picker UI, ZIP change reset, and confirmation summary."), ["prescriptions", "accessibility"]),
    ("PRM-44", "PRM-32", "S5", "story", "done", "medium", 3, "Look up recent prescriptions by exact RX reference in clinical operations",
     desc("Allow clinical operations to look up a prescription summary by its complete RX reference (e.g. RX-4012) without exposing full patient diagnosis notes or raw insurance subscriber numbers.",
          ["Exact case-insensitive RX reference returns status, timestamps, and dispensing state; partial strings do not enumerate nearby prescriptions."],
          "3 — Exact lookup endpoint, redacted response DTO, and access tests."), ["prescriptions", "security"]),
    ("PRM-45", "PRM-32", "S5", "task", "done", "high", 3, "Document and enforce the prescription fulfilment state machine",
     desc("Define allowed transitions across draft, signed, dispensed, cancelled, reversal_pending, and reversed so support, cancellation (PRM-13), and pharmacy reversal (PRM-20) share one rule set.",
          ["State table enforces valid transitions from draft through dispensed, cancelled, or reversed and returns 409 on invalid transitions."],
          "3 — State transition validator and append-only transition log."), ["prescriptions"]),
    ("PRM-46", "PRM-32", "S5", "task", "done", "medium", 3, "Send patient notification when prescription status changes",
     desc("Queue an asynchronous SMS/email update when a prescription moves to signed, cancelled, or reversed without delaying the clinical transaction if the notification worker is slow.",
          ["Prescription state change commits before queuing notification; retries never expose diagnosis details."],
          "3 — Outbox notification job, retry policy, and privacy-safe template."), ["prescriptions"]),
    ("PRM-47", "PRM-33", "S6", "story", "done", "high", 5, "Edit formulary drug entries in clinical pharmacist tools",
     desc("Let an authorized clinical pharmacist update drug name, dosage form, maximum daily dose, active status, and allergy cross-reactivity tier through validated operations rather than raw SQL.",
          ["Authorized clinical pharmacist can edit agreed formulary fields with field-level validation.", "Dosage and reference rules match CSV import validation and reject zero or negative max daily doses.", "Accepted edits update the formulary, trigger a triage search refresh, and record before, after, actor, and clinical reason."],
          "5 — Pharmacist edit form, server validation, cache/search refresh event, and audit write."), ["formulary", "triage-search"]),
    ("PRM-48", "PRM-33", "S6", "task", "done", "high", 5, "Restrict formulary edit permissions to clinical pharmacists on the server",
     desc("Verify on the server that the caller holds the clinical-pharmacist role in the workspace before creating or modifying any formulary drug entry. Triage nurses, patients, operations staff, and requests with a forged role header receive the same safe 403 Forbidden response.",
          ["Only verified clinical-pharmacist members can create or modify formulary entries.", "Patients, triage nurses, operations users, and forged role headers receive an identical safe 403 Forbidden response."],
          "5 — Server-side role enforcement, gateway claim check, and negative authorization tests."), ["formulary", "security", "auth"]),
    ("PRM-49", "PRM-33", "S6", "story", "done", "medium", 5, "Upload medication patient-education PDF leaflets to managed object storage",
     desc("Let clinical pharmacists attach a validated patient-education PDF leaflet to a formulary drug in MinIO/S3 object storage while keeping the previous leaflet intact if validation fails.",
          ["Valid PDF leaflets within size limits store under a generated key; invalid or oversized files leave the current leaflet unchanged."],
          "5 — Multipart upload, MIME/size validation, and generated storage key."), ["formulary"]),
    ("PRM-50", "PRM-33", "S6", "story", "done", "highest", 8, "Run bulk formulary CSV imports in the background with all-or-nothing validation",
     desc("Process a three-thousand-row clinic formulary CSV in a background worker, report progress counts (rows read, accepted, and rejected), and apply the update only if every row passes validation. Any row with a conflicting duplicate NDC or invalid maximum daily dose aborts the apply step so the live formulary is never left half-updated.",
          ["Valid formulary CSV upload queues a background job and returns an import job ID immediately.", "Progress endpoint reports rows read, accepted, and rejected without exposing another clinic's upload.", "A file with any invalid row writes nothing to the live formulary; a 100% valid file updates the formulary once and refreshes triage search."],
          "8 — Durable background worker, restart-safe progress counters, all-or-nothing database apply, and search refresh trigger."), ["formulary", "triage-search", "performance"]),
    ("PRM-51", "PRM-33", "S6", "story", "done", "medium", 3, "Show formulary bulk-import progress and downloadable rejected-row report",
     desc("Provide clinical pharmacists an accessible progress view for queued, validating, failed, and applied formulary imports with a downloadable CSV of rejected line numbers.",
          ["Progress page polls job phases, provides rejected-row CSV on failure, and links applied jobs to their compliance audit entry."],
          "3 — Polling status UI, screen-reader phase announcements, and error CSV download."), ["formulary", "accessibility"]),
    ("PRM-52", "PRM-33", "S6", "task", "done", "high", 5, "Audit clinical pharmacist formulary edits and bulk imports",
     desc("Record an append-only compliance audit entry for every accepted formulary drug edit and bulk import without storing full uploaded CSV files in the audit table.",
          ["Every accepted edit and bulk import records pharmacist actor, timestamp, source, clinical reason, and affected NDC codes.", "Compliance officers can filter audit rows by pharmacist, NDC, change type, and date without viewing credentials.", "Audit rows are append-only and remain intact even after a medication is marked inactive."],
          "5 — Normalized audit schema, event consumer, compliance filter API, and retention tests."), ["formulary", "security"]),
    ("PRM-53", "PRM-3", "S7", "task", "in_progress", "high", 5, "Complete multi-clinic accessibility pass for intake, waiting room, and prescription summary",
     desc("Close keyboard navigation, focus management, ARIA labeling, and contrast gaps across patient intake, video waiting room, and prescription summary screens.",
          ["Complete intake-to-prescription flow is keyboard-operable with visible focus.", "Dynamic queue and validation messages announce cleanly to screen readers."],
          "5 — Shared form labels, focus restoration, and VoiceOver/NVDA check matrix."), ["accessibility", "hipaa-pilot"]),
    ("PRM-54", "PRM-53", "S7", "subtask", "in_progress", "high", None, "Move keyboard focus to intake validation error summary on submit",
     desc("Focused subtask under PRM-53 so submitting an incomplete intake questionnaire moves keyboard focus to the top error summary with links to each invalid field.",
          ["Invalid intake submission moves focus to the error summary and announces once in VoiceOver."],
          "No story points — subtask under PRM-53 isolating intake error focus behavior."), ["accessibility", "intake"]),
    ("PRM-55", "PRM-31", "S4", "bug", "done", "medium", 2, "Clinician shift-handoff badge clips 12-hour timestamp on 320px mobile screens",
     desc("On narrow 320px mobile viewports, a two-line clinician name overlapped the AM/PM shift timestamp badge in the triage header.",
          ["Clinician name and shift timestamp wrap cleanly at 320px without overlapping the action button."],
          "2 — Flex-wrap CSS fix and mobile viewport regression check.",
          ("iOS Safari at 320px width", "Open triage queue with a long clinician display name", "Name and timestamp badge stay readable without overlap", "Timestamp badge clipped over the Sign out button")), ["video-room", "accessibility"]),
]

ISSUES_PART3 = [
    ("PRM-56", None, None, "epic", "done", "medium", None, "Establish staging clinical environments and synthetic patient fixtures",
     desc("Provide repeatable staging environments, synthetic HL7/FHIR fixtures, and CI smoke checks with zero real PHI.",
          ["Staging environments use 100% synthetic patient fixtures and pass CI smoke checks."],
          "No story points — foundational platform epic."), ["hipaa-pilot", "security"]),
    ("PRM-57", "PRM-56", "S1", "task", "done", "medium", 3, "Build synthetic patient MRN and HL7 intake fixture generator",
     desc("Generate deterministic synthetic patients, MRNs, allergies, and insurance cards so staging and load tests never touch real PHI.",
          ["Fixture generator produces deterministic synthetic patients with zero real PHI."],
          "3 — Deterministic seed script and PHI guard check."), ["intake", "security"]),
    ("PRM-58", "PRM-56", "S1", "task", "done", "low", 2, "Add staging healthcheck endpoint for clinical API and PostgreSQL pool",
     desc("Expose `/healthz` reporting database pool readiness and migration version without leaking connection strings.",
          ["Health endpoint returns 200 when pool and migrations are ready and omits credentials."],
          "2 — FastAPI/Spring health probe and smoke test."), ["hipaa-pilot"]),
    ("PRM-59", "PRM-1", "S1", "subtask", "done", "medium", None, "Verify intake form keyboard tab order on mobile and desktop",
     desc("Subtask under PRM-6 to confirm keyboard tab order and focus rings across the intake questionnaire.",
          ["Tab order follows visual top-to-bottom sequence across all intake fields."],
          "No story points — subtask under PRM-6."), ["intake", "accessibility"]),
    ("PRM-60", "PRM-2", "S2", "story", "done", "medium", 3, "Show clinician a pre-call audio and camera check before joining video room",
     desc("Let clinicians test microphone and camera permissions before entering the patient video waiting room.",
          ["Pre-call check verifies audio/video tracks and warns if permissions are blocked."],
          "3 — MediaStream preview component and permission fallback guidance."), ["video-room"]),
    ("PRM-61", "PRM-2", "S2", "subtask", "done", "medium", None, "Add regression test for Safari back-button state after check-in lock",
     desc("Subtask under PRM-16 ensuring iOS Safari Back navigation resets the pending Join button state cleanly.",
          ["Browser Back after timeout resets the Join waiting room button."],
          "No story points — subtask under PRM-16."), ["video-room"]),
    ("PRM-62", "PRM-2", "S3", "task", "done", "medium", 3, "Add payment-copay authorization status column to clinical ops visit queue",
     desc("Let clinical operations see whether copay authorization succeeded or timed out when triaging duplicate room reports from PRM-14.",
          ["Ops queue shows copay state (authorized, pending, failed) without exposing raw card digits."],
          "3 — Redacted status column and role check."), ["video-room", "prescriptions"]),
    ("PRM-63", "PRM-2", "S3", "story", "done", "medium", 3, "Allow patient to download visit receipt and work/school excuse note PDF",
     desc("Provide a downloadable visit summary and work/school excuse PDF once the clinician signs off the encounter.",
          ["Signed-off visit exposes a generated PDF receipt restricted to the authenticated patient."],
          "3 — PDF template renderer and patient ownership check."), ["intake", "prescriptions"]),
    ("PRM-64", "PRM-3", "S3", "subtask", "done", "medium", None, "Benchmark triage search query latency on 10,000 synthetic charts",
     desc("Subtask under PRM-24 verifying P95 search latency stays under 120ms across 10,000 synthetic patient charts.",
          ["P95 triage search latency stays below 120ms on 10k synthetic charts."],
          "No story points — subtask under PRM-24."), ["triage-search", "performance"]),
    ("PRM-65", "PRM-31", "S4", "story", "done", "medium", 3, "Sign in and sign out clinicians with clinic-scoped session binding",
     desc("Allow clinicians to sign in and sign out cleanly, clearing both browser cookies and server-side clinic queue subscriptions on sign-out.",
          ["Sign-out invalidates browser cookie and server session; protected triage routes immediately return 401."],
          "3 — Session teardown handler and redirect tests."), ["auth", "security"]),
    ("PRM-66", "PRM-31", "S4", "subtask", "done", "medium", None, "Scan gateway and clinical API logs for accidental token or MRN leakage",
     desc("Subtask under PRM-38 verifying structured log redaction across auth, intake, and prescription routes.",
          ["Automated log scan finds zero raw session tokens, cookies, or unmasked MRNs."],
          "No story points — subtask under PRM-38."), ["security", "auth"]),
    ("PRM-67", "PRM-32", "S5", "bug", "done", "high", 3, "Pharmacy directory timeout leaves spinner stuck on prescription summary screen",
     desc("When the stubbed pharmacy directory timed out after 5 seconds, the pharmacy selector stayed on an infinite spinner instead of showing the fallback message.",
          ["Directory timeout clears the spinner within 5 seconds and displays plain retry guidance."],
          "3 — AbortController timeout cleanup and UI state test.",
          ("Staging pharmacy picker with simulated 5s directory timeout", "Enter valid ZIP code while directory upstream is paused", "Spinner clears after 5s and shows retry banner", "Loading spinner remained stuck indefinitely and blocked pharmacy selection")), ["prescriptions", "accessibility"]),
    ("PRM-68", "PRM-32", "S5", "story", "done", "medium", 3, "Display clear prescription fulfilment timeline to patients in portal",
     desc("Show patients whether their prescription is signed, sent to pharmacy, dispensed, cancelled, or reversed using plain non-clinical language.",
          ["Patient portal displays current prescription status and next step without internal payer codes."],
          "3 — Status timeline component and copy review."), ["prescriptions"]),
    ("PRM-69", "PRM-32", "S5", "subtask", "done", "medium", None, "Verify idempotency of clinician slot release on repeated visit cancellation",
     desc("Subtask under PRM-13/PRM-45 confirming repeated cancellation requests release the physician calendar slot only once.",
          ["Repeated DELETE/cancel calls return the first cancellation record and never double-free calendar capacity."],
          "No story points — subtask under PRM-45."), ["prescriptions", "video-room"]),
    ("PRM-70", "PRM-33", "S6", "story", "done", "medium", 3, "Filter clinical formulary table by drug class and active status",
     desc("Let clinicians and pharmacists filter the formulary view by therapeutic class and active status while hiding deactivated drugs from standard prescribing.",
          ["Formulary search filters by drug class; deactivated drugs stay hidden from default prescribing picker."],
          "3 — Query filter parameters and prescribing picker guard."), ["formulary", "triage-search"]),
    ("PRM-71", "PRM-33", "S6", "bug", "done", "medium", 2, "Formulary CSV error export omits UTF-8 BOM for accented generic drug names",
     desc("Downloaded rejected-row CSV files rendered accented characters incorrectly in spreadsheet viewers because the response lacked a UTF-8 signature.",
          ["Rejected-row CSV export includes UTF-8 BOM and preserves characters accurately."],
          "2 — CSV response header and encoding test.",
          ("Staging formulary import failure download", "Download rejected-row CSV containing accented drug notes", "Spreadsheet viewer renders UTF-8 text cleanly", "Missing UTF-8 header caused garbled characters")), ["formulary"]),
    ("PRM-72", "PRM-33", "S6", "subtask", "done", "medium", None, "Test background formulary worker restart mid-validation on 3,000-row CSV",
     desc("Subtask under PRM-50 proving that restarting the worker mid-validation resumes cleanly without writing partial formulary rows.",
          ["Worker restart mid-run resumes validation and preserves all-or-nothing database apply."],
          "No story points — subtask under PRM-50."), ["formulary", "performance"]),
    ("PRM-73", "PRM-3", "S7", "story", "in_progress", "high", 5, "Expose multi-clinic operational metrics dashboard without patient PHI",
     desc("Provide pilot clinic leads a live dashboard of check-in queue depth, WebRTC TURN error rate, and prescription sign-off latency with zero patient PHI.",
          ["Dashboard displays queue depth, TURN errors, and P95 latency per clinic with zero PHI fields."],
          "5 — Prometheus/Grafana metric aggregation and PHI redaction check."), ["hipaa-pilot", "performance", "security"]),
    ("PRM-74", "PRM-3", "S7", "task", "planned", "medium", 3, "Document multi-clinic pilot rollback runbook and on-call escalation path",
     desc("Write and verify the step-by-step rollback procedure and clinical ops escalation path if pilot signaling or e-prescribing degrades.",
          ["Runbook documents 5-minute rollback steps and named clinical + engineering on-call owners."],
          "3 — Runbook drill in staging and sign-off."), ["hipaa-pilot"]),
    ("PRM-75", "PRM-3", "S7", "task", "planned", "low", 2, "Archive pre-pilot synthetic load-test artifacts and reset staging counters",
     desc("Clean up temporary load-test rooms and archive benchmark reports once PRM-28 and PRM-30 sign off.",
          ["Staging load-test rooms are archived and baseline counters are reset cleanly."],
          "2 — Cleanup script and retention verification."), ["hipaa-pilot", "performance"]),
]

ISSUES = ISSUES_PART1 + ISSUES_PART2 + ISSUES_PART3

COMMENTS = [
    ("PRM-14", "<p><strong>Incident trace</strong></p><p>Attached redacted gateway trace <code>duplicate-checkin-requests.log</code>: two separate request IDs reached the clinical API 180ms apart and both returned 201 Created; there was zero proxy or gateway retry. Follow-ups split into PRM-15 (idempotency key), PRM-16 (disable Join button), and PRM-21 (audited calendar slot repair).</p>"),
    ("PRM-15", "<p><strong>Verification result</strong></p><p>PASS — Repeated check-in requests with the same appointment idempotency token now return the existing room (VISIT-2041) and physician calendar hold before any new allocation occurs.</p>"),
    ("PRM-17", "<p><strong>Verification result</strong></p><p>PASS — Simulated 504 payer timeout during prescription sign-off stops prescription creation immediately and leaves the draft medication order untouched.</p>"),
    ("PRM-19", "<p><strong>Verification result</strong></p><p>PASS — Replaying the renewal webhook callback on a 10mg chart with a +20mg adjustment now returns the persisted first snapshot (30mg) instead of doubling to 60mg.</p>"),
    ("PRM-20", "<p><strong>Verification result</strong></p><p>PASS — The pharmacy reversal key is committed to PostgreSQL before calling the external e-prescribe network. Timeout retries look up and return the first reversal record without sending a duplicate network request.</p>"),
    ("PRM-21", "<p><strong>Verification result</strong></p><p>PASS — Direct SQL schedule edits are blocked by policy; the audited clinical correction endpoint records visit reference, clinical reason, staff actor, and slot/dosage delta in the immutable compliance log.</p>"),
    ("PRM-22", "<p><strong>Reopened after verification</strong></p><p>Reopened after a patient reported the second SMS reminder still opened the old pre-reschedule room VISIT-1902 instead of VISIT-1909. Both initial send and resend now read the canonical appointment version ID, and regression tests pass.</p>"),
    ("PRM-25", "<p><strong>Verification result</strong></p><p>PASS — Clinical safety set a 15-second upper bound for allergy-flag search visibility after edit; the shared idempotent indexing handler now covers both intake and chart-edit events with lag and dead-letter metrics.</p>"),
    ("PRM-26", "<p><strong>Work update</strong></p><p>PRM-27 caught one stale clinic-queue list badge during bulk night-shift roster import while single-chart invalidation stayed within 2 seconds; keeping PRM-26 in_progress until list invalidation is closed.</p>"),
    ("PRM-28", "<p><strong>Work update</strong></p><p>Waiting-room and chart reads pass concurrency targets, but end-to-end visit sign-off exceeds P95 while WebRTC TURN credentials and payer eligibility checks are cold (tracked in PRM-29).</p>"),
    ("PRM-38", "<p><strong>Verification result</strong></p><p>PASS — Session cookies enforce Secure, HttpOnly, SameSite=Strict, and the 15-minute HIPAA idle timeout without losing server-side clinical note drafts. Log scan confirmed zero raw tokens.</p>"),
    ("PRM-39", "<p><strong>Verification result</strong></p><p>PASS — Repeated sign-in failures trigger a temporary 429 keyed on source IP and hashed account/MRN identifier; valid sign-in resets the counter and no raw email or MRN appears in logs.</p>"),
    ("PRM-47", "<p><strong>Verification result</strong></p><p>PASS — Clinical pharmacists can update drug name, dosage form, max daily dose, active state, and allergy cross-reactivity via validated API calls that refresh triage search and write a before/after audit record.</p>"),
    ("PRM-48", "<p><strong>Verification result</strong></p><p>PASS — Server-side role check restricts formulary mutations to verified clinical-pharmacist members; triage nurses, patients, ops staff, and forged headers receive an identical 403 Forbidden response.</p>"),
    ("PRM-50", "<p><strong>Verification result</strong></p><p>PASS — 3,000-row formulary CSV runs in the background worker, reports read/accepted/rejected counts, aborts completely if any row has a conflicting NDC or invalid dose, and updates the formulary once when 100% valid.</p>"),
    ("PRM-52", "<p><strong>Verification result</strong></p><p>PASS — Append-only compliance audit table records pharmacist actor, timestamp, source, clinical reason, and changed NDC codes without storing raw CSV uploads.</p>"),
]

ISSUE_LINKS = [
    ("PRM-11", "PRM-37", "blocks"),
    ("PRM-14", "PRM-15", "blocks"),
    ("PRM-14", "PRM-16", "blocks"),
    ("PRM-14", "PRM-21", "relates to"),
    ("PRM-45", "PRM-13", "blocks"),
    ("PRM-13", "PRM-20", "blocks"),
    ("PRM-46", "PRM-22", "relates to"),
    ("PRM-48", "PRM-47", "blocks"),
    ("PRM-50", "PRM-51", "blocks"),
    ("PRM-24", "PRM-25", "relates to"),
    ("PRM-26", "PRM-28", "relates to"),
    ("PRM-53", "PRM-30", "blocks"),
    ("PRM-28", "PRM-30", "blocks"),
]

CARRYOVERS = [
    ("PRM-40", "PulseRoom Sprint 4", "PulseRoom Sprint 5", "Forged clinician-header CI test needed deterministic fixture cleanup before closing."),
    ("PRM-22", "PulseRoom Sprint 3", "PulseRoom Sprint 4", "Reopened after resent SMS reminder after reschedule still read cached legacy booking ID VISIT-1902."),
    ("PRM-70", "PulseRoom Sprint 6", "PulseRoom Sprint 7", "Formulary drug-class filter query passed, carried final prescribing-picker guard verification."),
]

ESTIMATE_CHANGES = [
    ("PRM-37", "3", "5", "Walk-in chart merge needed idempotent allergy deduplication across concurrent sign-in callbacks."),
    ("PRM-20", "3", "5", "Pre-call pharmacy reversal key persistence and timeout recovery expanded scope beyond happy path."),
    ("PRM-50", "5", "8", "Restart-safe progress tracking and all-or-nothing 3,000-row formulary apply required durable worker state."),
]


def seed() -> int:
    now = datetime(2026, 11, 2, 16, 0, tzinfo=timezone.utc)
    with psycopg.connect(DB_URL) as conn:
        users = [r[0] for r in conn.execute("SELECT id FROM users WHERE username LIKE 'atlascart.%' ORDER BY id").fetchall()]
        owner_id = users[0] if users else conn.execute("SELECT id FROM users ORDER BY id LIMIT 1").fetchone()[0]
        if not users:
            users = [owner_id]

        existing = conn.execute("SELECT id FROM spaces WHERE space_key=%s AND deleted_at IS NULL", (SPACE_KEY,)).fetchone()
        if existing:
            space_id = existing[0]
            for tbl in ("comments", "issue_history", "issue_labels", "issue_attachments", "issue_code_links"):
                conn.execute(f"DELETE FROM {tbl} WHERE issue_id IN (SELECT id FROM issues WHERE space_id=%s)", (space_id,))
            conn.execute("DELETE FROM issue_links WHERE source_issue_id IN (SELECT id FROM issues WHERE space_id=%s) OR target_issue_id IN (SELECT id FROM issues WHERE space_id=%s)", (space_id, space_id))
            conn.execute("DELETE FROM sprint_issue_history WHERE sprint_id IN (SELECT id FROM sprints WHERE space_id=%s)", (space_id,))
            conn.execute("UPDATE issues SET parent_id=NULL WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM issues WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM labels WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM sprints WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM space_members WHERE space_id=%s", (space_id,))
            conn.execute("DELETE FROM space_groups WHERE space_id=%s", (space_id,))
            conn.execute("UPDATE spaces SET name=%s, color=%s, owner_id=%s, issue_counter=%s, updated_at=%s WHERE id=%s",
                         (SPACE_NAME, SPACE_COLOR, owner_id, len(ISSUES), now, space_id))
        else:
            space_id = conn.execute(
                "INSERT INTO spaces (name, space_key, color, owner_id, issue_counter, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (SPACE_NAME, SPACE_KEY, SPACE_COLOR, owner_id, len(ISSUES), now, now),
            ).fetchone()[0]

        atc = conn.execute("SELECT id FROM spaces WHERE space_key='ATC' AND deleted_at IS NULL LIMIT 1").fetchone()
        member_roles: dict[int, str] = {owner_id: "owner"}
        if atc:
            for uid, role in conn.execute("SELECT user_id, role FROM space_members WHERE space_id=%s", (atc[0],)).fetchall():
                member_roles[uid] = role
            for (gid,) in conn.execute("SELECT group_id FROM space_groups WHERE space_id=%s", (atc[0],)).fetchall():
                conn.execute("INSERT INTO space_groups (space_id, group_id, added_at) VALUES (%s, %s, %s)", (space_id, gid, now))
        for (uid,) in conn.execute("SELECT id FROM users WHERE username IN ('alice','john','charles','michael23fire')").fetchall():
            member_roles.setdefault(uid, "admin")
        for uid, role in member_roles.items():
            conn.execute("INSERT INTO space_members (space_id, user_id, role, joined_at) VALUES (%s, %s, %s, %s)", (space_id, uid, role, now))

        sprint_ids: dict[str, int] = {}
        for code, name, goal, sd, ed, st, order, ip, cp, fp, ic, cc, fc in SPRINTS:
            sid = conn.execute(
                "INSERT INTO sprints (space_id, name, goal, start_date, end_date, status, sprint_order, "
                "initial_committed_points, initial_completed_points, completed_points, final_scope_points, "
                "initial_issue_count, completed_issue_count, final_issue_count, unestimated_issue_count, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (space_id, name, goal, sd, ed, st, order, ip, cp, cp, fp, ic, cc, fc, 0 if ip else None, now, now),
            ).fetchone()[0]
            sprint_ids[code] = sid

        label_ids: dict[str, int] = {}
        for lbl in LABELS:
            lid = conn.execute(
                "INSERT INTO labels (space_id, name, normalized_name, created_at) VALUES (%s,%s,%s,%s) RETURNING id",
                (space_id, lbl, lbl.lower(), now),
            ).fetchone()[0]
            label_ids[lbl] = lid

        issue_ids: dict[str, int] = {}
        for idx, (key, parent_key, scode, itype, status, prio, pts, title, description, lbls) in enumerate(ISSUES, 1):
            sp_id = sprint_ids.get(scode) if scode else None
            assignee_id = users[idx % len(users)]
            reporter_id = users[(idx + 3) % len(users)]
            created = now - timedelta(days=max(1, 90 - idx))
            iid = conn.execute(
                "INSERT INTO issues (issue_key, space_id, sprint_id, parent_id, title, description, issue_type, status, priority, assignee_id, reporter_id, story_points, issue_order, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (key, space_id, sp_id, None, title, description, itype, status, prio, assignee_id, reporter_id, pts, idx * 10, created, created + timedelta(hours=6)),
            ).fetchone()[0]
            issue_ids[key] = iid
            for lbl in lbls:
                conn.execute("INSERT INTO issue_labels (issue_id, label_id) VALUES (%s,%s)", (iid, label_ids[lbl]))
            conn.execute(
                "INSERT INTO issue_history (issue_id, actor_id, event_type, description, created_at) VALUES (%s,%s,'issue_created','created issue with acceptance criteria and estimate reasoning',%s)",
                (iid, reporter_id, created),
            )
            if status in ("in_progress", "done"):
                conn.execute(
                    "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES (%s,%s,'field_change','status','planned','in_progress','Started implementation',%s)",
                    (iid, assignee_id, created + timedelta(hours=2)),
                )
            if status == "done":
                conn.execute(
                    "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES (%s,%s,'field_change','status','in_progress','done','Closed after AC-by-AC verification',%s)",
                    (iid, users[(idx + 5) % len(users)], created + timedelta(hours=5)),
                )

        for key, parent_key, *_ in ISSUES:
            if parent_key:
                conn.execute("UPDATE issues SET parent_id=%s WHERE id=%s", (issue_ids[parent_key], issue_ids[key]))

        for key, text in COMMENTS:
            conn.execute("INSERT INTO comments (issue_id, author_id, content, created_at, updated_at) VALUES (%s,%s,%s,%s,%s)",
                         (issue_ids[key], owner_id, text, now, now))

        for src, dst, kind in ISSUE_LINKS:
            conn.execute("INSERT INTO issue_links (source_issue_id, target_issue_id, link_type, created_at) VALUES (%s,%s,%s,%s)",
                         (issue_ids[src], issue_ids[dst], kind, now))

        for key, old_sp, new_sp, reason in CARRYOVERS:
            conn.execute("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES (%s,%s,'field_change','sprint',%s,%s,%s,%s)",
                         (issue_ids[key], owner_id, old_sp, new_sp, f"Carried over — {reason}", now))

        for key, old_pt, new_pt, reason in ESTIMATE_CHANGES:
            conn.execute("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES (%s,%s,'field_change','storyPoints',%s,%s,%s,%s)",
                         (issue_ids[key], owner_id, old_pt, new_pt, f"Updated storyPoints — {reason}", now))

        conn.execute(
            "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES (%s,%s,'field_change','status','done','in_progress','Reopened: resend SMS still used pre-reschedule room VISIT-1902',%s)",
            (issue_ids["PRM-22"], owner_id, now),
        )
        conn.commit()
    print(f"Seeded {SPACE_KEY} ({SPACE_NAME}, space_id={space_id}) with {len(ISSUES)} issues, {len(SPRINTS)} sprints, and {len(member_roles)} members.")
    return space_id


if __name__ == "__main__":
    seed()



