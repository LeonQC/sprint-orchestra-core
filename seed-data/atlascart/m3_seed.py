#!/usr/bin/env python3
"""Generate, validate, install, and remove cumulative AtlasCart Month-3 data.

M3 contains the approved M1 event ledger as its historical prefix. The M1
cutoff is shifted eight weeks before the final M3 snapshot, then four more
sprints continue the same product and team story. Standard-library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from html import escape as html_escape
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


FINAL_AS_OF = date.fromisoformat(os.environ.get("ATLASCART_AS_OF", datetime.now().date().isoformat()))
os.environ["ATLASCART_M3_AS_OF"] = FINAL_AS_OF.isoformat()
os.environ["ATLASCART_AS_OF"] = (FINAL_AS_OF - timedelta(days=56)).isoformat()

import m1_seed as m1  # noqa: E402  (must see the shifted M1 cutoff)
from m3_data import (  # noqa: E402
    CANONICAL_LINKS,
    DATASET_ID,
    DATASET_VERSION,
    FIXTURE_MARKER,
    FIXTURE_NAME,
    ISSUE_BY_KEY,
    ISSUES,
    LINK_REASONS,
    M1_CONTINUATIONS,
    M3_ASSIGNEE_CHANGES,
    M3_AS_OF,
    M3_CARRYOVERS,
    M3_ESTIMATE_CHANGES,
    M3_EARLY_IMPLEMENTATION_UPDATES,
    M3_STATUS_OVERRIDES,
    NEW_LABELS,
    SNAPSHOT_AT,
    SPRINTS,
    THREE_AC_KEYS,
    TWO_AC_KEYS,
    USER_TIMEZONES,
    dt,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generated" / "m3"
ATTACHMENT_DIR = OUTPUT_DIR / "attachments"
INJECT_SQL_PATH = OUTPUT_DIR / "inject_atlas_cart_m3.sql"
CLEANUP_SQL_PATH = OUTPUT_DIR / "cleanup_atlas_cart_m3.sql"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
VALIDATION_PATH = OUTPUT_DIR / "validation-report.md"
REVIEW_PATH = OUTPUT_DIR / "jira-review-path.md"

TOTAL_ISSUES = len(m1.ISSUES) + len(ISSUES)
DEFAULT_ATTACHMENT_RATIO = 30.0
DEFAULT_LINK_RATIO = 20.0
ATTACHMENT_RATIO = DEFAULT_ATTACHMENT_RATIO
LINK_RATIO = DEFAULT_LINK_RATIO
M3_ATTACHMENTS: list[dict[str, Any]] = []
M3_LINKS: list[tuple[str, str, str, datetime]] = []
M3_COMMENTS: list[dict[str, Any]] = []

VERIFIER_BY_ISSUE = {
    "ATC-53": "U11", "ATC-54": "U10", "ATC-55": "U10", "ATC-56": "U10",
    "ATC-57": "U03", "ATC-58": "U03", "ATC-59": "U10", "ATC-60": "U11",
    "ATC-61": "U10", "ATC-62": "U09", "ATC-63": "U12", "ATC-64": "U12",
    "ATC-65": "U12", "ATC-66": "U10", "ATC-67": "U03", "ATC-68": "U10",
    "ATC-69": "U10", "ATC-70": "U03", "ATC-71": "U03", "ATC-72": "U10",
    "ATC-73": "U11", "ATC-74": "U09", "ATC-75": "U12", "ATC-76": "U10",
}


def verifier_for(item: dict[str, Any]) -> str:
    if item["status"] != "done":
        raise ValueError(f"{item['key']} has no completion verifier because it is not Done")
    return VERIFIER_BY_ISSUE[item["key"]]


def q(value: Any) -> str:
    return m1.q(value)


def user_expr(user_id: str | None) -> str:
    if user_id is None:
        return "NULL"
    return f"(SELECT id FROM users WHERE username = {q(m1.USER_BY_ID[user_id]['username'])})"


def space_expr() -> str:
    return f"(SELECT id FROM spaces WHERE space_key = 'ATC' AND name = {q(FIXTURE_NAME)} AND deleted_at IS NULL)"


def issue_expr(key: str | None) -> str:
    return "NULL" if key is None else f"(SELECT id FROM issues WHERE issue_key = {q(key)})"


def sprint_expr(sprint_id: str | None) -> str:
    if sprint_id is None:
        return "NULL"
    sprint = next(row for row in SPRINTS if row["id"] == sprint_id)
    return f"(SELECT id FROM sprints WHERE space_id = {space_expr()} AND name = {q(sprint['name'])})"


def label_expr(label: str) -> str:
    return f"(SELECT id FROM labels WHERE space_id = {space_expr()} AND normalized_name = {q(label.lower())})"


def description_html(item: dict[str, Any]) -> str:
    parts = ["<h3>Context</h3>", f"<p>{html_escape(item['context'])}</p>"]
    if item.get("bug"):
        report = item["bug"]
        parts.extend([
            "<h3>Bug report</h3>", "<ul>",
            f"<li><strong>When / environment:</strong> {html_escape(report['when'])}</li>",
            f"<li><strong>Steps to reproduce:</strong> {html_escape(report['steps'])}</li>",
            f"<li><strong>Expected behavior:</strong> {html_escape(report['expected'])}</li>",
            f"<li><strong>Actual behavior:</strong> {html_escape(report['actual'])}</li>", "</ul>",
        ])
    parts.extend(["<h3>Acceptance criteria</h3>", "<ul>"])
    parts.extend(
        f"<li><strong>AC{index}</strong> — {html_escape(criterion)}</li>"
        for index, criterion in enumerate(item["acceptance_criteria"], 1)
    )
    parts.extend(["</ul>", "<h3>Estimate reasoning</h3>", f"<p>{html_escape(item['estimate_rationale'])}</p>"])
    return "\n".join(parts)


def coverage_target(ratio: float) -> int:
    if not math.isfinite(ratio) or ratio < 0 or ratio > 100:
        raise ValueError("Coverage ratio must be a number from 0 through 100")
    return min(TOTAL_ISSUES, math.ceil(TOTAL_ISSUES * ratio / 100))


def select_extension_links(target_nodes: int) -> list[tuple[str, str, str, datetime]]:
    if target_nodes <= 0:
        return []
    max_nodes = len({key for source, target, _kind, _at in CANONICAL_LINKS for key in (source, target)})
    if target_nodes > max_nodes:
        raise ValueError(f"The M3 semantic extension can cover at most {max_nodes} additional issues")
    memo: set[tuple[int, frozenset[str]]] = set()

    def choose(index: int, nodes: frozenset[str]) -> tuple[int, ...] | None:
        if len(nodes) == target_nodes:
            return ()
        if index == len(CANONICAL_LINKS) or len(nodes) > target_nodes:
            return None
        state = (index, nodes)
        if state in memo:
            return None
        memo.add(state)
        source, target, _kind, _at = CANONICAL_LINKS[index]
        expanded = nodes | {source, target}
        if expanded != nodes and len(expanded) <= target_nodes:
            rest = choose(index + 1, expanded)
            if rest is not None:
                return (index,) + rest
        return choose(index + 1, nodes)

    indexes = choose(0, frozenset())
    if indexes is None:
        raise ValueError(f"No meaningful M3 link subset covers exactly {target_nodes} additional issues")
    return [CANONICAL_LINKS[index] for index in indexes]


def frontend_visuals(item: dict[str, Any]) -> list[dict[str, Any]]:
    if "frontend" not in item["labels"]:
        return []
    slug = "-".join(re.findall(r"[a-z0-9]+", item["title"].lower())[:5])
    result: list[dict[str, Any]] = []
    for index, criterion in enumerate(item["acceptance_criteria"], 1):
        before_name = f"{slug}-ac{index}-before.png"
        result.append({
            "issue": item["key"], "uploader": item["reporter"], "filename": before_name,
            "content_type": "image/png", "kind": "screenshot", "placement": "description",
            "created_at": item["created_at"] + timedelta(minutes=4 + index),
            "purpose": f"Before screenshot supplied with {item['key']} AC{index}, showing the current UI gap the acceptance criterion must change: {criterion}",
            "caption": f"Before {item['key']} AC{index}: {criterion}", "screenshot_type": "m3_frontend",
            "ac_index": index, "phase": "before", "context": f"{item['key']} / AC{index} / BEFORE",
        })
        if item["status"] == "done":
            after_name = f"{slug}-ac{index}-after.png"
            result.append({
                "issue": item["key"], "uploader": verifier_for(item), "filename": after_name,
                "content_type": "image/png", "kind": "screenshot", "placement": "comment",
                "created_at": item["done_at"] - timedelta(minutes=5 - index),
                "purpose": f"After screenshot attached to {item['key']} close verification for AC{index}, proving the implemented UI behavior: {criterion}",
                "caption": f"After {item['key']} AC{index}: {criterion}", "screenshot_type": "m3_frontend",
                "ac_index": index, "phase": "after", "context": f"{item['key']} / AC{index} / AFTER",
            })
    if item["status"] == "done" and item["key"] in {"ATC-54", "ATC-62", "ATC-74"}:
        regression_name = f"{slug}-regression-pass.png"
        result.append({
            "issue": item["key"], "uploader": verifier_for(item), "filename": regression_name,
            "content_type": "image/png", "kind": "screenshot", "placement": "comment",
            "created_at": item["done_at"] - timedelta(minutes=3),
            "purpose": f"Regression-run screenshot attached to the {item['key']} close verification, showing the issue-specific frontend checks passed alongside the visual acceptance result.",
            "caption": f"{item['key']} regression suite passed for {item['title']}",
            "screenshot_type": "m3_regression", "phase": "after",
            "evidence_role": "regression", "context": f"{item['key']} / REGRESSION / PASS",
        })
    return result


def bug_diagnostic(item: dict[str, Any]) -> dict[str, Any]:
    report = item["bug"]
    service = {"ATC-59": "cart-api", "ATC-68": "notification-worker", "ATC-76": "search-indexer"}.get(item["key"], "atlas-service")
    error = {
        "ATC-59": "merge_callback_replayed guest_token=gt_7f2 prior_result=missing quantity_applied_twice=true",
        "ATC-68": "order_link_reference_mismatch event_order=BETA-1128 expected_path=/orders/BETA-1128 rendered_path=/orders/BETA-1123",
        "ATC-76": "catalog_refresh_handler_missing event=product.updated sku=C-311 index_lag_seconds=43186",
    }.get(item["key"], "acceptance_mismatch detected=true")
    captured_at = item["created_at"] + timedelta(minutes=12)
    content = "\n".join([
        f"{captured_at.isoformat()} level=ERROR service={service} issue={item['key']} request_id=req_synthetic_{item['key'][4:]}",
        f"WHEN: {report['when']}",
        f"STEPS: {report['steps']}",
        f"EXPECTED: {report['expected']}",
        f"ACTUAL: {report['actual']}",
        f"ERROR: {error}",
        "DATA_SAFETY: synthetic fixture values only; no customer data or credential is present.",
        "",
    ])
    slug = "-".join(re.findall(r"[a-z0-9]+", item["title"].lower())[:7])
    return {
        "issue": item["key"], "uploader": item["reporter"], "filename": f"{slug}-diagnostic.log",
        "content_type": "text/plain", "kind": "log", "placement": "description",
        "created_at": captured_at,
        "purpose": f"Issue-opening diagnostic for {item['key']}, preserving the observed error plus explicit expected and actual output so engineering can reproduce the bug without guessing.",
        "content": content, "evidence_role": "bug_diagnostic",
    }


def curated_attachment(item: dict[str, Any]) -> list[dict[str, Any]]:
    key = item["key"]
    common = {"issue": key, "uploader": verifier_for(item) if item["status"] == "done" else item["reporter"], "placement": "comment", "comment_mode": "append"}
    if key == "ATC-53":
        return [common | {"filename": "gateway-auth-boundary.pdf", "content_type": "application/pdf", "kind": "pdf", "created_at": item["done_at"] - timedelta(minutes=8), "purpose": "Reviewed gateway authentication boundary documenting public routes, protected routes, trusted headers, failure behavior, and security checks used to close ATC-53.", "title": "AtlasCart gateway authentication boundary", "sections": [
            {"heading": "1. Purpose and trust boundary", "lines": ["The gateway owns customer session validation.", "Browser identity headers are untrusted and removed.", "Catalog routes remain public while account and order routes require a verified session."]},
            {"heading": "2. Route behavior", "lines": ["The table is the reviewed contract used by gateway, API, and QA."], "table": {"headers": ["Route", "Session", "Result", "Identity"], "rows": [["/products", "none", "allow", "none"], ["/account", "valid", "allow", "verified ID"], ["/orders", "expired", "401", "none"], ["/orders", "forged header", "401", "stripped"]]}},
            {"heading": "3. Verification", "lines": ["Public catalog smoke tests passed.", "Expired and forged cases return the same safe response.", "Logs contain request IDs but no cookie or session value."]},
        ]}]
    if key == "ATC-56":
        return [common | {"filename": "guest-cart-merge-matrix.xlsx", "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "kind": "xlsx", "created_at": item["done_at"] - timedelta(minutes=8), "purpose": "QA merge matrix recording guest-only, account-only, overlapping, stock-limited, and repeated-callback results used to verify ATC-56.", "sheet": "Cart merge", "rows": [["Case", "Guest", "Account", "Stock", "Expected", "Observed", "Result"], ["Different SKUs", "F-612 x1", "C-311 x1", "enough", "both once", "both once", "PASS"], ["Overlap", "F-612 x1", "F-612 x2", "9", "F-612 x3", "F-612 x3", "PASS"], ["Stock limit", "C-311 x3", "C-311 x2", "4", "x4 + note", "x4 + note", "PASS"], ["Callback retry", "F-612 x1", "F-612 x2", "9", "x3 once", "x3 once", "PASS"]]}]
    if key == "ATC-63":
        return [common | {"filename": "beta-order-state-model.pdf", "content_type": "application/pdf", "kind": "pdf", "created_at": item["done_at"] - timedelta(minutes=8), "purpose": "Product and operations decision record defining allowed beta order transitions, responsible actors, customer communication, and invalid-transition behavior for ATC-63.", "title": "AtlasCart beta order state model", "sections": [
            {"heading": "1. Scope", "lines": ["The model covers placed, processing, shipped, cancelled, refund pending, and refunded.", "It does not define returns, exchanges, or partial refunds."]},
            {"heading": "2. Allowed transitions", "lines": ["Every accepted transition creates an append-only history record."], "table": {"headers": ["From", "To", "Actor", "Customer note"], "rows": [["placed", "processing", "system", "order accepted"], ["placed", "cancelled", "operations", "stock restored"], ["processing", "shipped", "operations", "tracking available"], ["cancelled", "refund pending", "operations", "refund started"], ["refund pending", "refunded", "system", "refund completed"]]}},
            {"heading": "3. Rejected transitions", "lines": ["Shipped orders cannot be cancelled.", "Repeated transitions return the existing result.", "A 409 response leaves both status and inventory unchanged."]},
        ]}]
    if key == "ATC-73":
        return [common | {"filename": "background-import-run.csv", "content_type": "text/csv", "kind": "csv", "created_at": item["done_at"] - timedelta(minutes=8), "purpose": "Test results for the five-thousand-row background import, showing queue time, validation counts, catalog changes, search refresh, and failures for ATC-73.", "content": "phase,processed,valid,rejected,catalog_rows_written,elapsed_seconds,result\nqueued,0,0,0,0,0,PASS\nvalidating,1000,1000,0,0,4,PASS\nvalidating,3000,2998,2,0,11,PASS\nfailed,5000,4998,2,0,19,PASS\napplied,5000,5000,0,5000,24,PASS\nsearch_refresh,5000,5000,0,5000,31,PASS\n"}]
    return []


def supplemental_attachment(item: dict[str, Any], index: int) -> dict[str, Any]:
    done = item["status"] == "done"
    created = (item["done_at"] - timedelta(minutes=8)) if done else min(SNAPSHOT_AT - timedelta(hours=1), (item["started_at"] or item["created_at"]) + timedelta(hours=3))
    content = [f"# {item['key']} — {item['title']}", "", "## Purpose", "", f"This {'completion record' if done else 'working note'} stays within the issue acceptance criteria and uses synthetic AtlasCart data.", "", "## Estimate", "", item["estimate_rationale"], "", "## Acceptance checks", ""]
    for ac_index, criterion in enumerate(item["acceptance_criteria"], 1):
        content.extend([f"### AC{ac_index}", "", f"- Expected: {criterion}", f"- Result: {'PASS' if done else 'PENDING'}", ""])
    content.extend(["## Reproduction context", "", "```text", f"dataset={DATASET_ID}", f"issue={item['key']}", f"snapshot={M3_AS_OF.isoformat()}", "```", "", "## Data handling", "", "No real customer, credential, address, or payment data is present."])
    slug = "-".join(re.findall(r"[a-z0-9]+", item["title"].lower())[:7])
    return {"issue": item["key"], "uploader": verifier_for(item) if done else item["reporter"], "filename": f"{slug}-{'acceptance' if done else 'work-note'}.md", "content_type": "text/markdown", "kind": "markdown", "placement": "comment", "comment_mode": "append", "created_at": created, "purpose": f"Issue-specific {'completion evidence' if done else 'working reference'} for {item['key']}, preserving its estimate boundary and every defined plain-English acceptance check without adding hidden scope.", "content": "\n".join(content) + "\n"}


def make_comment(issue: str, author: str, at: datetime, content: str, **meta: Any) -> dict[str, Any]:
    local_at = at.astimezone(ZoneInfo(USER_TIMEZONES[author]))
    return {"issue": issue, "author": author, "created_at": local_at, "content": content, **meta}


def build_comments() -> list[dict[str, Any]]:
    selected_files = {attachment["filename"] for attachment in M3_ATTACHMENTS}
    comments: list[dict[str, Any]] = []
    for key, continuation in M1_CONTINUATIONS.items():
        comments.append(make_comment(key, continuation["author"], continuation["at"], f"<p><strong>Sprint 3 completion</strong></p><p>{html_escape(continuation['comment'])}</p>"))
    for key, carry in M3_CARRYOVERS.items():
        comments.append(make_comment(key, "U03", carry["at"], f"<p><strong>Carried from {carry['from']} to {carry['to']}</strong></p><p>{html_escape(carry['reason'])}</p><p>The unfinished acceptance work remains visible on this issue; completed scope is not counted until the destination sprint.</p>", carryover=True))
    for key, change in M3_ESTIMATE_CHANGES.items():
        comments.append(make_comment(key, "U03", change["at"], f"<p><strong>Estimate changed from {change['from']} to {change['to']}</strong></p><p>{html_escape(change['reason'])}</p><p>What we plan to deliver has not changed. The higher estimate covers work we found during refinement.</p>", estimate_change=True))
    for key, change in M3_ASSIGNEE_CHANGES.items():
        comments.append(make_comment(key, "U03", change["at"], f"<p><strong>Assignee changed</strong></p><p>{html_escape(change['reason'])}</p><p>{m1.USER_BY_ID[change['to']]['name']} now owns the remaining work and handoff.</p>", assignee_change=True))
    comments.append(make_comment("ATC-68", "U10", dt(-19, 11, 50), "<p><strong>First verification</strong></p><p>PASS — the first email opened BETA-1128, and three nearby order numbers each opened the right page.</p><p>I closed the issue based on those checks. The resent-email case was not part of this run.</p>", initial_verification=True))
    comments.append(make_comment("ATC-68", "U12", dt(-18, 8, 30), "<p><strong>New customer report</strong></p><p>A beta shopper received the tracking email again this morning. The View order button opened BETA-1123 instead of BETA-1128.</p><p>I removed the shopper details and kept only the two test order numbers.</p>", customer_observation=True))
    comments.append(make_comment("ATC-68", "U10", dt(-18, 9), "<p><strong>Reopened after verification</strong></p><p>I reproduced Jack's example when the email job sends the message a second time. The first email is fixed, but the resent email still opens the wrong order.</p><p>Reopening until both email paths pass and the account check still blocks other shoppers.</p>", reopened=True))
    comments.append(make_comment("ATC-68", "U06", dt(-18, 11), "<p><strong>Cause found</strong></p><p>The first email uses the new link, but the resend path was still reading an older saved value.</p><p>I am updating that second path and adding the resend case to the same regression test.</p>", followup_diagnosis=True))
    comments.append(make_comment("ATC-68", "U06", dt(-17, 13), "<p><strong>Follow-up fix ready</strong></p><p>The first send and resend now build the link from the same full order number. Tests also confirm that a shopper cannot open an order from another account.</p><p>The follow-up pull request is ready for the final QA run.</p>", followup_fix=True))
    for child in (item for item in ISSUES if item["type"] == "subtask"):
        parent = ISSUE_BY_KEY[child["parent"]]
        comments.append(make_comment(parent["key"], "U03", child["created_at"] + timedelta(minutes=10), f"<p><strong>Created focused subtask {child['key']}</strong></p><p>{html_escape(child['context'])}</p><p>{child['key']} is unestimated because {parent['key']} retains the parent estimate and delivery risk.</p>", subtask_created=child["key"]))
    for item in ISSUES:
        if item["type"] == "epic":
            comments.append(make_comment(item["key"], item["reporter"], item["created_at"] + timedelta(minutes=20), f"<p><strong>What this epic covers</strong></p><p>{html_escape(item['context'])}</p><p>Each child issue should have one clear result that a reviewer can check.</p>"))
            continue
        if item["started_at"]:
            update = M3_EARLY_IMPLEMENTATION_UPDATES.get(item["key"], item["implementation"])
            comments.append(make_comment(item["key"], item["assignee"], item["started_at"] + timedelta(hours=3), f"<p><strong>Work update</strong></p><p>{html_escape(update)}</p><p><strong>What is included</strong><br>{html_escape(item['estimate_rationale'])}</p>"))
        if item["status"] == "done":
            parts = ["<p><strong>Verification result</strong></p>", "<p>PASS — the checks below passed in the test environment.</p>", "<p><strong>Acceptance criteria checked</strong></p>", "<ul>"]
            for index, criterion in enumerate(item["acceptance_criteria"], 1):
                parts.append(f"<li><strong>AC{index} — PASS</strong><br>{html_escape(criterion)}</li>")
            parts.extend(["</ul>", "<p><strong>Close decision</strong><br>All acceptance criteria passed. Any follow-up work stays on its own issue.</p>"])
            after = sorted([a for a in M3_ATTACHMENTS if a["issue"] == item["key"] and a.get("phase") == "after"], key=lambda a: a.get("ac_index", 999))
            for attachment in after:
                if attachment.get("evidence_role") == "regression":
                    parts.append(f"<p><strong>Regression run</strong><br>The frontend regression checks passed with the accepted visual state.</p><p>[[ATTACHMENT:{attachment['filename']}]]</p>")
                else:
                    parts.append(f"<p><strong>Visual evidence AC{attachment['ac_index']}</strong></p><p>[[ATTACHMENT:{attachment['filename']}]]</p>")
            comments.append(make_comment(item["key"], verifier_for(item), item["done_at"] - timedelta(minutes=2), "\n".join(parts), verification=True))
    for source, target, kind, linked_at in M3_LINKS:
        reason = LINK_REASONS[(source, target, kind)]
        source_item, target_item = ISSUE_BY_KEY[source], ISSUE_BY_KEY[target]
        if kind == "blocks":
            comments.append(make_comment(source, source_item["assignee"] or source_item["reporter"], linked_at + timedelta(minutes=2), f"<p><strong>Dependency created for {target}</strong></p><p>{html_escape(reason)}</p><p>{target} can continue its separate work, but cannot finish until this result is ready.</p>", link_context=(source, target, kind), link_phase="blocked-source"))
            comments.append(make_comment(target, target_item["assignee"] or target_item["reporter"], linked_at + timedelta(minutes=4), f"<p><strong>Blocked by {source}</strong></p><p>{html_escape(reason)}</p><p>This issue cannot finish until the team working on {source} shares a verified result.</p>", link_context=(source, target, kind), link_phase="blocked-target"))
            if source_item["status"] == "done":
                clear_at = source_item["done_at"] - timedelta(minutes=10)
                comments.append(make_comment(source, source_item["assignee"] or source_item["reporter"], clear_at, f"<p><strong>Dependency cleared for {target}</strong></p><p>The required behavior passed its acceptance checks and is ready for integration.</p>", link_context=(source, target, kind), link_phase="unblocked-source"))
                comments.append(make_comment(target, target_item["assignee"] or target_item["reporter"], clear_at + timedelta(minutes=2), f"<p><strong>Unblocked by {source}</strong></p><p>The result from {source} passed its checks, so this work can continue.</p>", link_context=(source, target, kind), link_phase="unblocked-target"))
        else:
            comments.append(make_comment(source, source_item["assignee"] or source_item["reporter"], linked_at + timedelta(minutes=2), f"<p><strong>Related work: {target}</strong></p><p>{html_escape(reason)}</p><p>This shares context but does not create a hidden dependency.</p>", link_context=(source, target, kind), link_phase="related-source"))
            comments.append(make_comment(target, target_item["assignee"] or target_item["reporter"], linked_at + timedelta(minutes=4), f"<p><strong>Related to {source}</strong></p><p>{html_escape(reason)}</p><p>Each issue still keeps its own work and verification.</p>", link_context=(source, target, kind), link_phase="related-target"))
    # Append-only evidence goes to the issue's latest comment and therefore has no placeholder.
    for comment in comments:
        for marker in re.findall(r"\[\[ATTACHMENT:([^]]+)\]\]", comment["content"]):
            if marker not in selected_files:
                comment["content"] = comment["content"].replace(f"<p>[[ATTACHMENT:{marker}]]</p>", "")
    return sorted(comments, key=lambda row: (row["created_at"], row["issue"]))


def configure(attachment_ratio: float, link_ratio: float) -> None:
    global LINK_RATIO, ATTACHMENT_RATIO, M3_ATTACHMENTS, M3_LINKS, M3_COMMENTS
    LINK_RATIO = link_ratio
    total_attachment_target = coverage_target(attachment_ratio)
    mandatory_extension_keys = {item["key"] for item in ISSUES if "frontend" in item["labels"] or item["type"] == "bug"}
    minimum_base_coverage = math.ceil(len(m1.ISSUES) * m1.DEFAULT_ATTACHMENT_RATIO / 100)
    minimum_total_coverage = minimum_base_coverage + len(mandatory_extension_keys)
    if total_attachment_target < minimum_total_coverage:
        minimum_ratio = math.ceil((minimum_total_coverage - 1) / TOTAL_ISSUES * 100)
        raise ValueError(
            f"M3 requires at least {minimum_ratio:g}% attachment coverage: the approved M1 prefix covers "
            f"{minimum_base_coverage} issues and all {len(mandatory_extension_keys)} M3 frontend/bug issues require evidence"
        )
    # Keep the approved M1 default prefix at low/normal M3 profiles. Increase
    # old coverage only when all 32 extension issues are already covered.
    base_attachment_target = max(minimum_base_coverage, total_attachment_target - len(ISSUES))
    # Stay infinitesimally below the exact integer boundary because M1 rounds
    # coverage up with ceil(); binary float noise must not add one old issue.
    base_attachment_ratio = (base_attachment_target / len(m1.ISSUES) * 100 - 1e-9) if base_attachment_target else 0
    total_link_target = coverage_target(link_ratio)
    base_link_capacity = len({key for s, t, _k, _a in m1.CANONICAL_ISSUE_LINKS for key in (s, t)})
    extension_link_capacity = len({key for s, t, _k, _a in CANONICAL_LINKS for key in (s, t)})
    if total_link_target == 1:
        raise ValueError("A semantic issue link necessarily covers at least two issues; use 0 or at least 2.5 percent")
    if total_link_target > base_link_capacity + extension_link_capacity:
        max_ratio = (base_link_capacity + extension_link_capacity) / TOTAL_ISSUES * 100
        raise ValueError(f"The reviewed semantic link graph can cover at most {base_link_capacity + extension_link_capacity}/{TOTAL_ISSUES} issues ({max_ratio:.2f} percent)")
    base_link_target = 0 if total_link_target == 0 else max(2, min(base_link_capacity, math.ceil(len(m1.ISSUES) * link_ratio / 100)))
    base_link_ratio = (base_link_target / len(m1.ISSUES) * 100 - 1e-9) if base_link_target else 0
    m1.configure_dataset(base_attachment_ratio, base_link_ratio)

    ATTACHMENT_RATIO = attachment_ratio
    base_covered = len({attachment["issue"] for attachment in m1.ATTACHMENTS})
    needed = total_attachment_target - base_covered
    priority = ["ATC-53", "ATC-56", "ATC-63", "ATC-73", "ATC-57", "ATC-60", "ATC-61", "ATC-65", "ATC-66", "ATC-67"]
    optional = priority + [item["key"] for item in ISSUES if item["key"] not in priority and item["key"] not in mandatory_extension_keys]
    selected_keys = list(sorted(mandatory_extension_keys, key=lambda key: int(key.split("-")[1])))
    selected_keys.extend(optional[:needed - len(selected_keys)])
    selected: list[dict[str, Any]] = []
    for index, key in enumerate(selected_keys):
        item = ISSUE_BY_KEY[key]
        visual = frontend_visuals(item)
        curated = curated_attachment(item)
        selected.extend(visual)
        selected.extend(curated)
        diagnostic = [bug_diagnostic(item)] if item["type"] == "bug" else []
        selected.extend(diagnostic)
        if not visual and not curated and not diagnostic:
            selected.append(supplemental_attachment(item, index))
    M3_ATTACHMENTS = sorted(selected, key=lambda row: (row["created_at"], row["issue"], row["filename"]))

    base_link_nodes = len({key for s, t, _k, _a in m1.ISSUE_LINKS for key in (s, t)})
    M3_LINKS = select_extension_links(total_link_target - base_link_nodes)
    M3_COMMENTS = build_comments()


configure(DEFAULT_ATTACHMENT_RATIO, DEFAULT_LINK_RATIO)


def attachment_storage_name(attachment: dict[str, Any]) -> str:
    if attachment["issue"] in m1.ISSUE_BY_KEY:
        return m1.attachment_storage_name(attachment)
    safe = attachment["filename"].lower().replace(" ", "-")
    return f"atlascart-m3-{attachment['issue'].lower()}-{safe}"


def all_attachments() -> list[dict[str, Any]]:
    return list(m1.ATTACHMENTS) + M3_ATTACHMENTS


def validate() -> list[str]:
    checks = [f"M1 prefix validation passed: {len(m1.validate())} checks"]
    assert [item["key"] for item in ISSUES] == [f"ATC-{number}" for number in range(49, 84)]
    assert len(ISSUES) == 35 and TOTAL_ISSUES == 83
    checks.append("35 causal extension issues produce 83 total sequential issues")
    m2_s5 = next(sprint for sprint in SPRINTS if sprint["id"] == "S5")
    m3_s6 = next(sprint for sprint in SPRINTS if sprint["id"] == "S6")
    assert m2_s5["status"] == "completed"
    assert date.fromisoformat(m2_s5["end_date"]) < date.fromisoformat(m3_s6["start_date"])
    checks.append("The logical M2 governance checkpoint closes Sprint 5 before M3 work starts in Sprint 6; no duplicate M2 fixture is required")
    for key, continuation in M1_CONTINUATIONS.items():
        original = m1.ISSUE_BY_KEY[key]
        effective_start = continuation.get("started_at") or original["started_at"]
        assert effective_start and original["created_at"] <= effective_start < continuation["at"]
        if original["status"] == "planned":
            assert continuation.get("started_at"), f"{key} needs an explicit M3 start event"
        else:
            assert original["status"] == "in_progress" and original["started_at"]
    checks.append("Every M1-to-M3 Done continuation has a start date and a complete planned-to-in-progress-to-done status path")
    ac_distribution = Counter(len(item["acceptance_criteria"]) for item in ISSUES)
    assert ac_distribution == {1: 23, 2: 9, 3: 3}
    assert {item["key"] for item in ISSUES if len(item["acceptance_criteria"]) == 3} == THREE_AC_KEYS
    assert {item["key"] for item in ISSUES if len(item["acceptance_criteria"]) == 2} == TWO_AC_KEYS
    checks.append("AC distribution is enforced: 23 issues have one AC, 9 have two, and only 3 cross-system issues have three")
    for item in ISSUES:
        assert 1 <= len(item["acceptance_criteria"]) <= 3
        assert len(item["estimate_rationale"]) >= 100
        assert item["created_at"] <= SNAPSHOT_AT
        assert item["status"] in {"planned", "in_progress", "done"}
        assert all(f"<strong>AC{index}</strong>" in description_html(item) for index in range(1, len(item["acceptance_criteria"]) + 1))
        if item["type"] == "bug":
            assert item["bug"] and set(item["bug"]) == {"when", "steps", "expected", "actual"}
        if item["status"] == "done":
            assert item["started_at"] and item["done_at"] and item["started_at"] < item["done_at"] < SNAPSHOT_AT
    checks.append("Every extension issue has one to three plain-English ACs, unique estimate reasoning, and valid lifecycle timestamps")
    assert Counter(item["type"] for item in ISSUES)["subtask"] == 3
    assert all(item["parent"] in ISSUE_BY_KEY and item["points"] is None for item in ISSUES if item["type"] == "subtask")
    assert {comment.get("subtask_created") for comment in M3_COMMENTS if comment.get("subtask_created")} == {item["key"] for item in ISSUES if item["type"] == "subtask"}
    checks.append("Three active-sprint subtasks expose focused cache, accessibility, and load-test work without double-counting parent points")
    verification_keys = {comment["issue"] for comment in M3_COMMENTS if comment.get("verification")}
    done_keys = {item["key"] for item in ISSUES if item["status"] == "done"}
    assert verification_keys == done_keys
    for comment in M3_COMMENTS:
        assert comment["created_at"] <= SNAPSHOT_AT
        assert comment["content"].startswith("<p>")
    checks.append(f"{len(M3_COMMENTS)} readable extension comments include one AC-by-AC close result for every Done issue")
    verifier_counts = Counter(comment["author"] for comment in M3_COMMENTS if comment.get("verification"))
    assert set(verifier_counts) == {"U03", "U09", "U10", "U11", "U12"}
    assert max(verifier_counts.values()) / sum(verifier_counts.values()) < 0.5
    for item in (row for row in ISSUES if row["status"] == "done"):
        assert verifier_for(item) != item["assignee"]
    checks.append(f"Completion verification is distributed by domain across five people: {dict(verifier_counts)}")

    highest = [item for item in ISSUES if item["type"] != "epic" and item["priority"] == "highest"]
    assert {item["key"] for item in highest} == {"ATC-56", "ATC-59", "ATC-66", "ATC-68"}
    assert len(highest) / len([item for item in ISSUES if item["type"] != "epic"]) < 0.15
    checks.append("Highest priority is limited to two confirmed incidents and two money-adjacent features that the team raised before beta")

    assert len(M3_CARRYOVERS) == 3 and len(M3_ESTIMATE_CHANGES) == 3 and len(M3_ASSIGNEE_CHANGES) == 3
    assert sum(1 for events in M3_STATUS_OVERRIDES.values() for _at, old, new in events if old == "done" and new == "in_progress") == 1
    assert {comment["issue"] for comment in M3_COMMENTS if comment.get("carryover")} == set(M3_CARRYOVERS)
    assert {comment["issue"] for comment in M3_COMMENTS if comment.get("estimate_change")} == set(M3_ESTIMATE_CHANGES)
    assert {comment["issue"] for comment in M3_COMMENTS if comment.get("assignee_change")} == set(M3_ASSIGNEE_CHANGES)
    assert {comment["issue"] for comment in M3_COMMENTS if comment.get("reopened")} == set(M3_STATUS_OVERRIDES)
    checks.append("M3 imperfection budget is enforced: three carry-overs, three estimate changes, three handoffs, and one reopened bug")

    atc68_comments = [comment for comment in M3_COMMENTS if comment["issue"] == "ATC-68"]
    assert len([comment for comment in atc68_comments if comment.get("initial_verification")]) == 1
    assert len([comment for comment in atc68_comments if comment.get("customer_observation") and comment["author"] == "U12"]) == 1
    assert len([comment for comment in atc68_comments if comment.get("reopened") and comment["author"] == "U10"]) == 1
    assert len([comment for comment in atc68_comments if comment.get("followup_diagnosis") and comment["author"] == "U06"]) == 1
    assert len([comment for comment in atc68_comments if comment.get("followup_fix") and comment["author"] == "U06"]) == 1
    atc68_code_links = sorted((link for link in extension_code_links() if link["issue"] == "ATC-68"), key=lambda link: link["created_at"])
    first_done_at = next(at for at, old, new in M3_STATUS_OVERRIDES["ATC-68"] if old == "in_progress" and new == "done")
    reopen_at = next(at for at, old, new in M3_STATUS_OVERRIDES["ATC-68"] if old == "done" and new == "in_progress")
    assert len(atc68_code_links) == 2
    assert atc68_code_links[0]["last_activity_at"] < first_done_at < reopen_at
    assert reopen_at < atc68_code_links[1]["created_at"] < atc68_code_links[1]["last_activity_at"] < ISSUE_BY_KEY["ATC-68"]["done_at"]
    checks.append("ATC-68 preserves the full first-close, customer report, QA reopen, engineering diagnosis, follow-up PR, and final verification sequence")

    cumulative_prs: defaultdict[str, list[tuple[datetime, int, str]]] = defaultdict(list)
    for link in m1.CODE_LINKS:
        if link.kind == "pull_request":
            cumulative_prs[link.repo].append((link.created_at, int(link.ref_id), link.issue))
    for link in extension_code_links():
        cumulative_prs[link["repo"]].append((link["created_at"], int(link["ref_id"]), link["issue"]))
    for repo, pull_requests in cumulative_prs.items():
        chronological = sorted(pull_requests, key=lambda row: (row[0], row[1]))
        numbers = [row[1] for row in chronological]
        assert all(left < right for left, right in zip(numbers, numbers[1:])), f"{repo} PR numbers do not follow creation order"
    checks.append("Every repository assigns strictly increasing pull-request numbers in creation-time order across the cumulative M1 and M3 history")

    jack_forbidden = ("serialized", "idempotency", "worker replay", "atomic apply", "transaction", "payload")
    for comment in (row for row in M3_COMMENTS if row["author"] == "U12" and not row.get("verification")):
        assert not any(term in comment["content"].lower() for term in jack_forbidden), f"Jack guesses at a technical cause in {comment['issue']}"
    needless_phrases = (
        "acceptance boundary", "independently reviewable", "owning issue",
        "owning team", "final integration", "synthetic staging",
        "scope reminder", "evidence-backed",
    )
    reader_text = "\n".join([*(description_html(item) for item in ISSUES), *(comment["content"] for comment in M3_COMMENTS)]).lower()
    assert not any(phrase in reader_text for phrase in needless_phrases)
    jargon_terms = (
        "adapter", "payload", "idempotency", "serialization", "authorization",
        "atomic", "asynchronous", "durable", "transaction", "gateway",
        "callback", "concurrency", "backfill", "replay", "contract",
        "fixture", "mutation", "attribution", "multipart", "redaction",
    )
    for item in ISSUES:
        issue_text = " ".join((item["context"], *item["acceptance_criteria"], item["estimate_rationale"], item.get("implementation") or "")).lower()
        assert sum(term in issue_text for term in jargon_terms) <= 3, f"{item['key']} packs too many technical terms into reader-facing text"
    checks.append("Plain-language guardrails reject formal filler, limit jargon density, and keep Jack's comments free of guessed technical causes")

    human_events: list[tuple[str, str, datetime]] = []
    for item in ISSUES:
        human_events.append((f"{item['key']} created", item["reporter"], item["created_at"]))
        if item["started_at"]:
            human_events.append((f"{item['key']} started", item["assignee"], item["started_at"]))
        if item["done_at"]:
            human_events.append((f"{item['key']} done", item["assignee"], item["done_at"]))
    human_events.extend((f"{comment['issue']} comment", comment["author"], comment["created_at"]) for comment in M3_COMMENTS)
    human_events.extend((f"{row['issue']} attachment", row["uploader"], row["created_at"]) for row in M3_ATTACHMENTS)
    human_events.extend((f"{link['issue']} code link", link["creator"], link["created_at"]) for link in extension_code_links())
    human_events.extend((f"{source}->{target} link", ISSUE_BY_KEY[source]["assignee"] or ISSUE_BY_KEY[source]["reporter"], at) for source, target, _kind, at in M3_LINKS)
    human_events.extend((f"{key} carry-over", "U03", row["at"]) for key, row in M3_CARRYOVERS.items())
    human_events.extend((f"{key} estimate change", "U03", row["at"]) for key, row in M3_ESTIMATE_CHANGES.items())
    human_events.extend((f"{key} assignee change", "U03", row["at"]) for key, row in M3_ASSIGNEE_CHANGES.items())
    for key, events in M3_STATUS_OVERRIDES.items():
        for at, old, new in events:
            actor = "U10" if old == "done" and new == "in_progress" else ISSUE_BY_KEY[key]["assignee"]
            human_events.append((f"{key} {old}->{new}", actor, at))
    for label, actor, at in human_events:
        local = at.astimezone(ZoneInfo(USER_TIMEZONES[actor]))
        assert local.weekday() < 5, f"Weekend human event without incident reason: {label} at {local.isoformat()}"
        assert 7 <= local.hour <= 20, f"Human event outside working window: {label} at {local.isoformat()}"
    assert len({at.utcoffset() for _label, _actor, at in human_events}) >= 3
    checks.append(f"All {len(human_events)} M3 human events occur on weekdays in the actor's Pacific, Mountain, Central, or Eastern work window")
    selected_link_ids = {(s, t, k) for s, t, k, _a in M3_LINKS}
    assert selected_link_ids <= set(LINK_REASONS)
    for identity in selected_link_ids:
        phases = {c.get("link_phase") for c in M3_COMMENTS if c.get("link_context") == identity}
        assert {"blocked-source", "blocked-target"} <= phases if identity[2] == "blocks" else phases == {"related-source", "related-target"}
    total_link_nodes = len({key for s, t, _k, _a in list(m1.ISSUE_LINKS) + M3_LINKS for key in (s, t)})
    assert total_link_nodes == coverage_target(LINK_RATIO)
    checks.append(f"Semantic links cover {total_link_nodes}/{TOTAL_ISSUES} issues for requested {LINK_RATIO:g}% with matching lifecycle comments")
    attachments = all_attachments()
    covered = {attachment["issue"] for attachment in attachments}
    assert len(covered) == coverage_target(ATTACHMENT_RATIO)
    markers = {marker for comment in M3_COMMENTS for marker in re.findall(r"\[\[ATTACHMENT:([^]]+)\]\]", comment["content"])}
    assert markers <= {attachment["filename"] for attachment in M3_ATTACHMENTS}
    for attachment in M3_ATTACHMENTS:
        binary = m1.attachment_bytes(attachment)
        assert binary and len(attachment["purpose"]) >= 60
        if attachment["kind"] == "pdf":
            assert binary.startswith(b"%PDF-1.4") and binary.count(b"/Type /Page ") >= 3
            assert len(binary) >= 4000 and b"/BaseFont /Helvetica-Bold" in binary
            assert binary.count(b"0.07 0.12 0.22 rg") >= 3
            assert binary.count(b"0.05 0.55 0.52 rg") >= 3
        if attachment["kind"] == "xlsx":
            import zipfile
            with zipfile.ZipFile(BytesIO(binary)) as book:
                assert book.testzip() is None
        if attachment["kind"] == "csv":
            rows = list(csv.reader(attachment["content"].splitlines()))
            assert len(rows) >= 5
    checks.append(f"Attachment coverage is {len(covered)}/{TOTAL_ISSUES} issues with real, issue-specific M1 and M3 files")
    frontend_selected = {a["issue"] for a in M3_ATTACHMENTS if a.get("screenshot_type") == "m3_frontend"}
    frontend_keys = {item["key"] for item in ISSUES if "frontend" in item["labels"]}
    assert frontend_selected == frontend_keys
    for key in frontend_keys:
        item = ISSUE_BY_KEY[key]
        before = [a for a in M3_ATTACHMENTS if a["issue"] == key and a.get("phase") == "before"]
        after = [a for a in M3_ATTACHMENTS if a["issue"] == key and a.get("phase") == "after" and a.get("evidence_role") != "regression"]
        assert len(before) == len(item["acceptance_criteria"])
        assert len(after) == (len(item["acceptance_criteria"]) if item["status"] == "done" else 0)
        if item["status"] == "done":
            verification = next(comment for comment in M3_COMMENTS if comment["issue"] == key and comment.get("verification"))
            assert all(attachment["filename"] in verification["content"] for attachment in after)
    regression_keys = {a["issue"] for a in M3_ATTACHMENTS if a.get("evidence_role") == "regression"}
    assert regression_keys == {"ATC-54", "ATC-62", "ATC-74"}
    checks.append("Every M3 frontend issue has current PNG evidence; Done frontend work has after PNGs, with regression-run screenshots on representative stories")
    for item in (row for row in ISSUES if row["type"] == "bug"):
        diagnostic = [a for a in M3_ATTACHMENTS if a["issue"] == item["key"] and a.get("evidence_role") == "bug_diagnostic"]
        assert len(diagnostic) == 1
        content = diagnostic[0]["content"]
        assert all(token in content for token in ("EXPECTED:", "ACTUAL:", "ERROR:"))
        if "frontend" in item["labels"]:
            assert any(a["issue"] == item["key"] and a.get("phase") == "before" for a in M3_ATTACHMENTS)
    checks.append("Every M3 bug carries an issue-opening diagnostic with expected, actual, and error output; frontend bugs additionally require PNG evidence")
    history_rows = extension_sprint_history_rows()
    for sprint_id in ("S4", "S5", "S6"):
        metrics = sprint_metrics(sprint_id)
        assert metrics[3] < metrics[2], f"{sprint_id} should not complete its entire final scope"
        carried = [row for row in history_rows if row["sprint"] == sprint_id and row["outcome"] == "carried_over"]
        assert len(carried) == 1
    assert {row["issue"] for row in history_rows if row["outcome"] == "carried_over"} == set(M3_CARRYOVERS)
    checks.append("Sprints 4–6 each close with one explained carry-over; final completed points remain below final scope")
    assert Counter(sprint["status"] for sprint in SPRINTS) == {"completed": 3, "active": 1}
    active_start = date.fromisoformat(SPRINTS[-1]["start_date"])
    active_end = date.fromisoformat(SPRINTS[-1]["end_date"])
    assert active_start <= M3_AS_OF <= active_end + timedelta(days=2)
    assert M3_AS_OF <= active_end or M3_AS_OF.weekday() in {5, 6}
    checks.append("M3 ends inside active Sprint 7 or its weekend close grace after six cumulative closed sprints")
    return checks


def _with_m3_fixture(builder: Any, *args: Any, **kwargs: Any) -> str:
    """Run an M1 SQL builder against the cumulative M3 fixture name."""
    original = m1.FIXTURE_NAME
    try:
        m1.FIXTURE_NAME = FIXTURE_NAME
        return builder(*args, **kwargs)
    finally:
        m1.FIXTURE_NAME = original


def updated_at_by_issue() -> dict[str, datetime]:
    values = {item["key"]: item["created_at"] for item in ISSUES}
    for comment in M3_COMMENTS:
        if comment["issue"] in values:
            values[comment["issue"]] = max(values[comment["issue"]], comment["created_at"])
    for attachment in M3_ATTACHMENTS:
        values[attachment["issue"]] = max(values[attachment["issue"]], attachment["created_at"])
    for source, target, _kind, at in M3_LINKS:
        values[source] = max(values[source], at)
        values[target] = max(values[target], at)
    return values


def extension_code_links() -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for item in (row for row in ISSUES if row["type"] != "epic" and row["started_at"]):
        labels = set(item["labels"])
        repo = "atlascart-web" if "frontend" in labels else "atlascart-infra" if labels & {"platform", "security"} else "atlascart-api"
        created = item["started_at"] + timedelta(hours=5)
        if item["key"] == "ATC-68":
            last = dt(-19, 11, 40)
        else:
            last = (item["done_at"] - timedelta(minutes=20)) if item["done_at"] else min(SNAPSHOT_AT - timedelta(minutes=30), created + timedelta(days=2))
        assignee = m1.USER_BY_ID[item["assignee"]]
        links.append({
            "issue": item["key"], "creator": item["assignee"], "repo": repo,
            "title": f"{item['key']}: {item['title']}",
            "state": "MERGED" if item["status"] == "done" else "OPEN",
            "author_login": assignee.get("github") or assignee["username"],
            "created_at": created, "last_activity_at": last,
        })
    # ATC-68 was reopened after the first PR passed only the direct-email
    # checks. The resent-email path was fixed and reviewed separately.
    ethan = m1.USER_BY_ID["U06"]
    links.append({
        "issue": "ATC-68", "creator": "U06", "repo": "atlascart-api",
        "title": "ATC-68: Fix links in resent tracking emails",
        "state": "MERGED", "author_login": ethan.get("github") or ethan["username"],
        "created_at": dt(-18, 11, 30), "last_activity_at": dt(-17, 14, 30),
    })

    previous_max = {
        repo: max(
            (int(link.ref_id) for link in m1.CODE_LINKS if link.kind == "pull_request" and link.repo == repo),
            default=100,
        )
        for repo in {link["repo"] for link in links}
    }
    for repo in sorted(previous_max):
        repo_links = sorted((link for link in links if link["repo"] == repo), key=lambda link: (link["created_at"], link["issue"], link["title"]))
        for number, link in enumerate(repo_links, previous_max[repo] + 1):
            link["ref_id"] = str(number)
            link["url"] = f"https://github.com/northstar-labs-sim/{repo}/pull/{number}"
    return sorted(links, key=lambda link: (link["created_at"], link["repo"], int(link["ref_id"])))


def extension_sprint_history_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sprint_by_id = {sprint["id"]: sprint for sprint in SPRINTS}
    for item in ISSUES:
        if not item["sprint"]:
            continue
        carry = M3_CARRYOVERS.get(item["key"])
        sprint_ids = [carry["from"], carry["to"]] if carry else [item["sprint"]]
        for position, sprint_id in enumerate(sprint_ids):
            sprint = sprint_by_id[sprint_id]
            start = date.fromisoformat(sprint["start_date"])
            is_origin = carry is not None and position == 0
            initial = item["created_at"].astimezone(ZoneInfo("America/Los_Angeles")).date() <= start if is_origin or not carry else True
            completed_sprint = sprint["status"] == "completed"
            outcome = "carried_over" if is_origin else ("completed" if completed_sprint and item["status"] == "done" else None)
            status_at_end = "in_progress" if is_origin else ("done" if outcome == "completed" else None)
            points_start = item["initial_points"] if initial and is_origin else (item["points"] if initial else None)
            points_end = item["points"] if completed_sprint else None
            added_at = datetime.combine(start, datetime.min.time().replace(hour=10, minute=30), tzinfo=ZoneInfo("America/Los_Angeles")) if initial else item["created_at"]
            rows.append({
                "sprint": sprint_id, "issue": item["key"], "type": item["type"],
                "initial_scope": initial, "points_at_start": points_start,
                "points_at_end": points_end, "status_at_end": status_at_end,
                "outcome": outcome, "added_at": added_at, "removed_at": None,
            })
    return rows


def sprint_metrics(sprint_id: str) -> tuple[int, int, int, int, int, int, int, int]:
    rows = [row for row in extension_sprint_history_rows() if row["sprint"] == sprint_id]
    estimatable = [row for row in rows if row["type"] in {"story", "task", "bug"}]
    initial = [row for row in rows if row["initial_scope"]]
    completed = [row for row in rows if row["outcome"] == "completed"]
    return (
        sum(row["points_at_start"] or 0 for row in estimatable if row["initial_scope"]),
        sum(row["points_at_start"] or 0 for row in estimatable if row["initial_scope"] and row["outcome"] == "completed"),
        sum(row["points_at_end"] or 0 for row in estimatable),
        sum(row["points_at_end"] or 0 for row in estimatable if row["outcome"] == "completed"),
        len(initial), len(completed), len(rows),
        sum(row["points_at_end"] is None for row in estimatable),
    )


def _attachment_sql(attachment: dict[str, Any]) -> list[str]:
    storage_name = attachment_storage_name(attachment)
    size = len(m1.attachment_bytes(attachment))
    list_in_panel = attachment["placement"] == "panel"
    lines = [
        "INSERT INTO issue_attachments (issue_id, uploader_id, original_filename, storage_filename, content_type, size_bytes, list_in_attachment_panel, created_at) VALUES "
        f"({issue_expr(attachment['issue'])}, {user_expr(attachment['uploader'])}, {q(attachment['filename'])}, {q(storage_name)}, {q(attachment['content_type'])}, {size}, {q(list_in_panel)}, {q(attachment['created_at'])});"
    ]
    safe_name = html_escape(attachment["filename"], quote=True).replace("'", "''")
    if attachment["content_type"].startswith("image/"):
        embed = "'<img data-attachment-id=\"' || a.id || '\" data-attachment-name=\"" + safe_name + "\" alt=\"" + safe_name + "\" class=\"ticket-rich-image\" />'"
    else:
        embed = "'<span data-attachment-id=\"' || a.id || '\" data-attachment-name=\"" + safe_name + "\">[file] " + safe_name + "</span>'"
    purpose = html_escape(attachment["purpose"], quote=True).replace("'", "''")
    if attachment["placement"] == "description":
        lines.append(
            "UPDATE issues i SET description = i.description || E'\\n\\n<p><strong>Attached reference:</strong> " + purpose + " ' || "
            f"{embed} || '</p>' FROM issue_attachments a WHERE a.issue_id = i.id AND i.issue_key = {q(attachment['issue'])} AND a.storage_filename = {q(storage_name)};"
        )
    elif attachment["placement"] == "comment" and attachment.get("comment_mode") != "append":
        placeholder = f"[[ATTACHMENT:{attachment['filename']}]]"
        lines.append(
            f"UPDATE comments c SET content = REPLACE(c.content, {q(placeholder)}, {embed}) FROM issue_attachments a, issues i "
            f"WHERE c.issue_id = i.id AND a.issue_id = i.id AND i.issue_key = {q(attachment['issue'])} AND a.storage_filename = {q(storage_name)} AND c.content LIKE {q('%' + placeholder + '%')};"
        )
    elif attachment["placement"] == "comment":
        lines.append(
            "UPDATE comments c SET content = c.content || E'\\n\\n<p><strong>Attached file:</strong> " + purpose + " ' || "
            f"{embed} || '</p>' FROM issue_attachments a, issues i WHERE c.issue_id = i.id AND a.issue_id = i.id "
            f"AND i.issue_key = {q(attachment['issue'])} AND a.storage_filename = {q(storage_name)} AND c.id = "
            "(SELECT c2.id FROM comments c2 WHERE c2.issue_id = i.id ORDER BY c2.created_at DESC, c2.id DESC LIMIT 1);"
        )
    return lines


def extension_sql() -> str:
    lines: list[str] = [
        "-- Promote the approved M1 prefix into the cumulative M3 snapshot.",
        f"UPDATE spaces SET issue_counter = {TOTAL_ISSUES}, updated_at = {q(SNAPSHOT_AT)} WHERE id = {space_expr()};",
        f"UPDATE sprints SET status = 'completed', updated_at = {q(dt(-51, 16, 30))} WHERE space_id = {space_expr()} AND name = 'AtlasCart Sprint 3';",
    ]
    for key, continuation in M1_CONTINUATIONS.items():
        original = m1.ISSUE_BY_KEY[key]
        effective_start = continuation.get("started_at") or original["started_at"]
        if effective_start is None:
            raise ValueError(f"Done continuation {key} has no start timestamp")
        lines.extend([
            f"UPDATE issues SET status = 'done', start_date = {q(effective_start.date().isoformat())}, updated_at = {q(continuation['at'])} WHERE issue_key = {q(key)} AND space_id = {space_expr()};",
            "INSERT INTO comments (issue_id, author_id, content, created_at, updated_at) VALUES "
            f"({issue_expr(key)}, {user_expr(continuation['author'])}, {q('<p><strong>Sprint 3 completion</strong></p><p>' + html_escape(continuation['comment']) + '</p>')}, {q(continuation['at'])}, {q(continuation['at'])});",
            "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES "
            f"({issue_expr(key)}, {user_expr(continuation['author'])}, 'field_change', 'status', 'in_progress', 'done', 'Updated status after Sprint 3 verification', {q(continuation['at'] - timedelta(minutes=1))});",
            "INSERT INTO issue_history (issue_id, actor_id, event_type, description, created_at) VALUES "
            f"({issue_expr(key)}, {user_expr(continuation['author'])}, 'comment_created', 'added a Sprint 3 completion comment', {q(continuation['at'])});",
        ])
        if original["status"] == "planned":
            lines.append(
                "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES "
                f"({issue_expr(key)}, {user_expr(original['assignee'])}, 'field_change', 'status', 'planned', 'in_progress', 'Started work during Sprint 3', {q(effective_start)});"
            )
    lines.extend([
        "UPDATE sprint_issue_history h SET points_at_end = i.story_points, status_at_end = 'done', outcome = 'completed' FROM issues i, sprints sp WHERE h.issue_id = i.id AND h.sprint_id = sp.id AND sp.space_id = " + space_expr() + " AND sp.name = 'AtlasCart Sprint 3';",
        "UPDATE sprints sp SET initial_committed_points = x.initial_points, initial_completed_points = x.initial_completed, final_scope_points = x.final_points, completed_points = x.completed_points, initial_issue_count = x.initial_count, completed_issue_count = x.completed_count, final_issue_count = x.final_count, unestimated_issue_count = x.unestimated FROM (SELECT h.sprint_id, COALESCE(SUM(h.points_at_start),0) initial_points, COALESCE(SUM(h.points_at_start) FILTER (WHERE h.status_at_end='done'),0) initial_completed, COALESCE(SUM(h.points_at_end),0) final_points, COALESCE(SUM(h.points_at_end) FILTER (WHERE h.status_at_end='done'),0) completed_points, COUNT(*) FILTER (WHERE h.initial_scope) initial_count, COUNT(*) FILTER (WHERE h.status_at_end='done') completed_count, COUNT(*) final_count, COUNT(*) FILTER (WHERE h.points_at_end IS NULL AND h.issue_type NOT IN ('epic','subtask')) unestimated FROM sprint_issue_history h GROUP BY h.sprint_id) x WHERE sp.id=x.sprint_id AND sp.space_id=" + space_expr() + " AND sp.name='AtlasCart Sprint 3';",
    ])
    for sprint in SPRINTS:
        metrics = sprint_metrics(sprint["id"]) if sprint["status"] == "completed" else (None,) * 8
        values = ", ".join("NULL" if value is None else str(value) for value in metrics)
        lines.append(
            "INSERT INTO sprints (space_id, name, goal, start_date, end_date, status, sprint_order, initial_committed_points, initial_completed_points, final_scope_points, completed_points, initial_issue_count, completed_issue_count, final_issue_count, unestimated_issue_count, created_at, updated_at) VALUES "
            f"({space_expr()}, {q(sprint['name'])}, {q(sprint['goal'])}, {q(sprint['start_date'])}, {q(sprint['end_date'])}, {q(sprint['status'])}, {sprint['order']}, {values}, {q(sprint['created_at'])}, {q(sprint['updated_at'])});"
        )
    for label in NEW_LABELS:
        lines.append(f"INSERT INTO labels (space_id, name, normalized_name, created_at, deleted_at) VALUES ({space_expr()}, {q(label)}, {q(label)}, {q(dt(-52, 11, 5))}, NULL) ON CONFLICT DO NOTHING;")
    updated = updated_at_by_issue()
    for index, item in enumerate(ISSUES, len(m1.ISSUES)):
        due = next((sprint["end_date"] for sprint in SPRINTS if sprint["id"] == item["sprint"]), None)
        start_date = item["started_at"].date().isoformat() if item["started_at"] else None
        lines.append(
            "INSERT INTO issues (issue_key, space_id, sprint_id, parent_id, title, description, issue_type, status, priority, assignee_id, reporter_id, story_points, start_date, due_date, issue_order, created_at, updated_at) VALUES "
            f"({q(item['key'])}, {space_expr()}, {sprint_expr(item['sprint'])}, {issue_expr(item['parent'])}, {q(item['title'])}, {q(description_html(item))}, {q(item['type'])}, {q(item['status'])}, {q(item['priority'])}, {user_expr(item['assignee'])}, {user_expr(item['reporter'])}, {q(item['points'])}, {q(start_date)}, {q(due)}, {index * 10}, {q(item['created_at'])}, {q(updated[item['key']])});"
        )
        for label in item["labels"]:
            lines.append(f"INSERT INTO issue_labels (issue_id, label_id) VALUES ({issue_expr(item['key'])}, {label_expr(label)});")
        lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, description, created_at) VALUES " f"({issue_expr(item['key'])}, {user_expr(item['reporter'])}, 'issue_created', 'created the issue with acceptance criteria and estimate reasoning', {q(item['created_at'])});")
        if item["started_at"] and item["key"] not in M3_STATUS_OVERRIDES:
            lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(item['key'])}, {user_expr(item['assignee'])}, 'field_change', 'status', 'planned', 'in_progress', 'Started implementation', {q(item['started_at'])});")
        if item["done_at"] and item["key"] not in M3_STATUS_OVERRIDES:
            lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(item['key'])}, {user_expr(verifier_for(item))}, 'field_change', 'status', 'in_progress', 'done', 'Closed after AC-by-AC verification', {q(item['done_at'])});")
    for key, events in M3_STATUS_OVERRIDES.items():
        for at, old, new in events:
            actor = "U10" if old == "done" and new == "in_progress" else (verifier_for(ISSUE_BY_KEY[key]) if new == "done" else ISSUE_BY_KEY[key]["assignee"])
            description = "Daniel reopened the issue after the resent-email check failed" if old == "done" and new == "in_progress" else "Updated status during implementation and verification"
            lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(key)}, {user_expr(actor)}, 'field_change', 'status', {q(old)}, {q(new)}, {q(description)}, {q(at)});")
    for key, change in M3_ESTIMATE_CHANGES.items():
        lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(key)}, {user_expr('U03')}, 'field_change', 'storyPoints', {q(str(change['from']))}, {q(str(change['to']))}, {q('Updated storyPoints — ' + change['reason'])}, {q(change['at'])});")
    for key, change in M3_ASSIGNEE_CHANGES.items():
        lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(key)}, {user_expr('U03')}, 'field_change', 'assignee', {q(m1.USER_BY_ID[change['from']]['name'])}, {q(m1.USER_BY_ID[change['to']]['name'])}, {q('Updated assignee — ' + change['reason'])}, {q(change['at'])});")
    for key, carry in M3_CARRYOVERS.items():
        old_name = next(sprint["name"] for sprint in SPRINTS if sprint["id"] == carry["from"])
        new_name = next(sprint["name"] for sprint in SPRINTS if sprint["id"] == carry["to"])
        lines.append("INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(key)}, {user_expr('U03')}, 'field_change', 'sprint', {q(old_name)}, {q(new_name)}, {q('Updated sprint — ' + carry['reason'])}, {q(carry['at'])});")
    for comment in M3_COMMENTS:
        if comment["issue"] in M1_CONTINUATIONS:
            continue
        lines.extend([
            "INSERT INTO comments (issue_id, author_id, content, created_at, updated_at) VALUES " f"({issue_expr(comment['issue'])}, {user_expr(comment['author'])}, {q(comment['content'])}, {q(comment['created_at'])}, {q(comment['created_at'])});",
            "INSERT INTO issue_history (issue_id, actor_id, event_type, description, created_at) VALUES " f"({issue_expr(comment['issue'])}, {user_expr(comment['author'])}, 'comment_created', 'added a structured project update', {q(comment['created_at'])});",
        ])
    for source, target, kind, at in M3_LINKS:
        actor = ISSUE_BY_KEY[source]["assignee"] or ISSUE_BY_KEY[source]["reporter"]
        lines.extend([
            f"INSERT INTO issue_links (source_issue_id, target_issue_id, link_type, created_at) VALUES ({issue_expr(source)}, {issue_expr(target)}, {q(kind)}, {q(at)});",
            "INSERT INTO issue_history (issue_id, actor_id, event_type, field_name, from_value, to_value, description, created_at) VALUES " f"({issue_expr(source)}, {user_expr(actor)}, 'link_created', 'Link', 'None', {q(kind + ' ' + target)}, 'Created a reasoned issue relationship', {q(at)});",
        ])
    for link in extension_code_links():
        lines.extend([
            "INSERT INTO issue_code_links (issue_id, creator_id, url, kind, provider, owner, repo, ref_id, title, state, author_login, created_at, last_activity_at) VALUES " f"({issue_expr(link['issue'])}, {user_expr(link['creator'])}, {q(link['url'])}, 'pull_request', 'github', 'northstar-labs-sim', {q(link['repo'])}, {q(link['ref_id'])}, {q(link['title'])}, {q(link['state'])}, {q(link['author_login'])}, {q(link['created_at'])}, {q(link['last_activity_at'])});",
            "INSERT INTO issue_history (issue_id, actor_id, event_type, description, created_at) VALUES " f"({issue_expr(link['issue'])}, {user_expr(link['creator'])}, 'code_link_added', {q('linked pull request ' + link['ref_id'])}, {q(link['created_at'])});",
        ])
    for attachment in M3_ATTACHMENTS:
        lines.extend(_attachment_sql(attachment))
    for row in extension_sprint_history_rows():
        lines.append(
            "INSERT INTO sprint_issue_history (sprint_id, issue_id, issue_key, issue_type, initial_scope, points_at_start, points_at_end, status_at_end, outcome, added_at, removed_at) VALUES "
            f"({sprint_expr(row['sprint'])}, {issue_expr(row['issue'])}, {q(row['issue'])}, {q(row['type'])}, {q(row['initial_scope'])}, {q(row['points_at_start'])}, {q(row['points_at_end'])}, {q(row['status_at_end'])}, {q(row['outcome'])}, {q(row['added_at'])}, {q(row['removed_at'])});"
        )
    lines.extend([
        "DO $$ DECLARE issue_count INTEGER; sprint_count INTEGER; active_count INTEGER; BEGIN",
        f"SELECT COUNT(*) INTO issue_count FROM issues WHERE space_id={space_expr()};",
        f"SELECT COUNT(*) INTO sprint_count FROM sprints WHERE space_id={space_expr()};",
        f"SELECT COUNT(*) INTO active_count FROM sprints WHERE space_id={space_expr()} AND status='active';",
        f"IF issue_count<>{TOTAL_ISSUES} OR sprint_count<>7 OR active_count<>1 THEN RAISE EXCEPTION 'AtlasCart M3 count mismatch: issues %, sprints %, active %', issue_count, sprint_count, active_count; END IF;",
        "END $$;",
    ])
    return "\n".join(lines)


def build_sql(*, commit: bool) -> str:
    validate()
    base = _with_m3_fixture(m1.build_sql, commit=True)
    return "\n".join(["\\set ON_ERROR_STOP on", "BEGIN;", m1.transaction_body(base), extension_sql(), "COMMIT;" if commit else "ROLLBACK;", ""])


def build_cleanup_sql(*, commit: bool) -> str:
    return _with_m3_fixture(m1.build_cleanup_sql, commit=commit)


def lifecycle_check_sql() -> str:
    return "\n".join(["\\set ON_ERROR_STOP on", "BEGIN;", m1.transaction_body(build_sql(commit=True)), "DROP TABLE IF EXISTS _atlascart_issues;", "DROP TABLE IF EXISTS _atlascart_sprints;", "DROP TABLE IF EXISTS _atlascart_spaces;", m1.transaction_body(build_cleanup_sql(commit=True)), f"DO $$ BEGIN IF EXISTS (SELECT 1 FROM spaces WHERE space_key='ATC' AND name={q(FIXTURE_NAME)}) THEN RAISE EXCEPTION 'M3 cleanup failed'; END IF; END $$;", "ROLLBACK;", ""])


def status_sql() -> str:
    return "\n".join(["\\set ON_ERROR_STOP on", "SELECT CASE", f"WHEN EXISTS (SELECT 1 FROM spaces WHERE space_key='ATC' AND name={q(FIXTURE_NAME)} AND deleted_at IS NULL) THEN 'installed'", "WHEN EXISTS (SELECT 1 FROM spaces WHERE space_key='ATC' AND deleted_at IS NULL) THEN 'key-used-by-another-space'", "ELSE 'not-installed' END AS atlascart_m3_status;", f"SELECT COUNT(DISTINCT i.id) issues, COUNT(DISTINCT c.id) comments, COUNT(DISTINCT a.id) attachments FROM spaces s LEFT JOIN issues i ON i.space_id=s.id LEFT JOIN comments c ON c.issue_id=i.id LEFT JOIN issue_attachments a ON a.issue_id=i.id WHERE s.space_key='ATC' AND s.name={q(FIXTURE_NAME)};", ""])


def preflight_sql() -> str:
    original = m1.FIXTURE_NAME
    try:
        m1.FIXTURE_NAME = FIXTURE_NAME
        return m1.preflight_sql()
    finally:
        m1.FIXTURE_NAME = original


def attachment_runtime_rows() -> list[dict[str, Any]]:
    import subprocess
    sql = (
        "SELECT a.id, i.id, i.issue_key, s.id, a.original_filename, COALESCE(a.content_type,''), a.size_bytes, a.storage_filename "
        "FROM issue_attachments a JOIN issues i ON i.id=a.issue_id JOIN spaces s ON s.id=i.space_id "
        f"WHERE s.space_key='ATC' AND s.name={q(FIXTURE_NAME)} ORDER BY a.id"
    )
    result = subprocess.run(["docker", "exec", "orchestra-postgres", "psql", "-qAt", "-F", "\t", "-U", "poc", "-d", "pocdb", "-c", sql], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Could not read M3 attachment metadata: {result.stderr.strip()}")
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 8:
            raise SystemExit(f"Unexpected M3 attachment metadata row: {line}")
        rows.append({"attachmentId": int(fields[0]), "issueId": int(fields[1]), "issueKey": fields[2], "spaceId": int(fields[3]), "filename": fields[4], "mimeType": fields[5], "byteSize": int(fields[6]), "storageKey": fields[7]})
    return rows


def upload_attachments() -> None:
    m1.docker_run(["docker", "exec", "orchestra-minio", "mc", "alias", "set", "atlascart-local", "http://127.0.0.1:9000", "minioadmin", "minioadmin"])
    m1.docker_run(["docker", "exec", "orchestra-minio", "mc", "mb", "--ignore-existing", "atlascart-local/jira-attachments"])
    for attachment in all_attachments():
        key = attachment_storage_name(attachment)
        m1.docker_run(["docker", "exec", "-i", "orchestra-minio", "mc", "pipe", f"atlascart-local/jira-attachments/{key}"], input_bytes=m1.attachment_bytes(attachment))
        m1.docker_run(["docker", "exec", "orchestra-minio", "mc", "stat", f"atlascart-local/jira-attachments/{key}"])


def remove_attachments() -> None:
    import subprocess
    alias_result = subprocess.run(["docker", "exec", "orchestra-minio", "mc", "alias", "set", "atlascart-local", "http://127.0.0.1:9000", "minioadmin", "minioadmin"], capture_output=True, text=True)
    if alias_result.returncode != 0:
        raise SystemExit(f"Could not configure local MinIO: {alias_result.stderr.strip()}")
    for prefix in ("atlascart-m1-", "atlascart-m3-"):
        result = subprocess.run(["docker", "exec", "orchestra-minio", "mc", "rm", "--recursive", "--force", f"atlascart-local/jira-attachments/{prefix}"], capture_output=True, text=True)
        combined = (result.stderr + result.stdout).lower()
        if result.returncode != 0 and "does not exist" not in combined and "not found" not in combined:
            raise SystemExit(f"M3 MinIO objects could not be removed: {(result.stderr or result.stdout).strip()}")


def write_outputs() -> list[str]:
    checks = validate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ATTACHMENT_DIR.mkdir(parents=True, exist_ok=True)
    INJECT_SQL_PATH.write_text(build_sql(commit=True), encoding="utf-8")
    CLEANUP_SQL_PATH.write_text(build_cleanup_sql(commit=True), encoding="utf-8")
    expected = {attachment_storage_name(row) for row in all_attachments()}
    for existing in ATTACHMENT_DIR.iterdir():
        if existing.is_file() and existing.name not in expected:
            existing.unlink()
    for attachment in all_attachments():
        (ATTACHMENT_DIR / attachment_storage_name(attachment)).write_bytes(m1.attachment_bytes(attachment))
    linked = {key for source, target, _kind, _at in list(m1.ISSUE_LINKS) + M3_LINKS for key in (source, target)}
    covered = {row["issue"] for row in all_attachments()}
    manifest = {
        "dataset_id": DATASET_ID, "dataset_version": DATASET_VERSION,
        "space_key": "ATC", "space_name": FIXTURE_NAME,
        "snapshot_at": SNAPSHOT_AT.isoformat(), "as_of_date": M3_AS_OF.isoformat(),
        "historical_prefix": {"dataset": "atlascart-m1", "cutoff": m1.AS_OF_DATE.isoformat(), "shifted_days_before_m3": 56},
        "logical_m2_checkpoint": {"through_sprint": "AtlasCart Sprint 5", "cutoff": next(sprint["end_date"] for sprint in SPRINTS if sprint["id"] == "S5"), "separate_fixture": False},
        "active_sprint": "AtlasCart Sprint 7", "closed_sprints": 6,
        "requested_attachment_ratio": ATTACHMENT_RATIO, "attachment_covered_issue_count": len(covered), "actual_attachment_ratio": round(len(covered) / TOTAL_ISSUES * 100, 2),
        "requested_link_ratio": LINK_RATIO, "link_covered_issue_count": len(linked), "actual_link_ratio": round(len(linked) / TOTAL_ISSUES * 100, 2),
        "generation_rules": {
            "ac_distribution": {"one_ac": 23, "two_ac": 9, "three_ac": 3},
            "three_ac_issue_keys": sorted(THREE_AC_KEYS),
            "minimum_attachment_ratio": 25,
            "frontend_evidence": "current PNG required; Done work requires after PNG; representative work includes regression evidence",
            "bug_evidence": "EXPECTED, ACTUAL, and ERROR diagnostic required; frontend bugs additionally require current PNG",
            "human_time": "weekdays only; actor-local Pacific, Mountain, Central, or Eastern 07:00-20:59",
            "imperfection_budget": {"carryovers": 3, "estimate_changes": 3, "assignee_changes": 3, "reopened_issues": 1, "subtasks": 3},
            "verification": "domain-based assignment across five qualified verifiers; no verifier may close half of Done extension work",
        },
        "login_password": "123", "recommended_review_user": "atlascart.liam",
        "counts": {"users": len(m1.USERS), "sprints": 7, "issues": TOTAL_ISSUES, "comments": len(m1.COMMENTS) + len(M3_COMMENTS), "history_minimum": len(m1.HISTORIES) + len(M3_COMMENTS), "issue_links": len(m1.ISSUE_LINKS) + len(M3_LINKS), "code_links": len(m1.CODE_LINKS) + len(extension_code_links()), "attachments": len(all_attachments())},
        "attachments": [{"issue_key": row["issue"], "filename": row["filename"], "content_type": row["content_type"], "byte_size": len(m1.attachment_bytes(row)), "placement": row["placement"], "purpose": row["purpose"], "local_path": f"attachments/{attachment_storage_name(row)}", "minio_bucket": "jira-attachments", "storage_key": attachment_storage_name(row)} for row in all_attachments()],
        "review_issue_keys": ["ATC-53", "ATC-56", "ATC-59", "ATC-63", "ATC-66", "ATC-68", "ATC-73", "ATC-76", "ATC-78", "ATC-80"],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    VALIDATION_PATH.write_text("\n".join(["# AtlasCart M3 Validation Report", "", f"Dataset: `{DATASET_ID}` version `{DATASET_VERSION}`", "", *[f"- PASS — {check}" for check in checks], ""]), encoding="utf-8")
    REVIEW_PATH.write_text(f"""# AtlasCart M3 Jira Review Path

Dataset as-of date: `{M3_AS_OF.isoformat()}`

Login as `atlascart.liam` with password `123`, then open `ATC — {FIXTURE_NAME}`.

## 1. Confirm the cumulative timeline

The project has 83 issues, six completed sprints, and active Sprint 7. Sprints 1–3 preserve the approved M1 history; M3 continues the same team and product instead of replacing that story.

## 2. Review Sprint 4 — customer identity

Follow ATC-53, ATC-56, and bug ATC-59. Check that the gateway contract enables sign-in and cart merge, and that the retry bug has a concrete example, evidence, and AC-by-AC close result.

## 3. Review Sprint 5 — order operations

Follow ATC-63 to ATC-68. The order state decision constrains cancellation and refund work. Tracking-email bug ATC-68 should read like a specific beta incident rather than generic filler.

## 4. Review Sprint 6 — seller catalog

Follow ATC-70, ATC-73, ATC-74, and ATC-76. Confirm permission, background import, progress UI, search freshness, CSV evidence, comments, and pull requests tell one causal story.

## 5. Review active Sprint 7

ATC-77, ATC-78, and ATC-80 are in progress. ATC-79 remains To Do and is blocked by real readiness evidence, so the current sprint deliberately does not look artificially complete.
""", encoding="utf-8")
    return checks


def print_summary(action: str) -> None:
    types = Counter(item["type"] for item in list(m1.ISSUES) + ISSUES)
    attachments = all_attachments()
    covered = {row["issue"] for row in attachments}
    linked = {key for source, target, _kind, _at in list(m1.ISSUE_LINKS) + M3_LINKS for key in (source, target)}
    print(f"AtlasCart cumulative M3 {action} succeeded.")
    print(f"Space: ATC — {FIXTURE_NAME}")
    print(f"Issues: {TOTAL_ISSUES} {dict(types)} | Sprints: 6 completed + 1 active")
    print(f"Comments: {len(m1.COMMENTS) + len(M3_COMMENTS)} | Code links: {len(m1.CODE_LINKS) + len(extension_code_links())} | Attachments: {len(attachments)}")
    print(f"Attachment coverage: {len(covered)}/{TOTAL_ISSUES} ({len(covered)/TOTAL_ISSUES*100:.2f}%), requested {ATTACHMENT_RATIO:g}%")
    print(f"Issue-link coverage: {len(linked)}/{TOTAL_ISSUES} ({len(linked)/TOTAL_ISSUES*100:.2f}%), requested {LINK_RATIO:g}%")
    print("Review login: atlascart.liam / 123")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and install the cumulative AtlasCart M3 Jira dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_profiles(target: argparse.ArgumentParser) -> None:
        target.add_argument("--attachment-ratio", type=float, default=DEFAULT_ATTACHMENT_RATIO, metavar="PERCENT", help="Target issue attachment coverage (25-100, default 30; mandatory evidence floor applies)")
        max_link_ratio = (len({key for s, t, _k, _a in m1.CANONICAL_ISSUE_LINKS for key in (s, t)}) + len({key for s, t, _k, _a in CANONICAL_LINKS for key in (s, t)})) / TOTAL_ISSUES * 100
        target.add_argument("--link-ratio", type=float, default=DEFAULT_LINK_RATIO, metavar="PERCENT", help=f"Target semantic issue-link coverage (0-{max_link_ratio:.2f}, default 20)")

    for name, help_text in (("generate", "Generate local SQL, files, manifest, and reports"), ("validate", "Run deterministic in-memory checks"), ("check-db", "Exercise inject and cleanup in one rolled-back transaction")):
        command = subparsers.add_parser(name, help=help_text)
        add_profiles(command)
    subparsers.add_parser("status", help="Read M3 status without changing data")
    inject = subparsers.add_parser("inject", help="Inject or refresh only the M3 fixture")
    add_profiles(inject)
    inject.add_argument("--yes", action="store_true")
    inject.add_argument("--skip-attachments", action="store_true")
    inject.add_argument("--skip-kafka", action="store_true")
    inject.add_argument("--kafka-topic", default="jira.attachment.ingestion")
    cleanup = subparsers.add_parser("cleanup", help="Remove M3 DB rows, objects, and Kafka topic")
    cleanup.add_argument("--yes", action="store_true")
    cleanup.add_argument("--skip-attachments", action="store_true")
    cleanup.add_argument("--skip-kafka", action="store_true")
    kafka = cleanup.add_mutually_exclusive_group()
    kafka.add_argument("--delete-kafka-topic", action="store_true", help="Explicitly select the default topic deletion behavior")
    kafka.add_argument("--keep-kafka-topic", action="store_true", help="Keep topic and publish deletion events")
    cleanup.add_argument("--kafka-topic", default="jira.attachment.ingestion")
    args = parser.parse_args()
    if hasattr(args, "attachment_ratio"):
        try:
            configure(args.attachment_ratio, args.link_ratio)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == "validate":
        for check in validate():
            print(f"PASS: {check}")
        print_summary("validation")
        return
    if args.command == "status":
        m1.check_containers(require_minio=False)
        m1.run_sql(status_sql())
        return
    write_outputs()
    if args.command == "generate":
        print_summary("generation")
        print(f"Injection SQL: {INJECT_SQL_PATH}\nCleanup SQL: {CLEANUP_SQL_PATH}\nReview guide: {REVIEW_PATH}")
        return
    if args.command == "check-db":
        m1.check_containers(require_minio=False)
        m1.run_sql(lifecycle_check_sql())
        print_summary("database check")
        print("The complete lifecycle passed inside a rolled-back transaction; no database data changed.")
        return
    if not args.yes:
        raise SystemExit(f"No data was changed. Re-run with `{args.command} --yes` to confirm.")
    if args.command == "inject":
        use_kafka = not args.skip_attachments and not args.skip_kafka
        m1.check_containers(require_minio=not args.skip_attachments, require_kafka=use_kafka)
        m1.run_sql(preflight_sql())
        old_rows = attachment_runtime_rows() if use_kafka else []
        if not args.skip_attachments:
            remove_attachments()
            upload_attachments()
        m1.run_sql(build_sql(commit=True))
        if use_kafka:
            deleted = m1.publish_attachment_tracking(old_rows, "attachment_deleted", args.kafka_topic)
            uploaded = m1.publish_attachment_tracking(attachment_runtime_rows(), "attachment_uploaded", args.kafka_topic)
            print(f"Kafka: published {deleted} replacement deletions and {uploaded} uploads to {args.kafka_topic}.")
        print_summary("injection")
        return
    if args.delete_kafka_topic and args.skip_kafka:
        parser.error("--delete-kafka-topic and --skip-kafka cannot be used together")
    delete_topic = not args.skip_kafka and not args.keep_kafka_topic
    publish_deleted = not args.skip_attachments and not args.skip_kafka and args.keep_kafka_topic
    m1.check_containers(require_minio=not args.skip_attachments, require_kafka=delete_topic or publish_deleted)
    rows = attachment_runtime_rows() if publish_deleted else []
    m1.run_sql(build_cleanup_sql(commit=True))
    if not args.skip_attachments:
        remove_attachments()
    if delete_topic:
        m1.delete_kafka_topic(args.kafka_topic)
        print(f"Kafka: deleted the entire topic {args.kafka_topic}.")
    elif publish_deleted:
        count = m1.publish_attachment_tracking(rows, "attachment_deleted", args.kafka_topic)
        print(f"Kafka: published {count} attachment_deleted events to {args.kafka_topic}.")
    print("AtlasCart M3 cleanup succeeded; unrelated spaces and referenced shared users were preserved.")


if __name__ == "__main__":
    main()
