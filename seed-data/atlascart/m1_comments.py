"""Human-written comment fixtures for the canonical AtlasCart M1 story."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

from m1_scenario import (
    ISSUE_BY_KEY,
    ISSUE_LINKS,
    LINK_NARRATIVES,
    STATUS_OVERRIDES,
    USER_BY_ID,
    ts,
)
from m1_visual_evidence import FRONTEND_VISUAL_EVIDENCE, verification_filenames


def c(issue: str, author: str, at, content: str, *, verification: bool = False) -> dict:
    local_at = at.astimezone(ZoneInfo(USER_BY_ID[author]["timezone"]))
    return {"issue": issue, "author": author, "created_at": local_at, "content": content, "verification": verification}


COMMENTS = [
    c("ATC-1", "U02", ts(2, 9, 12), "Month 1 only needs the path from the catalog to checkout. Wish lists and recommendations stay in the backlog."),
    c("ATC-2", "U05", ts(2, 9, 18), "I will keep the first import strict. If a row is unclear, we should show the seller what to fix instead of guessing."),
    c("ATC-3", "U01", ts(2, 9, 22), "Guest checkout is fine for the beta. We should not hold the first order behind account creation."),
    c("ATC-4", "U12", ts(2, 9, 27), "For the first beta I need the order reference, status, total, and time. I do not need a full customer profile page."),
    c("ATC-5", "U03", ts(2, 9, 32), "Let us keep one API deployment for now. We can split services after we have a measured reason."),
    c("ATC-6", "U10", ts(2, 9, 37), "I will test the main path in Chrome, Firefox, Safari, and one narrow-phone size. We can add a wider browser matrix later."),

    c("ATC-7", "U07", ts(2, 13, 20), "Routing and the shared page shell are in place. I left the header simple so catalog work can move without waiting for final branding."),
    c("ATC-7", "U09", ts(3, 11, 5), "The plain header is fine for beta. Please keep enough room for the cart count on the right."),
    c("ATC-8", "U04", ts(3, 15, 10), "The first migration creates products and stock. Health checks now verify that the API can reach PostgreSQL."),
    c("ATC-8", "U11", ts(4, 14, 5), "Staging sees the health endpoint and fails the deploy if the database check is down."),
    c("ATC-9", "U05", ts(3, 11, 30), "Price is stored as a decimal amount in USD. I added database checks for negative price and stock values."),
    c("ATC-9", "U10", ts(6, 14, 25), "Verified valid products, zero stock, and both negative-value checks. The migration error is clear enough to diagnose."),

    c("ATC-10", "U05", ts(4, 11, 20), "The attached sample shows the two problems in the supplier file: a blank price and the same SKU with different prices."),
    c("ATC-10", "U05", ts(9, 15, 40), "Blank values are handled now. I am blocked on the duplicate rule because keeping the first or last row would silently pick a price."),
    c("ATC-10", "U03", ts(12, 10, 18), "Created ATC-22 so the duplicate-SKU product decision is explicit before importer validation can finish."),
    c("ATC-10", "U02", ts(12, 11, 30), "Reject conflicting duplicates and report both line numbers. That is easier for a seller to understand than an automatic choice."),
    c("ATC-10", "U03", ts(12, 11, 38), "With the duplicate report and tests, this is closer to eight points than five. Move the unfinished part with ATC-22 into Sprint 2."),
    c("ATC-10", "U05", ts(16, 13, 15), "The duplicate validation is working. I am cleaning up the error response so the UI can show both rows."),
    c("ATC-10", "U10", ts(18, 15, 20), "Verified blank prices, exact duplicate rows, and conflicting duplicate rows. No partial catalog is saved when validation fails."),

    c("ATC-11", "U05", ts(6, 13, 20), "The query is a basic case-insensitive name match. We are not doing relevance scoring in this sprint."),
    c("ATC-11", "U08", ts(9, 15, 20), "The response has everything the card needs. I do not need another endpoint for the list page."),
    c("ATC-12", "U09", ts(5, 10, 25), "Attached the list note as product-list-requirements.pdf in the panel. The description now has one before screenshot per acceptance criterion so QA can see the current gaps in card fields and list states."),
    c("ATC-12", "U08", ts(9, 16, 10), "The empty and error states are in staging. Long product names still need the narrow-screen fix in ATC-19."),
    c("ATC-12", "U10", ts(10, 15, 55), "The list works with 0, 1, and 50 fixture products. I linked the one layout bug instead of reopening this story."),
    c("ATC-13", "U05", ts(9, 13, 45), "Inactive and unknown SKUs both return 404. The response does not reveal whether a seller removed the product."),
    c("ATC-14", "U09", ts(10, 9, 40), "The description now includes one before screenshot per AC: the raw detail layout for AC1 and the enabled zero-stock button for AC2. Verification will attach the matching after screenshots at close."),
    c("ATC-14", "U08", ts(12, 14, 25), "The Add to cart button is disabled at zero stock and still shows why. Ready for the catalog smoke test."),
    c("ATC-15", "U06", ts(6, 11, 25), "The API already returns an empty image value. Lucas can finish the browser fallback with the rest of the catalog UI."),
    c("ATC-15", "U08", ts(9, 15, 45), "Missing and failed image URLs now use the same placeholder. I kept the product name as real text, not part of the image."),
    c("ATC-16", "U11", ts(3, 10, 10), "Backend tests and frontend checks run in parallel. The build will not publish an image if either one fails."),
    c("ATC-16", "U03", ts(5, 13, 10), "All required checks passed on run 184, and the forced frontend failure still blocks merge. I am attaching the final run because it is the evidence used to close this task: [[ATTACHMENT:ci-pipeline-run-184.png]]"),
    c("ATC-17", "U11", ts(9, 14, 30), "The rollback command points staging at the previous image tag. It does not change the database automatically."),
    c("ATC-17", "U10", ts(11, 15, 5), "Catalog smoke test passed on the staging build from commit 7be31a2."),
    c("ATC-18", "U10", ts(9, 14, 15), "The checklist now covers bad price data and a missing image. I will run the narrow-screen pass after the list page deploys."),
    c("ATC-18", "U10", ts(12, 16, 10), "Catalog smoke test passed. ATC-10 is the only catalog item carrying into the next sprint."),
    c("ATC-19", "U10", ts(9, 10, 25), "Reproduced at 375px with the long product name 'Adjustable Bamboo Monitor Stand'. The AC1 before screenshot is attached in the description because it is part of the original bug report; the price covers the stock label."),
    c("ATC-19", "U08", ts(10, 11, 55), "The card now grows with the title and keeps price and stock on separate lines. Checked 320px and 375px."),
    c("ATC-20", "U10", ts(10, 11, 10), "I reduced the supplier file to five rows so the importer behavior is easy to reproduce. Row 18 is blank and becomes zero; the same file keeps valid zero distinct from missing price: [[ATTACHMENT:invalid-price-reproduction.csv]]"),
    c("ATC-20", "U05", ts(11, 14, 25), "Blank, negative, and non-numeric prices now reject the row with its original line number."),
    c("ATC-21", "U10", ts(11, 9, 48), "Captured the exact visual failure in Chrome 126 with cache disabled and Slow 3G. The AC1 before screenshot is attached in the description because the native icon appears only briefly before our placeholder."),
    c("ATC-21", "U08", ts(12, 11, 40), "The image is hidden until load succeeds, so the browser icon no longer flashes before the placeholder."),
    c("ATC-22", "U02", ts(12, 11, 15), "Decision: reject every conflicting duplicate and show both line numbers. Exact repeated rows are also rejected for now so the behavior stays simple."),
    c("ATC-22", "U05", ts(16, 11, 15), "I have the rule. The remaining work is in the importer and tests under ATC-10."),

    c("ATC-23", "U04", ts(16, 13, 5), "The cart response includes current price and available stock. The browser still owns the guest cart ID for Month 1."),
    c("ATC-23", "U06", ts(18, 14, 30), "I checked the contract from the web app. Update and remove return the full new cart, which keeps totals simple."),
    c("ATC-24", "U07", ts(13, 9, 5), "Created ATC-40 for the header cart count so badge updates have their own owner and QA evidence without expanding this story's scope."),
    c("ATC-24", "U07", ts(17, 11, 20), "Add to cart works from both product views. The header count stays in ATC-40 so this story can remain focused on the two entry points."),
    c("ATC-24", "U10", ts(20, 15, 0), "Verified the available and out-of-stock paths. The current page stays in place after adding an item."),
    c("ATC-25", "U07", ts(18, 15, 15), "Invalid stored cart data is cleared with one warning in the console. The shopper sees an empty cart, not an error page."),
    c("ATC-25", "U10", ts(20, 16, 20), "Refresh and browser restart both restore valid cart items. Corrupt fixture data falls back to empty as expected."),
    c("ATC-26", "U06", ts(20, 12, 45), "Quantity zero now removes the row. I am keeping the API response as the source for totals so the browser does not calculate money."),
    c("ATC-26", "U10", ts(23, 11, 10), "Update, remove, and zero quantity pass. I opened ATC-39 for the separate 320px overlap."),
    c("ATC-27", "U05", ts(16, 10, 8), "The first estimate only covered a lookup. The confirmation check needs to run with order creation so stock cannot change in between."),
    c("ATC-27", "U04", ts(23, 11, 35), "Please use the same transaction as ATC-30 for the final check. Returning the changed SKU is enough for the UI message."),
    c("ATC-27", "U05", ts(27, 12, 5), "The transaction check is in review. I still need to cover two carts buying the last unit at the same time."),
    c("ATC-28", "U08", ts(19, 13, 15), "Required fields are clear on blur and submit. State is a short text field for beta; a full address library can wait."),
    c("ATC-28", "U11", ts(23, 15, 5), "Confirmed the shipping fields do not appear in application logs or the request-error message."),
    c("ATC-29", "U04", ts(18, 15, 40), "The adapter maps accepted, rejected, and timed-out responses. Provider payloads stay inside the adapter."),
    c("ATC-29", "U04", ts(13, 9, 10), "Created ATC-41 so accepted, rejected, and timeout adapter cases have visible contract evidence without live sandbox calls in CI."),
    c("ATC-29", "U10", ts(23, 16, 5), "Accepted and rejected sandbox cards pass. Timeout coverage is tracked in ATC-41."),

    c("ATC-30", "U04", ts(16, 10, 15), "The first estimate missed price snapshots and safe retries. I am moving this to eight before implementation starts."),
    c("ATC-30", "U04", ts(24, 14, 35), "Order creation and item snapshots are in staging. This reviewed diagram records where the checkout key must be checked relative to payment, order insert, and stock change: [[ATTACHMENT:checkout-idempotency-architecture.png]]"),
    c("ATC-30", "U10", ts(25, 14, 40), "One normal checkout creates one order and reduces stock once. Marking the planned happy path verified."),
    c("ATC-30", "U03", ts(26, 10, 22), "Reopening this because the beta incident shows that one browser action can submit twice. The story is not complete until the API handles a repeated request safely."),
    c("ATC-30", "U04", ts(27, 15, 10), "The order endpoint now uses the checkout idempotency key before insert. The focused API test passes; the slow-browser regression remains under ATC-47."),
    c("ATC-31", "U07", ts(23, 15, 30), "The page reads the existing order result after refresh. It does not call Place order again."),
    c("ATC-31", "U10", ts(25, 14, 20), "Confirmed one order reference after checkout and after refresh. The slow double-click case is tracked separately now."),
    c("ATC-32", "U03", ts(16, 10, 22), "Ethan will take this because it crosses the API and the internal page. Five points includes both sides."),
    c("ATC-32", "U06", ts(24, 11, 10), "The recent-order API is ready. The page shows reference, status, total, and time without the full address."),
    c("ATC-32", "U12", ts(27, 13, 45), "The list is enough to find the duplicate example. I still need a clearer way to tell which order has the successful payment."),
    c("ATC-33", "U11", ts(19, 12, 20), "Request IDs now cross the web proxy and API logs. The error response includes the ID but no cart or address fields."),
    c("ATC-33", "U04", ts(23, 13, 20), "I used the request ID during a failed sandbox call and found the API entry without searching by shopper data."),
    c("ATC-34", "U11", ts(25, 9, 30), "Beta build beta-1.0.0 is deployed. Health checks and the catalog smoke path are green."),
    c("ATC-34", "U10", ts(25, 15, 35), "Happy-path checkout passed on the beta build. Invitations can go out to the small group."),
    c("ATC-35", "U10", ts(23, 13, 20), "Attached the current test plan. The slow-response case is included, but I have not run it against the beta build yet."),
    c("ATC-35", "U10", ts(26, 9, 45), "I can reproduce the report with Slow 3G and two clicks before the spinner is visible. I split the regression automation into ATC-47."),
    c("ATC-36", "U02", ts(26, 15, 33), "Move this to Sprint 3. The raw sandbox message is not ideal, but the incident fixes are more important today."),
    c("ATC-37", "U09", ts(19, 15, 15), "The description includes the 320px empty-cart screenshot from design review. The red box marks unused space, and Browse products begins below the initial viewport. This issue changes spacing and button position only."),
    c("ATC-37", "U03", ts(26, 15, 37), "Moving this to Sprint 3. Sophia can pick it up after the checkout incident work is stable."),
    c("ATC-38", "U12", ts(24, 10, 20), "The inconsistent reference is annoying when I compare the page with logs, but I can still find the order by time for this beta."),
    c("ATC-38", "U02", ts(26, 15, 39), "Back to the backlog. This does not block incident mitigation or beta support today."),
    c("ATC-39", "U10", ts(16, 15, 35), "Reproduced at 320px with the two-line product name and quantity 10 from the bug example. The before screenshot is attached in the description and shows the increment control occupying the same space as Remove."),
    c("ATC-39", "U03", ts(20, 10, 27), "Lucas is off today. Move this to Sprint 3 instead of interrupting Sophia's checkout work."),
    c("ATC-40", "U07", ts(18, 11, 25), "The count comes from the same cart state as the page. It disappears at zero instead of showing a badge with 0."),
    c("ATC-41", "U04", ts(20, 14, 15), "The contract test uses fixture responses for accepted, rejected, and timeout cases. No sandbox network call is made."),
    c("ATC-42", "U12", ts(18, 9, 20), "I will use only the synthetic BETA order references. No names or addresses need to be in the examples."),
    c("ATC-42", "U10", ts(20, 14, 5), "Examples are ready and the screenshot fields are redacted. Customer operations reviewed them."),

    c("ATC-43", "U12", ts(26, 9, 20), "The shopper wanted one desk lamp but the operations page showed BETA-1042 and BETA-1043. This redacted screenshot shows the two records and matching time/amount without the shopper's name or address: [[ATTACHMENT:duplicate-orders.png]]"),
    c("ATC-43", "U10", ts(26, 9, 42), "Reproduced in Chrome with Slow 3G. Click Place order twice before the spinner appears. Two order rows show up and stock drops by two."),
    c("ATC-43", "U11", ts(26, 10, 18), "Both requests reached the API and returned 201. There is no proxy retry in the request logs."),
    c("ATC-43", "U04", ts(26, 10, 35), "I found the second insert. We create the order before checking whether this checkout request was already handled."),
    c("ATC-43", "U03", ts(26, 10, 52), "Split the follow-up into ATC-44 for the API idempotency fix, ATC-45 for immediate browser mitigation, and ATC-47 for the slow-response regression exit criteria. Stock correction stays on ATC-46."),
    c("ATC-43", "U07", ts(26, 16, 25), "ATC-45 is live in beta and disables Place order as soon as submission starts. This reduces repeat clicks, but ATC-44 still owns the permanent fix."),
    c("ATC-43", "U12", ts(27, 9, 5), "I told the beta shopper that we found the duplicate record. No second sandbox charge was captured."),
    c("ATC-43", "U04", ts(27, 13, 20), "ATC-44 now returns the first order result for a repeated key. The attached duplicate-checkout-requests.log is the redacted gateway/API evidence that ruled out a proxy retry: two request IDs reached the API with retry_count=0. I am keeping the incident open until Daniel finishes ATC-47."),
    c("ATC-43", "U10", ts(27, 16, 10), "One-click and repeated-request API checks pass. The full browser regression is still in progress under ATC-47."),
    c("ATC-44", "U04", ts(26, 12, 10), "The key is created once in the browser and stored with the order. A repeated key returns the existing order before stock changes."),
    c("ATC-44", "U11", ts(27, 12, 50), "Deployed the API fix to beta as build beta-1.0.2. Request logs show the same key returning one order ID."),
    c("ATC-44", "U10", ts(27, 13, 5), "Focused API verification passed with five repeated submissions. One order and one stock reduction."),
    c("ATC-45", "U07", ts(26, 12, 20), "The button disables before the request starts and re-enables only after an error. Successful checkout moves to confirmation."),
    c("ATC-45", "U10", ts(26, 16, 15), "Verified with keyboard activation and repeated mouse clicks. I cannot send a second browser request while the first is pending."),
    c("ATC-46", "U05", ts(26, 14, 28), "I corrected the beta fixture stock manually for now. Sprint 3 needs a recorded operation instead of a direct database update."),
    c("ATC-46", "U12", ts(26, 14, 32), "Please include the order reference and reason in the correction record. Support should be able to explain who changed stock."),
    c("ATC-47", "U10", ts(27, 9, 15), "The API-level repeat test is stable. I am adding the browser case with a delayed response and two rapid clicks."),
    c("ATC-47", "U07", ts(27, 14, 35), "I added a test hook for the pending state so the browser test does not depend on a random network delay."),
    c("ATC-48", "U10", ts(27, 10, 30), "This came from a second pass through the checkout plan. It describes the same missing final stock check already covered by ATC-27."),
    c("ATC-48", "U02", ts(27, 11, 12), "Closing as a duplicate of ATC-27. Keep the reproduction note here and track the fix in the original story."),

    c("ATC-36", "U12", ts(30, 10, 15), "Two beta shoppers said the raw timeout message made them think the order was lost. I attached the redacted customer request: [[ATTACHMENT:beta-checkout-feedback.pdf]]"),
    c("ATC-43", "U03", ts(30, 11, 5), "Carrying the incident into the active sprint. The immediate fixes are live, but we are not closing it until stock recovery and the browser regression are complete."),
    c("ATC-32", "U12", ts(31, 9, 40), "Operations can find recent orders now. The next useful improvement is filtering by order reference without exposing the full shipping address."),
    c("ATC-46", "U05", ts(32, 11, 20), "Engineering and support finished the correction workflow review. This report records the required fields, permission boundary, retry rule, audit table, and exclusions we will implement: [[ATTACHMENT:inventory-correction-requirement.pdf]]"),
    c("ATC-35", "U10", ts(33, 15, 40), "I attached the current execution matrix to this status update. It records six passes and the Safari Back/timeout failure, which remains owned by ATC-47: [[ATTACHMENT:checkout-verification-results.csv]]"),
    c("ATC-27", "U10", ts(34, 13, 25), "I ran two carts against the last unit. One order succeeded and the other received the changed-SKU response. Detailed runs are attached: [[ATTACHMENT:stock-concurrency-results.xlsx]]"),
    c("ATC-27", "U05", ts(34, 15, 5), "The database lock works for the two-cart case. I am checking that a timed-out request releases it before marking this done."),
    c("ATC-43", "U12", ts(34, 15, 35), "No new duplicate orders have been reported since the fix. Support still needs the stock correction under ATC-46 before this is fully recoverable."),
    c("ATC-47", "U10", ts(34, 16, 5), "Chrome and Firefox pass the slow-response test. Safari sometimes keeps the pending button disabled after the simulated timeout, so this remains in progress."),
    c("ATC-47", "U07", ts(34, 16, 35), "I can reproduce the Safari state after a timeout followed by Back. I am resetting the pending flag when the page is restored."),
]


def team_for(user_id: str | None) -> str:
    if user_id == "U10":
        return "AtlasCart Quality"
    if user_id in {"U01", "U02", "U09", "U12"}:
        return "AtlasCart Product"
    return "AtlasCart Engineering"


def first_done_after(issue_key: str, after) -> object | None:
    item = ISSUE_BY_KEY[issue_key]
    candidates = []
    if item.get("done_at") and item["done_at"] >= after:
        candidates.append(item["done_at"])
    candidates.extend(
        at for at, _old, new in STATUS_OVERRIDES.get(issue_key, [])
        if new == "done" and at >= after
    )
    return min(candidates) if candidates else None


# Linked work is not left as an unexplained graph edge. Both issue timelines
# record why the link exists; blocking links also record who cleared the
# dependency and what work can resume.
assert set(LINK_NARRATIVES) == {(source, target, kind) for source, target, kind, _at in ISSUE_LINKS}
for source, target, kind, linked_at in ISSUE_LINKS:
    reason = LINK_NARRATIVES[(source, target, kind)]
    source_item = ISSUE_BY_KEY[source]
    target_item = ISSUE_BY_KEY[target]
    source_owner = source_item["assignee"] or source_item["reporter"]
    target_owner = target_item["assignee"] or target_item["reporter"]
    source_name = USER_BY_ID[source_owner]["name"]
    target_name = USER_BY_ID[target_owner]["name"]

    if kind == "blocks":
        COMMENTS.append(c(
            source, source_owner, linked_at + timedelta(minutes=3),
            f"Dependency update for {target}: this work blocks {target}. {reason} "
            f"{source_name} in {team_for(source_owner)} owns the dependency and will update {target_name} when the usable result is ready.",
        ) | {"link_context": (source, target, kind), "link_phase": "blocked-source"})
        COMMENTS.append(c(
            target, target_owner, linked_at + timedelta(minutes=5),
            f"Blocked by {source}: {reason} {target_name} can continue with local fixtures or design work, "
            f"but final integration and acceptance must wait for {source_name} in {team_for(source_owner)}.",
        ) | {"link_context": (source, target, kind), "link_phase": "blocked-target"})
        cleared_at = first_done_after(source, linked_at)
        if cleared_at:
            source_clear_at = cleared_at - timedelta(minutes=6)
            target_clear_at = cleared_at - timedelta(minutes=4)
            COMMENTS.append(c(
                source, source_owner, source_clear_at,
                f"Dependency cleared for {target}: {source_name} from {team_for(source_owner)} delivered the behavior described in this issue and shared the verified contract with {target_name}. "
                f"{target} no longer needs to wait on {source}.",
            ) | {"link_context": (source, target, kind), "link_phase": "unblocked-source"})
            COMMENTS.append(c(
                target, target_owner, target_clear_at,
                f"Unblocked by {source}: {source_name} from {team_for(source_owner)} provided the required result. "
                f"{target_name} in {team_for(target_owner)} can now resume final integration and the acceptance checks that depended on it.",
            ) | {"link_context": (source, target, kind), "link_phase": "unblocked-target"})
    elif kind == "relates to":
        COMMENTS.append(c(
            source, source_owner, linked_at + timedelta(minutes=3),
            f"Related work note for {target}: {reason} The link is for shared context and coordinated testing; it does not make either issue a hidden dependency.",
        ) | {"link_context": (source, target, kind), "link_phase": "related-source"})
        COMMENTS.append(c(
            target, target_owner, linked_at + timedelta(minutes=5),
            f"Related to {source}: {reason} Keep results and follow-up decisions on their owning issues so the two scopes do not drift together.",
        ) | {"link_context": (source, target, kind), "link_phase": "related-target"})
    else:
        COMMENTS.append(c(
            source, source_owner, linked_at + timedelta(minutes=2),
            f"Duplicate review: {reason} This issue keeps its reproduction details, but implementation and final verification remain on {target}.",
        ) | {"link_context": (source, target, kind), "link_phase": "duplicate-source"})
        COMMENTS.append(c(
            target, target_owner, linked_at + timedelta(minutes=4),
            f"Duplicate linked from {source}: {reason} No extra scope is added here; the existing owner and acceptance criteria remain authoritative.",
        ) | {"link_context": (source, target, kind), "link_phase": "duplicate-target"})


# Every issue that is currently Done has one explicit, outcome-based verification
# result. These are separate from implementation updates so the rule can be
# enforced mechanically and reviewed naturally in the issue timeline.
VERIFICATION_RESULTS = {
    "ATC-7": "PASS — routing, shared layout, and API client load in the staging shell; direct URL refresh also works.",
    "ATC-8": "PASS — migrations apply on an empty database and the health check fails when PostgreSQL is unavailable.",
    "ATC-9": "PASS — valid products save correctly, while negative price and stock values are rejected.",
    "ATC-10": "PASS — blank prices and both duplicate-SKU cases return row errors without saving a partial catalog.",
    "ATC-11": "PASS — active products, empty results, and case-insensitive name queries return the agreed fields.",
    "ATC-12": "PASS — loading, error, empty, and 50-product states work on desktop and a 375px screen; ATC-19 tracks the separate card bug.",
    "ATC-13": "PASS — active SKUs return details and both unknown and inactive SKUs return the same safe 404 response.",
    "ATC-14": "PASS — price and stock render correctly, and Add to cart is disabled with a reason at zero stock.",
    "ATC-15": "PASS — blank, failed, and slow image URLs show the neutral placeholder without hiding the product name.",
    "ATC-16": "PASS — backend and frontend failures each block merge, while a green run produces the application build.",
    "ATC-17": "PASS — the catalog build deploys to staging, records its image tag, and the rollback command restores the prior tag.",
    "ATC-18": "PASS — the catalog smoke checklist completes; the known duplicate-SKU work remains explicitly tracked under ATC-10.",
    "ATC-19": "PASS — the long product name no longer overlaps price or stock at 320px and 375px.",
    "ATC-20": "PASS — blank, negative, and non-numeric prices are rejected with the original CSV line number.",
    "ATC-21": "PASS — Chrome on a slow connection moves directly to the placeholder without flashing a broken-image icon.",
    "ATC-22": "PASS — product and engineering agree that conflicting and exact duplicate SKU rows are rejected with both line numbers.",
    "ATC-23": "PASS — add, update, remove, and read examples match the web client and return current price and stock.",
    "ATC-24": "PASS — available products can be added from list and detail views, unavailable products cannot, and the current page is preserved.",
    "ATC-25": "PASS — refresh and browser restart restore a valid cart, while corrupt stored data safely returns an empty cart.",
    "ATC-26": "PASS — update, remove, and quantity zero all return correct server totals without a page reload.",
    "ATC-28": "PASS — required address fields validate, a valid US beta address submits, and address values are absent from logs.",
    "ATC-29": "PASS — accepted, rejected, and timed-out sandbox responses map to the agreed internal states without storing card data.",
    "ATC-30": "PASS — a normal request and five repeated requests produce one order, one item snapshot set, and one stock reduction.",
    "ATC-31": "PASS — confirmation shows the same order after checkout and refresh without submitting checkout again.",
    "ATC-33": "PASS — one request ID connects proxy and API logs, appears in the error response, and exposes no cart or address values.",
    "ATC-34": "PASS — beta-1.0.0 is healthy and the catalog-to-confirmation smoke path completes in the private-beta environment.",
    "ATC-40": "PASS — the header count updates after add, update, and remove, and disappears when the cart is empty.",
    "ATC-41": "PASS — accepted, rejected, and timeout contract cases run without any live sandbox network call.",
    "ATC-42": "PASS — QA and customer operations confirm every example uses synthetic references and contains no name, address, email, or payment detail.",
    "ATC-44": "PASS — five submissions with one idempotency key return one order ID and reduce stock once.",
    "ATC-45": "PASS — keyboard and repeated mouse activation cannot send a second request while checkout is pending.",
    "ATC-48": "PASS — triage confirms the reproduction and expected fix are already covered by ATC-27; no separate implementation is needed.",
}

VERIFIER_OVERRIDES = {
    "ATC-8": "U11", "ATC-16": "U03", "ATC-17": "U11", "ATC-22": "U02",
    "ATC-33": "U04", "ATC-34": "U11", "ATC-42": "U12",
}

VERIFICATION_ATTACHMENTS = {
    "ATC-13": "product-detail-api-contract.md",
}


def verification_evidence_lines(issue_key: str) -> str:
    item = ISSUE_BY_KEY[issue_key]
    if issue_key in FRONTEND_VISUAL_EVIDENCE:
        visual_files = verification_filenames(issue_key, done=item["status"] == "done")
        if visual_files:
            return "\n".join(
                f"Evidence AC{index}: [[ATTACHMENT:{filename}]]"
                for index, filename in enumerate(visual_files, start=1)
            )
    legacy = VERIFICATION_ATTACHMENTS.get(issue_key)
    return f"Evidence: [[ATTACHMENT:{legacy}]]" if legacy else ""


def completion_time(issue: dict) -> object:
    if issue["done_at"] is not None:
        return issue["done_at"]
    done_events = [at for at, _old, new in STATUS_OVERRIDES.get(issue["key"], []) if new == "done"]
    if not done_events:
        raise ValueError(f"Done issue {issue['key']} has no completion timestamp")
    return max(done_events)


for issue_key, result in VERIFICATION_RESULTS.items():
    item = ISSUE_BY_KEY[issue_key]
    ac_results = "\n".join(
        f"AC{index} — PASS — {criterion}"
        for index, criterion in enumerate(item["acceptance_criteria"], start=1)
    )
    evidence_line = verification_evidence_lines(issue_key)
    evidence_suffix = f"\n{evidence_line}" if evidence_line else ""
    COMMENTS.append(c(
        issue_key,
        VERIFIER_OVERRIDES.get(issue_key, "U10"),
        completion_time(item) - timedelta(minutes=3),
        f"Verification result: {result}\n{ac_results}\nClose decision: every acceptance criterion above passed; any separate follow-up remains on its linked issue.{evidence_suffix}",
        verification=True,
    ))
