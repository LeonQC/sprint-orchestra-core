"""Cumulative Month-3 AtlasCart facts layered on top of the approved M1 prefix."""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


TZ = ZoneInfo("America/Los_Angeles")
M3_AS_OF = date.fromisoformat(os.environ.get("ATLASCART_M3_AS_OF", datetime.now(TZ).date().isoformat()))
SNAPSHOT_AT = datetime.combine(M3_AS_OF, time(12, 0), tzinfo=TZ)
M3_WEEK_MONDAY = M3_AS_OF - timedelta(days=M3_AS_OF.weekday())
DATASET_ID = "atlascart-m3"
DATASET_VERSION = "1.5.0"
FIXTURE_NAME = "AtlasCart (Synthetic M3)"
FIXTURE_MARKER = "[atlascart-m3-fixture]"


def dt(days: int, hour: int, minute: int = 0) -> datetime:
    """Return a timestamp relative to the final M3 snapshot date."""
    event_date = M3_WEEK_MONDAY + timedelta(days=days)
    if event_date.weekday() == 5:
        event_date += timedelta(days=2)
    elif event_date.weekday() == 6:
        event_date += timedelta(days=1)
    return datetime.combine(event_date, time(hour, minute), tzinfo=TZ)


def ds(days: int) -> str:
    return (M3_WEEK_MONDAY + timedelta(days=days)).isoformat()


USER_TIMEZONES = {
    "U01": "America/Los_Angeles", "U02": "America/New_York",
    "U03": "America/Los_Angeles", "U04": "America/Los_Angeles",
    "U05": "America/New_York", "U06": "America/Chicago",
    "U07": "America/Los_Angeles", "U08": "America/New_York",
    "U09": "America/New_York", "U10": "America/Los_Angeles",
    "U11": "America/Denver", "U12": "America/New_York",
}


SPRINTS = [
    {"id": "S4", "name": "AtlasCart Sprint 4", "goal": "Let returning beta shoppers sign in without losing their guest cart.", "start_date": ds(-49), "end_date": ds(-38), "status": "completed", "order": 3, "created_at": dt(-52, 11), "updated_at": dt(-38, 16, 45)},
    {"id": "S5", "name": "AtlasCart Sprint 5", "goal": "Give shoppers and operations a safe order lifecycle from shipping choice through refund.", "start_date": ds(-35), "end_date": ds(-24), "status": "completed", "order": 4, "created_at": dt(-40, 11), "updated_at": dt(-24, 16, 50)},
    {"id": "S6", "name": "AtlasCart Sprint 6", "goal": "Let sellers maintain a searchable catalog with an auditable import workflow.", "start_date": ds(-21), "end_date": ds(-10), "status": "completed", "order": 5, "created_at": dt(-27, 11), "updated_at": dt(-10, 16, 55)},
    {"id": "S7", "name": "AtlasCart Sprint 7", "goal": "Prove the storefront is accessible, observable, and ready for a larger public beta.", "start_date": ds(-7), "end_date": ds(4), "status": "active", "order": 6, "created_at": dt(-13, 11), "updated_at": SNAPSHOT_AT},
]

NEW_LABELS = ["auth", "security", "shipping", "refunds", "seller-tools", "search", "performance", "public-beta"]

# Real Jira issues normally carry one clear acceptance outcome. Two ACs are
# useful when there are genuinely separate paths; three are reserved for the
# few cross-system stories whose failure modes must remain independently
# reviewable. The generator validates this exact M3 distribution.
THREE_AC_KEYS = {"ATC-56", "ATC-66", "ATC-73"}
TWO_AC_KEYS = {"ATC-49", "ATC-50", "ATC-51", "ATC-52", "ATC-53", "ATC-63", "ATC-70", "ATC-78", "ATC-80"}


def acceptance_count(key: str) -> int:
    if key in THREE_AC_KEYS:
        return 3
    if key in TWO_AC_KEYS:
        return 2
    return 1


def issue(
    key: str, title: str, issue_type: str, parent: str | None, sprint: str | None,
    status: str, priority: str, assignee: str | None, reporter: str, points: int | None,
    created_at: datetime, context: str, labels: list[str], ac: tuple[str, str, str],
    rationale: str, *, started_at: datetime | None = None, done_at: datetime | None = None,
    bug: dict[str, str] | None = None, implementation: str | None = None,
    initial_points: int | None = None, initial_assignee: str | None = None,
) -> dict:
    estimate = rationale if points is None else f"{points} — {rationale}"
    created_at = created_at.astimezone(ZoneInfo(USER_TIMEZONES[reporter]))
    owner = assignee or reporter
    started_at = started_at.astimezone(ZoneInfo(USER_TIMEZONES[owner])) if started_at else None
    done_at = done_at.astimezone(ZoneInfo(USER_TIMEZONES[owner])) if done_at else None
    return {
        "key": key, "title": title, "type": issue_type, "parent": parent, "sprint": sprint,
        "status": status, "priority": priority, "assignee": assignee, "reporter": reporter,
        "points": points, "initial_points": points if initial_points is None else initial_points,
        "initial_assignee": initial_assignee or assignee, "created_at": created_at, "context": context,
        "labels": labels, "acceptance_criteria": list(ac[:acceptance_count(key)]), "estimate_rationale": estimate,
        "started_at": started_at, "done_at": done_at, "bug": bug,
        "implementation": implementation or context,
    }


EPICS = [
    issue("ATC-49", "Give returning shoppers a secure account", "epic", None, None, "in_progress", "high", "U07", "U01", None, dt(-52, 9), "Add the smallest useful customer identity layer without weakening guest checkout or exposing private order data.", ["auth", "security"], ("A beta shopper can create and use an account without blocking guest checkout.", "A signed-in shopper sees only orders that belong to that account.", "Authentication failures remain understandable and never expose credentials or internal provider details."), "No story points — this epic groups identity outcomes across several sprints and must be planned through its child work."),
    issue("ATC-50", "Support shipping, cancellation, and refunds", "epic", None, None, "in_progress", "high", "U06", "U01", None, dt(-52, 9, 5), "Extend the first checkout into a supportable order lifecycle while keeping payment and address handling deliberately narrow.", ["shipping", "refunds", "orders"], ("A shopper can choose a beta shipping option and understand the order status that follows.", "Operations can cancel or refund only when the current order state permits it.", "Every lifecycle change records who acted, why it changed, and what the shopper should expect next."), "No story points — shipping and refunds contain independent API, UI, operations, and provider work that should not be hidden in one estimate."),
    issue("ATC-51", "Give sellers safe catalog operations", "epic", None, None, "in_progress", "medium", "U05", "U02", None, dt(-27, 9), "Move beyond one supplier file by giving invited sellers controlled editing, background imports, search refresh, and an audit trail.", ["seller-tools", "catalog", "search"], ("An authorized seller can update catalog data without direct database access.", "Large imports report progress and row failures without leaving a partial catalog.", "Storefront search and catalog audit history reflect accepted seller changes."), "No story points — this epic spans permissions, asynchronous processing, UI, indexing, and audit behavior that must be estimated separately."),
    issue("ATC-52", "Prepare AtlasCart for a larger public beta", "epic", None, None, "in_progress", "highest", "U03", "U01", None, dt(-13, 9), "Prove the current storefront can serve more shoppers with measurable performance, accessibility, security, and release controls.", ["public-beta", "performance", "accessibility", "platform"], ("The agreed catalog-to-order path meets its public-beta latency and accessibility thresholds.", "Operational dashboards expose errors, saturation, and release health without customer data leakage.", "A documented go or no-go review names every accepted risk and remaining blocker."), "No story points — readiness is an outcome assembled from load, accessibility, observability, and release work rather than one deliverable."),
]


WORK = [
    issue("ATC-53", "Add customer authentication at the gateway", "task", "ATC-49", "S4", "done", "high", "U04", "U03", 5, dt(-51, 9), "Validate customer sessions at the gateway and forward one trusted customer identifier to the API while public catalog routes remain anonymous.", ["auth", "security", "backend"], ("Public catalog and product routes remain available without a session.", "Protected account and order routes reject missing or expired sessions with one safe 401 response.", "The gateway forwards a verified customer identifier and removes any client-supplied identity header."), "The change touches gateway filters, session validation, downstream headers, and integration tests. The identity provider contract already exists, so account UI and authorization policy are outside this task.", started_at=dt(-49, 10), done_at=dt(-45, 15), implementation="The gateway now strips client identity headers before adding the verified customer ID, and public catalog smoke tests still pass."),
    issue("ATC-54", "Create a customer account", "story", "ATC-49", "S4", "done", "high", "U06", "U01", 5, dt(-51, 9, 5), "Let an invited shopper create an account with email and password after accepting the beta terms; checkout must still work without an account.", ["auth", "frontend", "backend"], ("A valid invited email creates one account and starts a signed-in session.", "Duplicate email and weak password errors explain the correction without revealing another account.", "Declining account creation leaves guest catalog, cart, and checkout available."), "This includes a form, API validation, password hashing, duplicate handling, and a signed-in session. Email verification and social login are intentionally deferred.", started_at=dt(-49, 11), done_at=dt(-44, 14), implementation="The account API hashes passwords with the approved work factor, and the web form keeps guest checkout visible after a validation error."),
    issue("ATC-55", "Sign in and sign out returning shoppers", "story", "ATC-49", "S4", "done", "high", "U07", "U01", 3, dt(-51, 9, 10), "Add plain sign-in and sign-out behavior for beta accounts without introducing password recovery in this sprint.", ["auth", "frontend"], ("A valid account starts a session and returns the shopper to the page they requested.", "An invalid email or password produces the same safe message and does not say which value was wrong.", "Sign out clears the session and protected pages immediately require sign-in again."), "The provider and gateway session contract are already available. Work is limited to two screens, redirect state, cookie clearing, and browser tests; recovery and multi-factor authentication are excluded.", started_at=dt(-47, 9), done_at=dt(-43, 16), implementation="Sign-in returns shoppers to their original product or order route, and sign-out invalidates both the browser cookie and server session."),
    issue("ATC-56", "Merge a guest cart after sign-in", "story", "ATC-49", "S4", "done", "highest", "U04", "U02", 8, dt(-51, 9, 15), "When a guest signs in, combine the browser cart with the account cart once. Use current price and stock, and never silently double a quantity.", ["auth", "cart", "backend"], ("Products that appear in only one cart appear once in the merged cart.", "The same product in both carts combines up to current stock and explains any quantity that could not be kept.", "Refreshing or retrying sign-in returns the same merged cart without adding the guest items again."), "This work joins a browser cart, a saved account cart, and current stock. It also needs tests for retries and two requests arriving at nearly the same time. Promotions and carts saved on several devices are not included.", started_at=dt(-47, 10), done_at=dt(-40, 15), implementation="The merge marks the guest cart as used, checks stock again, and records any quantity it could not keep so the page can explain the change."),
    issue("ATC-57", "Harden customer session cookies", "task", "ATC-49", "S4", "done", "high", "U11", "U03", 3, dt(-49, 9), "Set the beta session lifetime and cookie controls, then prove logs and browser storage do not reveal the session value.", ["auth", "security", "platform"], ("The session cookie is Secure, HttpOnly, SameSite=Lax, and scoped to the application path.", "Idle and absolute expiration both return the shopper to sign-in without losing a server-side account cart.", "Application, gateway, and audit logs contain no raw cookie or session token."), "The code change is small, but configuration spans gateway, browser behavior, and log redaction. Token rotation and cross-domain login are not part of the beta.", started_at=dt(-48, 10), done_at=dt(-44, 11), implementation="Cookie flags are shared across environments, while beta uses a shorter idle timeout; a log scan confirmed no raw session values."),
    issue("ATC-58", "Rate-limit repeated sign-in failures", "task", "ATC-49", "S4", "done", "medium", "U11", "U03", 3, dt(-47, 9), "Slow automated password guessing without locking a real shopper out after one ordinary typo.", ["auth", "security", "platform"], ("Repeated failures from one source receive a temporary 429 after the configured threshold.", "A successful sign-in clears the account failure counter without exposing the previous attempts.", "Metrics show limited requests and logs use a hashed account key rather than an email address."), "This adds a bounded gateway counter, expiry, metrics, and deterministic tests. Distributed global enforcement and bot scoring are deferred until traffic justifies them.", started_at=dt(-45, 10), done_at=dt(-41, 14), implementation="The limiter keys on source and a hashed account identifier, expires automatically, and exports counts without email addresses."),
    issue("ATC-59", "Cart merge doubles a matching SKU", "bug", "ATC-49", "S4", "done", "highest", "U04", "U10", 3, dt(-43, 9), "A shopper with two mugs in an account cart and one in a guest cart sees six after sign-in because the merge callback runs twice.", ["auth", "cart", "incident-follow-up"], ("The example carts merge to quantity three, not six, when stock is available.", "Refreshing or retrying the callback returns the same merged cart without a second quantity change.", "A regression test covers two overlapping SKUs and one guest-only SKU."), "The faulty calculation is straightforward, but the fix must preserve idempotency under callback retry and verify multiple overlap patterns. No schema migration is required.", started_at=dt(-43, 10), done_at=dt(-40, 12), bug={"when": "Private beta, Chrome 127, account cart quantity 2 and guest cart quantity 1 for SKU F-612.", "steps": "Add one mug as a guest, sign into an account that already has two, then refresh the callback route.", "expected": "The merged cart contains three mugs once.", "actual": "The first callback shows three and the retry doubles the account quantity to six."}, implementation="The callback now stores and returns the first merge result before clearing the guest token; repeated callbacks no longer recalculate quantities."),
    issue("ATC-60", "Add authentication contract tests", "task", "ATC-49", "S4", "done", "medium", "U10", "U03", 3, dt(-47, 9, 20), "Test the gateway and API with no session, a valid session, an expired session, and a fake customer identity.", ["auth", "private-beta"], ("Public catalog requests work without sign-in, while account-order requests require a valid session.", "Expired sessions return the agreed safe 401 response, and the API ignores a customer identity supplied by the browser."), "Test data already stands in for the identity provider. This task adds four gateway-to-API cases to CI; browser sign-in remains covered by the sign-in stories.", started_at=dt(-43, 9), done_at=dt(-39, 15), implementation="CI now checks anonymous, valid, expired, and fake-identity requests with test sessions and stops the merge if an API response changes."),

    issue("ATC-61", "Return beta shipping quotes", "story", "ATC-50", "S5", "done", "high", "U04", "U01", 5, dt(-37, 9), "Return standard and expedited shipping choices from cart weight and destination. Use saved test responses instead of calling a real shipping company.", ["shipping", "backend"], ("A supported US address receives named shipping choices with a USD price and delivery range.", "An unsupported destination shows one plain message without exposing the shipping company's response."), "This adds one connection to the shipping service, translates its response into AtlasCart choices, handles timeouts, and tests checkout with those choices. International shipping, negotiated rates, and label purchase are not included.", started_at=dt(-35, 10), done_at=dt(-30, 15), implementation="Saved shipping responses now become AtlasCart choices. A timeout or unknown service stops order creation and leaves the cart unchanged."),
    issue("ATC-62", "Choose a shipping option at checkout", "story", "ATC-50", "S5", "done", "high", "U08", "U01", 5, dt(-37, 9, 5), "Show shipping choices after a valid address and keep the selected price visible through order confirmation.", ["shipping", "checkout", "frontend"], ("Standard and expedited choices show price and delivery range in a keyboard-usable control.", "Changing the address refreshes quotes and clearly resets an option that is no longer valid.", "The selected option and price appear unchanged in the checkout total and confirmation."), "The story includes quote loading and failure states, accessible selection, total integration, and confirmation display. Address validation and provider mapping are separate work.", started_at=dt(-34, 9), done_at=dt(-29, 16), implementation="The checkout step resets stale quotes after address edits and reads the final shipping price from the server order preview rather than recalculating it."),
    issue("ATC-63", "Define the beta order status lifecycle", "task", "ATC-50", "S5", "done", "high", "U02", "U03", 3, dt(-37, 9, 10), "Agree on the small set of order states and transitions that support, cancellation, shipment, and refund work will use.", ["orders", "shipping", "refunds"], ("The lifecycle defines allowed transitions from placed through shipped, cancelled, or refunded.", "Every transition names its owning actor and whether customer communication is required.", "Invalid transitions return one conflict response and leave status history unchanged."), "The output is a reviewed state table and API rule set, not implementation of every transition. Product, operations, and engineering must resolve cancellation and refund boundaries.", started_at=dt(-35, 11), done_at=dt(-32, 14), implementation="Product and operations approved the state table, including the rule that a shipped order cannot be cancelled and must use refund handling."),
    issue("ATC-64", "Search recent orders by reference", "story", "ATC-50", "S5", "done", "medium", "U06", "U12", 3, dt(-37, 9, 15), "Let customer operations find one synthetic beta order by its complete reference without exposing address or payment fields.", ["orders", "backend", "frontend"], ("An exact case-insensitive reference returns the matching order summary.", "No match shows an empty result and a partial reference does not enumerate nearby orders.", "The result contains status, total, created time, and payment state but no full address or provider detail."), "The recent-order view and index already exist. Work is one exact query, safe response mapping, a small search field, and access tests; fuzzy customer search is excluded.", started_at=dt(-34, 10), done_at=dt(-30, 14), implementation="Operations can search the complete BETA reference, while partial values deliberately return no suggestions and no sensitive fields."),
    issue("ATC-65", "Cancel an order before fulfillment", "story", "ATC-50", "S5", "done", "high", "U06", "U12", 5, dt(-35, 9), "Allow operations to cancel a paid order before shipment, restore stock once, and record the support reason.", ["orders", "inventory", "refunds"], ("An authorized operator can cancel a paid unshipped order with an order reference and reason.", "Cancellation restores each item quantity exactly once and records actor, reason, and timestamp.", "A shipped, already cancelled, or unknown order returns a conflict or not-found response without stock change."), "Cancellation combines permission checks, lifecycle transition, inventory compensation, idempotency, and audit history. Payment refund remains a separate provider operation.", started_at=dt(-33, 10), done_at=dt(-27, 15), implementation="Cancellation and inventory restoration now share one transaction and repeated requests return the existing cancellation audit record."),
    issue("ATC-66", "Refund a cancelled beta order", "story", "ATC-50", "S5", "done", "highest", "U04", "U12", 8, dt(-35, 9, 5), "Send one full refund through the payment sandbox after an eligible cancellation and keep provider details behind the adapter.", ["refunds", "orders", "backend"], ("An eligible cancelled paid order creates one refund request and records the safe provider reference.", "Retrying the operation returns the first refund and never submits a second provider request.", "Rejected or timed-out refunds remain visible to operations with a safe recovery state and request ID."), "This crosses order state, payment adapter, idempotency, failure recovery, audit, and operations UI. Partial refunds and charge disputes are explicitly excluded from the beta.", started_at=dt(-32, 9), done_at=dt(-25, 15), implementation="The refund key is stored before the provider call, allowing timeout recovery to query the first result instead of issuing another refund."),
    issue("ATC-67", "Send email when an order status changes", "task", "ATC-50", "S5", "done", "medium", "U11", "U03", 3, dt(-34, 9), "Send a local test email after an order is placed, cancelled, or refunded. A slow or failed email must not delay the order update.", ["orders", "platform"], ("Each supported status sends one email with the order number and a plain next step.", "If sending fails, the system tries again later without undoing the accepted order change."), "This work adds three email templates, a message after each order change, retry rules, and tests that private payment data never appears. Marketing preferences and a real email company are not included.", started_at=dt(-31, 10), done_at=dt(-27, 14), implementation="The order update is saved before the email job starts. A failed test email can be sent again using only the order number and the minimum shipping details."),
    issue("ATC-68", "Tracking link opens a different beta order", "bug", "ATC-50", "S5", "done", "highest", "U06", "U10", 3, dt(-29, 9), "The confirmation email for BETA-1128 opens BETA-1123 because the last digit is missing from the link.", ["shipping", "orders", "incident-follow-up"], ("The BETA-1128 email opens only the BETA-1128 status page.", "Orders with similar numbers still open the right page, including emails that the background job sends again.", "A regression test covers at least three nearby synthetic order numbers."), "The broken link is a small change, but we must check both the first email and a resent email. We must also confirm that signing in to one account never allows that shopper to open another account's order.", started_at=dt(-29, 10), done_at=dt(-27, 12), bug={"when": "Private beta after order BETA-1128 moved to Placed and its confirmation email arrived.", "steps": "Open the test email and select View order while signed in as the BETA-1128 shopper.", "expected": "The link opens BETA-1128 after the account check.", "actual": "The link ends in BETA-1123 and opens the wrong test order."}, implementation="Both the first email and a resent email now keep the full order number in the link. The order page also checks that the signed-in shopper owns that order."),

    issue("ATC-69", "Edit product details in seller tools", "story", "ATC-51", "S6", "done", "high", "U05", "U02", 5, dt(-23, 9), "Let an invited seller update product name, description, price, active state, and stock through validated operations rather than SQL.", ["seller-tools", "catalog", "frontend", "backend"], ("An authorized seller can edit the agreed product fields and sees field-specific validation errors.", "Price and stock rules match import validation and reject negative values.", "An accepted edit appears in product detail and records before, after, actor, and reason."), "This spans seller form, API validation, transaction handling, storefront cache refresh, and audit history. Variant products and bulk editing remain outside scope.", started_at=dt(-21, 10), done_at=dt(-16, 15), implementation="Seller edits reuse catalog validation, publish a cache-refresh event, and store a field-level audit record with the supplied reason."),
    issue("ATC-70", "Restrict seller catalog permissions", "task", "ATC-51", "S6", "done", "highest", "U04", "U03", 5, dt(-23, 9, 5), "Check seller permissions on the server before any product is created or changed. Customer and operations access must stay separate.", ["seller-tools", "security", "backend"], ("Only a seller member with catalog-edit permission can create or change a product.", "Customers, operations users, and requests with a fake seller role receive the same safe forbidden response."), "The check uses gateway sign-in details, space membership, API rules, failure tests, and the audit record. Building a general role editor is not included.", started_at=dt(-21, 11), done_at=dt(-17, 14), implementation="The server now looks up seller membership before a catalog change and ignores role or space values sent by the browser."),
    issue("ATC-71", "Upload product images to managed storage", "story", "ATC-51", "S6", "done", "medium", "U11", "U02", 5, dt(-23, 9, 10), "Store seller product images in managed object storage and keep the existing neutral placeholder for invalid or unavailable images.", ["seller-tools", "catalog", "platform"], ("An authorized seller can upload a PNG or JPEG within the documented size and dimension limits.", "Invalid type, oversized content, and failed scanning leave the current product image unchanged.", "Accepted objects use generated storage keys and the product API returns a stable image reference."), "The story covers multipart upload, validation, object storage, generated keys, and cleanup on failure. Image editing, CDN transformation, and video are deferred.", started_at=dt(-20, 9), done_at=dt(-15, 15), implementation="Uploads stream to a temporary object, pass type and size validation, then promote to a generated key before the product reference changes."),
    issue("ATC-72", "Index products for catalog search", "story", "ATC-51", "S6", "done", "high", "U05", "U02", 5, dt(-23, 9, 15), "Replace the basic product-name match with a small search index for SKU, name, and description. Inactive products must still stay hidden.", ["search", "catalog", "backend"], ("A search finds active products by SKU, name, or description without caring about letter case.", "Inactive products never appear, and an empty search keeps the normal catalog order."), "This defines the search fields, loads the current catalog, handles new changes, and tests which products should appear first. Personalization, synonyms, typo correction, and recommendations are not included.", started_at=dt(-20, 10), done_at=dt(-14, 16), implementation="The search index receives accepted catalog changes, hides inactive products, and passed the agreed SKU, name, and description examples."),
    issue("ATC-73", "Run large catalog imports in the background", "story", "ATC-51", "S6", "done", "highest", "U05", "U02", 8, dt(-23, 9, 20), "Process a five-thousand-row seller file in the background, show progress, and update the catalog only after the whole file passes validation.", ["seller-tools", "catalog", "platform"], ("A valid seller file enters a queue and returns an import number before processing begins.", "The progress page shows how many rows were read, accepted, and rejected without exposing another seller's file.", "A file with any invalid row changes nothing; a fully valid file updates the catalog once and refreshes search."), "The job must survive a restart, keep useful progress, validate every row, update the whole catalog once, and recover safely after a retry. Applying only part of a file is not allowed.", started_at=dt(-19, 9), done_at=dt(-12, 15), implementation="The job saves its validation progress but does not change catalog rows until every row passes. It then updates the catalog once and asks search to refresh."),
    issue("ATC-74", "Show catalog import progress", "story", "ATC-51", "S6", "done", "medium", "U08", "U02", 3, dt(-21, 9), "Give sellers a readable progress page for queued, validating, failed, and applied imports with downloadable row errors.", ["seller-tools", "catalog", "frontend"], ("The page shows current phase and processed counts without requiring a browser refresh.", "A failed import explains the summary and provides the issue-specific rejected-row file.", "An applied import links to its audit entry and never presents a stale Apply action."), "The API already exposes job status. Work is polling with backoff, four states, error download, audit link, and accessible status announcements; cancellation is excluded.", started_at=dt(-18, 10), done_at=dt(-14, 15), implementation="The progress page backs off while queued, announces phase changes for assistive technology, and links completed jobs to their audit entry."),
    issue("ATC-75", "Audit seller catalog changes", "task", "ATC-51", "S6", "done", "high", "U06", "U03", 5, dt(-21, 9, 5), "Give operations a searchable record of seller edits and imports without storing full uploaded files in the audit table.", ["seller-tools", "catalog", "security"], ("Each accepted edit and import records seller, time, source, reason, and changed product references.", "Operations can filter by seller, SKU, change type, and date without reading private credentials.", "Audit records are append-only through the application and remain after a product is deactivated."), "This includes a normalized audit model, event consumers, filters, permission checks, and retention behavior. Full file archival and legal export are separate concerns.", started_at=dt(-17, 9), done_at=dt(-12, 14), implementation="Catalog events now create append-only audit rows, and operations filters use stable seller and SKU identifiers rather than copied product text."),
    issue("ATC-76", "Edited product remains stale in search", "bug", "ATC-51", "S6", "done", "high", "U05", "U10", 3, dt(-15, 9), "After a seller renames SKU C-311, product detail updates immediately but search keeps the old name until the nightly rebuild.", ["search", "catalog", "incident-follow-up"], ("An accepted product rename is searchable by the new name within sixty seconds.", "The old unique name stops matching after the refresh completes.", "A failed refresh is visible in metrics and can be replayed without repeating the seller edit."), "The missing event handler is limited, but the fix needs freshness timing, replay safety, and failure observability across edit and import event shapes.", started_at=dt(-15, 10), done_at=dt(-12, 11), bug={"when": "Seller staging after renaming C-311 from Monitor Stand to Bamboo Display Riser.", "steps": "Save the edit, confirm product detail, then search both names for five minutes.", "expected": "The new name appears within sixty seconds and the old unique name stops matching.", "actual": "Product detail is current, but search returns the old name until nightly rebuild."}, implementation="Edit and import events now share the same idempotent indexing handler, with lag and dead-letter metrics for replay."),

    issue("ATC-77", "Cache the public product catalog safely", "story", "ATC-52", "S7", "in_progress", "high", "U05", "U03", 5, dt(-9, 9), "Reduce repeated catalog reads for the public beta without serving inactive products or hiding a seller update for too long.", ["performance", "catalog", "backend"], ("Repeated public catalog requests meet the cache-hit latency target under the agreed load.", "Deactivate, edit, and import events invalidate affected entries within the freshness target.", "Cache failure falls back to PostgreSQL with bounded load and no customer-visible raw error."), "This combines cache keys, invalidation, fallback, load tests, and observability. Personalized catalog responses and multi-region cache replication are excluded.", started_at=dt(-7, 10), implementation="Read-through caching is working for list and detail routes; the team is still testing invalidation during a large import."),
    issue("ATC-78", "Complete the public-beta accessibility pass", "task", "ATC-52", "S7", "in_progress", "highest", "U08", "U09", 5, dt(-9, 9, 5), "Close the keyboard, focus, labeling, and contrast gaps on catalog, account, cart, checkout, and order status paths.", ["public-beta", "accessibility", "frontend"], ("The complete catalog-to-order path is operable by keyboard with visible and logical focus.", "Forms and dynamic status changes expose useful labels and announcements to a screen reader.", "Automated checks and the manual matrix contain no critical or high-severity accessibility failure."), "The work spans five flows, shared components, automated checks, and a manual screen-reader pass. A full external WCAG audit is not included.", started_at=dt(-6, 9), implementation="Shared form labels and focus restoration are merged; the checkout error summary and manual Safari/VoiceOver pass remain open."),
    issue("ATC-79", "Run the public-beta release readiness review", "task", "ATC-52", "S7", "planned", "highest", "U03", "U01", 3, dt(-8, 9), "Create one clear go or no-go record for features, incidents, speed, accessibility, security, rollback, and accepted risks. Every decision needs a link to the supporting result.", ["public-beta", "platform", "private-beta"], ("Every release requirement links to a current test, dashboard, issue, or named risk owner.", "Any open highest-priority blocker makes the result no-go.", "The final decision records who attended, which build was reviewed, the environment, the rollback version, and each follow-up owner."), "This is a review and evidence-gathering task after the related work finishes. Fixing a blocker and running the load test belong to their existing issues."),
    issue("ATC-80", "Load-test catalog through checkout", "task", "ATC-52", "S7", "in_progress", "highest", "U11", "U03", 8, dt(-9, 9, 10), "Exercise the public catalog, sign-in, cart, quote, and checkout path at the agreed beta concurrency while protecting the payment sandbox.", ["public-beta", "performance", "platform"], ("The workload reaches the target request mix and concurrency with synthetic shoppers and fixture payments.", "P95 latency and error rate remain within the agreed threshold for every critical route.", "The run records bottlenecks, resource saturation, request IDs, and a repeatable environment without customer data."), "This requires workload modeling, isolated fixtures, environment controls, dashboards, bottleneck analysis, and safe payment stubbing. Internet-scale capacity and destructive stress are excluded.", started_at=dt(-6, 10), implementation="Catalog and sign-in workloads are stable at target concurrency; checkout currently exceeds the P95 goal while shipping quotes are cold."),
]

# Active Sprint 7 includes focused subtasks rather than hiding every technical
# step inside large parent tickets. They deliberately carry no story points;
# the estimate stays on the parent.
WORK.extend([
    issue("ATC-81", "Exercise catalog cache invalidation during import", "subtask", "ATC-77", "S7", "in_progress", "high", "U05", "U03", None, dt(-5, 9), "Run the large-import path while catalog caching is enabled and record whether edited or deactivated products remain stale.", ["performance", "catalog", "backend"], ("The import scenario records cache invalidation time and identifies any SKU that exceeds the freshness target.", "The run can be repeated with the same synthetic catalog.", "Any cache miss or stale result keeps its request ID."), "No story points — ATC-77 carries the cache design, invalidation, fallback, and load-test estimate; this subtask isolates one reproducible verification slice.", started_at=dt(-5, 10), implementation="The first run found one stale list entry while detail invalidation stayed within target."),
    issue("ATC-82", "Fix checkout focus after validation errors", "subtask", "ATC-78", "S7", "in_progress", "high", "U08", "U09", None, dt(-5, 9, 10), "Move keyboard focus to the checkout error summary and keep each message connected to its field.", ["accessibility", "frontend"], ("Submitting invalid checkout data moves focus to a readable summary whose links return to the affected fields.", "VoiceOver announces the summary once.", "Valid checkout behavior remains unchanged."), "No story points — ATC-78 already includes this accessibility work in its estimate. This subtask gives the remaining checkout problem its own owner and test result.", started_at=dt(-4, 9), implementation="Focus now moves to the summary in Chrome; Safari and VoiceOver verification remains open."),
    issue("ATC-83", "Add a cold shipping-quote load scenario", "subtask", "ATC-80", "S7", "planned", "medium", "U11", "U03", None, dt(-4, 9, 10), "Add a controlled cold-cache shipping quote step to the public-beta workload so the current checkout P95 can be reproduced.", ["performance", "shipping", "platform"], ("The workload can run cold shipping quotes without calling a production provider.", "Results separate quote latency from the rest of checkout.", "Fixture cleanup remains repeatable."), "No story points — ATC-80 carries workload modeling and environment risk; this subtask isolates the cold-quote scenario discovered during the first run."),
])

ISSUES = EPICS + WORK

M3_PRIORITY_OVERRIDES = {
    "ATC-52": "high", "ATC-70": "high", "ATC-73": "high",
    "ATC-78": "high", "ATC-79": "high", "ATC-80": "high",
}
M3_CARRYOVERS = {
    "ATC-60": {"from": "S4", "to": "S5", "at": dt(-38, 16, 20), "reason": "The forged-header CI case was flaky on the Sprint 4 close build, so the team carried the test task instead of calling an intermittent pass complete."},
    "ATC-68": {"from": "S5", "to": "S6", "at": dt(-24, 16, 25), "reason": "The first link fix passed locally, but Sprint 5 ended before the team checked emails sent a second time and confirmed that another account still could not open the order."},
    "ATC-72": {"from": "S6", "to": "S7", "at": dt(-10, 16, 25), "reason": "Search queries worked, but the initial backfill and replay check exceeded the Sprint 6 window and moved into Sprint 7."},
}
M3_ESTIMATE_CHANGES = {
    "ATC-56": {"from": 5, "to": 8, "at": dt(-47, 13), "reason": "Refinement exposed stock revalidation and callback idempotency across both guest and account carts."},
    "ATC-66": {"from": 5, "to": 8, "at": dt(-32, 13), "reason": "Timeout recovery and provider idempotency made the refund story larger than the original happy-path estimate."},
    "ATC-73": {"from": 5, "to": 8, "at": dt(-19, 13), "reason": "We need to keep progress after a restart, avoid partial catalog changes, and safely retry a stopped import. That is more than one small background-job change."},
}
M3_ASSIGNEE_CHANGES = {
    "ATC-71": {"from": "U05", "to": "U11", "at": dt(-20, 11), "reason": "Olivia took ownership when object validation and cleanup became the main risk."},
    "ATC-74": {"from": "U07", "to": "U08", "at": dt(-18, 11), "reason": "Lucas already owned the seller-tools UI and could reuse its accessible status components."},
    "ATC-80": {"from": "U10", "to": "U11", "at": dt(-6, 11), "reason": "Olivia owns the load environment and dashboards; Daniel remains the test reviewer."},
}
M3_STATUS_OVERRIDES = {
    "ATC-68": [
        (dt(-29, 10), "planned", "in_progress"),
        (dt(-19, 12), "in_progress", "done"),
        (dt(-18, 9), "done", "in_progress"),
        (dt(-17, 15), "in_progress", "done"),
    ]
}

# Most implementation summaries describe the final state. A reopened issue
# needs an earlier note that honestly reflects only the first attempted fix.
M3_EARLY_IMPLEMENTATION_UPDATES = {
    "ATC-68": "I fixed the link used by the first email and added tests for three nearby order numbers. I still need to check what happens when the email job sends the message again.",
}

for _item in ISSUES:
    if _item["key"] in M3_PRIORITY_OVERRIDES:
        _item["priority"] = M3_PRIORITY_OVERRIDES[_item["key"]]
    if _item["key"] in M3_ESTIMATE_CHANGES:
        _item["initial_points"] = M3_ESTIMATE_CHANGES[_item["key"]]["from"]
    if _item["key"] in M3_ASSIGNEE_CHANGES:
        _item["initial_assignee"] = M3_ASSIGNEE_CHANGES[_item["key"]]["from"]
    if _item["key"] in M3_CARRYOVERS:
        _item["sprint"] = M3_CARRYOVERS[_item["key"]]["to"]
    if _item["key"] == "ATC-60":
        _item["done_at"] = dt(-33, 15).astimezone(ZoneInfo(USER_TIMEZONES[_item["assignee"]]))
    elif _item["key"] == "ATC-68":
        _item["done_at"] = dt(-17, 15).astimezone(ZoneInfo(USER_TIMEZONES[_item["assignee"]]))
    elif _item["key"] == "ATC-72":
        _item["done_at"] = dt(-5, 15).astimezone(ZoneInfo(USER_TIMEZONES[_item["assignee"]]))

ISSUE_BY_KEY = {item["key"]: item for item in ISSUES}

# The M1 snapshot remains immutable. These are later facts about work that was
# open at that cutoff, including an explicit carry into Sprint 4 where needed.
M1_CONTINUATIONS = {
    "ATC-27": {"status": "done", "sprint": "S3", "at": dt(-52, 14), "author": "U10", "comment": "Final timeout and lock-release checks passed. The stock transaction releases cleanly after a simulated payment timeout, so all ATC-27 acceptance criteria are now complete."},
    "ATC-32": {"status": "done", "sprint": "S3", "at": dt(-53, 15), "author": "U12", "comment": "Operations approved the payment-state column and can distinguish the successful order during duplicate-order support without opening customer details."},
    "ATC-35": {"status": "done", "sprint": "S3", "at": dt(-52, 15), "author": "U10", "comment": "The Safari Back and timeout case now passes with the pending-state reset from ATC-47. The complete checkout matrix is attached to the Sprint 3 close record."},
    "ATC-36": {"status": "done", "sprint": "S3", "started_at": dt(-54, 9), "at": dt(-54, 14), "author": "U12", "comment": "Customer operations approved the plain timeout and rejection wording after confirming the cart and next action remain clear on mobile."},
    "ATC-37": {"status": "done", "sprint": "S3", "started_at": dt(-54, 9, 15), "at": dt(-54, 15), "author": "U10", "comment": "The empty-cart action remains inside the first 320px viewport and the layout passes keyboard and narrow-screen verification."},
    "ATC-39": {"status": "done", "sprint": "S3", "started_at": dt(-53, 9), "at": dt(-53, 14), "author": "U10", "comment": "Quantity 10 and a two-line product name no longer overlap Remove at 320px; the normal one-digit layout is unchanged."},
    "ATC-43": {"status": "done", "sprint": "S3", "at": dt(-51, 16), "author": "U03", "comment": "Closing the duplicate-order incident after idempotency, pending-button mitigation, stock recovery design, and slow-browser regression all reached verified owners. No duplicate beta order was reported after beta-1.0.2."},
    "ATC-46": {"status": "done", "sprint": "S3", "at": dt(-52, 13), "author": "U12", "comment": "Operations completed a correction through the audited workflow and confirmed the order reference, reason, actor, and stock delta are understandable without a direct database edit."},
    "ATC-47": {"status": "done", "sprint": "S3", "at": dt(-52, 14, 30), "author": "U10", "comment": "Chrome, Firefox, and Safari now pass delayed response, repeated activation, timeout, and browser Back cases. The Safari pending-state regression is closed."},
}

CANONICAL_LINKS = [
    ("ATC-53", "ATC-55", "blocks", dt(-49, 9, 10)),
    ("ATC-53", "ATC-56", "blocks", dt(-49, 9, 15)),
    ("ATC-56", "ATC-59", "relates to", dt(-43, 9, 10)),
    ("ATC-63", "ATC-65", "blocks", dt(-35, 11, 10)),
    ("ATC-65", "ATC-66", "blocks", dt(-32, 9, 10)),
    ("ATC-63", "ATC-67", "blocks", dt(-34, 9, 10)),
    ("ATC-67", "ATC-68", "relates to", dt(-29, 9, 10)),
    ("ATC-70", "ATC-69", "blocks", dt(-21, 11, 10)),
    ("ATC-71", "ATC-69", "relates to", dt(-20, 9, 10)),
    ("ATC-73", "ATC-74", "blocks", dt(-18, 10, 10)),
    ("ATC-72", "ATC-76", "relates to", dt(-15, 9, 10)),
    ("ATC-77", "ATC-80", "relates to", dt(-7, 10, 10)),
    ("ATC-78", "ATC-79", "blocks", dt(-6, 9, 10)),
    ("ATC-80", "ATC-79", "blocks", dt(-6, 10, 10)),
]

LINK_REASONS = {
    ("ATC-53", "ATC-55", "blocks"): "Sign-in cannot trust a browser session until the gateway validates and forwards the customer identity.",
    ("ATC-53", "ATC-56", "blocks"): "Cart merge needs the verified account identity produced by the gateway boundary.",
    ("ATC-56", "ATC-59", "relates to"): "The bug is a retry failure in the merge behavior delivered by ATC-56, while its reproduction and verification stay on the bug.",
    ("ATC-63", "ATC-65", "blocks"): "Cancellation rules require the approved lifecycle to distinguish cancellable and shipped orders.",
    ("ATC-65", "ATC-66", "blocks"): "The beta refund begins only after an eligible order has a recorded cancellation.",
    ("ATC-63", "ATC-67", "blocks"): "Notification templates need stable status names and customer next actions from the lifecycle decision.",
    ("ATC-67", "ATC-68", "relates to"): "The wrong tracking link was introduced in the status-email adapter and is verified separately as an incident follow-up.",
    ("ATC-70", "ATC-69", "blocks"): "Seller editing must not ship until server-side seller membership and catalog permission checks exist.",
    ("ATC-71", "ATC-69", "relates to"): "Managed image upload supplies one editable field on the seller product form but can be released independently.",
    ("ATC-73", "ATC-74", "blocks"): "The progress UI requires durable import job phases and counters from the background worker.",
    ("ATC-72", "ATC-76", "relates to"): "The stale-result bug is a freshness failure in the search indexing path delivered by ATC-72.",
    ("ATC-77", "ATC-80", "relates to"): "The load test measures catalog cache behavior, but it also covers uncached checkout and remains independently owned.",
    ("ATC-78", "ATC-79", "blocks"): "The readiness review cannot approve public beta while critical accessibility verification remains open.",
    ("ATC-80", "ATC-79", "blocks"): "The readiness decision requires current latency, error, and saturation evidence from the load test.",
}
