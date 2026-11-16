"""Canonical AtlasCart M1 scenario facts.

This module contains data, not database code. Logical story dates are shifted so
the snapshot always lands inside the current active sprint.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from m1_quality import (
    ACCEPTANCE_CRITERIA,
    BUG_REPORT_DETAILS,
    LINK_NARRATIVES,
    POINT_RATIONALES as DETAILED_POINT_RATIONALES,
    UNESTIMATED_RATIONALES,
)


TZ = ZoneInfo("America/Los_Angeles")
REFERENCE_START = date(2026, 3, 2)
REFERENCE_SNAPSHOT = date(2026, 4, 6)
_configured_as_of = os.environ.get("ATLASCART_AS_OF")
AS_OF_DATE = date.fromisoformat(_configured_as_of) if _configured_as_of else datetime.now(TZ).date()
# Keep the canonical Monday-Friday story calendar stable for every rolling
# snapshot. The snapshot may be any day inside the active two-week sprint, but
# human events must not drift onto weekends merely because the CLI ran Tuesday
# instead of Monday.
AS_OF_WEEK_MONDAY = AS_OF_DATE - timedelta(days=AS_OF_DATE.weekday())
DATE_SHIFT = AS_OF_WEEK_MONDAY - REFERENCE_SNAPSHOT
FIXTURE_NAME = "AtlasCart (Synthetic M1)"
FIXTURE_MARKER = "[atlascart-m1-fixture]"
SPACE_KEY = "ATC"
DATASET_ID = "atlascart-m1"
DATASET_VERSION = "1.8.0"


def ts(day: int, hour: int, minute: int = 0) -> datetime:
    """Map a logical story day onto the rolling as-of timeline."""
    logical_date = date(2026, 3, 1) + timedelta(days=day - 1)
    event_date = logical_date + DATE_SHIFT
    if event_date.weekday() == 5:
        event_date += timedelta(days=2)
    elif event_date.weekday() == 6:
        event_date += timedelta(days=1)
    return datetime.combine(event_date, time(hour, minute), tzinfo=TZ)


def ds(day: int) -> str:
    logical_date = date(2026, 3, 1) + timedelta(days=day - 1)
    return (logical_date + DATE_SHIFT).isoformat()


if _configured_as_of:
    SNAPSHOT_AT = datetime.combine(AS_OF_DATE, time(12, 0), tzinfo=TZ)
else:
    SNAPSHOT_AT = datetime.now(TZ).replace(second=0, microsecond=0)


USERS = [
    {"id": "U01", "username": "atlascart.maya", "name": "Maya Chen", "email": "maya.chen@atlascart.example.test", "color": "#7C3AED", "role": "owner"},
    {"id": "U02", "username": "atlascart.liam", "name": "Liam Ortiz", "email": "liam.ortiz@atlascart.example.test", "color": "#2563EB", "role": "admin"},
    {"id": "U03", "username": "atlascart.emma", "name": "Emma Brooks", "email": "emma.brooks@atlascart.example.test", "color": "#0891B2", "role": "admin"},
    {"id": "U04", "username": "atlascart.noah", "name": "Noah Kim", "email": "noah.kim@atlascart.example.test", "color": "#059669", "role": "member"},
    {"id": "U05", "username": "atlascart.ava", "name": "Ava Patel", "email": "ava.patel@atlascart.example.test", "color": "#16A34A", "role": "member"},
    {"id": "U06", "username": "atlascart.ethan", "name": "Ethan Wright", "email": "ethan.wright@atlascart.example.test", "color": "#65A30D", "role": "member"},
    {"id": "U07", "username": "atlascart.sophia", "name": "Sophia Nguyen", "email": "sophia.nguyen@atlascart.example.test", "color": "#CA8A04", "role": "member"},
    {"id": "U08", "username": "atlascart.lucas", "name": "Lucas Martin", "email": "lucas.martin@atlascart.example.test", "color": "#EA580C", "role": "member"},
    {"id": "U09", "username": "atlascart.mia", "name": "Mia Johnson", "email": "mia.johnson@atlascart.example.test", "color": "#DC2626", "role": "member"},
    {"id": "U10", "username": "atlascart.daniel", "name": "Daniel Park", "email": "daniel.park@atlascart.example.test", "color": "#DB2777", "role": "member"},
    {"id": "U11", "username": "atlascart.olivia", "name": "Olivia Garcia", "email": "olivia.garcia@atlascart.example.test", "color": "#9333EA", "role": "member"},
    {"id": "U12", "username": "atlascart.jack", "name": "Jack Wilson", "email": "jack.wilson@atlascart.example.test", "color": "#4F46E5", "role": "member"},
]

USER_TIMEZONES = {
    "U01": "America/Los_Angeles", "U02": "America/New_York",
    "U03": "America/Los_Angeles", "U04": "America/Los_Angeles",
    "U05": "America/New_York", "U06": "America/Chicago",
    "U07": "America/Los_Angeles", "U08": "America/New_York",
    "U09": "America/New_York", "U10": "America/Los_Angeles",
    "U11": "America/Denver", "U12": "America/New_York",
}
for _user in USERS:
    _user["timezone"] = USER_TIMEZONES[_user["id"]]

USER_BY_ID = {user["id"]: user for user in USERS}

GROUPS = [
    {"name": "AtlasCart Product", "description": f"{FIXTURE_MARKER} Product, design, and beta feedback", "owner": "U01", "members": ["U01", "U02", "U09", "U12"]},
    {"name": "AtlasCart Engineering", "description": f"{FIXTURE_MARKER} Product delivery and operations", "owner": "U03", "members": ["U03", "U04", "U05", "U06", "U07", "U08", "U11"]},
    {"name": "AtlasCart Quality", "description": f"{FIXTURE_MARKER} Test coordination and beta verification", "owner": "U10", "members": ["U10", "U06", "U12"]},
]

SPRINTS = [
    {
        "id": "S1",
        "name": "AtlasCart Sprint 1",
        "goal": "Load a small seller catalog and let a shopper browse product details in staging.",
        "start_date": ds(2),
        "end_date": ds(13),
        "status": "completed",
        "order": 0,
        "created_at": ts(2, 8, 15),
        "updated_at": ts(13, 16, 45),
    },
    {
        "id": "S2",
        "name": "AtlasCart Sprint 2",
        "goal": "Let an invited shopper place a test order and let operations see what happened.",
        "start_date": ds(16),
        "end_date": ds(27),
        "status": "completed",
        "order": 1,
        "created_at": ts(12, 14, 0),
        "updated_at": ts(27, 16, 55),
    },
    {
        "id": "S3",
        "name": "AtlasCart Sprint 3",
        "goal": "Make beta orders easier to support and recover.",
        "start_date": ds(30),
        "end_date": ds(41),
        "status": "active",
        "order": 2,
        "created_at": ts(20, 11, 0),
        "updated_at": SNAPSHOT_AT,
    },
]

LABELS = [
    "catalog", "storefront", "cart", "checkout", "orders", "inventory",
    "frontend", "backend", "platform", "private-beta", "accessibility",
    "incident-follow-up",
]


def issue(
    title: str,
    issue_type: str,
    parent: str | None,
    sprint: str | None,
    status: str,
    priority: str,
    assignee: str | None,
    reporter: str,
    points: int | None,
    created_at: datetime,
    description: str,
    labels: list[str],
    *,
    started_at: datetime | None = None,
    done_at: datetime | None = None,
    initial_assignee: str | None = None,
    initial_points: int | None = None,
) -> dict:
    created_at = created_at.astimezone(ZoneInfo(USER_TIMEZONES[reporter]))
    work_owner = assignee or reporter
    started_at = started_at.astimezone(ZoneInfo(USER_TIMEZONES[work_owner])) if started_at else None
    done_at = done_at.astimezone(ZoneInfo(USER_TIMEZONES[work_owner])) if done_at else None
    return {
        "title": title,
        "type": issue_type,
        "parent": parent,
        "sprint": sprint,
        "status": status,
        "priority": priority,
        "assignee": assignee,
        "reporter": reporter,
        "points": points,
        "created_at": created_at,
        "description": description.strip(),
        "labels": labels,
        "started_at": started_at,
        "done_at": done_at,
        "initial_assignee": initial_assignee or assignee,
        "initial_points": points if initial_points is None else initial_points,
    }


ISSUES = [
    issue("Build the first AtlasCart storefront", "epic", None, None, "planned", "high", None, "U02", None, ts(2, 8, 30), "Give beta shoppers a simple storefront that works on desktop and mobile. Month 1 covers the product list, product detail page, cart entry points, and clear empty states.", ["storefront"]),
    issue("Create a reliable product catalog", "epic", None, None, "in_progress", "high", "U05", "U02", None, ts(2, 8, 35), "Import a small seller catalog and keep product names, prices, images, and stock understandable when supplier data is incomplete.", ["catalog", "backend"]),
    issue("Let shoppers complete checkout", "epic", None, None, "in_progress", "highest", "U07", "U01", None, ts(2, 8, 40), "Build the cart and checkout path for the private beta. The first version supports guest checkout, USD, one shipping option, and the payment sandbox.", ["cart", "checkout"]),
    issue("Give operations a usable order view", "epic", None, None, "in_progress", "high", "U06", "U02", None, ts(2, 8, 45), "Create orders from checkout and give customer operations enough information to understand and correct beta problems without editing the database.", ["orders"]),
    issue("Make staging safe to release", "epic", None, None, "in_progress", "medium", "U11", "U03", None, ts(2, 8, 50), "Keep CI, staging deploys, request logs, and rollback steps simple enough for the small team to use during the beta.", ["platform"]),
    issue("Keep the private beta testable", "epic", None, None, "in_progress", "high", "U10", "U02", None, ts(2, 8, 55), "Cover the main shopper flow, supported browsers, narrow screens, and the failures most likely to confuse a beta shopper.", ["private-beta"]),

    issue("Set up the web application shell", "task", "ATC-1", "S1", "done", "medium", "U07", "U03", 2, ts(2, 9, 0), "Create the React and TypeScript application shell with routing, shared layout, and a small API client. Keep the first version intentionally plain.", ["storefront", "frontend"], started_at=ts(2, 11), done_at=ts(3, 15, 20)),
    issue("Set up the API and database migrations", "task", "ATC-1", "S1", "done", "high", "U04", "U03", 3, ts(2, 9, 5), "Create the Spring Boot API shell, PostgreSQL connection, and first migrations. Add a health endpoint that staging can check.", ["backend", "platform"], started_at=ts(2, 10, 45), done_at=ts(4, 14, 10)),
    issue("Store products and current stock", "story", "ATC-2", "S1", "done", "high", "U05", "U02", 5, ts(2, 9, 10), "Store a product SKU, name, short description, price in USD, image reference, and one current stock count. Reject negative prices and stock values.", ["catalog", "inventory", "backend"], started_at=ts(3, 9, 15), done_at=ts(6, 15, 40)),
    issue("Import the supplier catalog CSV", "story", "ATC-2", "S2", "done", "high", "U05", "U02", 8, ts(2, 9, 15), "Import the beta supplier CSV and return useful row errors. The file may contain blank descriptions and missing images. Duplicate SKUs need an explicit rule instead of silent overwrites.", ["catalog", "backend"], started_at=ts(4, 9, 20), done_at=ts(18, 15, 30), initial_points=5),
    issue("Return products from the catalog API", "story", "ATC-2", "S1", "done", "high", "U05", "U02", 3, ts(2, 9, 20), "Return active products with price, stock, and image information. Support a simple text query but do not build advanced ranking yet.", ["catalog", "backend"], started_at=ts(6, 10), done_at=ts(9, 14, 45)),
    issue("Show the product list", "story", "ATC-1", "S1", "done", "high", "U08", "U02", 5, ts(2, 9, 25), "Show product cards with name, price, image, and availability. The page needs a loading state, an error state, and a useful empty result.", ["storefront", "catalog", "frontend"], started_at=ts(5, 9), done_at=ts(10, 16, 10)),
    issue("Add the product detail endpoint", "task", "ATC-2", "S1", "done", "medium", "U05", "U03", 3, ts(2, 9, 30), "Return one active product by SKU with its description, price, stock, and image reference. Missing and inactive products should return a clear not-found response.", ["catalog", "backend"], started_at=ts(9, 9, 10), done_at=ts(10, 14, 20)),
    issue("Show product details", "story", "ATC-1", "S1", "done", "high", "U08", "U02", 3, ts(2, 9, 35), "Let a shopper open a product page from the catalog. Show the current price and stock, and disable Add to cart when the product is unavailable.", ["storefront", "catalog", "frontend"], started_at=ts(10, 10), done_at=ts(12, 14, 40)),
    issue("Use a placeholder for missing product images", "story", "ATC-2", "S1", "done", "medium", "U08", "U09", 3, ts(2, 9, 40), "Use one neutral placeholder when the supplier file has no image or the image cannot load. Keep the product name visible and avoid a broken-image icon.", ["catalog", "storefront", "frontend"], started_at=ts(6, 13), done_at=ts(9, 16), initial_assignee="U06"),
    issue("Run checks for every pull request", "task", "ATC-5", "S1", "done", "medium", "U11", "U03", 3, ts(2, 9, 45), "Run backend tests, frontend checks, and build steps in GitHub Actions. A failed check should block merge.", ["platform"], started_at=ts(2, 12, 30), done_at=ts(5, 13, 20)),
    issue("Deploy the catalog build to staging", "task", "ATC-5", "S1", "done", "medium", "U11", "U03", 2, ts(2, 9, 50), "Deploy the web app and API to staging from the main branch. Record the image version and keep a short rollback command.", ["platform"], started_at=ts(9, 11), done_at=ts(11, 15, 15)),
    issue("Write the catalog smoke test", "task", "ATC-6", "S1", "done", "medium", "U10", "U02", 2, ts(2, 9, 55), "Check import, product list, product detail, missing images, invalid prices, and one narrow-screen layout before the catalog build is accepted.", ["catalog", "private-beta"], started_at=ts(9, 14), done_at=ts(12, 16, 20)),
    issue("Price wraps under the image on small screens", "bug", "ATC-6", "S1", "done", "medium", "U08", "U10", 2, ts(9, 10, 20), "At 375px wide, a long product name pushes the price under the image and over the stock label. The card should grow without overlapping either value.", ["storefront", "frontend", "accessibility"], started_at=ts(9, 13), done_at=ts(10, 12, 10)),
    issue("Reject catalog rows without a valid price", "bug", "ATC-2", "S1", "done", "high", "U05", "U10", 2, ts(10, 11, 5), "A blank price currently becomes 0.00 during import. Reject the row and include its line number so the seller can fix the file.", ["catalog", "backend"], started_at=ts(10, 13), done_at=ts(11, 14, 35)),
    issue("Missing image returns a raw browser error", "bug", "ATC-1", "S1", "done", "medium", "U08", "U10", 2, ts(11, 9, 40), "When an image URL returns 404, Chrome shows the broken-image icon before the placeholder appears. Swap to the placeholder without showing the raw error state.", ["catalog", "storefront", "frontend"], started_at=ts(11, 11), done_at=ts(12, 11, 50)),
    issue("Define how duplicate supplier SKUs are handled", "subtask", "ATC-10", "S2", "done", "high", "U02", "U05", None, ts(12, 10, 15), "Choose one clear import rule for duplicate SKU rows. For the beta, reject every conflicting duplicate and report both line numbers.", ["catalog"], started_at=ts(12, 11), done_at=ts(16, 11, 20)),

    issue("Create the cart API contract", "story", "ATC-3", "S2", "done", "high", "U04", "U02", 5, ts(12, 13, 0), "Define add, update, remove, and read operations for a guest cart. Return current product price and stock so the UI can explain changes.", ["cart", "backend"], started_at=ts(16, 11), done_at=ts(18, 14, 40)),
    issue("Add products to the cart", "story", "ATC-3", "S2", "done", "high", "U07", "U02", 5, ts(12, 13, 5), "Let a shopper add an available product from the list or detail page. Show a useful cart count and do not lose the current page.", ["cart", "frontend"], started_at=ts(17, 9), done_at=ts(20, 15, 10)),
    issue("Keep the cart after a page refresh", "story", "ATC-3", "S2", "done", "medium", "U07", "U02", 3, ts(12, 13, 10), "Keep the guest cart in browser storage and restore it after refresh. If stored data is invalid, start with an empty cart instead of breaking the page.", ["cart", "frontend"], started_at=ts(18, 9, 30), done_at=ts(20, 16, 30)),
    issue("Update and remove cart items", "story", "ATC-3", "S2", "done", "high", "U06", "U02", 3, ts(12, 13, 15), "Let a shopper change quantity or remove an item. Quantity zero removes the item, and totals update without a full page reload.", ["cart", "frontend", "backend"], started_at=ts(18, 10), done_at=ts(23, 11, 20)),
    issue("Check stock before checkout", "story", "ATC-3", "S3", "in_progress", "high", "U05", "U02", 5, ts(12, 13, 20), "Check current stock when checkout starts and again before the order is confirmed. Explain which item changed instead of returning a generic error.", ["cart", "inventory", "backend"], started_at=ts(23, 9), initial_points=3),
    issue("Collect the beta shipping address", "story", "ATC-3", "S2", "done", "high", "U08", "U02", 3, ts(12, 13, 25), "Collect name, street, city, state, postal code, and country for US beta orders. Validate required fields and do not log the address.", ["checkout", "frontend"], started_at=ts(19, 9), done_at=ts(23, 15, 20)),
    issue("Connect the payment sandbox", "story", "ATC-3", "S2", "done", "highest", "U04", "U02", 5, ts(12, 13, 30), "Create a small payment adapter for the provider sandbox. Store provider references and status, but never store card details.", ["checkout", "backend", "private-beta"], started_at=ts(18, 13), done_at=ts(23, 16, 15)),
    issue("Create one order from checkout", "story", "ATC-4", "S2", "done", "highest", "U04", "U02", 8, ts(12, 13, 35), "Create an order after the payment sandbox accepts the request. Save item price snapshots, shipping details, payment reference, and a readable order number. One checkout attempt must create one order.", ["checkout", "orders", "backend"], started_at=ts(19, 10), done_at=ts(27, 15, 20), initial_points=5),
    issue("Show the order confirmation", "story", "ATC-3", "S2", "done", "high", "U07", "U02", 3, ts(12, 13, 40), "After a successful order, show the order reference, item summary, shipping summary, and next step. Refreshing the page must not submit checkout again.", ["checkout", "orders", "frontend"], started_at=ts(23, 10), done_at=ts(25, 14, 30)),
    issue("Show recent orders to operations", "story", "ATC-4", "S3", "in_progress", "high", "U06", "U12", 5, ts(12, 13, 45), "Give customer operations a simple recent-order list with order reference, status, total, and created time. Hide full shipping details from the list view.", ["orders", "backend", "frontend"], started_at=ts(23, 11), initial_assignee="U04", initial_points=3),
    issue("Add request IDs to checkout logs", "task", "ATC-5", "S2", "done", "high", "U11", "U03", 3, ts(12, 13, 50), "Attach one request ID to checkout API logs and return it in error responses. Do not log cart contents or shipping details.", ["checkout", "platform"], started_at=ts(19, 10), done_at=ts(23, 13, 30)),
    issue("Deploy the private-beta build", "task", "ATC-5", "S2", "done", "highest", "U11", "U01", 2, ts(12, 13, 55), "Deploy the approved catalog, cart, checkout, and order changes to the private-beta environment. Record the version and complete the smoke test before sending invitations.", ["platform", "private-beta"], started_at=ts(24, 9), done_at=ts(25, 15, 40)),
    issue("Run the checkout test plan", "task", "ATC-6", "S3", "in_progress", "high", "U10", "U02", 3, ts(12, 14, 0), "Test the happy path, failed payment, repeated clicks, refresh behavior, out-of-stock items, and supported browsers. Use synthetic shopper details only.", ["checkout", "private-beta"], started_at=ts(23, 13)),
    issue("Explain a failed payment to the shopper", "story", "ATC-3", "S3", "planned", "medium", "U07", "U02", 3, ts(12, 14, 5), "Keep the cart and show a useful message when the payment sandbox rejects or times out. Do not show the provider's raw error.", ["checkout", "frontend"]),
    issue("Polish the empty cart on narrow screens", "bug", "ATC-6", "S3", "planned", "low", "U07", "U09", 2, ts(12, 14, 10), "On a narrow phone, the Browse products button sits below the visible area with too much empty space above it. Match the updated mobile note.", ["cart", "frontend", "accessibility"], initial_assignee="U08"),
    issue("Use a consistent beta order reference", "task", "ATC-4", None, "planned", "low", "U06", "U12", 2, ts(12, 14, 15), "Use the same order-reference format in confirmation, logs, and the operations view. This is useful, but it does not block the first beta order.", ["orders"]),
    issue("Cart controls overlap on a narrow screen", "bug", "ATC-6", "S3", "planned", "medium", "U07", "U10", 2, ts(16, 15, 30), "At 320px wide, the quantity controls overlap the Remove button when the product name uses two lines.", ["cart", "frontend", "accessibility"], initial_assignee="U08"),
    issue("Show the cart count in the header", "subtask", "ATC-24", "S2", "done", "medium", "U07", "U09", None, ts(13, 9, 0), "Update the header count after add, update, and remove actions. Hide the count when the cart is empty.", ["cart", "frontend"], started_at=ts(18, 11), done_at=ts(20, 12, 20)),
    issue("Add a payment adapter contract test", "subtask", "ATC-29", "S2", "done", "high", "U04", "U03", None, ts(13, 9, 5), "Cover accepted, rejected, and timed-out sandbox responses without making a real provider call in the test suite.", ["checkout", "backend"], started_at=ts(20, 10), done_at=ts(23, 14, 20)),
    issue("Prepare redacted beta order examples", "task", "ATC-6", None, "done", "medium", "U10", "U12", 2, ts(18, 9, 15), "Prepare synthetic order examples for QA and customer operations. Screenshots must not contain a real name, address, email, or payment detail.", ["orders", "private-beta"], started_at=ts(18, 10), done_at=ts(20, 14, 10)),

    issue("Checkout can create two orders from a double click", "bug", "ATC-4", "S3", "in_progress", "highest", "U04", "U12", 5, ts(26, 9, 12), "During the private beta, a slow checkout allowed two Place order clicks. Two order records were created and stock was reduced twice. Only one sandbox payment succeeded. We need a safe mitigation and follow-up fixes.", ["checkout", "orders", "private-beta", "incident-follow-up"], started_at=ts(26, 10, 5), initial_points=3),
    issue("Reject repeated checkout requests", "subtask", "ATC-43", "S2", "done", "highest", "U04", "U03", None, ts(26, 11, 20), "Send an idempotency key with checkout and return the first result when the same key is submitted again. The check must happen before a second order insert.", ["checkout", "orders", "backend", "incident-follow-up"], started_at=ts(26, 11, 35), done_at=ts(27, 13, 10)),
    issue("Disable Place order while checkout is pending", "subtask", "ATC-43", "S2", "done", "highest", "U07", "U03", None, ts(26, 11, 28), "Disable the Place order button as soon as submission starts and keep it disabled until the request succeeds or fails. This reduces repeat clicks but does not replace the API fix.", ["checkout", "frontend", "incident-follow-up"], started_at=ts(26, 11, 45), done_at=ts(26, 16, 20)),
    issue("Restore stock after a duplicate beta order", "bug", "ATC-4", "S3", "in_progress", "high", "U05", "U12", 3, ts(26, 14, 20), "The duplicate order reduced stock twice. Add a safe operations action for this beta case and record why the correction was made.", ["orders", "inventory", "incident-follow-up"], started_at=ts(32, 9, 20)),
    issue("Cover slow checkout in regression tests", "subtask", "ATC-43", "S3", "in_progress", "high", "U10", "U03", None, ts(26, 14, 35), "Add a repeatable slow-response test that clicks once and twice. Verify one order, one stock reduction, and one visible confirmation.", ["checkout", "private-beta", "incident-follow-up"], started_at=ts(27, 9, 10)),
    issue("Checkout accepts more items than current stock", "bug", "ATC-3", None, "done", "medium", "U05", "U10", 2, ts(27, 10, 25), "The report describes checkout accepting a quantity above current stock. This is already covered by ATC-27, including the second stock check before confirmation.", ["cart", "inventory", "backend"]),
]

for number, item in enumerate(ISSUES, start=1):
    item["key"] = f"ATC-{number}"

ISSUE_BY_KEY = {item["key"]: item for item in ISSUES}

# Reviewed with the M1 sizing rubric: 1 = tiny/local, 2 = small and isolated,
# 3 = normal single-area work, 5 = cross-layer/risky, 8 = large or uncertain.
# Keeping this map explicit makes every estimate reviewable instead of random.
REVIEWED_POINT_OVERRIDES = {
    "ATC-7": 3, "ATC-8": 5, "ATC-15": 2, "ATC-17": 3,
    "ATC-21": 1, "ATC-23": 3, "ATC-24": 3, "ATC-26": 5,
    "ATC-28": 5, "ATC-37": 1, "ATC-38": 3, "ATC-42": 1,
    "ATC-46": 5, "ATC-48": 1,
}
for key, reviewed_points in REVIEWED_POINT_OVERRIDES.items():
    ISSUE_BY_KEY[key]["points"] = reviewed_points
    ISSUE_BY_KEY[key]["initial_points"] = reviewed_points

# The full planning analysis is stored in each issue description as well as the audit report.
POINT_RATIONALES = DETAILED_POINT_RATIONALES

assert set(ACCEPTANCE_CRITERIA) == set(ISSUE_BY_KEY)
assert set(UNESTIMATED_RATIONALES) == set(ISSUE_BY_KEY) - set(POINT_RATIONALES)
for item in ISSUES:
    key = item["key"]
    item["context"] = item["description"]
    sections = ["Context", item["context"]]
    if key in BUG_REPORT_DETAILS:
        report = BUG_REPORT_DETAILS[key]
        sections.extend([
            "", "Bug example",
            f"When / environment: {report['when']}",
            f"Steps to reproduce: {report['steps']}",
            f"Expected behavior: {report['expected']}",
            f"Actual behavior: {report['actual']}",
        ])
    sections.extend(["", "Acceptance criteria"])
    sections.extend(f"AC{index}: {criterion}" for index, criterion in enumerate(ACCEPTANCE_CRITERIA[key], start=1))
    if key in POINT_RATIONALES:
        item["estimate_rationale"] = POINT_RATIONALES[key]
    else:
        item["estimate_rationale"] = UNESTIMATED_RATIONALES[key]
    sections.extend(["", "Estimate reasoning", item["estimate_rationale"]])
    item["acceptance_criteria"] = list(ACCEPTANCE_CRITERIA[key])
    item["description"] = "\n".join(sections)

# Confirmation was accepted after the first ATC-30 completion, before the
# later incident reopened order creation for idempotency work.
ISSUE_BY_KEY["ATC-31"]["done_at"] = ts(25, 15, 10)


S1_INITIAL = [f"ATC-{n}" for n in range(7, 19)]
S1_ADDED = ["ATC-19", "ATC-20", "ATC-21", "ATC-22"]
S1_CARRIED = ["ATC-10", "ATC-22"]

S2_INITIAL = ["ATC-10", "ATC-22"] + [f"ATC-{n}" for n in range(23, 39)] + ["ATC-40", "ATC-41"]
S2_ADDED = ["ATC-39", "ATC-43", "ATC-44", "ATC-45", "ATC-47"]
S2_CARRIED = ["ATC-27", "ATC-32", "ATC-35", "ATC-43", "ATC-47"]
S2_CARRY_REASONS = {
    "ATC-27": "The two-cart check passed, but timeout and lock-release coverage was not finished.",
    "ATC-32": "Operations could see recent orders, but payment-state clarity still needed product review.",
    "ATC-35": "The normal checkout matrix passed, while slow-response and last-item cases remained open.",
    "ATC-43": "The immediate fixes shipped, but the incident stayed open for stock recovery and browser regression.",
    "ATC-47": "The API repeat test passed, but the slow-browser regression was not complete.",
}
S2_REMOVED = {
    "ATC-36": ("S3", ts(26, 15, 35), "Moved to Sprint 3 to make room for incident work"),
    "ATC-37": ("S3", ts(26, 15, 38), "Moved to Sprint 3 to make room for incident work"),
    "ATC-38": (None, ts(26, 15, 40), "Returned to the backlog because it does not block beta recovery"),
    "ATC-39": ("S3", ts(20, 10, 30), "Moved after Lucas's planned day off reduced frontend capacity"),
}
S3_INITIAL = ["ATC-27", "ATC-32", "ATC-35", "ATC-36", "ATC-37", "ATC-39", "ATC-43", "ATC-46", "ATC-47"]

ESTIMATE_CHANGES = {
    "ATC-10": (5, 8, ts(12, 11, 35), "Duplicate SKU handling makes the import larger than the first estimate."),
    "ATC-27": (3, 5, ts(16, 10, 5), "The second stock check needs transaction-level coverage."),
    "ATC-30": (5, 8, ts(16, 10, 12), "Order snapshots and safe retry behavior add backend work."),
    "ATC-32": (3, 5, ts(16, 10, 18), "The operations view needs both API and UI changes."),
    "ATC-43": (3, 5, ts(26, 10, 50), "The incident needs mitigation, investigation, and verification."),
}

ASSIGNEE_CHANGES = {
    "ATC-15": ("U06", "U08", ts(6, 11, 30), "Lucas can finish the browser fallback with the catalog UI work."),
    "ATC-32": ("U04", "U06", ts(16, 10, 20), "Ethan will own the cross-stack operations view."),
    "ATC-37": ("U08", "U07", ts(20, 10, 25), "Sophia will pick this up after the incident work settles."),
    "ATC-39": ("U08", "U07", ts(20, 10, 28), "Lucas is away today, so Sophia will take the follow-up next sprint."),
}

# ATC-30 was complete before the beta report, then reopened and fixed again.
STATUS_OVERRIDES = {
    "ATC-30": [
        (ts(19, 10), "planned", "in_progress"),
        (ts(25, 14, 50), "in_progress", "done"),
        (ts(26, 10, 20), "done", "in_progress"),
        (ts(27, 15, 20), "in_progress", "done"),
    ],
    "ATC-48": [
        (ts(27, 11, 15), "planned", "done"),
    ],
}

ISSUE_LINKS = [
    ("ATC-8", "ATC-9", "blocks", ts(2, 10, 20)),
    ("ATC-9", "ATC-10", "blocks", ts(2, 10, 25)),
    ("ATC-9", "ATC-11", "blocks", ts(2, 10, 30)),
    ("ATC-11", "ATC-12", "blocks", ts(2, 10, 35)),
    ("ATC-13", "ATC-14", "blocks", ts(2, 10, 40)),
    ("ATC-15", "ATC-12", "relates to", ts(6, 11, 45)),
    ("ATC-15", "ATC-14", "relates to", ts(10, 9, 20)),
    ("ATC-22", "ATC-10", "blocks", ts(12, 10, 30)),
    ("ATC-23", "ATC-24", "blocks", ts(12, 14, 25)),
    ("ATC-23", "ATC-26", "blocks", ts(12, 14, 30)),
    ("ATC-24", "ATC-25", "relates to", ts(12, 14, 35)),
    ("ATC-27", "ATC-30", "relates to", ts(16, 10, 25)),
    ("ATC-28", "ATC-30", "blocks", ts(16, 10, 30)),
    ("ATC-29", "ATC-30", "blocks", ts(16, 10, 35)),
    ("ATC-30", "ATC-31", "blocks", ts(16, 10, 40)),
    ("ATC-30", "ATC-32", "blocks", ts(16, 10, 45)),
    ("ATC-33", "ATC-34", "blocks", ts(19, 9, 30)),
    ("ATC-43", "ATC-30", "relates to", ts(26, 10, 25)),
    ("ATC-44", "ATC-43", "blocks", ts(26, 11, 25)),
    ("ATC-45", "ATC-43", "blocks", ts(26, 11, 32)),
    ("ATC-43", "ATC-46", "relates to", ts(26, 14, 25)),
    ("ATC-43", "ATC-47", "relates to", ts(26, 14, 40)),
    ("ATC-48", "ATC-27", "duplicates", ts(27, 11, 10)),
]


CURATED_ATTACHMENTS = [
    {
        "issue": "ATC-10", "uploader": "U05", "filename": "supplier-catalog-sample.csv",
        "content_type": "text/csv", "kind": "csv", "placement": "panel", "created_at": ts(4, 11, 15),
        "purpose": "Original supplier sample used to reproduce blank prices and conflicting duplicate SKUs before importer work began.",
        "content": "sku,name,description,price,currency,stock,image_url,active\nA-104,Desk Lamp,Adjustable warm desk lamp,39.00,USD,12,,true\nA-104,Desk Lamp,Same supplier SKU with a conflicting price,41.00,USD,12,,true\nB-208,Oak Tray,Small solid-oak desk tray,,USD,8,https://assets.example.test/b-208.jpg,true\nC-311,Monitor Stand,Adjustable bamboo monitor stand,74.50,USD,4,https://assets.example.test/c-311.jpg,true\nD-402,Cable Clips,Pack of six silicone clips,-4.00,USD,25,https://assets.example.test/d-402.jpg,true\nE-510,Felt Desk Pad,,28.00,USD,0,,true\nF-612,Travel Mug,Double-wall steel mug,not_available,USD,9,https://assets.example.test/f-612.jpg,true\n",
    },
    {
        "issue": "ATC-12", "uploader": "U09", "filename": "product-list-requirements.pdf",
        "content_type": "application/pdf", "kind": "pdf", "placement": "panel", "created_at": ts(5, 10, 20),
        "purpose": "Product and design requirements supplied at the start of ATC-12 to define the list layout, states, and acceptance boundary.",
        "title": "Beta product list requirements",
        "sections": [
            {"heading": "1. Customer problem and scope", "lines": [
                "Prepared after the product review with Maya Chen and Jack Wilson.",
                "Beta shoppers need to decide whether a product is relevant without opening every detail page.",
                "The first release serves a small seller catalog in USD and does not include recommendations, reviews, wish lists, or advanced ranking.",
                "The list must remain understandable when a supplier omits an image or description.",
                "Desktop target: four cards per row when space allows. Mobile target: one card per row.",
                "Primary users: invited beta shoppers. Secondary users: QA and customer operations reproducing shopper reports.",
            ]},
            {"heading": "2. Functional and content requirements", "lines": [
                "Each card shows product name, USD price, current availability, and product image or neutral placeholder.",
                "Price and stock remain visible without opening the product detail page.",
                "Long names may wrap to two lines and must not overlap price, stock, or the next card.",
                "Selecting a card opens the matching product detail route.",
                "Loading uses a stable card skeleton so the page does not jump when results arrive.",
                "API failure shows a plain-language retry message and keeps the page navigation usable.",
                "An empty query result explains that no products matched and offers a way to show all products.",
                "Out-of-stock products remain visible but clearly say Out of stock.",
            ]},
            {"heading": "3. Acceptance, accessibility, and exclusions", "lines": [
                "Accept with 0, 1, 12, and 50 synthetic fixture products.",
                "Verify desktop Chrome and Firefox plus a 375px mobile viewport.",
                "Product names are real text with a useful heading level; important state is not conveyed only by color.",
                "Keyboard focus reaches each card in reading order and remains visible.",
                "A missing or failed image never removes the product name.",
                "Known follow-up: ATC-19 covers the long-name card collision discovered during QA.",
                "Excluded from this issue: search ranking, filters, pagination, seller editing, recommendations, and analytics.",
                "Product owner approval means these checks pass in staging with synthetic data.",
            ], "table": {
                "headers": ["Scenario", "Viewport", "Expected", "Owner"],
                "rows": [
                    ["Empty catalog", "Desktop", "Helpful empty state", "Product"],
                    ["Long name", "375px", "No overlap", "Frontend"],
                    ["Image 404", "375px", "Placeholder", "Frontend"],
                    ["API failure", "Desktop", "Retry remains usable", "QA"],
                ],
            }},
        ],
    },
    {
        "issue": "ATC-7", "uploader": "U10", "filename": "web-shell-staging.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(3, 15, 16),
        "purpose": "Final visual verification attached at close, showing the shared header, routed catalog content, and readable API-unavailable banner in staging.",
        "caption": "AtlasCart staging shell after implementation: shared header, routed content, and readable error boundary.",
        "screenshot_type": "web_shell",
    },
    {
        "issue": "ATC-12", "uploader": "U10", "filename": "product-list-staging.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(10, 16, 6),
        "purpose": "Final visual verification attached at close, showing the implemented mobile product list against the requirement PDF and its readable stock states.",
        "caption": "Product list in staging at 375px after implementation, with readable price and stock states.",
        "screenshot_type": "product_list_fixed",
    },
    {
        "issue": "ATC-13", "uploader": "U10", "filename": "product-detail-api-contract.md",
        "content_type": "text/markdown", "kind": "markdown", "placement": "comment", "created_at": ts(10, 14, 16),
        "purpose": "Backend verification evidence attached at close; request and response examples prove the active, unknown, and inactive SKU contract without pretending an API task needs a UI screenshot.",
        "content": "# Product detail endpoint verification\n\n## Why this evidence exists\n\nATC-13 is a backend endpoint task. The useful proof is the HTTP contract used by the web client, not a browser screenshot. All values below are synthetic.\n\n## Active SKU\n\n```http\nGET /api/products/C-311\n200 OK\nContent-Type: application/json\n\n{\"sku\":\"C-311\",\"name\":\"Adjustable Bamboo Monitor Stand\",\"price\":74.50,\"currency\":\"USD\",\"stock\":4,\"active\":true}\n```\n\nThe response contains the description, current price, stock, and image reference needed by ATC-14.\n\n## Unknown SKU\n\n```http\nGET /api/products/NO-SUCH-SKU\n404 Not Found\n\n{\"code\":\"PRODUCT_NOT_FOUND\",\"message\":\"Product not found\"}\n```\n\n## Inactive SKU\n\n```http\nGET /api/products/OLD-901\n404 Not Found\n\n{\"code\":\"PRODUCT_NOT_FOUND\",\"message\":\"Product not found\"}\n```\n\nThe unknown and inactive cases intentionally return the same safe shape.\n\n## Acceptance result\n\n- AC1: PASS — the active product returns all agreed fields.\n- AC2: PASS — unknown and inactive SKUs use the same 404 response.\n- AC3: PASS — the response does not reveal whether a seller removed the product.\n\n## Reproduction\n\n```text\nbuild=atlascart-api:catalog-s1-27\nenvironment=staging\nfixture=synthetic-catalog-v3\n```\n",
    },
    {
        "issue": "ATC-14", "uploader": "U09", "filename": "product-detail-current.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(2, 9, 37),
        "purpose": "Current-state screenshot supplied during planning, showing the plain staging response before the shopper-facing product detail layout is implemented.",
        "caption": "Current staging state before ATC-14: product data is present but not arranged for shoppers.",
        "screenshot_type": "product_detail_current",
    },
    {
        "issue": "ATC-14", "uploader": "U09", "filename": "product-detail-expected.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(2, 9, 38),
        "purpose": "Expected-state design supplied during planning, showing the agreed mobile order, stock message, and disabled Add to cart behavior.",
        "caption": "Expected ATC-14 mobile layout: image, name, price, stock message, then Add to cart.",
        "screenshot_type": "product_detail_expected",
    },
    {
        "issue": "ATC-14", "uploader": "U10", "filename": "product-detail-fixed.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(12, 14, 36),
        "purpose": "Final visual verification attached at close, showing the implemented staging page matches the expected layout and explains the zero-stock disabled button.",
        "caption": "ATC-14 staging verification: implemented product detail layout with clear zero-stock behavior.",
        "screenshot_type": "product_detail_fixed",
    },
    {
        "issue": "ATC-19", "uploader": "U10", "filename": "narrow-product-card.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(9, 10, 22),
        "purpose": "Before screenshot supplied with the original bug report, showing the exact 375px price-and-stock collision the engineer must fix.",
        "caption": "Staging at 375px: price overlaps the stock label on the second product card.",
        "screenshot_type": "bug_ui",
    },
    {
        "issue": "ATC-19", "uploader": "U10", "filename": "narrow-product-card-fixed.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(10, 12, 6),
        "purpose": "After screenshot attached to the close verification, showing the long name, price, and stock on separate non-overlapping lines at 375px.",
        "caption": "Staging after the fix at 375px: the long product card has no overlap.",
        "screenshot_type": "bug_ui_fixed",
    },
    {
        "issue": "ATC-16", "uploader": "U11", "filename": "ci-pipeline-run-184.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(5, 13, 5),
        "purpose": "Final CI evidence attached to the completion comment to show each required check and the merge gate passed.",
        "caption": "GitHub Actions run 184: backend tests, frontend checks, image build, and merge gate passed.",
        "screenshot_type": "ci_pipeline",
    },
    {
        "issue": "ATC-21", "uploader": "U10", "filename": "missing-image-flash.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(11, 9, 45),
        "purpose": "Before screenshot supplied with the bug report, captured under Slow 3G to show Chrome's broken-image icon before the AtlasCart placeholder appears.",
        "caption": "Chrome Slow 3G: native broken-image icon flashes before the AtlasCart placeholder.",
        "screenshot_type": "missing_image",
    },
    {
        "issue": "ATC-21", "uploader": "U10", "filename": "missing-image-fixed.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(12, 11, 46),
        "purpose": "After screenshot attached to the close verification, proving the neutral placeholder appears directly under the same Chrome Slow 3G setup.",
        "caption": "Chrome Slow 3G after the fix: the neutral placeholder appears without a broken icon flash.",
        "screenshot_type": "missing_image_fixed",
    },
    {
        "issue": "ATC-20", "uploader": "U10", "filename": "invalid-price-reproduction.csv",
        "content_type": "text/csv", "kind": "csv", "placement": "comment", "created_at": ts(10, 11, 8),
        "purpose": "Small reproduction file attached to QA's bug report to show the blank, negative, and non-numeric price cases the importer must reject.",
        "content": "test_case,source_row,sku,input_price,expected,actual_before_fix\nblank price,18,B-208,,reject row 18 and save nothing,imported as USD 0.00\nnegative price,19,D-402,-4.00,reject row 19 and save nothing,rejected without row number\ntext price,20,F-612,not_available,reject row 20 and save nothing,generic parse error\nvalid decimal,21,C-311,74.50,accept row at USD 74.50,accepted\nzero price,22,G-701,0.00,accept row at USD 0.00,accepted\n",
    },
    {
        "issue": "ATC-37", "uploader": "U09", "filename": "empty-cart-mobile-spacing.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(19, 15, 12),
        "purpose": "Current-state screenshot supplied with the design issue, showing the oversized empty area that pushes Browse products below the initial 320px viewport.",
        "caption": "Private beta empty cart at 320px: the primary action begins below the visible area.",
        "screenshot_type": "empty_cart",
    },
    {
        "issue": "ATC-39", "uploader": "U10", "filename": "cart-controls-overlap-320px.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "description", "created_at": ts(16, 15, 32),
        "purpose": "Before screenshot supplied with the bug report, showing the quantity increment control overlapping Remove at the exact viewport and quantity in the report.",
        "caption": "Private beta cart at 320px: quantity 10 makes the increment control overlap Remove.",
        "screenshot_type": "cart_controls",
    },
    {
        "issue": "ATC-27", "uploader": "U10", "filename": "stock-concurrency-results.xlsx",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "kind": "xlsx", "placement": "comment", "created_at": ts(34, 13, 20),
        "purpose": "QA execution workbook attached to the active verification update; it records each two-cart run and lock-release result.",
        "sheet": "Concurrency results",
        "rows": [
            ["Run", "Build", "Database", "Starting stock", "Cart A quantity", "Cart B quantity", "First result", "Second result", "Orders created", "Ending stock", "Lock released (ms)", "Result"],
            [1, "beta-1.0.2", "postgres-15", 1, 1, 1, "201 BETA-1061", "409 SKU_CHANGED", 1, 0, 42, "pass"],
            [2, "beta-1.0.2", "postgres-15", 2, 2, 1, "201 BETA-1062", "409 SKU_CHANGED", 1, 0, 47, "pass"],
            [3, "beta-1.0.2", "postgres-15", 3, 2, 2, "201 BETA-1063", "409 SKU_CHANGED", 1, 1, 44, "pass"],
            [4, "beta-1.0.2", "postgres-15", 1, 1, 1, "client timeout", "201 BETA-1064", 1, 0, 138, "pass"],
            [5, "beta-1.0.2", "postgres-15", 1, 1, 1, "payment rejected", "201 BETA-1065", 1, 0, 51, "pass"],
        ],
    },
    {
        "issue": "ATC-30", "uploader": "U04", "filename": "checkout-idempotency-architecture.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(24, 14, 20),
        "purpose": "Implementation-review diagram showing where the checkout key is checked before order insert and stock change.",
        "caption": "Implementation diagram reviewed for checkout idempotency and one-order creation.",
        "screenshot_type": "architecture",
    },
    {
        "issue": "ATC-35", "uploader": "U10", "filename": "checkout-verification-results.csv",
        "content_type": "text/csv", "kind": "csv", "placement": "comment", "created_at": ts(33, 15, 35),
        "purpose": "Current checkout execution matrix attached to QA's status update, including the open Safari failure rather than an all-green summary.",
        "content": "Test case,Build,Environment,Browser,Viewport,Starting condition,Action,Expected behavior,Observed behavior,Result,Tester\nCHK-001,beta-1.0.2,private-beta,Chrome 126,1440x900,one in-stock item and accepted sandbox card,click Place order once,one order and one stock reduction,one order BETA-1054 and stock minus one,PASS,Daniel Park\nCHK-002,beta-1.0.2,private-beta,Firefox 127,1440x900,rejected sandbox card,submit checkout,cart remains and provider detail stays hidden,cart preserved and plain rejection message shown,PASS,Daniel Park\nCHK-003,beta-1.0.2,private-beta,Safari 17.5,390x844,slow response followed by browser Back,return to checkout,pending state resets and retry is available,button remains disabled after page restore,FAIL,Daniel Park\nCHK-004,beta-1.0.2,private-beta,Chrome 126,375x812,last unit bought by another cart,confirm checkout,changed SKU is named and no order is created,SKU C-311 named and no order created,PASS,Daniel Park\nCHK-005,beta-1.0.2,private-beta,Chrome 126,1440x900,same checkout idempotency key,submit five times,one order and one stock reduction,one BETA order and stock minus one,PASS,Daniel Park\nCHK-006,beta-1.0.2,private-beta,Firefox 127,1440x900,confirmed order exists,refresh confirmation,no checkout resubmit,same reference shown and no new request,PASS,Daniel Park\nCHK-007,beta-1.0.2,private-beta,Safari 17.5,390x844,keyboard-only navigation,activate checkout controls,focus visible and only one request,focus visible and one request,PASS,Daniel Park\n",
    },
    {
        "issue": "ATC-36", "uploader": "U12", "filename": "beta-checkout-feedback.pdf",
        "content_type": "application/pdf", "kind": "pdf", "placement": "comment", "created_at": ts(30, 10, 10),
        "purpose": "Redacted customer-operations source document explaining why failed-payment wording and recovery behavior are needed.",
        "title": "Private beta checkout feedback",
        "sections": [
            {"heading": "1. Request summary", "lines": [
                "Source: private beta feedback summarized by customer operations.",
                "Audience: product, checkout engineering, and QA.",
                "Privacy: shopper names, emails, addresses, and payment details were removed before this document was attached.",
                "Three shoppers understood a rejected payment and tried again.",
                "Two shoppers saw a timeout message and were unsure whether the order existed or whether their cart was still saved.",
                "Customer request: keep the cart, avoid provider language, and explain the next safe action.",
            ]},
            {"heading": "2. Observed sessions", "lines": [
                "Session A: rejected sandbox payment. Shopper read the message, changed the test card, and completed checkout.",
                "Session B: simulated timeout. Shopper returned to the cart because the page did not say whether items were preserved.",
                "Session C: simulated timeout on mobile. Shopper pressed Back and then wondered whether retrying would create a duplicate.",
                "Session D: rejected payment in Firefox. Cart remained in memory, but the raw provider phrase looked like a system failure.",
                "No session in this report used real payment credentials or real customer information.",
                "Related risk: the duplicate-order incident makes a clear retry instruction especially important.",
            ], "table": {
                "headers": ["Session", "Browser", "Condition", "Outcome"],
                "rows": [
                    ["A", "Chrome", "Rejected card", "Recovered"],
                    ["B", "Chrome", "Timeout", "Unsure"],
                    ["C", "Safari mobile", "Timeout + Back", "Unsafe retry concern"],
                    ["D", "Firefox", "Rejected card", "Raw wording confused"],
                ],
            }},
            {"heading": "3. Requested behavior", "lines": [
                "Rejected payment: say the payment did not complete and invite the shopper to try another method.",
                "Timeout: say the result is still being checked and prevent a second submit while that check runs.",
                "Failure paths keep all valid cart items and the entered shipping form in the current browser session.",
                "Never display the provider response body, provider status code, or internal request exception.",
                "If an existing order is found for the checkout attempt, open that confirmation instead of asking for another payment.",
                "The message must work at 375px and remain understandable without color alone.",
            ]},
            {"heading": "4. Acceptance and follow-up", "lines": [
                "QA verifies rejected, timed-out, and recovered responses in Chrome, Firefox, and Safari.",
                "After a failure, the cart contains the same SKU, quantity, and price snapshot shown before submission.",
                "Repeated activation while pending creates no second browser request.",
                "Refresh after a confirmed order shows the existing order and does not submit checkout again.",
                "Logging contains a request ID but no address, cart body, payment detail, or provider error body.",
                "ATC-47 owns the remaining slow-response browser regression. This request does not replace that work.",
                "Product acceptance requires plain wording reviewed by customer operations.",
            ]},
        ],
    },
    {
        "issue": "ATC-43", "uploader": "U12", "filename": "duplicate-orders.png",
        "content_type": "image/png", "kind": "screenshot", "placement": "comment", "created_at": ts(26, 9, 18),
        "purpose": "Redacted incident evidence showing the two order rows created from the single shopper attempt described by support.",
        "caption": "Redacted operations view showing BETA-1042 and BETA-1043 for one checkout attempt.",
        "screenshot_type": "order_incident",
    },
    {
        "issue": "ATC-43", "uploader": "U11", "filename": "duplicate-checkout-requests.log",
        "content_type": "text/plain", "kind": "log", "placement": "panel", "created_at": ts(27, 11, 40),
        "purpose": "Redacted request log used to prove two browser requests reached the API and that the gateway did not retry either request.",
        "content": "2026-03-26T09:41:58.210-07:00 INFO gateway request_id=req-7f21 method=POST path=/api/checkout status=201 upstream_ms=1842 retry_count=0\n2026-03-26T09:41:58.244-07:00 INFO order-api request_id=req-7f21 checkout_key=chk-a91 result=created order=BETA-1042 stock_change=-1\n2026-03-26T09:41:58.391-07:00 INFO gateway request_id=req-7f22 method=POST path=/api/checkout status=201 upstream_ms=1768 retry_count=0\n2026-03-26T09:41:58.427-07:00 WARN order-api request_id=req-7f22 checkout_key=chk-a91 result=created order=BETA-1043 stock_change=-1 duplicate_key_check=missing\n2026-03-26T10:18:03.006-07:00 INFO investigation conclusion=\"two client requests; no gateway retry\" pii_redacted=true\n",
    },
    {
        "issue": "ATC-46", "uploader": "U06", "filename": "inventory-correction-requirement.pdf",
        "content_type": "application/pdf", "kind": "pdf", "placement": "comment", "created_at": ts(32, 11, 15),
        "purpose": "Refinement report attached after engineering and support agreed on the safe correction workflow, audit fields, and exclusions.",
        "title": "Inventory correction requirement",
        "sections": [
            {"heading": "1. Problem and operating context", "lines": [
                "The duplicate checkout incident created two order records and reduced stock twice for one intended purchase.",
                "Customer operations can identify the duplicate, but today an engineer must edit fixture stock directly.",
                "The beta needs a narrow correction action that is safe enough to explain during support review.",
                "This is not a general inventory-management feature and does not replace the stock transaction in checkout.",
                "Primary actor: an authorized operations member. Reviewer: engineering or QA during the private beta.",
            ]},
            {"heading": "2. Correction workflow", "lines": [
                "Operations opens the affected order and selects Restore stock.",
                "The form requires the duplicate or cancelled order reference, affected SKU, quantity, and a plain-language reason.",
                "The system shows the current stock count and the proposed count before confirmation.",
                "Quantity must be positive and cannot exceed the quantity recorded on that order line.",
                "The action refuses an order that is not cancelled or identified as the duplicate recovery case.",
                "Submitting the same correction twice returns the existing result and does not increment stock again.",
            ]},
            {"heading": "3. Audit, access, and failure behavior", "lines": [
                "Each successful correction records order reference, SKU, old count, new count, quantity, reason, actor, and timestamp.",
                "The audit entry never stores a shopper address, email, payment detail, or full provider payload.",
                "A failed database transaction changes neither stock nor the audit record.",
                "An unauthorized member receives a clear forbidden response with no stock detail beyond the existing order view.",
                "Concurrent checkout and correction operations use the same stock consistency boundary.",
                "Application logs include request ID and order reference but no customer personal data.",
            ]},
            {"heading": "4. Acceptance criteria and exclusions", "lines": [
                "One valid correction restores exactly the duplicate order quantity and creates one audit record.",
                "A repeated correction request changes nothing and returns the first correction reference.",
                "Invalid quantity, unknown order, non-cancelled order, and missing reason are rejected.",
                "Operations can read the correction result without database access.",
                "QA verifies old and new stock values plus the actor and reason in the audit view.",
                "Excluded: bulk corrections, warehouse-level stock, supplier reconciliation, general returns, and automated refunds.",
                "The manual fixture correction mentioned in the incident comment remains historical context only.",
            ], "table": {
                "headers": ["Field", "Required", "Example", "Stored in audit"],
                "rows": [
                    ["Order reference", "Yes", "BETA-1043", "Yes"],
                    ["SKU", "Yes", "A-104", "Yes"],
                    ["Quantity", "Yes", "1", "Yes"],
                    ["Reason", "Yes", "Duplicate recovery", "Yes"],
                    ["Address", "No", "Not collected", "No"],
                ],
            }},
        ],
    },
]


REPOSITORIES = [
    ("northstar-labs-sim", "atlascart-web"),
    ("northstar-labs-sim", "atlascart-api"),
    ("northstar-labs-sim", "atlascart-infra"),
]
