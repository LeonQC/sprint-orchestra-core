"""Per-AC before/after PNG evidence for frontend issues."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any


@dataclass(frozen=True)
class AcVisualShot:
    ac_index: int
    before_filename: str
    before_type: str
    before_caption: str
    before_purpose: str
    before_context: str
    after_filename: str | None = None
    after_type: str | None = None
    after_caption: str | None = None
    after_purpose: str | None = None
    after_context: str | None = None


# Frontend-labelled issues use one description PNG per AC (current problem) and,
# when Done, one verification PNG per AC (fix verified).
FRONTEND_VISUAL_EVIDENCE: dict[str, list[AcVisualShot]] = {
    "ATC-7": [
        AcVisualShot(
            1,
            "web-shell-ac1-before.png",
            "frontend_evidence",
            "Staging before ATC-7: no shared header, layout shell, or readable API error boundary.",
            "Before screenshot supplied during planning, showing the app booted without the shared shell or error boundary required by AC1.",
            "WEB SHELL / AC1 / BEFORE",
            "web-shell-ac1-after.png",
            "web_shell",
            "Staging after ATC-7 AC1: shared header, routed content area, and readable API-unavailable banner.",
            "After screenshot attached at close for AC1, showing the shared header, layout shell, and error boundary in staging.",
            "WEB SHELL / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "web-shell-ac2-before.png",
            "frontend_evidence",
            "Direct refresh on /products returned a blank page before route handling was wired.",
            "Before screenshot supplied during planning, showing catalog and product-detail routes failing on direct browser refresh (AC2).",
            "WEB SHELL / AC2 / BEFORE",
            "web-shell-ac2-after.png",
            "frontend_evidence",
            "Direct refresh on /products and /products/C-311 now renders after route handling was added.",
            "After screenshot attached at close for AC2, showing catalog and product-detail routes survive direct refresh.",
            "WEB SHELL / AC2 / AFTER",
        ),
    ],
    "ATC-12": [
        AcVisualShot(
            1,
            "product-list-ac1-before.png",
            "frontend_evidence",
            "Early staging list omitted price, stock, or placeholder image on several cards.",
            "Before screenshot supplied with the story, showing incomplete product cards before AC1 fields were implemented.",
            "PRODUCT LIST / AC1 / BEFORE",
            "product-list-ac1-after.png",
            "product_list_fixed",
            "Product list at 375px after AC1: each card shows name, USD price, stock state, and image or placeholder.",
            "After screenshot attached at close for AC1, showing readable name, price, availability, and placeholder behavior.",
            "PRODUCT LIST / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "product-list-ac2-before.png",
            "frontend_evidence",
            "Loading, API failure, and empty-result states were missing or unusable in early staging.",
            "Before screenshot supplied with the story, showing missing loading, error, and empty states required by AC2.",
            "PRODUCT LIST / AC2 / BEFORE",
            "product-list-ac2-after.png",
            "frontend_evidence",
            "Loading skeleton, API retry message, and empty-result copy are all readable after AC2 implementation.",
            "After screenshot attached at close for AC2, showing distinct loading, failure, and empty states.",
            "PRODUCT LIST / AC2 / AFTER",
        ),
    ],
    "ATC-14": [
        AcVisualShot(
            1,
            "product-detail-ac1-before.png",
            "product_detail_current",
            "Current staging detail page exposed raw values without shopper-facing layout.",
            "Before screenshot supplied during planning, showing product data present but not arranged for shoppers (AC1).",
            "PRODUCT DETAIL / AC1 / BEFORE",
            "product-detail-ac1-after.png",
            "frontend_evidence",
            "Staging detail page shows name, description, price, image, and availability in the agreed order.",
            "After screenshot attached at close for AC1, showing the implemented shopper-facing product detail layout.",
            "PRODUCT DETAIL / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "product-detail-ac2-before.png",
            "frontend_evidence",
            "Add to cart stayed enabled at zero stock with no explanation for the shopper.",
            "Before screenshot supplied during planning, showing the action enabled at zero stock before AC2 behavior existed.",
            "PRODUCT DETAIL / AC2 / BEFORE",
            "product-detail-ac2-after.png",
            "frontend_evidence",
            "Add to cart is disabled at zero stock and the page explains why.",
            "After screenshot attached at close for AC2, showing the disabled action and zero-stock explanation.",
            "PRODUCT DETAIL / AC2 / AFTER",
        ),
    ],
    "ATC-15": [
        AcVisualShot(
            1,
            "missing-image-ac1-before.png",
            "missing_image",
            "Chrome Slow 3G showed the native broken-image icon before the placeholder appeared.",
            "Before screenshot supplied with the story, showing the browser broken-image flash on a 404 image URL.",
            "MISSING IMAGE / AC1 / BEFORE",
            "missing-image-ac1-after.png",
            "missing_image_fixed",
            "The neutral AtlasCart placeholder appears directly with no broken-image flash.",
            "After screenshot attached at close for AC1, proving the placeholder replaces failed images without a browser icon flash.",
            "MISSING IMAGE / AC1 / AFTER",
        ),
    ],
    "ATC-19": [
        AcVisualShot(
            1,
            "narrow-product-card-ac1-before.png",
            "bug_ui",
            "At 375px the long product name let price overlap the stock label.",
            "Before screenshot supplied with the bug report, showing the exact 375px price-and-stock collision (AC1).",
            "PRODUCT CARD / AC1 / BEFORE",
            "narrow-product-card-ac1-after.png",
            "bug_ui_fixed",
            "At 320px and 375px the card keeps name, price, and stock on separate readable lines.",
            "After screenshot attached at close for AC1, showing no overlap at narrow phone widths.",
            "PRODUCT CARD / AC1 / AFTER",
        ),
    ],
    "ATC-21": [
        AcVisualShot(
            1,
            "missing-image-flash-ac1-before.png",
            "missing_image",
            "Chrome Slow 3G painted the native broken-image icon before the placeholder loaded.",
            "Before screenshot supplied with the bug report, captured under Slow 3G to show the broken-image flash (AC1).",
            "IMAGE FLASH / AC1 / BEFORE",
            "missing-image-flash-ac1-after.png",
            "missing_image_fixed",
            "The placeholder appears directly under the same Slow 3G setup with no broken icon.",
            "After screenshot attached at close for AC1, proving the image stays hidden until load success.",
            "IMAGE FLASH / AC1 / AFTER",
        ),
    ],
    "ATC-24": [
        AcVisualShot(
            1,
            "add-to-cart-ac1-before.png",
            "frontend_evidence",
            "Add to cart was missing from one entry point and did not keep the shopper on the current page.",
            "Before screenshot supplied during planning, showing incomplete add-to-cart entry points before AC1.",
            "ADD TO CART / AC1 / BEFORE",
            "add-to-cart-ac1-after.png",
            "frontend_evidence",
            "Available products can be added from both list and detail without leaving the current page.",
            "After screenshot attached at close for AC1, showing add-to-cart from list and detail views.",
            "ADD TO CART / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "add-to-cart-ac2-before.png",
            "frontend_evidence",
            "Out-of-stock products could still be added with no shopper-facing reason.",
            "Before screenshot supplied during planning, showing unavailable products addable before AC2 blocking existed.",
            "ADD TO CART / AC2 / BEFORE",
            "add-to-cart-ac2-after.png",
            "frontend_evidence",
            "Unavailable products stay unaddable and the shopper sees why.",
            "After screenshot attached at close for AC2, showing the blocked unavailable path with a clear reason.",
            "ADD TO CART / AC2 / AFTER",
        ),
    ],
    "ATC-25": [
        AcVisualShot(
            1,
            "cart-restore-ac1-before.png",
            "frontend_evidence",
            "Guest cart contents were lost after refresh or browser restart.",
            "Before screenshot supplied during planning, showing cart loss after refresh before AC1 persistence existed.",
            "CART RESTORE / AC1 / BEFORE",
            "cart-restore-ac1-after.png",
            "frontend_evidence",
            "Valid guest cart items restore after refresh and after reopening the browser.",
            "After screenshot attached at close for AC1, showing restored cart items after refresh.",
            "CART RESTORE / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "cart-restore-ac2-before.png",
            "frontend_evidence",
            "Corrupt stored cart data broke the page instead of opening an empty cart.",
            "Before screenshot supplied during planning, showing corrupt local storage crashing the cart flow (AC2).",
            "CART RESTORE / AC2 / BEFORE",
            "cart-restore-ac2-after.png",
            "frontend_evidence",
            "Corrupt or outdated stored data clears safely and opens an empty cart.",
            "After screenshot attached at close for AC2, showing corrupt-data fallback to an empty cart.",
            "CART RESTORE / AC2 / AFTER",
        ),
    ],
    "ATC-26": [
        AcVisualShot(
            1,
            "cart-update-ac1-before.png",
            "frontend_evidence",
            "Quantity changes required a reload and totals were calculated in the browser.",
            "Before screenshot supplied during planning, showing stale totals and reload-dependent quantity updates (AC1).",
            "CART UPDATE / AC1 / BEFORE",
            "cart-update-ac1-after.png",
            "frontend_evidence",
            "Quantity changes update server-calculated item and cart totals without a reload.",
            "After screenshot attached at close for AC1, showing live totals after quantity change.",
            "CART UPDATE / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "cart-update-ac2-before.png",
            "frontend_evidence",
            "Invalid quantities cleared the cart or showed a generic failure.",
            "Before screenshot supplied during planning, showing destructive invalid-quantity handling before AC2.",
            "CART UPDATE / AC2 / BEFORE",
            "cart-update-ac2-after.png",
            "frontend_evidence",
            "Invalid or unavailable quantities show a clear message and keep other cart items.",
            "After screenshot attached at close for AC2, showing a recoverable invalid-quantity message.",
            "CART UPDATE / AC2 / AFTER",
        ),
    ],
    "ATC-28": [
        AcVisualShot(
            1,
            "shipping-form-ac1-before.png",
            "frontend_evidence",
            "Required address fields lacked inline validation or clear missing-field messages.",
            "Before screenshot supplied during planning, showing unclear shipping validation before AC1.",
            "SHIPPING FORM / AC1 / BEFORE",
            "shipping-form-ac1-after.png",
            "frontend_evidence",
            "Each missing or invalid US beta address field is explained on blur and submit.",
            "After screenshot attached at close for AC1, showing inline validation on required fields.",
            "SHIPPING FORM / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "shipping-form-ac2-before.png",
            "frontend_evidence",
            "Address values appeared in error payloads visible to the shopper.",
            "Before screenshot supplied during planning, showing address values leaking into visible error output (AC2).",
            "SHIPPING FORM / AC2 / BEFORE",
            "shipping-form-ac2-after.png",
            "frontend_evidence",
            "Failure messages stay generic and do not echo address values to the shopper.",
            "After screenshot attached at close for AC2, showing safe error copy with no address echo.",
            "SHIPPING FORM / AC2 / AFTER",
        ),
    ],
    "ATC-31": [
        AcVisualShot(
            1,
            "order-confirmation-ac1-before.png",
            "frontend_evidence",
            "Confirmation omitted item summary, shipping summary, or next-step guidance.",
            "Before screenshot supplied during planning, showing incomplete confirmation content before AC1.",
            "CONFIRMATION / AC1 / BEFORE",
            "order-confirmation-ac1-after.png",
            "frontend_evidence",
            "Confirmation shows reference, items, total, shipping summary, and next step.",
            "After screenshot attached at close for AC1, showing the full confirmation layout.",
            "CONFIRMATION / AC1 / AFTER",
        ),
        AcVisualShot(
            2,
            "order-confirmation-ac2-before.png",
            "frontend_evidence",
            "Refreshing confirmation resubmitted checkout or showed a blank page.",
            "Before screenshot supplied during planning, showing unsafe refresh behavior before AC2.",
            "CONFIRMATION / AC2 / BEFORE",
            "order-confirmation-ac2-after.png",
            "frontend_evidence",
            "Refresh or revisit loads the existing order without submitting checkout again.",
            "After screenshot attached at close for AC2, showing the same order after refresh.",
            "CONFIRMATION / AC2 / AFTER",
        ),
    ],
    "ATC-40": [
        AcVisualShot(
            1,
            "header-cart-count-ac1-before.png",
            "frontend_evidence",
            "Header cart count did not update after add, update, or remove actions.",
            "Before screenshot supplied during planning, showing a stale or missing header count before AC1.",
            "HEADER COUNT / AC1 / BEFORE",
            "header-cart-count-ac1-after.png",
            "frontend_evidence",
            "Header count updates after add, quantity update, and remove, then hides when empty.",
            "After screenshot attached at close for AC1, showing live header count changes.",
            "HEADER COUNT / AC1 / AFTER",
        ),
    ],
    "ATC-45": [
        AcVisualShot(
            1,
            "checkout-pending-ac1-before.png",
            "frontend_evidence",
            "Place order stayed clickable while checkout was pending and allowed repeat submits.",
            "Before screenshot supplied during the incident follow-up, showing repeat-click exposure before AC1 mitigation.",
            "CHECKOUT PENDING / AC1 / BEFORE",
            "checkout-pending-ac1-after.png",
            "frontend_evidence",
            "Place order disables before the first request and cannot send a second while pending.",
            "After screenshot attached at close for AC1, showing disabled Place order during pending checkout.",
            "CHECKOUT PENDING / AC1 / AFTER",
        ),
    ],
    "ATC-37": [
        AcVisualShot(
            1,
            "empty-cart-ac1-before.png",
            "empty_cart",
            "At 320px the Browse products button sat below the viewport with excess empty space.",
            "Before screenshot supplied with the bug report, showing the empty-cart spacing problem at 320px (AC1).",
            "EMPTY CART / AC1 / BEFORE",
        ),
    ],
    "ATC-39": [
        AcVisualShot(
            1,
            "cart-controls-ac1-before.png",
            "cart_controls",
            "At 320px quantity 10 made the increment control overlap Remove.",
            "Before screenshot supplied with the bug report, showing control overlap at 320px with quantity 10 (AC1).",
            "CART CONTROLS / AC1 / BEFORE",
        ),
    ],
}


def frontend_visual_issue_keys() -> set[str]:
    return set(FRONTEND_VISUAL_EVIDENCE)


def verification_filenames(issue_key: str, *, done: bool) -> list[str]:
    if not done:
        return []
    prefix = f"atlascart-m1-{issue_key.lower()}"
    return [
        f"{prefix}-{shot.after_filename}"
        for shot in FRONTEND_VISUAL_EVIDENCE[issue_key]
        if shot.after_filename
    ]


def build_visual_attachments(issue: dict[str, Any], shot: AcVisualShot) -> list[dict[str, Any]]:
    issue_key = issue["key"]
    prefix = f"atlascart-m1-{issue_key.lower()}"
    created_base = issue["created_at"]
    reporter = issue["reporter"]
    assignee = issue["assignee"] or issue["reporter"]
    attachments: list[dict[str, Any]] = []

    before = {
        "issue": issue_key,
        "uploader": reporter,
        "filename": f"{prefix}-{shot.before_filename}",
        "content_type": "image/png",
        "kind": "screenshot",
        "placement": "description",
        "created_at": created_base + timedelta(minutes=shot.ac_index),
        "purpose": shot.before_purpose,
        "caption": shot.before_caption,
        "screenshot_type": shot.before_type,
        "ac_index": shot.ac_index,
        "phase": "before",
        "context": shot.before_context,
    }
    attachments.append(before)

    if shot.after_filename and shot.after_type and shot.after_caption and shot.after_purpose:
        after_created = issue.get("done_at") or created_base
        attachments.append({
            "issue": issue_key,
            "uploader": assignee,
            "filename": f"{prefix}-{shot.after_filename}",
            "content_type": "image/png",
            "kind": "screenshot",
            "placement": "comment",
            "created_at": after_created - timedelta(minutes=10 - shot.ac_index),
            "purpose": shot.after_purpose,
            "caption": shot.after_caption,
            "screenshot_type": shot.after_type,
            "ac_index": shot.ac_index,
            "phase": "after",
            "context": shot.after_context or shot.before_context.replace("BEFORE", "AFTER"),
        })
    return attachments


def all_visual_attachments(issue_by_key: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for issue_key, shots in FRONTEND_VISUAL_EVIDENCE.items():
        issue = issue_by_key[issue_key]
        assert len(shots) == len(issue["acceptance_criteria"]), (
            f"{issue_key} visual evidence count must match AC count"
        )
        for shot in shots:
            rows.extend(build_visual_attachments(issue, shot))
    return rows
