"""Human-reviewed quality facts for the AtlasCart M1 Jira history.

These strings are intentionally issue-specific. They are source data for Jira,
not reusable prose templates.

Acceptance-criteria count follows a realistic mix: one or two AC per issue is
common; three AC is reserved for larger or cross-cutting work.
"""

ACCEPTANCE_CRITERIA = {
    "ATC-1": [
        "A beta shopper can move from the product list to a product page and start a cart on desktop or a narrow phone.",
        "Loading, empty, and recoverable error states tell the shopper what is happening in plain English.",
    ],
    "ATC-2": [
        "A seller catalog can be imported without guessing when price, stock, SKU, description, or image data is incomplete.",
        "Invalid supplier rows are reported clearly and do not leave a partially imported catalog.",
    ],
    "ATC-3": [
        "A guest shopper can add an available item, enter a US shipping address, use the payment sandbox, and receive one confirmation.",
        "A failed or delayed checkout keeps the cart and never exposes a provider's raw message.",
        "Repeated actions and stock changes do not create an extra order or sell unavailable stock.",
    ],
    "ATC-4": [
        "Every accepted checkout creates a readable order with item-price snapshots and one stock change.",
        "Customer operations can find the order reference, status, total, and time without seeing a full address in the list.",
    ],
    "ATC-5": [
        "Every pull request runs backend checks, frontend checks, and a build before merge.",
        "A released build has a recorded version, a health check, request IDs, and a tested rollback path.",
    ],
    "ATC-6": [
        "QA covers the main catalog-to-order path plus the failures most likely to confuse a beta shopper.",
        "Known failures are linked to owned follow-up issues instead of being hidden in a test report.",
    ],
    "ATC-7": [
        "The React and TypeScript app starts with the shared header, content layout, and error boundary.",
        "Catalog and product-detail routes work after both in-app navigation and a direct browser refresh.",
    ],
    "ATC-8": [
        "The Spring Boot API starts against a new PostgreSQL database and applies the first migrations in order.",
        "The health endpoint reports healthy only when the API can reach PostgreSQL.",
    ],
    "ATC-9": [
        "A product stores SKU, name, optional description, USD price, optional image reference, active state, and current stock.",
        "Negative price and negative stock are rejected by both application validation and the database boundary.",
    ],
    "ATC-10": [
        "A valid supplier CSV imports all rows and reports how many products were created or updated.",
        "Blank, negative, and non-numeric prices return the original row number and save no partial import.",
        "Exact or conflicting duplicate SKUs follow the decision in ATC-22 and name both affected rows.",
    ],
    "ATC-11": [
        "The catalog endpoint returns only active products with SKU, name, price, stock, and image information.",
        "A case-insensitive name query returns matching products, while no match returns an empty result rather than an error.",
    ],
    "ATC-12": [
        "Each product card shows name, USD price, availability, and an image or neutral placeholder.",
        "Loading, API failure, no products, and no search matches each have a clear and usable state.",
    ],
    "ATC-13": [
        "An active SKU returns its description, price, stock, and image reference, while unknown or inactive SKUs return the same safe not-found response.",
    ],
    "ATC-14": [
        "Opening a card shows the matching product name, description, price, image, and current availability.",
        "Add to cart is disabled at zero stock and the page explains why.",
    ],
    "ATC-15": [
        "A blank or failed image request shows the neutral product placeholder immediately without a browser broken-image icon.",
    ],
    "ATC-16": [
        "Each pull request runs backend tests, frontend type and lint checks, and the application build.",
        "Failure in either codebase blocks merge and prevents an image from being published.",
    ],
    "ATC-17": [
        "A main-branch build deploys the web app and API to staging with a recorded image tag.",
        "The staging health check confirms that the API and database are reachable.",
    ],
    "ATC-18": [
        "The smoke run covers import, product list, product detail, missing image, invalid price, and a narrow viewport.",
    ],
    "ATC-19": [
        "The product card keeps name, price, and stock readable at 320px and 375px without overlap.",
    ],
    "ATC-20": [
        "Blank, negative, or non-numeric prices are rejected with the original CSV row number and no rows from that file are saved.",
    ],
    "ATC-21": [
        "Chrome on a throttled connection never paints its native broken-image icon before the AtlasCart placeholder appears.",
    ],
    "ATC-22": [
        "Conflicting and exact duplicate SKUs are rejected and the seller sees both row numbers.",
    ],
    "ATC-23": [
        "The contract defines read, add, update, and remove operations for one guest cart ID.",
        "Every response returns the complete updated cart with server-calculated totals.",
    ],
    "ATC-24": [
        "An available product can be added from both list and detail views.",
        "An unavailable product cannot be added and the shopper sees the reason.",
    ],
    "ATC-25": [
        "A valid guest cart is restored after refresh and after reopening the browser.",
        "Corrupt or outdated stored data is cleared and opens an empty cart instead of breaking the page.",
    ],
    "ATC-26": [
        "A shopper can change quantity and see server-calculated item and cart totals without a page reload.",
        "Invalid or unavailable quantities return a clear message without losing other cart items.",
    ],
    "ATC-27": [
        "Stock is checked when checkout opens and again inside the order-creation transaction.",
        "When two carts request the last unit, only one order succeeds and stock never becomes negative.",
        "The losing checkout names the changed SKU, and a timeout releases the database lock.",
    ],
    "ATC-28": [
        "The form collects the six required US beta address fields and explains each missing or invalid value.",
        "Address values are absent from application logs, error payloads, and list views.",
    ],
    "ATC-29": [
        "Accepted, rejected, and timed-out sandbox responses map to the agreed internal payment states.",
        "Only the provider reference and safe status are stored; card details and raw provider payloads are not stored.",
    ],
    "ATC-30": [
        "An accepted checkout creates one order with item-price snapshots, shipping details, payment reference, and readable order number.",
        "The stock check, stock reduction, order insert, and idempotency record succeed or roll back together.",
        "Five submissions with the same checkout key return the first order and reduce stock once.",
    ],
    "ATC-31": [
        "Confirmation shows the order reference, items, total, shipping summary, and next step.",
        "Refreshing or revisiting confirmation loads the existing order without submitting checkout again.",
    ],
    "ATC-32": [
        "Operations can see recent order reference, status, total, and created time in newest-first order.",
        "Full shipping address and payment-provider details are not present in the list response or page.",
    ],
    "ATC-33": [
        "One request ID follows checkout through the gateway and API logs.",
        "The correlated logs contain no cart body, shipping address, card detail, or provider response body.",
    ],
    "ATC-34": [
        "The approved catalog, cart, checkout, and order build is deployed with a recorded beta version and passing health checks.",
    ],
    "ATC-35": [
        "The test matrix covers normal checkout, payment failure, repeat action, refresh, changed stock, and supported browsers.",
        "Every execution records build, environment, setup, expected behavior, observed behavior, result, and tester.",
        "Failures remain visible and link to follow-up issues rather than being reported as an all-green summary.",
    ],
    "ATC-36": [
        "A rejected payment keeps the cart and suggests a safe next action without showing provider wording.",
        "A timeout explains that the result is being checked and prevents an unsafe second submit.",
    ],
    "ATC-37": [
        "At 320px and 375px, the empty-cart message and Browse products button are visible without scrolling an empty gap.",
    ],
    "ATC-38": [
        "Confirmation, application logs, and the operations page display the same beta order reference.",
        "Existing beta orders remain searchable during any format transition.",
    ],
    "ATC-39": [
        "At 320px, a two-line product name and quantity 10 do not make quantity controls overlap Remove.",
    ],
    "ATC-40": [
        "The header count changes after add, quantity update, and remove without a refresh, and hides when the cart is empty.",
    ],
    "ATC-41": [
        "Contract tests cover accepted, rejected, and timed-out provider responses using fixtures and no live sandbox call.",
    ],
    "ATC-42": [
        "The examples use synthetic BETA order references and invented product data with no real customer or payment detail.",
    ],
    "ATC-43": [
        "The report records one intended checkout, the two resulting orders, the stock impact, and the payment outcome.",
        "Frontend mitigation and API idempotency stop a repeat action from creating another order.",
        "The incident remains open until stock recovery and slow-browser regression work are complete.",
    ],
    "ATC-44": [
        "Five repeated submissions produce one order ID and one stock reduction.",
    ],
    "ATC-45": [
        "Place order disables before the first network request begins and cannot send a second request while checkout is pending.",
    ],
    "ATC-46": [
        "An authorized operations member can restore exactly the quantity lost to the duplicate order.",
        "The audit records old stock, new stock, actor, reason, and time without customer personal data.",
    ],
    "ATC-47": [
        "A repeatable browser test controls the checkout response delay without depending on random network timing.",
        "One click and rapid repeated activation both end with one order, one stock reduction, and one confirmation.",
    ],
    "ATC-48": [
        "The issue closes as a duplicate of ATC-27 with no separate code change or conflicting owner.",
    ],
}


BUG_REPORT_DETAILS = {
    "ATC-19": {
        "when": "Staging build catalog-0.4.1, Chrome 126, viewport 375x812, with the product name 'Adjustable Bamboo Monitor Stand'.",
        "steps": "Open the catalog, scroll to SKU C-311, and let the two-line name finish rendering.",
        "expected": "The name may wrap, but price and stock remain on separate readable lines inside the card.",
        "actual": "USD 39.00 is painted across the left side of the IN STOCK label and the card height does not expand.",
    },
    "ATC-20": {
        "when": "Staging importer build catalog-api-0.4.1 with supplier CSV row 18 containing an empty price.",
        "steps": "Upload the file, wait for import completion, and query the product created from row 18.",
        "expected": "The import stops, reports row 18 as missing price, and saves none of the file's rows.",
        "actual": "The import reports success and the storefront shows the affected product at USD 0.00.",
    },
    "ATC-21": {
        "when": "Staging build web-0.4.2, Chrome 126 with Slow 3G, using SKU B-208 whose image URL returns 404.",
        "steps": "Disable cache, open the catalog, and watch the image area before the fallback finishes loading.",
        "expected": "The neutral AtlasCart placeholder is visible for the whole loading and failure sequence.",
        "actual": "Chrome's native broken-image icon flashes for roughly half a second before the placeholder replaces it.",
    },
    "ATC-37": {
        "when": "Private-beta empty cart at 320x700 and 375x812 after all items have been removed.",
        "steps": "Remove the final item, keep the phone at the top of the page, and look for Browse products.",
        "expected": "The empty-state message and button are visible together with normal spacing.",
        "actual": "A large empty block pushes Browse products below the initial viewport.",
    },
    "ATC-39": {
        "when": "Private-beta cart at 320x700 with 'Adjustable Bamboo Monitor Stand' and quantity 10.",
        "steps": "Open the cart and tab through decrement, quantity, increment, and Remove.",
        "expected": "Each control has its own visible area and the focus outline does not cover another action.",
        "actual": "The increment control overlaps Remove; both still click, but their labels and focus outlines collide.",
    },
    "ATC-43": {
        "when": "Private-beta build beta-1.0.0 in Chrome with Slow 3G and one Desk Lamp in the cart.",
        "steps": "Use the accepted sandbox card and click Place order twice before the pending state appears.",
        "expected": "One checkout attempt returns one order, captures one sandbox payment, and reduces stock once.",
        "actual": "Orders BETA-1042 and BETA-1043 are inserted and stock falls by two, although only one payment is captured.",
    },
    "ATC-46": {
        "when": "After duplicate-order incident ATC-43, Desk Lamp stock is one lower than the single captured payment justifies.",
        "steps": "Ask operations to restore the duplicate order quantity using the current internal tools.",
        "expected": "An authorized, repeat-safe action restores stock and records order, SKU, quantity, actor, reason, and time.",
        "actual": "Only an engineer can edit PostgreSQL directly, leaving no support-readable correction record.",
    },
    "ATC-48": {
        "when": "Checkout test pass with SKU C-311 at stock 1 while the cart requests quantity 2.",
        "steps": "Open checkout, allow another cart to buy the last unit, and confirm the original checkout.",
        "expected": "No order is created and the response names C-311 as changed; this behavior is owned by ATC-27.",
        "actual": "The reporter saw checkout continue with an outdated quantity, matching the already documented ATC-27 race.",
    },
}


POINT_RATIONALES = {
    "ATC-7": "3 — The shell changes routing, shared layout, direct-refresh handling, and the API client together. Each piece is familiar frontend work, but a mistake affects every later page; no design system or authentication work is included.",
    "ATC-8": "5 — This establishes the Spring Boot service, PostgreSQL connection, ordered migrations, health behavior, and the staging deployment check. It crosses backend and platform ownership and must be proven from an empty database, so it is larger than a normal single-endpoint task.",
    "ATC-9": "5 — Product and stock need a new schema, decimal-money choices, application validation, database constraints, and migration coverage. The model is small, but changing it later would affect import, catalog, cart, and orders.",
    "ATC-10": "8 — The importer must parse a supplier-controlled file, keep the write atomic, preserve source row numbers, handle missing values, and apply the new duplicate-SKU decision. Unknown supplier data and error-reporting behavior make this too uncertain for five points.",
    "ATC-11": "3 — This is one read endpoint and one basic case-insensitive query using the established product model. Response mapping, empty results, and inactive-product filtering need tests, but there is no ranking, pagination, or new persistence design.",
    "ATC-12": "5 — The page combines API loading, product-card composition, responsive layout, keyboard behavior, and distinct loading, error, and empty states. The API contract already exists, keeping it below eight, but the number of visible states makes three too small.",
    "ATC-13": "3 — The endpoint is limited to one SKU lookup, response mapping, and a shared safe 404 for unknown or inactive products. The schema already exists and no authorization or caching design is part of this task.",
    "ATC-14": "3 — The detail route, product rendering, stock-aware button, and narrow-screen order form one contained frontend path. It reuses the list API patterns and does not yet implement cart mutation.",
    "ATC-15": "2 — The work is localized to image loading and fallback state in an existing card and detail view. It needs blank, 404, and slow-load checks, but no API or schema change.",
    "ATC-16": "3 — Standard GitHub Actions steps already exist in the team's experience; the work is wiring two test suites, one build, and required merge status. Repository permissions add coordination, but there is no custom CI platform.",
    "ATC-17": "3 — Deployment follows one established container path, but it must record the image tag, gate on health, and prove rollback. Operational failure is more costly than the code size suggests, so it is not a two.",
    "ATC-18": "2 — QA is documenting and running six known checks against features that already exist. The work is bounded to one environment and one narrow viewport; automated cross-browser coverage is outside this issue.",
    "ATC-19": "2 — The faulty flex layout is confined to one product-card component. The fix must preserve 320px, 375px, and desktop layouts, but it does not change data or behavior.",
    "ATC-20": "2 — One importer validation path must stop coercing blank prices and return the original line number. Negative and non-numeric cases reuse the same boundary, so the change and regression surface stay small.",
    "ATC-21": "1 — The root cause is one image element being shown before load success. Hiding it until success and checking one throttled 404 path is a tiny, reversible frontend change.",
    "ATC-23": "3 — The deliverable is an agreed request and response contract for four cart operations, backed by examples and client review. Cart persistence and UI implementation are deliberately separate issues.",
    "ATC-24": "3 — Add to cart must work from two existing entry points and update shared cart state without navigation. The backend contract is ready, and the header badge is split into ATC-40, keeping this within one frontend area.",
    "ATC-25": "3 — Browser persistence, restore, version checking, and corrupt-data fallback must behave as one flow. It is frontend-only, but failure happens during startup and needs careful recovery tests.",
    "ATC-26": "5 — Quantity changes and removal coordinate frontend controls, backend mutations, server-calculated money totals, and unavailable-stock errors. The cross-layer consistency risk is the reason this is five rather than three.",
    "ATC-27": "5 — The final stock check must share the order transaction, lock the affected rows correctly, return the changed SKU, and survive two-cart and timeout tests. Concurrency risk dominates the relatively small code change.",
    "ATC-28": "5 — Six-field form behavior, client and server validation, state preservation, and log/privacy checks cross UI and API boundaries. A full address library is excluded, which keeps the estimate below eight.",
    "ATC-29": "5 — The adapter creates a safety boundary around an external payment contract and must map accepted, rejected, and timed-out responses without storing sensitive payloads. Fixture-based contract tests reduce but do not remove integration uncertainty.",
    "ATC-30": "8 — Order creation joins payment outcome, price snapshots, shipping data, stock locking, readable references, and idempotency in one atomic path. Failure can duplicate durable business data, and the incident proved that safe retry behavior cannot be deferred.",
    "ATC-31": "3 — The page renders one established order response and restores it after refresh without resubmission. Missing-order recovery needs testing, but no new transaction or broad API design is included.",
    "ATC-32": "5 — Operations needs a new privacy-limited API and an internal list UI, with payment-state clarity across both. The query is simple, but crossing backend, frontend, and support review makes three unrealistic.",
    "ATC-33": "3 — The request ID must be created or preserved through gateway and API boundaries, returned safely, and checked against privacy rules. It uses standard logging rather than a new observability stack.",
    "ATC-34": "2 — The team is deploying an already approved build through the path established in ATC-17 and running a focused smoke check. There is release responsibility but little new implementation.",
    "ATC-35": "3 — The matrix spans browsers and six checkout conditions and requires careful evidence, but it is manual QA on existing builds. Automated slow-response coverage is separately owned by ATC-47.",
    "ATC-36": "3 — Rejected and timed-out states need different wording and recovery behavior while preserving cart state and hiding provider detail. It is one frontend flow with product review, not a payment integration rewrite.",
    "ATC-37": "1 — The defect is an isolated empty-state spacing value and button position with three viewport checks. No shared component behavior or data changes are expected.",
    "ATC-38": "3 — A single reference format must be agreed, produced, logged, and displayed consistently in three surfaces while old beta references remain findable. The change is broad enough to require coordination but not new order behavior.",
    "ATC-39": "2 — One cart-row responsive layout needs control sizing and wrapping changes plus keyboard regression at two phone widths. Cart calculations and API behavior are unaffected.",
    "ATC-42": "1 — The task creates a few synthetic examples, applies an existing redaction checklist, and gets QA/support review. It has no production data access or code change.",
    "ATC-43": "5 — Investigation crosses browser behavior, gateway logs, order inserts, payment outcome, and stock impact. Mitigation and permanent fixes are split into subtasks, but coordination, customer communication, and exit verification remain on the incident.",
    "ATC-46": "5 — A correction changes stock and therefore needs authorization, order-based limits, idempotency, transactional audit, and an operations-facing result. A manual one-off SQL update would be smaller but would not meet the recovery requirement.",
    "ATC-48": "1 — QA only needs to reproduce the report, compare it with ATC-27, preserve the useful steps, and close the duplicate. No separate design, code, or regression suite is created here.",
}


SUBTASK_RATIONALE: dict[str, list[tuple[str, str]]] = {
    "ATC-10": [
        (
            "ATC-22",
            "Importer work stalled because product had not chosen how exact and conflicting duplicate SKUs should behave. "
            "The subtask makes that decision visible and blocks implementation until both affected row numbers are reported.",
        ),
    ],
    "ATC-24": [
        (
            "ATC-40",
            "Header cart count updates touch the shared shell and need a different owner from add-to-cart behavior. "
            "Splitting it out keeps this story focused on list and detail entry points while badge work stays testable on its own.",
        ),
    ],
    "ATC-29": [
        (
            "ATC-41",
            "Accepted, rejected, and timeout sandbox responses need repeatable contract evidence without live provider calls in CI. "
            "The subtask carries that QA proof while this story keeps the integration and storage scope.",
        ),
    ],
    "ATC-43": [
        (
            "ATC-44",
            "The permanent fix is an API idempotency check before order insert. "
            "Tracking it separately lets backend deploy the technical fix without mixing it into incident coordination.",
        ),
        (
            "ATC-45",
            "Browser repeat-click mitigation reduces customer exposure before the API fix lands. "
            "Frontend can ship the disabled Place order state while the incident remains open.",
        ),
        (
            "ATC-47",
            "The slow double-click reproduction needs a repeatable browser regression, not a one-off manual retest. "
            "The subtask owns automated exit criteria tied to the checkout test plan.",
        ),
    ],
}


UNESTIMATED_RATIONALES = {
    "ATC-1": "Not estimated — this epic is the Month 1 storefront outcome, not one deliverable. The team estimates its product-list, detail, shell, and responsive child issues so the same work is not counted twice.",
    "ATC-2": "Not estimated — this epic groups catalog storage, import, API, and data-quality work owned across multiple sprints. Forecasting uses the estimated children because their uncertainty changes independently.",
    "ATC-3": "Not estimated — this epic spans cart, address, payment, checkout, and recovery. One epic number would hide concurrency and provider risk already represented by the child estimates.",
    "ATC-4": "Not estimated — this epic is the operations and order-management outcome. Order creation, recent-order view, incident recovery, and stock correction are sized separately and roll up here.",
    "ATC-5": "Not estimated — this epic is a continuing release-safety theme across CI, deployment, logs, and rollback. The team estimates each concrete platform change instead of treating ongoing ownership as sprint scope.",
    "ATC-6": "Not estimated — this epic collects QA coverage and beta-readiness work. Test tasks and discovered bugs carry their own estimates, while the epic remains an outcome-level view.",
    "ATC-22": "Not estimated — this decision subtask is included in ATC-10's eight points. Adding separate points would double-count the product discussion that caused the parent estimate to increase.",
    "ATC-40": "Not estimated — this header-count subtask is part of the frontend delivery tracked by ATC-24. The split gives ownership and visibility, not extra scope for velocity.",
    "ATC-41": "Not estimated — the adapter contract test is included in ATC-29's integration estimate. It is tracked separately so QA evidence is visible without counting the same risk twice.",
    "ATC-44": "Not estimated — this API fix is a technical subtask of incident ATC-43 and the reopened ATC-30 scope. The incident team tracks its completion separately but does not add another velocity estimate mid-incident.",
    "ATC-45": "Not estimated — this browser mitigation is a subtask of ATC-43. Its urgency and owner are visible here, while the incident's five-point estimate carries the coordination and verification cost.",
    "ATC-47": "Not estimated — this regression subtask is part of ATC-43's exit work and ATC-35's test plan. Separate points would count the same incident verification in two places.",
}


LINK_NARRATIVES = {
    ("ATC-8", "ATC-9", "blocks"): "ATC-9 needs the first product migration and a working PostgreSQL test boundary before its model can be saved safely.",
    ("ATC-9", "ATC-10", "blocks"): "The importer can parse sample rows early, but it cannot commit products until ATC-9 defines price, stock, and database validation.",
    ("ATC-9", "ATC-11", "blocks"): "The catalog query needs the active-product schema and indexes created by ATC-9 before API integration can finish.",
    ("ATC-11", "ATC-12", "blocks"): "The list UI can use fixtures, but its loading, empty, and error behavior cannot be accepted until the real catalog response is stable.",
    ("ATC-13", "ATC-14", "blocks"): "The product page needs the detail response and safe not-found behavior before frontend integration and QA can finish.",
    ("ATC-15", "ATC-12", "relates to"): "Both issues render the same product-card image state; ATC-15 owns the reusable fallback while ATC-12 owns the full list experience.",
    ("ATC-15", "ATC-14", "relates to"): "The detail page reuses the missing-image fallback from ATC-15, but its stock-aware action remains independently testable.",
    ("ATC-22", "ATC-10", "blocks"): "Engineering cannot finish duplicate validation until product chooses whether exact and conflicting duplicate SKUs are rejected or merged.",
    ("ATC-23", "ATC-24", "blocks"): "The add-to-cart UI needs the agreed add response and guest cart identifier from ATC-23 before it can connect to the API.",
    ("ATC-23", "ATC-26", "blocks"): "Update and remove behavior depends on ATC-23 returning the complete server-calculated cart after each mutation.",
    ("ATC-24", "ATC-25", "relates to"): "ATC-24 creates the browser cart state that ATC-25 persists and restores; neither issue alone owns the whole cart lifecycle.",
    ("ATC-27", "ATC-30", "relates to"): "Both work inside order creation: ATC-30 owns the durable order transaction, while ATC-27 adds the final concurrency-safe stock decision.",
    ("ATC-28", "ATC-30", "blocks"): "Order creation cannot save its shipping snapshot until the validated, privacy-reviewed address shape from ATC-28 is available.",
    ("ATC-29", "ATC-30", "blocks"): "ATC-30 must receive one safe accepted payment result and provider reference from the adapter before inserting an order.",
    ("ATC-30", "ATC-31", "blocks"): "Confirmation needs a committed order response and a safe reload lookup before the page can be verified.",
    ("ATC-30", "ATC-32", "blocks"): "The operations list needs real order status, total, reference, and timestamps created by ATC-30.",
    ("ATC-33", "ATC-34", "blocks"): "The beta release requires request IDs in checkout failures so support can investigate invitations without searching by personal data.",
    ("ATC-43", "ATC-30", "relates to"): "The incident exposed the missing safe-retry assumption in ATC-30 and caused that completed story to be reopened for idempotency.",
    ("ATC-44", "ATC-43", "blocks"): "The incident cannot exit technical mitigation until the API returns the first order for a repeated checkout key before stock changes.",
    ("ATC-45", "ATC-43", "blocks"): "Customer exposure remains high until the browser disables Place order before a second request can start.",
    ("ATC-43", "ATC-46", "relates to"): "ATC-46 restores the extra stock reduction caused by the incident and supplies the missing support-readable audit trail.",
    ("ATC-43", "ATC-47", "relates to"): "ATC-47 turns the incident's slow double-click reproduction into a repeatable browser regression across supported browsers.",
    ("ATC-48", "ATC-27", "duplicates"): "The second report describes the same stale-stock race, expected response, and transaction boundary already owned by ATC-27.",
}
