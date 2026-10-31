#!/usr/bin/env python3
"""RAGAS scoring of Ask AI on the AtlasCart question set (50 questions with reference answers).

Metrics: faithfulness, answer relevancy, context precision, context recall. Judge: `gemini-judge`
through the LiteLLM gateway, a different vendor from the system under test (gpt-5.6-luna), to avoid
same-family self-preference. Floors live in eval/eval_gate.json; `scripts/run_rag_eval.sh gate` runs
this with both caches off.

Usage:
    python -m eval.ragas_eval
    python -m eval.ragas_eval --url http://localhost:8200 --out /tmp/ragas.json
    python -m eval.ragas_eval --reuse-answers /tmp/ragas.answers.json   # re-score saved answers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.llm.prompt_registry import versions_for

# --- Question set + reference answers ------------------------------------------------------------
# Reference answers come from the seeded data itself, not from what the system answered. They cover
# counts, root-cause narrative, subtask/epic relationships, reopen history and out-of-scope abstention.


@dataclass(frozen=True)
class RagasCase:
    question: str
    reference: str
    # Which retrieval behaviour the question is meant to exercise. Not passed to RAGAS — it exists so
    # the report can break the four metrics down by category, which is the only way to see that (say)
    # context_recall is healthy on key lookups and weak on unassisted semantic search. Averaging those
    # together hides exactly the thing worth knowing.
    category: str = ""
    # Issue keys the answer has to be built from, so eval/retrieval_hit.py can check, without an
    # LLM, that the retriever puts at least one of them in the top 5. Empty for questions a tool
    # answers (counts, lookups) and for abstention questions, which have no right source.
    required_issue_keys: tuple = ()


# --- The AtlasCart question set ------------------------------------------------------------------
#
# A RAGAS number is only comparable to another run against the SAME fixture: if the corpus is
# reseeded, re-derive the references before comparing runs.
#
# Distribution, 50 questions, 70% semantic / 30% tool — semantic is the larger share deliberately:
# it is the part that moves when the retrieval engine changes, and it is what "how accurate is this
# RAG system" actually asks. Tool calling is held to 30% because those questions mostly measure
# whether the agent picked the right structured call, which is a routing property, not a retrieval one.
#
#   Semantic search        70%  (35)   no identifier in the question; retrieval works from meaning
#     one-source           40%  (20)     the whole answer sits in one issue, comment or attachment
#     linked-sources       24%  (12)     several sources, each naming the next: the answer is
#                                        reachable only by following the trail
#     abstention            6%  (3)      the answer is genuinely absent; refusing is the correct
#                                        behaviour, and confident invention is the costly failure
#   Tool calling           30%  (15)   structured queries the agent should answer with a tool
#     issue lookup          8%  (4)      an explicit key is given
#     filter/aggregation   12%  (6)      counts, group-bys, status filters
#     user/project lookup   6%  (3)      assignees, sprints, workspace-level facts
#     multi-tool            4%  (2)      needs more than one tool call composed
#
# Every `reference` below was derived by querying the database, and aggregate answers were checked
# with SQL, because a wrong reference scores the judge, not the system.

CASES: List[RagasCase] = [
    # ============================================================================================
    # SEMANTIC — one-source (20). Nothing in the question says where the answer lives: no issue key,
    # no filename. Retrieval has to work from meaning alone. This is the class that actually moves
    # when the retrieval engine changes, which is why it is the largest one here.
    # ============================================================================================
    RagasCase(
        "A shopper on a slow connection clicked the order button twice and ended up with two "
        "orders. What went wrong in the system?",
        "Checkout created the order before checking whether that checkout request had already been "
        "handled, so a repeated submission inserted a second order. Orders BETA-1042 and BETA-1043 "
        "were both created and stock fell by two, even though only one sandbox payment was "
        "captured. This is issue ATC-43.",
        "semantic:one-source",
        required_issue_keys=("ATC-43",),
    ),
    RagasCase(
        "Someone signed in with one mug in their guest basket and two already saved to their "
        "account, but the basket then showed six. Why did that happen?",
        "The cart-merge callback ran twice and recalculated the quantities on the retry instead of "
        "returning the result of the first merge. The fix stores and returns the first merge result "
        "before clearing the guest token, so the example carts merge to three, not six. This is "
        "issue ATC-59.",
        "semantic:one-source",
        required_issue_keys=("ATC-59",),
    ),
    RagasCase(
        "A beta shopper received their tracking email a second time and the View order button "
        "opened a different order. What was the cause?",
        "The first email was built from the new link, but the resend path still read an older saved "
        "value, so the resent email opened BETA-1123 instead of BETA-1128. Both paths now build the "
        "link from the same full order number. This is issue ATC-68.",
        "semantic:one-source",
        required_issue_keys=("ATC-68",),
    ),
    RagasCase(
        "After a seller renames a product, why does the storefront search still show the old name?",
        "The indexing event handler was missing for that path. Edit and import events now share the "
        "same idempotent indexing handler, with lag and dead-letter metrics for replay, and an "
        "accepted rename becomes searchable by the new name within sixty seconds. This is issue "
        "ATC-76.",
        "semantic:one-source",
        required_issue_keys=("ATC-76",),
    ),
    RagasCase(
        "What happens to a shopper's basket if they close the browser and come back later?",
        "A valid guest cart is restored after a refresh or a browser restart. Invalid or corrupt "
        "stored cart data is cleared with a single console warning and the shopper sees an empty "
        "cart rather than an error page. This is issue ATC-25.",
        "semantic:one-source",
        required_issue_keys=("ATC-25",),
    ),
    RagasCase(
        "How does the importer handle a supplier file where the same product appears twice with "
        "different prices?",
        "It rejects the row rather than picking a price. Conflicting duplicate SKUs are rejected "
        "and both line numbers are reported, and exact repeated rows are rejected too so the "
        "behaviour stays simple. Nothing is saved when validation fails. The decision is ATC-22 and "
        "the importer work is ATC-10.",
        "semantic:one-source",
        required_issue_keys=("ATC-22", "ATC-10"),
    ),
    RagasCase(
        "What happens when a price cell is left blank in the supplier upload?",
        "The row is rejected and reported with its original CSV line number, as are negative and "
        "non-numeric prices. A valid zero price is kept distinct from a missing one. This is issue "
        "ATC-20.",
        "semantic:one-source",
        required_issue_keys=("ATC-20",),
    ),
    RagasCase(
        "Was there a decision about whether shoppers have to create an account before they can "
        "order?",
        "Yes. Guest checkout was accepted for the beta and the first order is deliberately not held "
        "behind account creation. The decision is recorded on the checkout epic ATC-3.",
        "semantic:one-source",
        required_issue_keys=("ATC-3",),
    ),
    RagasCase(
        "Did the team decide to split the backend into multiple services?",
        "No. The decision was to keep one API deployment for now and to split services only after "
        "there is a measured reason to. It is recorded on the staging epic ATC-5.",
        "semantic:one-source",
        required_issue_keys=("ATC-5",),
    ),
    RagasCase(
        "Which browsers and screen sizes were agreed for testing the beta?",
        "Chrome, Firefox, Safari and one narrow-phone size, with a wider browser matrix left for "
        "later. It is recorded on the beta-testability epic ATC-6.",
        "semantic:one-source",
        required_issue_keys=("ATC-6",),
    ),
    RagasCase(
        "Does the catalog search handle typos or synonyms?",
        "No. The catalog query is a basic case-insensitive name match and relevance scoring was "
        "explicitly out of scope for that sprint. This is issue ATC-11.",
        "semantic:one-source",
        required_issue_keys=("ATC-11",),
    ),
    RagasCase(
        "What is stopping the catalog caching work from being finished?",
        "Read-through caching works for the list and detail routes, but invalidation during a large "
        "import is still being tested. The focused subtask found one stale list entry while detail "
        "invalidation stayed within target. The story is ATC-77 and the subtask is ATC-81.",
        "semantic:one-source",
        required_issue_keys=("ATC-77", "ATC-81"),
    ),
    RagasCase(
        "Why is checkout missing its latency target in the load tests?",
        "Catalog and sign-in workloads are stable at the target concurrency, but checkout exceeds "
        "its P95 goal while shipping quotes are cold. A controlled cold-cache shipping-quote step "
        "was added to the workload to reproduce it. The load-test task is ATC-80 and the subtask is "
        "ATC-83.",
        "semantic:one-source",
        required_issue_keys=("ATC-80", "ATC-83"),
    ),
    RagasCase(
        "What did operations need before stock could be corrected without a direct database edit?",
        "An authorized, repeat-safe correction action that records the order reference, the reason, "
        "the actor and the stock change in an audit record support can read, rather than an "
        "engineer updating PostgreSQL directly. This is issue ATC-46.",
        "semantic:one-source",
        required_issue_keys=("ATC-46",),
    ),
    RagasCase(
        "Why does an unknown product SKU and one a seller removed return the same response?",
        "Both inactive and unknown SKUs return 404 on purpose, so the response does not reveal "
        "whether a seller removed the product. This is issue ATC-13.",
        "semantic:one-source",
        required_issue_keys=("ATC-13",),
    ),
    RagasCase(
        "What happens if the shipping quote service times out during checkout?",
        "A timeout or an unknown service stops order creation and leaves the cart unchanged, rather "
        "than completing the order without a quote. This is issue ATC-61.",
        "semantic:one-source",
        required_issue_keys=("ATC-61",),
    ),
    RagasCase(
        "Can operations cancel an order that has already shipped?",
        "No. The approved order lifecycle distinguishes cancellable from shipped orders: a shipped "
        "order cannot be cancelled and has to go through refund handling instead. This is issue "
        "ATC-63.",
        "semantic:one-source",
        required_issue_keys=("ATC-63",),
    ),
    RagasCase(
        "How does the system stop a retried refund from paying a customer twice?",
        "The refund key is stored before the provider call, so timeout recovery queries the first "
        "result instead of issuing another refund, and retrying returns the existing refund rather "
        "than submitting a second provider request. This is issue ATC-66.",
        "semantic:one-source",
        required_issue_keys=("ATC-66",),
    ),
    RagasCase(
        "What stops a browser from claiming to be a different signed-in customer?",
        "The gateway strips client-supplied identity headers before adding the verified customer "
        "ID, so the browser cannot supply its own identity. Public catalog routes still work "
        "without a session. This is issue ATC-53.",
        "semantic:one-source",
        required_issue_keys=("ATC-53",),
    ),
    RagasCase(
        "What was done about shoppers seeing the raw payment error message?",
        "Two beta shoppers reported that the raw timeout message made them think the order was "
        "lost, so the wording was replaced with plain timeout and rejection messages, confirmed to "
        "keep the cart and the next action clear on mobile. This is issue ATC-36.",
        "semantic:one-source",
        required_issue_keys=("ATC-36",),
    ),

    # ============================================================================================
    # SEMANTIC — linked-sources (12). The answer is spread across sources, and each one names the
    # next: an issue's comments point at the follow-ups, and those carry the rest. Tests whether
    # retrieval can follow a trail rather than land on a single best passage.
    # ============================================================================================
    RagasCase(
        "The incident where one click created two orders — which follow-up work came out of it, and "
        "what did each piece own?",
        "Four follow-ups were split out of ATC-43: ATC-44 was the permanent API idempotency fix "
        "(a repeated checkout key returns the existing order before stock changes), ATC-45 was the "
        "immediate browser mitigation that disables Place order while checkout is pending, ATC-47 "
        "covered the slow-response browser regression across Chrome, Firefox and Safari, and ATC-46 "
        "owned restoring the stock the duplicate order consumed.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-43", "ATC-44", "ATC-45", "ATC-47", "ATC-46"),
    ),
    RagasCase(
        "In the duplicate-order investigation, how did the team rule out a proxy retry?",
        "The gateway and API logs showed two separate request IDs both reaching the API and both "
        "returning 201, with no proxy retry present in the request logs — so the second order came "
        "from a second real browser submission, not a retry. The redacted evidence is the "
        "duplicate-checkout-requests.log attached to ATC-43.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-43",),
    ),
    RagasCase(
        "Which sprint goal was about making beta orders easier to support, and what work closed "
        "under it?",
        "AtlasCart Sprint 3, whose goal was to make beta orders easier to support and recover. The "
        "duplicate-order incident ATC-43 closed in it along with its follow-ups, the audited stock "
        "correction ATC-46, the operations order view ATC-32, the checkout test plan ATC-35 and the "
        "pre-checkout stock check ATC-27.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-43", "ATC-46", "ATC-32", "ATC-35", "ATC-27"),
    ),
    RagasCase(
        "Why did the supplier catalog import story's estimate go up, and what was moved out of it?",
        "Once the duplicate report and its tests were included the work was judged closer to eight "
        "points than five. The unfinished duplicate-SKU decision was moved out with ATC-22 into "
        "Sprint 2, so the product decision could be made explicitly before importer validation "
        "could finish. The story is ATC-10.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-10", "ATC-22"),
    ),
    RagasCase(
        "Which issue was reopened after a customer reported the same problem again, and what was "
        "different the second time?",
        "ATC-68. It was closed after the first tracking email was verified to open the right order, "
        "then reopened when a beta shopper received the email a second time: the first send was "
        "fixed but the resend path still opened the wrong order, because it read an older saved "
        "value rather than the full order number.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-68",),
    ),
    RagasCase(
        "A report came in that checkout was accepting more items than were in stock. How was it "
        "handled?",
        "It was triaged as a duplicate. ATC-48 describes the same stale-stock race, expected "
        "response and transaction boundary already owned by ATC-27, so it was closed as a duplicate "
        "of ATC-27 with its reproduction note kept, and no separate implementation was done.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-48", "ATC-27"),
    ),
    RagasCase(
        "Why did the guest-cart merge story get re-estimated, and what did the extra points cover?",
        "ATC-56 went from five points to eight. Refinement exposed stock revalidation and callback "
        "idempotency across both the guest and the account cart; what the team planned to deliver "
        "did not change, the higher estimate covered the work that was already implied.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-56",),
    ),
    RagasCase(
        "Which work was carried from one sprint into the next, and why in each case?",
        "Three. ATC-60 was carried from Sprint 4 to Sprint 5 because the forged-header CI case was "
        "flaky on the close build and the team would not call an intermittent pass complete. ATC-68 "
        "was carried from Sprint 5 to Sprint 6 because the sprint ended before resent emails were "
        "checked. ATC-72 was carried from Sprint 6 to Sprint 7 because the initial search backfill "
        "and replay check exceeded the Sprint 6 window.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-60", "ATC-68", "ATC-72"),
    ),
    RagasCase(
        "Which issues changed owner part-way through, and what was the reason each time?",
        "Three. ATC-71 moved to Olivia Garcia when object validation and cleanup became the main "
        "risk. ATC-74 moved to Lucas Martin because he already owned the seller-tools UI and could "
        "reuse its accessible status components. ATC-80 moved to Olivia Garcia because she owns the "
        "load environment and dashboards, with Daniel Park staying as test reviewer.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-71", "ATC-74", "ATC-80"),
    ),
    RagasCase(
        "The bug where merging a cart doubled a product's quantity — which story delivered the "
        "behaviour it broke?",
        "ATC-56, the story that merges a guest cart after sign-in. ATC-59 is a retry failure in the "
        "merge behaviour ATC-56 delivered; the two are linked as related work, deliberately not as "
        "a dependency, and the bug keeps its own reproduction and verification.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-56", "ATC-59"),
    ),
    RagasCase(
        "What had to be finished before sign-in and cart merge could be completed?",
        "ATC-53, the gateway customer authentication boundary. Sign-in could not trust a browser "
        "session and cart merge could not get a verified account identity until the gateway "
        "validated and forwarded the customer ID, so ATC-55 and ATC-56 were both blocked on it "
        "until it passed its acceptance checks.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-53", "ATC-55", "ATC-56"),
    ),
    RagasCase(
        "Find the work about restoring stock after a duplicate beta order and say what state it is "
        "in.",
        "ATC-46, a high-priority bug under the operations epic ATC-4, assigned to Ava Patel at five "
        "points in Sprint 3. It is done: operations completed a correction through the audited "
        "workflow and confirmed the order reference, reason, actor and stock delta are "
        "understandable without a direct database edit.",
        "semantic:linked-sources",
        required_issue_keys=("ATC-46",),
    ),

    # ============================================================================================
    # SEMANTIC — abstention (3). The honest answer is "not in the corpus". Kept small but present:
    # without it a set rewards a system that always produces something, and confident invention on
    # an absent fact is the most costly failure this agent can have.
    # ============================================================================================
    RagasCase(
        "Is there anything in this workspace about the office wifi?",
        "No. Nothing in this workspace covers office wifi; the corpus is the AtlasCart storefront "
        "project. The correct response is to say there is no information about this.",
        "semantic:abstention",
    ),
    RagasCase(
        "What was our AWS bill last month?",
        "No. The workspace holds issues, comments and attachments for the AtlasCart project and "
        "contains no billing or cost information. The correct response is to say there is no "
        "information about this.",
        "semantic:abstention",
    ),
    RagasCase(
        "What was the root cause of ATC-999?",
        "There is no issue ATC-999 in this workspace — the project's issues run from ATC-1 to "
        "ATC-90. The correct response is to say there is no information about this rather than to "
        "describe a plausible root cause.",
        "semantic:abstention",
    ),

    # ============================================================================================
    # TOOL CALLING (15 total). Structured questions the agent should answer with a tool rather than
    # by reading passages: an explicit key, a count, a filter, a workspace-level fact.
    # ============================================================================================
    # --- issue lookup (4): the key is handed to the agent ---
    RagasCase(
        "What is ATC-43 about?",
        "ATC-43 is a done bug titled 'Checkout can create two orders from a double click': a repeated "
        "checkout submission on a slow connection created a second order and reduced stock twice.",
        "tool:issue-lookup",
    ),
    RagasCase(
        "Which epic does ATC-59 belong to?",
        "ATC-49.",
        "tool:issue-lookup",
    ),
    RagasCase(
        "Which issues are the subtasks of ATC-43?",
        "Three: ATC-44 (reject repeated checkout requests), ATC-45 (disable Place order while "
        "checkout is pending) and ATC-47 (cover slow checkout in regression tests).",
        "tool:issue-lookup",
    ),
    RagasCase(
        "What is the status of ATC-72 and which sprint is it in?",
        "ATC-72 is done and sits in AtlasCart Sprint 7.",
        "tool:issue-lookup",
    ),

    # --- filter / aggregation (6): counts and group-bys over the whole space ---
    RagasCase(
        "How many bugs are there in total, and how many are still open?",
        "There are 11 bugs and none are still open — all 11 are done.",
        "tool:filter-aggregation",
    ),
    RagasCase(
        "How many issues are bugs, how many are stories, and how many are tasks?",
        "11 bugs, 31 stories and 24 tasks. The space also holds 10 epics and 9 subtasks, 85 issues "
        "in total.",
        "tool:filter-aggregation",
    ),
    RagasCase(
        "Which issues are currently blocked?",
        "None. No issue in this workspace is currently in blocked status.",
        "tool:filter-aggregation",
    ),
    RagasCase(
        "How many issues are marked highest priority, and name a few of them.",
        "Eleven: ATC-3, ATC-29, ATC-30, ATC-34, ATC-43, ATC-44, ATC-45, ATC-56, ATC-59, ATC-66 and "
        "ATC-68.",
        "tool:filter-aggregation",
    ),
    RagasCase(
        "How many issues are currently in progress?",
        "Seven: ATC-4, ATC-52, ATC-77, ATC-78, ATC-80, ATC-81 and ATC-82.",
        "tool:filter-aggregation",
    ),
    RagasCase(
        "How many issues currently have nobody assigned to them?",
        "Three: ATC-1, ATC-89 and ATC-90.",
        "tool:filter-aggregation",
    ),

    # --- user / project lookup (3): assignees, sprints, workspace-level facts ---
    RagasCase(
        "Who has the most issues assigned in this workspace?",
        "Ava Patel, with 16 issues. Noah Kim is next with 13, then Sophia Nguyen with 12.",
        "tool:user-project-lookup",
    ),
    RagasCase(
        "How many sprints are there in total, and how many are complete?",
        "Nine sprints. Six are completed (AtlasCart Sprint 1 through Sprint 6), AtlasCart Sprint 7 "
        "is the active one, and two future sprints are planned for saving cards for faster repeat "
        "checkout.",
        "tool:user-project-lookup",
    ),
    RagasCase(
        "What is the current sprint and what is its goal?",
        "AtlasCart Sprint 7, running 2026-08-31 to 2026-09-11. Its goal is to prove the storefront "
        "is accessible, observable and ready for a larger public beta.",
        "tool:user-project-lookup",
    ),

    # --- multi-tool (2): needs more than one structured call composed ---
    RagasCase(
        "Which issues are in the active sprint, and what is the status of each?",
        "Ten issues in AtlasCart Sprint 7: ATC-72 is done; ATC-77, ATC-78, ATC-80, ATC-81 and "
        "ATC-82 are in progress; ATC-79, ATC-83, ATC-89 and ATC-90 are planned.",
        "tool:multi-tool",
    ),
    RagasCase(
        "Between ATC-43 and ATC-59, which was completed first?",
        "ATC-43. It closed in AtlasCart Sprint 3 (6 July to 17 July 2026), while ATC-59 closed in "
        "AtlasCart Sprint 4 (20 July to 31 July 2026).",
        "tool:multi-tool",
    ),
]


# --- /ask HTTP client -----------------------------------------------------------------------------


# ai-service accepts a user JWT or the internal service token; eval runs are internal callers.
INTERNAL_HEADERS = {
    "X-Internal-Token": os.getenv("AI_INTERNAL_SERVICE_TOKEN", "dev-internal-service-token-min-32-chars-long!!")
}


def ask(base_url: str, space_ids: List[int], question: str, timeout: float = 400.0) -> dict:
    payload = json.dumps({"question": question, "spaceIds": space_ids}).encode()
    req = urllib.request.Request(
        base_url + "/ask", data=payload, headers={"Content-Type": "application/json", **INTERNAL_HEADERS}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def collect_samples(
    base_url: str, space_ids: List[int], cases: Optional[List[RagasCase]] = None, concurrency: int = 1
) -> tuple[List[dict], List[str], List[dict]]:
    """Returns (ragas_samples, excluded_questions, per_question_cost).

    RAGAS's faithfulness/context_precision/context_recall are all defined as "is the answer
    supported by this retrieved context" — originally read narrowly here as "retrieved TEXT" only
    (`citations`, from search_knowledge_base/get_issue_comments/get_issue_details), a vanilla
    chunk-based-RAG assumption this system's hybrid tool surface doesn't fully share: count/list/
    relationship questions are answered by a structured, deterministic tool call (query_issues/
    query_sprints/get_issue_history) that is self-grounding by design (`_ground_answer` explicitly
    does not require a citation for it — see crag_loop.py) and never populated `citations`. An earlier
    version of this function excluded every such question from RAGAS's scope entirely — correct in
    spirit (don't score a claim against context that can't support it), but wider than it needed to
    be: those structured tool calls' own result text was always available on `AgentAnswer`, it just
    wasn't being surfaced through the API or fed to RAGAS. Fixed live after finding it caused a false
    "unfaithful" score on a genuinely correct, tool-grounded answer ("is there a blocked issue" —
    the "blocked" status came from query_issues, not from the one retrieved text chunk RAGAS could
    see).

    `AskResponse.structured_evidence` now carries exactly that text, so it's included in
    `retrieved_contexts` alongside `citations` here. The exclusion path below should now rarely
    trigger — only for a genuine non-abstaining answer with neither citations nor any structured
    tool call behind it at all (in practice: hasn't happened once since this fix; kept as a safety
    net, not removed, since "score a claim against literally no context" would be worse than
    excluding it).
    """
    cases = cases if cases is not None else CASES
    samples = []
    excluded = []
    cost = []

    def _fetch(case: RagasCase) -> tuple[Optional[dict], float, Optional[str]]:
        t0 = time.time()
        try:
            return ask(base_url, space_ids, case.question), time.time() - t0, None
        except urllib.error.HTTPError as e:
            return None, time.time() - t0, f"HTTP {e.code}"

    # Answers don't depend on each other, so they can be fetched in parallel; results are still
    # processed in case order so the output file is identical to a serial run's.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        fetched = list(pool.map(_fetch, cases))

    for i, (case, (body, elapsed, error)) in enumerate(zip(cases, fetched), 1):
        if error:
            print(f"[{i}/{len(cases)}] ERROR {case.question!r}: {error}", file=sys.stderr)
            continue
        citations = body.get("citations", [])
        structured_evidence = body.get("structuredEvidence", [])
        abstained = bool(body.get("abstained"))
        print(
            f"[{i}/{len(cases)}] {round(elapsed, 1)}s "
            f"abstained={abstained} citations={len(citations)} structured_evidence={len(structured_evidence)} "
            f"— {case.question}"
        )
        contexts = [c.get("content", "") for c in citations] + list(structured_evidence)
        if not contexts:
            if abstained:
                # An accurate "no context" context — genuinely nothing was retrieved or queried.
                contexts = ["No relevant information was found in the knowledge base for this query."]
            else:
                # Safety net for a case that (as of this fix) shouldn't occur in practice: a
                # non-abstaining answer with no text citations AND no structured tool call behind it.
                excluded.append(case.question)
                continue
        samples.append(
            {
                "user_input": case.question,
                "response": body.get("answer", ""),
                "retrieved_contexts": contexts,
                "reference": case.reference,
            }
        )
        # Cost, recorded per question next to quality. Retrieval depth and the number of query
        # angles are the model's to choose and both spend tokens, so a quality gain that is really
        # just a spending increase has to be visible as one. Without this the two are
        # indistinguishable in the result file and "the numbers went up" is unfalsifiable.
        cost.append(
            {
                "question": case.question,
                "input_tokens": body.get("inputTokens", 0),
                "output_tokens": body.get("outputTokens", 0),
                "cached_input_tokens": body.get("cachedInputTokens", 0),
                "retrieval_rounds": body.get("retrievalRounds", 0),
                "queries_used": len(body.get("queriesUsed") or []),
                # Wall-clock, not token count. Extra query angles are issued concurrently and cost
                # almost no extra tokens, but they still add model round-trips — so a change can be
                # free on the token bill and still make the product feel slower. That is a different
                # cost, borne by a different party, and it needs its own number.
                "seconds": round(elapsed, 1),
            }
        )
    return samples, excluded, cost


def _print_category_plan(cases: List[RagasCase]) -> None:
    """Show the composition of the set before spending an hour scoring it.

    A set that has drifted toward key lookups still reports four confident-looking metrics, and
    nothing in them says it can no longer see retrieval quality.
    """
    by_family: dict = {}
    for case in cases:
        family = case.category.split(":")[0] or "uncategorized"
        by_family.setdefault(family, []).append(case.category)
    total = len(cases)
    print(f"Question set: {total} cases")
    for family, categories in by_family.items():
        print(f"  {family:9s} {len(categories):3d}  {len(categories) / total:5.1%}")
        seen: dict = {}
        for category in categories:
            seen[category] = seen.get(category, 0) + 1
        for category, count in seen.items():
            print(f"    {category.split(':', 1)[-1]:22s} {count:3d}")


def _print_category_breakdown(df, cases: List[RagasCase]) -> dict:
    """Mean of each metric per category, joined back onto the scored rows by question text.

    Joined on text rather than by position because collect_samples drops any question that produced
    no evidence at all, so the row count can be smaller than the case count and positional zipping
    would silently attribute scores to the wrong categories.
    """
    category_of = {case.question: case.category for case in cases}
    metrics = [
        column for column in
        ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        if column in df.columns
    ]
    if "user_input" not in df.columns or not metrics:
        return {}

    scored = df.assign(category=df["user_input"].map(category_of))
    per_category: dict = {}
    # Coverage per (category, metric), not just per category overall. The overall coverage warning
    # above catches "half the run failed"; it does NOT catch "every failure landed in one 3-question
    # category", which is the more common and more dangerous shape — a category can show a perfect
    # 1.000 that is really the mean of a single surviving sample. Found live: two of the eight
    # categories here had every metric's failure concentrated inside them, and one category's
    # context_recall "1.000" turned out to be n=1 after the other two samples failed to score.
    print("\n" + f"{'category':24s} {'n':>3s} " + " ".join(f"{m[:9]:>9s}" for m in metrics))
    low_coverage: list = []
    for category, group in scored.groupby("category", dropna=True):
        n = len(group)
        row: dict = {"n": n}
        cells = []
        row["scored"] = {}
        for metric in metrics:
            scored_n = int(group[metric].notna().sum())
            mean = group[metric].dropna().mean()
            row[metric] = round(float(mean), 3) if scored_n else None
            row["scored"][metric] = scored_n
            flag = "" if scored_n == n else f"({scored_n}/{n})"
            if scored_n < n:
                low_coverage.append((category, metric, scored_n, n))
            cells.append(f"{row[metric]:6.3f}{flag:>5s}" if row[metric] is not None else f"{'--':>11s}")
        per_category[category] = row
        print(f"{category:24s} {n:3d} " + " ".join(cells))
    if low_coverage:
        print("\nWARNING: the following category/metric cells are means over FEWER samples than "
              "the category's n — a clean-looking number can be a single surviving sample:")
        for category, metric, scored_n, n in low_coverage:
            print(f"  {category}.{metric}: {scored_n}/{n} scored")
    return per_category


def _print_cost_summary(cost: List[dict], cases: List[RagasCase]) -> dict:
    """Mean tokens, retrieval rounds and query count — overall and per category."""
    if not cost:
        return {}
    category_of = {case.question: case.category for case in cases}
    fields = ("input_tokens", "output_tokens", "retrieval_rounds", "queries_used", "seconds")

    def _means(rows: List[dict]) -> dict:
        out = {f: round(sum(r.get(f, 0) for r in rows) / len(rows), 1) for f in fields}
        out["n"] = len(rows)
        return out

    by_category: dict = {}
    for row in cost:
        by_category.setdefault(category_of.get(row["question"], "uncategorized"), []).append(row)

    summary = {"overall": _means(cost),
               "by_category": {c: _means(rows) for c, rows in sorted(by_category.items())}}
    o = summary["overall"]
    print(f"\ncost: {o['input_tokens']:.0f} in / {o['output_tokens']:.0f} out tokens per question, "
          f"{o['retrieval_rounds']:.1f} retrieval rounds, {o['queries_used']:.1f} queries, "
          f"{o['seconds']:.1f}s")
    print(f"{'category':24s} {'n':>3s} {'in_tok':>8s} {'out_tok':>8s} {'rounds':>7s} {'queries':>8s}")
    for category, row in summary["by_category"].items():
        print(f"{category:24s} {row['n']:3d} {row['input_tokens']:8.0f} {row['output_tokens']:8.0f} "
              f"{row['retrieval_rounds']:7.1f} {row['queries_used']:8.1f}")
    return summary


def main() -> None:
    from eval.atlascart_ids import atlascart_space_id

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8200")
    # Resolved, not pinned: m3_seed reinstalls mint a new space id (see eval/atlascart_ids.py).
    parser.add_argument("--space-ids", type=int, nargs="+", default=None)
    # gemini-judge: ~1s per evaluation, ~$0.34 per 50-question run.
    parser.add_argument(
        "--llm-base-url", default="http://localhost:4000/v1",
        help="OpenAI-compatible endpoint for the RAGAS judge model (the LiteLLM gateway, so a hosted "
        "judge like gemini-judge/claude-judge is reachable).",
    )
    parser.add_argument(
        "--llm-api-key", default=os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-local-master-key"),
        help="Auth for --llm-base-url (the gateway's master or a virtual key).",
    )
    parser.add_argument("--llm-model", default="gemini-judge")
    # Only answer_relevancy uses an embedding. Same model as retrieval, so there is one fewer
    # "which embedding produced this number" question per result.
    parser.add_argument("--embedding-model", default="openai-embedding-3-small")
    parser.add_argument(
        "--embedding-base-url", default="http://localhost:4000/v1",
        help="OpenAI-compatible endpoint for the judge's own answer_relevancy embedding call.",
    )
    parser.add_argument(
        "--embedding-api-key", default=os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-local-master-key"),
        help="Auth for --embedding-base-url.",
    )
    parser.add_argument(
        "--reasoning-effort", default="none",
        help="Passed through to the judge model. With reasoning on, a judge-sized call can spend its "
        "whole token budget thinking and return nothing, which RAGAS then retries.",
    )
    parser.add_argument(
        "--out", default=str(Path(__file__).parent / "results" / "ragas_eval_result.json"),
    )
    parser.add_argument(
        "--ask-concurrency", type=int, default=1,
        help="Parallel /ask calls while collecting answers. Independent of --max-workers (the judge's "
        "concurrency); the answers do not change with concurrency.",
    )
    parser.add_argument(
        "--reuse-answers", default=None,
        help="Skip /ask entirely and score answers saved by an earlier run (<out>.answers.json).",
    )
    parser.add_argument(
        "--max-workers", type=int, default=1,
        help="RAGAS RunConfig max_workers: concurrent judge requests. A failed judge call becomes a "
        "silent NaN (raise_exceptions=False), so check scored_samples/total_samples after raising it.",
    )
    args = parser.parse_args()

    cases = CASES
    _print_category_plan(cases)

    # Resolved from the space key: a pinned id goes stale the first time the fixture is reinstalled.
    if not args.space_ids:
        args.space_ids = [atlascart_space_id()]

    answers_path = Path(args.out).with_suffix(".answers.json")
    if args.reuse_answers:
        saved = json.loads(Path(args.reuse_answers).read_text())
        samples, excluded, cost = saved["samples"], saved["excluded"], saved["cost"]
        print(f"Reusing {len(samples)} saved answers from {args.reuse_answers} (no /ask calls)")
    else:
        print(
            f"Collecting {len(cases)} live /ask samples from {args.url} (space_ids={args.space_ids}, "
            f"concurrency={args.ask_concurrency}) ..."
        )
        samples, excluded, cost = collect_samples(args.url, args.space_ids, cases, args.ask_concurrency)
        # Saved before scoring, so a failed judge run can be re-scored without new /ask calls.
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(json.dumps({"samples": samples, "excluded": excluded, "cost": cost}, indent=2))
        print(f"Answers saved to {answers_path} (rerun scoring only with --reuse-answers {answers_path})")
    if excluded:
        print(
            f"\n{len(excluded)} question(s) excluded from RAGAS scope (no citations AND no "
            f"structured tool evidence at all — see collect_samples' docstring; should be rare):"
        )
        for q in excluded:
            print(f"  - {q}")
    if not samples:
        print("No groundable samples collected — aborting.", file=sys.stderr)
        sys.exit(1)

    # Imported lazily: these pull in langchain/ragas, which real Ask-AI request-serving code never
    # needs — keeping the import at call time means a plain `python -m uvicorn app.main:app` never
    # pays for or depends on this eval-only dependency set.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=args.llm_base_url, api_key=args.llm_api_key, model=args.llm_model, temperature=0.0,
            # See --reasoning-effort's help text: without this, a reasoning-capable judge silently
            # burns its whole budget thinking and returns nothing, which RAGAS then retries. Passed
            # as a named arg, not via model_kwargs — langchain_openai warns that it knows this
            # parameter explicitly and stuffing it into model_kwargs is the unsupported path.
            reasoning_effort=args.reasoning_effort,
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            base_url=args.embedding_base_url, api_key=args.embedding_api_key, model=args.embedding_model,
            # The gateway takes plain strings; skipping the tokenizer-based length check avoids
            # pulling in `transformers` for inputs that are always short.
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
    )

    dataset = EvaluationDataset.from_list(samples)
    print(f"\nRunning RAGAS metrics (judge={args.llm_model}) ...")
    # Generous timeout: faithfulness chains several sequential judge calls per sample.
    run_config = RunConfig(timeout=900, max_workers=args.max_workers, max_retries=3)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
        run_config=run_config,
    )

    df = result.to_pandas()
    print("\n" + df.to_string())

    # `semantic:abstention` is excluded from the means, and scored separately below, because RAGAS
    # structurally cannot score a refusal: faithfulness decomposes an answer into claims and checks each against the context, and
    # "there is no information about this" contains no verifiable claim (two of three scored 0.00);
    # answer_relevancy generates questions from the answer and scores their similarity to the
    # original, and a refusal generates nothing to compare. Leaving them in drags all four means down by punishing the exact behaviour the category
    # exists to reward. What matters for an abstention case is binary — did it refuse — so that is
    # what gets reported.
    abstention_questions = {c.question for c in cases if c.category == "semantic:abstention"}
    scored_df = df[~df["user_input"].isin(abstention_questions)] if abstention_questions else df
    means = {
        col: round(float(scored_df[col].dropna().mean()), 3)
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if col in scored_df.columns
    }
    print("\nMean scores (excluding %d abstention case(s)):" % len(abstention_questions), means)

    abstention_result = {}
    if abstention_questions:
        refused = 0
        for _, row in df[df["user_input"].isin(abstention_questions)].iterrows():
            answer = str(row.get("response", "") or "").lower()
            # The agent's own fixed refusal string (app/agent/crag_loop.py's ABSTENTION_PHRASE), plus
            # the looser hedges it also produces. A clarification question ("which report should I
            # use?") is deliberately NOT counted: the corpus has no such data, so asking for a
            # narrower pointer is not the same as saying so.
            if ("no information about this" in answer
                    or "did not find" in answer or "didn't find" in answer
                    or "no results" in answer):
                refused += 1
        abstention_result = {"refused": refused, "total": len(abstention_questions)}
        print(f"Abstention cases: {refused}/{len(abstention_questions)} correctly refused")

    # How many samples each mean is actually over. `dropna()` above silently discards anything the
    # judge failed to score, so a run where half the cohort errored still prints four confident
    # numbers — and they are not comparable with a run that scored every sample. Found live twice:
    # once when the judge model unloaded on a TTL mid-run, and once when it was reloaded with an 8k
    # context instead of 262k, so every question with many retrieved contexts overflowed and scored
    # NaN. Both times the means looked entirely plausible. Printing the coverage, and refusing to
    # stay quiet below 90%, is what makes that failure visible instead of publishable.
    scored = {
        col: int(df[col].notna().sum())
        for col in means
    }
    total = len(df)
    print("scored samples: " + ", ".join(f"{c}={n}/{total}" for c, n in scored.items()))
    worst = min(scored.values()) if scored else total
    if worst < total:
        pct = worst / total if total else 0
        banner = "!!! " if pct < 0.9 else ""
        print(f"{banner}WARNING: at least one metric scored only {worst}/{total} samples "
              f"({pct:.0%}). Means are over the scored subset only and are NOT comparable with a "
              f"fully-scored run. Check the judge model: context length, TTL, and whether it stayed "
              f"loaded for the whole run.{' ' + banner if banner else ''}")
    per_category = _print_category_breakdown(df, cases)
    cost_summary = _print_cost_summary(cost, cases)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"means": means,
         # Sample coverage per metric, so a partially-scored run cannot be mistaken for a complete
         # one after the fact. Anything below `n` here means the means are over a subset.
         "scored_samples": scored,
         "total_samples": total,
         # Per-category means alongside the overall ones. The overall number answers "is the
         # system good"; this answers "good at what", which is the only form in which a
         # retrieval change is legible — a gain concentrated in semantic:* and flat in tool:*
         # is a different result from the same overall delta spread evenly, and a single mean
         # cannot tell them apart.
         "per_category": per_category,
         # Token spend per question, overall and per category. Read this next to per_category: a
         # retrieval change that lifts a category's scores while multiplying its token cost is a
         # different result from one that lifts them at the same cost, and only this makes the two
         # distinguishable after the fact.
         "cost": cost_summary,
         # Which prompt earned these scores — see app/llm/prompt_registry.py. Read from this
         # checkout, not from the service under test: this eval drives ai-service over HTTP, so the
         # two agree as long as the running service is this working tree (it is, in every documented
         # way of running this) and the fingerprints are worth exactly that assumption.
         "prompt_versions": versions_for("crag_answer_system", "crag_router_system"),
         "abstention": abstention_result,
        "excluded_no_context": excluded,
         "per_sample": df.to_dict(orient="records")},
        indent=2,
    ))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
