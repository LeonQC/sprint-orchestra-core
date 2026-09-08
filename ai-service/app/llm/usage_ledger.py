"""Per-request cost records: who spent what, on which AI feature, at what rate.

For the month-end question — "a $1,000 invoice arrived; what was bought, by whom, and does the
provider's math match ours?" — each record carries:
  * who: the authenticated user (X-User-Id, set from the verified JWT);
  * which feature: ask_ai, summarize_comments, draft_task, ...;
  * which conversation: a chat is several /ask calls, and the conversation's cost is what users
    recognize;
  * at what rate: tokens x published price, itemized per bucket (pricing.py `CostComponent`).

Written as span attributes, not our own table: Phoenix is already a durable, queryable store, and a
second table would need its own writes and backups and would drift from the trace. `user.id`,
`session.id` and `metadata` follow OpenInference conventions, so Phoenix's session views just work.

The itemized breakdown is a JSON string under `metadata` (OTel attributes can't be nested);
`ai.usage.*` scalars are copied out for filtering and summing. Not `llm.token_count.*`: the SDK
spans underneath already set those, and repeating them would double-count. Keeping our numbers
separate also means a mismatch with Phoenix's own rollup is a real signal.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

from app.auth import user_quota
from app.llm.pricing import CostComponent, has_known_price, itemize_cost, resolve_model
from app.llm.types import Usage

# Feature names, matching what the product buttons say — this is what the cost report groups by,
# so it should read in product terms (`comment_digest` used to look like "someone left a comment").
# Constants, not literals: a typo would silently split one feature into two rows.
FEATURE_ASK_AI = "ask_ai"                                    # UI: "Ask AI"
FEATURE_INVESTIGATE = "multi_agent_investigation"             # Explicit /investigate endpoint
FEATURE_SUMMARIZE_COMMENTS = "summarize_comments"            # UI: "Summarize comments"
FEATURE_IMPROVE_DESCRIPTION = "improve_description"          # UI: "Improve description"
FEATURE_LINK_SIMILAR_WORK_ITEMS = "link_similar_work_items"  # UI: "Link similar work items"
FEATURE_DRAFT_TASK = "draft_task"                            # UI: "Draft Task"
FEATURE_PLAN_EPIC = "plan_epic"                              # UI: "Plan Epic"
FEATURE_PLAN_EPIC_REFINE = "plan_epic_refine"                # UI: "Plan Epic", the refine action
FEATURE_SPRINT_PACE = "sprint_pace"                          # UI: "Sprint Pace"
FEATURE_SPRINT_RECOVERY = "sprint_recovery"                  # UI: "Sprint Recovery"
FEATURE_SEMANTIC_SEARCH = "semantic_search"                  # UI: "Semantic search"

# This project's own namespace — deliberately not `llm.*`, see the module docstring.
FEATURE = "ai.feature"
USAGE_INPUT_TOKENS = "ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "ai.usage.output_tokens"
USAGE_CACHED_INPUT_TOKENS = "ai.usage.cached_input_tokens"
USAGE_CACHE_WRITE_TOKENS = "ai.usage.cache_write_tokens"
USAGE_TOTAL_TOKENS = "ai.usage.total_tokens"
USAGE_COST_USD = "ai.usage.cost_usd"
USAGE_PRICING_STATUS = "ai.usage.pricing_status"
USAGE_BILLED_MODELS = "ai.usage.billed_models"

# Worst first: a request with one priced and one unpriced model isn't fully priced. "unavailable" is
# a real hosted call we can't price, i.e. the total is an undercount.
_PRICING_STATUS_SEVERITY = {"unavailable": 2, "estimated": 1, "free_local": 0}


@dataclass(frozen=True, slots=True)
class Caller:
    """Request-scoped identity. Empty strings, never placeholders — see `caller_from_headers`."""

    user_id: str
    username: str
    session_id: str


@dataclass(frozen=True, slots=True)
class ModelCharge:
    """What one model, in one role, cost within a single request."""

    role: str  # answer_agent | router | structured_output
    provider: str
    model: str  # as requested — may be a gateway route alias
    billed_model: str  # what that alias actually bills as
    deployment: str  # local | hosted
    pricing_status: str  # free_local | estimated | unavailable
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    cost_usd: float
    components: List[CostComponent]

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens + self.output_tokens
            + self.cached_input_tokens + self.cache_write_tokens
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "billed_model": self.billed_model,
            "deployment": self.deployment,
            "pricing_status": self.pricing_status,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "components": [
                {
                    "bucket": c.bucket,
                    "tokens": c.tokens,
                    "rate_usd_per_million": c.rate_usd_per_million,
                    "cost_usd": round(c.cost_usd, 8),
                }
                for c in self.components
            ],
        }


def charge_for(role: str, provider: str, model: str, deployment: str, usage: Usage) -> ModelCharge:
    """Price one model role's usage, keeping the rate card used. `deployment == "local"` is $0 without
    looking at the table — a self-hosted model has no per-token bill, whatever its name.
    """
    if deployment == "local":
        pricing_status = "free_local"
    elif has_known_price(model):
        pricing_status = "estimated"
    else:
        pricing_status = "unavailable"

    components = (
        []
        if deployment == "local"
        else itemize_cost(
            model, usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        )
    )
    return ModelCharge(
        role=role,
        provider=provider,
        model=model,
        billed_model=resolve_model(model),
        deployment=deployment,
        pricing_status=pricing_status,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        cache_write_tokens=usage.cache_creation_input_tokens,
        cost_usd=sum(c.cost_usd for c in components),
        components=components,
    )


@dataclass(slots=True)
class UsageLedger:
    """One AI feature invocation's complete billing record."""

    feature: str
    caller: Caller
    request_id: str
    charges: List[ModelCharge]
    # Context that makes a line item recognizable (the question, cache hit, which tool answered).
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> Usage:
        total = Usage()
        for charge in self.charges:
            total = total + Usage(
                input_tokens=charge.input_tokens,
                output_tokens=charge.output_tokens,
                cache_creation_input_tokens=charge.cache_write_tokens,
                cache_read_input_tokens=charge.cached_input_tokens,
            )
        return total

    @property
    def cost_usd(self) -> float:
        return sum(charge.cost_usd for charge in self.charges)

    @property
    def total_tokens(self) -> int:
        return sum(charge.total_tokens for charge in self.charges)

    @property
    def pricing_status(self) -> str:
        if not self.charges:
            return "free_local"
        return max(
            (c.pricing_status for c in self.charges),
            key=lambda status: _PRICING_STATUS_SEVERITY.get(status, 0),
        )

    def as_dict(self) -> Dict[str, Any]:
        usage = self.usage
        return {
            "feature": self.feature,
            "user_id": self.caller.user_id,
            "username": self.caller.username,
            "session_id": self.caller.session_id,
            "request_id": self.request_id,
            "detail": self.detail,
            "totals": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cache_read_input_tokens,
                "cache_write_tokens": usage.cache_creation_input_tokens,
                "total_tokens": self.total_tokens,
                "cost_usd": round(self.cost_usd, 8),
                "pricing_status": self.pricing_status,
            },
            "charges": [charge.as_dict() for charge in self.charges],
        }

    def attach(self, span=None) -> None:
        """Write this record onto `span` (default: the current span).

        The explicit span is for /ask/stream, whose body runs after the handler returned, where the
        current span can't be relied on. Call this inside the span's `with` block — OTel drops writes
        to an ended span. With tracing off it's a no-op (non-recording span), so a cost record can
        never fail a request.
        """
        # Here because every ledger, HTTP or workflow, passes through attach() exactly once.
        user_quota.record_spend(self.caller.user_id, self.cost_usd)
        span = span if span is not None else trace.get_current_span()
        usage = self.usage
        span.set_attribute(FEATURE, self.feature)
        span.set_attribute(SpanAttributes.SESSION_ID, self.caller.session_id)
        # Only when known: an empty `user.id` would make Phoenix's user filter match every internal
        # eval/demo call as if it were one anonymous person, which is worse than an absent attribute.
        if self.caller.user_id:
            span.set_attribute(SpanAttributes.USER_ID, self.caller.user_id)
        span.set_attribute(SpanAttributes.TAG_TAGS, [self.feature])
        span.set_attribute(SpanAttributes.METADATA, json.dumps(self.as_dict()))
        span.set_attribute(USAGE_INPUT_TOKENS, usage.input_tokens)
        span.set_attribute(USAGE_OUTPUT_TOKENS, usage.output_tokens)
        span.set_attribute(USAGE_CACHED_INPUT_TOKENS, usage.cache_read_input_tokens)
        span.set_attribute(USAGE_CACHE_WRITE_TOKENS, usage.cache_creation_input_tokens)
        span.set_attribute(USAGE_TOTAL_TOKENS, self.total_tokens)
        span.set_attribute(USAGE_COST_USD, self.cost_usd)
        span.set_attribute(USAGE_PRICING_STATUS, self.pricing_status)
        span.set_attribute(
            USAGE_BILLED_MODELS,
            ",".join(sorted({c.billed_model for c in self.charges if c.billed_model})),
        )


def caller_from_headers(headers, history_key: str = "") -> Caller:
    """Caller identity from the X-User-Id / X-Username headers set by app/auth/request_auth.py.

    User fields stay empty for internal callers (eval scripts), so our own traffic isn't charged to a
    fake user. Session: an explicit `x-session-id` wins; otherwise a hash of the conversation's first
    question, which is the same on every turn of one chat (hashed, so no question text leaks). A
    first question with no history gets a fresh id.
    """
    session_id = (headers.get("x-session-id") or "").strip()
    user_id = headers.get("x-user-id") or ""
    if not session_id:
        if history_key:
            digest = hashlib.sha256(f"{user_id}\x00{history_key}".encode()).hexdigest()
            session_id = f"chat-{digest[:16]}"
        else:
            session_id = f"chat-{uuid.uuid4().hex[:16]}"
    return Caller(user_id=user_id, username=headers.get("x-username") or "", session_id=session_id)
