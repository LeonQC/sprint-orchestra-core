"""Static $/token prices, turning token counts into an estimated cost per request.

A hardcoded table, like most LLM observability tools use: Anthropic has no pricing API, prices
change rarely, and a stale price is fixable by editing the table. Self-hosted models cost $0/token;
their tokens are still counted for capacity planning.

Only the Standard tier is modeled (our traffic is short-context and interactive). The ledger records
the rate actually applied to each token bucket (usage_ledger.py), so an invoice billed at a different
rate shows up as a visible difference instead of hiding inside one total.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    # Anthropic prompt-cache multipliers on the input price (same for every model): writing a cache
    # entry 1.25x, reading a hit 0.1x. Properties, so input_per_million is the only number to update.
    @property
    def cache_write_per_million(self) -> float:
        return self.input_per_million * 1.25

    @property
    def cache_read_per_million(self) -> float:
        return self.input_per_million * 0.1


# $ per 1M tokens. Source: Anthropic's published pricing (see claude-api skill, "Current Models").
_ANTHROPIC_PRICING = {
    "claude-fable-5": ModelPrice(10.00, 50.00),
    "claude-mythos-5": ModelPrice(10.00, 50.00),
    "claude-opus-4-8": ModelPrice(5.00, 25.00),
    "claude-opus-4-7": ModelPrice(5.00, 25.00),
    "claude-opus-4-6": ModelPrice(5.00, 25.00),
    "claude-sonnet-5": ModelPrice(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice(1.00, 5.00),
}

# $ per 1M tokens, Standard tier, short context, checked against OpenAI's pricing page (2026-07-29,
# re-checked 2026-08-10). Not yet confirmed by a real bill. Long-context and Batch tiers aren't
# modeled. The 2026-08-10 check found gpt-5.6-luna (the model we run) priced 5x too high
# ($1.00/$6.00 vs $0.20/$1.20) and gpt-5.6-terra wrong too — the table really does drift.
_OPENAI_PRICING = {
    "gpt-5.6-sol": ModelPrice(5.00, 30.00),
    "gpt-5.6-terra": ModelPrice(2.00, 12.00),
    "gpt-5.6-luna": ModelPrice(0.20, 1.20),
    "gpt-5.5": ModelPrice(5.00, 30.00),
    "gpt-5.5-pro": ModelPrice(30.00, 180.00),
    "gpt-5.4": ModelPrice(2.50, 15.00),
    "gpt-5.4-mini": ModelPrice(0.75, 4.50),
    "gpt-5.4-nano": ModelPrice(0.20, 1.25),
    "gpt-5.4-pro": ModelPrice(30.00, 180.00),
}

_FREE = ModelPrice(0.0, 0.0)

# Gateway route aliases (litellm-gateway/config.yaml) -> real model names, so our /stats estimate
# prices the right model (LiteLLM may return the alias or the model). Keep in sync with config.yaml.
# Unlisted aliases fall through to $0. The gateway's own cost tracking is the authoritative one.
_GATEWAY_ALIASES = {
    "openai-production": "gpt-5.6-luna",
    "claude-production": "claude-sonnet-5",
    "claude-judge": "claude-sonnet-5",
}


def price_for_model(model: str) -> ModelPrice:
    """Model name -> $/1M prices; unknown or self-hosted models are $0.

    Looks up both provider tables by exact name (names never collide), after resolving gateway
    aliases. Unknown names return $0 instead of raising: underestimating is safer than breaking /ask
    over a price lookup — but it means a newly added model must be priced, or it reads as free.
    """
    model = _GATEWAY_ALIASES.get(model, model)
    return _ANTHROPIC_PRICING.get(model) or _OPENAI_PRICING.get(model) or _FREE


def has_known_price(model: str) -> bool:
    """Whether a hosted model has an explicit price rather than the safe $0 fallback."""
    model = _GATEWAY_ALIASES.get(model, model)
    return model in _ANTHROPIC_PRICING or model in _OPENAI_PRICING


def resolve_model(model: str) -> str:
    """The real model a gateway alias bills as (other names unchanged). Stored in the ledger because
    invoices list the real model, never our alias.
    """
    return _GATEWAY_ALIASES.get(model, model)


@dataclass(frozen=True, slots=True)
class CostComponent:
    """One billed token bucket of one call: tokens x rate = dollars.

    Itemized because a single total can't be checked against an invoice, and buckets differ a lot
    (output is 6x input on gpt-5.6-luna; a cache read is a tenth of fresh input).
    """

    bucket: str  # input | output | cache_write | cache_read
    tokens: int
    rate_usd_per_million: float
    cost_usd: float


def itemize_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> List[CostComponent]:
    """Break one call's cost into billed buckets with their rates. Empty buckets are left out; the rate
    is kept even at $0, so "free" is visibly a priced-at-zero decision.
    """
    price = price_for_model(model)
    buckets = (
        ("input", input_tokens, price.input_per_million),
        ("output", output_tokens, price.output_per_million),
        ("cache_write", cache_creation_input_tokens, price.cache_write_per_million),
        ("cache_read", cache_read_input_tokens, price.cache_read_per_million),
    )
    return [
        CostComponent(bucket, tokens, rate, (tokens / 1_000_000) * rate)
        for bucket, tokens, rate in buckets
        if tokens
    ]


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Total $ for one call, defined as the sum of `itemize_cost`, so the total and its line items can
    never disagree.
    """
    return sum(
        component.cost_usd
        for component in itemize_cost(
            model, input_tokens, output_tokens,
            cache_creation_input_tokens, cache_read_input_tokens,
        )
    )
