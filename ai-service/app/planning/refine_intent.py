"""Read what a refine instruction asks for before deciding whether to read history.

  * New work ("add a backend test task and a frontend one") is a late requirement: it gets a history
    search and a grounded estimate, like any requirement.
  * An edit ("make issue 3 five points", "merge 2 and 4") specifies nothing new, so searching would
    only cost time and hand the planner irrelevant examples.

The result is the LIST of new work, each searched separately — better than the old single query mixing
the epic and the instruction. If this call fails, history is read anyway: wasted time beats an
ungrounded issue.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import Field

from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.llm.prompt_registry import template_loader
from app.llm.structured_output import StructuredOutput
from app.tracing import llm_step

logger = logging.getLogger(__name__)

_prompts = template_loader(__file__)
_MAX_VALIDATION_RETRIES = 2
# One short list and a sentence. Two orders of magnitude smaller than the planning call, which is the
# whole point: this exists to avoid a much more expensive read of the history.
_MAX_OUTPUT_TOKENS = 600


class RefineIntent(StructuredOutput):
    """The new work an instruction asks for, if any."""

    new_work: List[str] = Field(
        default_factory=list,
        description="Each distinct piece of NEW work the instruction asks to add, in its own words, "
                    "one per entry. Empty when the instruction only changes, merges, reorders, "
                    "re-estimates or removes issues that already exist — an empty list is the common "
                    "case and is never a failure.",
    )
    reasoning: str = Field(
        ..., min_length=1,
        description="One short sentence on why this instruction does or does not add work.",
    )


@dataclass(slots=True)
class RefineIntentResult:
    intent: RefineIntent
    degraded: bool = False
    error: Optional[str] = None
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def adds_work(self) -> bool:
        return bool(self.intent.new_work)


def _degraded(reason: str, error: Optional[str] = None) -> RefineIntentResult:
    """Fall back to "assume it adds work", so the caller retrieves as it always did."""
    return RefineIntentResult(
        intent=RefineIntent(new_work=[], reasoning=reason), degraded=True, error=error,
    )


async def read_refine_intent(client, model: str, instruction: str) -> RefineIntentResult:
    start = time.perf_counter()
    if not instruction.strip():
        return RefineIntentResult(
            intent=RefineIntent(new_work=[], reasoning="No instruction was given."),
        )
    try:
        with llm_step("plan_epic.read_refine_intent"):
            intent, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=RefineIntent,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": _prompts.render("refine_intent_system.jinja")},
                    {"role": "user", "content": instruction},
                ],
            )
    except Exception as exc:  # noqa: BLE001 - a failed read must not fail the edit
        logger.warning("could not read the refine instruction's intent", exc_info=True)
        return _degraded("The instruction could not be read; history was consulted anyway.", str(exc))

    usage = usage_from_completion(completion)
    cleaned = [text.strip() for text in intent.new_work if text and text.strip()]
    return RefineIntentResult(
        intent=intent.model_copy(update={"new_work": cleaned}),
        latency_seconds=time.perf_counter() - start,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cache_read_input_tokens,
        estimated_cost_usd=estimate_cost_usd(
            served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        ),
    )
