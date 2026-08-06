"""AI task drafting: one LLM call -> validated title / labels / estimate / dependencies.

If instructor's retries run out (model down, or a small model that can't produce valid output), it
never returns a 500: it falls back to a valid TaskDraft from the raw text (first line as title) with
`degraded=True`, so the user still gets a usable starting point.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from app.drafting.prompts import render_task_draft_prompt
from app.drafting.schemas import TaskDraft
from app.llm.instructor_usage import usage_from_completion
from app.llm.fallback import served_model
from app.llm.pricing import estimate_cost_usd
from app.tracing import llm_step

logger = logging.getLogger(__name__)

# instructor's re-ask retries (on top of HTTP retries). Small, because the user is waiting.
_MAX_VALIDATION_RETRIES = 2
# Generous on purpose: a reasoning model can spend many tokens thinking before the
# JSON, and at 500 and then 1500 it ran out mid-output on long descriptions, falsely triggering the
# fallback. It's a ceiling, not a target, so short calls aren't slower.
_MAX_OUTPUT_TOKENS = 4000


@dataclass(slots=True)
class DraftResult:
    draft: TaskDraft
    degraded: bool
    latency_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # of input_tokens, the prompt-cache hit (billed ~10%); see app/llm/instructor_usage.py
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None


def _fallback_draft(description: str) -> TaskDraft:
    first_line = description.strip().splitlines()[0] if description.strip() else ""
    title = first_line[:120] or "Untitled task"
    return TaskDraft(title=title, issue_type="task", labels=[], estimate_story_points=None, dependencies=[])


async def draft_task(
    client, model: str, description: str, existing_labels: Optional[List[str]] = None
) -> DraftResult:
    system_prompt = render_task_draft_prompt(existing_labels)
    start = time.perf_counter()
    try:
        # `create_with_completion` also returns the raw response, the only place token counts live.
        # Without it this feature's spend was missing from /stats and the usage ledger.
        with llm_step("draft_task.draft"):
            draft, completion = await client.chat.completions.create_with_completion(
                model=model,
                response_model=TaskDraft,
                max_retries=_MAX_VALIDATION_RETRIES,
                max_tokens=_MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": description},
                ],
            )
        usage = usage_from_completion(completion)
        return DraftResult(
            draft=draft,
            degraded=False,
            latency_seconds=time.perf_counter() - start,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens,
            estimated_cost_usd=estimate_cost_usd(
                served_model(model, getattr(completion, "model", None)), usage.input_tokens, usage.output_tokens,
                usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a structured-output failure must degrade, never 500
        logger.warning(
            "task draft generation failed after retries; falling back to a safe default",
            extra={"error": str(exc)},
        )
        return DraftResult(
            draft=_fallback_draft(description),
            degraded=True,
            latency_seconds=time.perf_counter() - start,
            error=str(exc),
        )
