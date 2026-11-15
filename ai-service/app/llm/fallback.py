"""Which model actually answered an LLM call, so a gateway fallback is priced and counted correctly.

LiteLLM echoes the requested route alias (`openai-production`) in a response's `model` field, and
reports the fallback deployment's real model name (`gemini-3.8-flash`) when it fell back, on both
/v1/responses and /chat/completions. Without this, fallback tokens would be priced at the requested
model's rate, and a fallback /ask would be recorded at a fraction of what it cost.
"""
from __future__ import annotations

import logging

from app.llm.pricing import resolve_model
from app.observability import LLM_FALLBACKS_TOTAL

logger = logging.getLogger(__name__)


def served_model(requested: str, reported: str | None) -> str:
    """The model that answered: `requested` unless the response names a different one."""
    if not reported or reported == requested:
        return requested
    # A provider called directly reports a dated snapshot ("gpt-5.6-luna-2026-...") of the same model.
    if reported.startswith(resolve_model(requested)):
        return requested
    LLM_FALLBACKS_TOTAL.labels(requested, reported).inc()
    logger.warning("LLM call answered by a fallback model", extra={"requested": requested, "served": reported})
    return reported
