"""Turn instructor's raw provider response into `Usage`, for every caller of `create_with_completion`.

This existed twice, and only one copy handled OpenAI's cached tokens: OpenAI's `prompt_tokens`
includes them while Anthropic reports them separately, so the other copy overstated fresh input and
cost. One shared copy now.
"""
from __future__ import annotations

from app.llm.types import Usage


def usage_from_completion(completion) -> Usage:
    """Anthropic: input_tokens/output_tokens (+ cache counters). OpenAI-compatible: prompt_tokens /
    completion_tokens, cached under prompt_tokens_details.cached_tokens. Missing values count as 0 —
    a server reporting no usage means "unmeasured", never an error.
    """
    raw = getattr(completion, "usage", None)
    if raw is None:
        return Usage()

    def _int(*names: str) -> int:
        for name in names:
            value = getattr(raw, name, None)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    cached = 0
    details = getattr(raw, "prompt_tokens_details", None)
    if details is not None and isinstance(getattr(details, "cached_tokens", None), (int, float)):
        cached = int(details.cached_tokens)
    input_tokens = _int("input_tokens", "prompt_tokens")
    return Usage(
        # OpenAI's prompt_tokens includes cached tokens; subtract them so both providers are priced the
        # same way (cached input costs a tenth).
        input_tokens=max(0, input_tokens - cached) if cached else input_tokens,
        output_tokens=_int("output_tokens", "completion_tokens"),
        cache_creation_input_tokens=_int("cache_creation_input_tokens"),
        cache_read_input_tokens=_int("cache_read_input_tokens") or cached,
    )
