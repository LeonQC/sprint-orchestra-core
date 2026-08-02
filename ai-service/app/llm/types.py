"""Provider-agnostic LLM types.

The agent loop uses Anthropic's Messages-API shape (content blocks as dicts: text, tool_use,
tool_result) because that's the production target. AnthropicClient is nearly a pass-through;
OpenAICompatibleClient is the one place that converts to OpenAI's shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Protocol

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]


@dataclass(slots=True)
class Usage:
    """Token counts for one call, normalized for cost tracking (pricing.py).

    Default 0 when a server doesn't report usage — counting must never break the agent. The cache
    fields are Anthropic-only (0 elsewhere) and kept separate from `input_tokens` because they're
    priced differently.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # The part of the totals above that a gateway fallback model answered, keyed by that model, so it
    # can be priced at its own rate (app/llm/fallback.py).
    fallback: Dict[str, "Usage"] = field(default_factory=dict)

    def __add__(self, other: "Usage") -> "Usage":
        merged = dict(self.fallback)
        for model, usage in other.fallback.items():
            merged[model] = merged[model] + usage if model in merged else usage
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
            merged,
        )


@dataclass(slots=True)
class LLMTurn:
    """One assistant turn, normalized across providers."""

    stop_reason: Literal["tool_use", "end_turn"]
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    # The exact {"role": "assistant", "content": [...]} dict to append to `messages` for the next
    # call — content blocks are plain dicts in Anthropic's native shape (see module docstring).
    assistant_message: Dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    async def next_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | None = None,
        # When given, called with each answer-text fragment as it is generated (POST /ask/stream's
        # final-answer synthesis only). Purely a surfacing hook — the returned LLMTurn is identical
        # whether or not it is passed. A client that cannot stream may ignore it and answer whole.
        on_text_delta: Callable[[str], None] | None = None,
    ) -> LLMTurn: ...

    async def aclose(self) -> None: ...
