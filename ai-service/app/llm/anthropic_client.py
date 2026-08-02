"""Production LLM client: the official `anthropic` SDK (default model claude-opus-4-8).

Non-streaming is fine at 2048 max output tokens. Extended thinking is off for now: it would mean
carrying thinking blocks across turns, which complicates the loop — a natural next step.

Prompt caching, with two breakpoints:
  1. the system prompt — identical across calls (time rounded to 5 minutes), so shared across tool
     rounds, turns and even users within the window;
  2. the last block of the last message — caches "the conversation so far", so each next call pays
     full price only for what's new.
A cache write costs ~1.25x input, a read ~0.1x, so savings start from the second call. Cache tokens
are tracked separately from `input_tokens` because they're priced differently (pricing.py, types.py).
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List

import anthropic

from app.llm.types import LLMTurn, ToolCall, Usage

_CACHE_CONTROL = {"cache_control": {"type": "ephemeral"}}


def _cached_system(system: str) -> List[Dict[str, Any]]:
    return [{"type": "text", "text": system, **_CACHE_CONTROL}]


def _with_trailing_cache_breakpoint(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a NEW list with cache_control on the last message's last block (only that message is
    copied). Never mutates the caller's list — it's the loop's running conversation, and a stray
    marker would stick around in later turns.
    """
    if not messages:
        return messages
    *head, last = messages
    last = copy.deepcopy(last)
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, **_CACHE_CONTROL}]
    elif isinstance(content, list) and content:
        content[-1] = {**content[-1], **_CACHE_CONTROL}
    return [*head, last]


class AnthropicClient:
    def __init__(self, api_key: str, model: str, max_output_tokens: int):
        # Empty api_key falls through to the SDK's own env/profile resolution (ANTHROPIC_API_KEY,
        # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) rather than forcing a bad empty key.
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self._model = model
        self._max_tokens = max_output_tokens

    async def next_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> LLMTurn:
        request = dict(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_cached_system(system),
            messages=_with_trailing_cache_breakpoint(messages),
            tools=tools,
        )
        if tool_choice:
            request["tool_choice"] = {"type": "tool", "name": tool_choice}
        if on_text_delta is not None:
            # Streaming path (POST /ask/stream's final-answer synthesis). `messages.stream()`
            # accumulates the same message `messages.create()` would return, so `get_final_message()`
            # feeds the identical parsing below — the callback is the only added effect.
            async with self._client.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    if text:
                        on_text_delta(text)
                response = await stream.get_final_message()
        else:
            response = await self._client.messages.create(**request)

        blocks: List[Dict[str, Any]] = []
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                blocks.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        return LLMTurn(
            stop_reason=stop_reason,
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": blocks},
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                # Absent entirely on SDK responses where caching wasn't in play (older account/API
                # version); getattr keeps that a clean 0 rather than an AttributeError.
                cache_creation_input_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            ),
        )

    async def aclose(self) -> None:
        await self._client.close()
