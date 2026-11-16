"""LLM client for the LiteLLM gateway (or any OpenAI-compatible server), over the Responses API.

The default provider (`AI_LLM_PROVIDER=openai_compatible`). This is the one place that converts between the Anthropic message/tool shape the agent uses and
OpenAI's shape.

Uses the official `openai` SDK, not raw httpx, because OpenInference traces by patching that SDK —
raw HTTP calls never show up in Phoenix.

Uses /v1/responses because /v1/chat/completions rejects function tools combined with any
reasoning effort other than "none" (seen with gpt-5.6-luna), and reasoning helps tool selection.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List

from openai import AsyncOpenAI

from app.llm.circuit_breaker import breaker_for, is_upstream_failure
from app.llm.fallback import served_model
from app.llm.types import LLMTurn, ToolCall, Usage
from app.observability import LLM_RESPONSE_CACHE_HITS_TOTAL, get_billing_user

logger = logging.getLogger(__name__)

# LiteLLM sets this response header only when the reply was replayed from its response cache.
_CACHE_HIT_HEADER = "x-litellm-cache-key"


_TEXT_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_scalar(value: str) -> Any:
    """Parse JSON-shaped parameter values while leaving ordinary model text as a string."""
    value = value.strip()
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def _parse_text_tool_calls(text: str) -> tuple[str, List[ToolCall]]:
    """Recover tool calls a model wrote into its text output.

    Some servers don't convert the model's `<tool_call>` syntax into `tool_calls`, which leaks markup
    into the UI and skips the tool. Handles the XML-like and JSON-in-`<tool_call>` forms. Only
    complete, parseable blocks are removed; malformed ones stay visible for debugging.
    """
    calls: List[ToolCall] = []
    consumed_spans: List[tuple[int, int]] = []

    for index, match in enumerate(_TEXT_TOOL_CALL_RE.finditer(text), start=1):
        payload = match.group(1).strip()
        name: str | None = None
        arguments: Dict[str, Any] | None = None

        xml_match = _XML_FUNCTION_RE.fullmatch(payload)
        if xml_match:
            name = xml_match.group(1)
            arguments = {
                parameter.group(1): _parse_scalar(parameter.group(2))
                for parameter in _XML_PARAMETER_RE.finditer(xml_match.group(2))
            }
        else:
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                function = decoded.get("function")
                if isinstance(function, dict):
                    name = function.get("name")
                    raw_arguments = function.get("arguments", {})
                else:
                    name = decoded.get("name")
                    raw_arguments = decoded.get("arguments", decoded.get("input", {}))
                if isinstance(raw_arguments, str):
                    try:
                        raw_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        raw_arguments = None
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments

        if name and arguments is not None:
            calls.append(ToolCall(id=f"text_tool_call_{index}", name=name, input=arguments))
            consumed_spans.append(match.span())

    if not consumed_spans:
        return text, []

    pieces: List[str] = []
    cursor = 0
    for start, end in consumed_spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    cleaned = "".join(pieces).strip()
    return cleaned, calls


# GPT-5 / gpt-oss "harmony" channel markup. Tool calls should arrive as structured items, but
# somewhere in the model -> LiteLLM -> Responses chain a commentary-channel call
# (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}`) gets flattened into
# the answer text (seen with gpt-5.6-luna via the gateway). It arrives more or less mangled, down to
# a bare `: {args}` glued to the answer. Both forms are handled; the call is recovered when that
# tool was offered.
_HARMONY_JSON = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
_HARMONY_STREAM_RE = re.compile(r"\bto=(?:functions\.)?[A-Za-z_][A-Za-z0-9_]*")
# The constrain marker has also arrived as a few garbage characters right before "json"
# (e.g. "ოციjson"), sometimes after an HTML space entity. Narrow and start-anchored: only treated as
# debris when it leads straight into a JSON object welded to answer text.
_HARMONY_LEADING_NOISE = (
    r"[\s:;,.\-`]*"
    r"(?:(?:&(?:#x(?:20|a0)|#(?:32|160)|nbsp);)[\s]*)*"
    r"(?:<\|[a-z_]+\|>[\s]*)*"
    r"(?:[^\x00-\x7F\s]{1,12})?"
)
# A JSON blob at the very start of the answer, welded to the following text — the fully stripped
# form of the leak (":\n{\"issue_keys\":[...]}Yes. Checkout-related issues include:"). Real answers
# don't start like that, so it's safe to strip from the front only (`\A`).
_HARMONY_LEADING_JSON_RE = re.compile(
    r"\A" + _HARMONY_LEADING_NOISE + r"(?:json\b\s*)?(?:<\|[a-z_]+\|>\s*)*"
    r"(?P<args>" + _HARMONY_JSON + r")(?:\s*<\|[a-z_]+\|>)*(?=\s*\S)",
    re.IGNORECASE,
)
_HARMONY_LEADING_STREAM_RE = re.compile(
    r"\A" + _HARMONY_LEADING_NOISE + r"(?:json\b\s*)?(?:<\|[a-z_]+\|>\s*)*\{",
    re.IGNORECASE,
)
# Several junk tokens before "json" ("Zusammen 代ന്തjson", "Japgolly 代 Аԥс json"), sometimes
# with a stray word welded on after the blob ("}-offsetofThe ..."). Only when that prefix holds a
# non-ASCII character, so an English answer that happens to open with a few words, "json" and an
# object is left alone. Case-sensitive on purpose: the welded
# word is lowercase and the answer starts uppercase.
_HARMONY_GARBLED_PREFIX = (
    r"[\s:;,.\-`]*"
    r"(?=(?:(?!json)[^\n]){0,80}?[^\x00-\x7F])"
    r"(?:\S{1,16}[ \t]+){0,4}\S{0,16}?"
)
_HARMONY_GARBLED_JSON_RE = re.compile(
    r"\A" + _HARMONY_GARBLED_PREFIX + r"json\s*(?P<args>" + _HARMONY_JSON + r")"
    r"(?:[-_][a-z]{1,12}(?=[A-Z*]))?(?=\s*\S)"
)
_HARMONY_GARBLED_STREAM_RE = re.compile(r"\A" + _HARMONY_GARBLED_PREFIX + r"json\s*\{")


def _looks_like_harmony_stream_leak(text: str) -> bool:
    """Recognize protocol debris before any buffered preview text reaches the browser."""
    return bool(
        "<tool_call" in text
        or "<|channel|>" in text
        or ("to=" in text and _HARMONY_STREAM_RE.search(text))
        or _HARMONY_LEADING_STREAM_RE.search(text)
        or _HARMONY_GARBLED_STREAM_RE.search(text)
    )


def _strip_leading_blobs(text: str, known_tool_names: "frozenset[str]", calls: List[ToolCall]) -> str:
    """Strip leaked JSON blobs from the front of the answer (the `to=...` header already gone).

    Looped: the same blob has arrived twice in a row (":\n{...} :\n{...}The answer"). The first blob
    becomes a recovered call only when exactly one tool was on offer; otherwise it is just removed.
    """
    for _ in range(5):
        leading = _HARMONY_LEADING_JSON_RE.match(text) or _HARMONY_GARBLED_JSON_RE.match(text)
        if not leading:
            break
        try:
            decoded = json.loads(leading.group("args"))
        except json.JSONDecodeError:
            decoded = None
        if not calls and isinstance(decoded, dict) and len(known_tool_names) == 1:
            calls.append(
                ToolCall(id="harmony_tool_call_0", name=next(iter(known_tool_names)), input=decoded)
            )
        text = text[leading.end():].lstrip()
    return text


def _parse_harmony_tool_calls(
    text: str, known_tool_names: "frozenset[str]" = frozenset()
) -> tuple[str, List[ToolCall]]:
    """Strip leaked harmony markup, recovering the tool call it encodes when possible.

    Only spans that clearly look like a channel header are removed; a plain `to=word` in prose stays,
    and anything unparseable stays visible for debugging.
    """
    calls: List[ToolCall] = []

    text = _strip_leading_blobs(text, known_tool_names, calls)

    name_alt = (
        "|".join(sorted((re.escape(n) for n in known_tool_names), key=len, reverse=True))
        or r"[A-Za-z_][A-Za-z0-9_]*"
    )
    # A header with free-form debris before its JSON on the same line: "to=query_issues  (JSON): {...}",
    # "to=query_issues 代 geschniegeltասպանjson  code\n{...}". The JSON is required, so a plain
    # `to=word` in prose is never touched.
    debris_header = re.compile(
        r"(?:<\|[a-z_]+\|>[ \t]*)*(?:(?:commentary|analysis|final)\b[ \t]*)?"
        r"to=(?:functions\.)?(?P<name>" + name_alt + r")\b[^{\n]{0,48}\n?[ \t]*"
        r"(?P<args>" + _HARMONY_JSON + r")",
        re.IGNORECASE,
    )
    debris_matches = list(debris_header.finditer(text))
    for match in reversed(debris_matches):
        try:
            decoded = json.loads(match.group("args"))
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict) and match.group("name") in known_tool_names:
            calls.append(ToolCall(id=f"harmony_tool_call_d{match.start()}", name=match.group("name"), input=decoded))
        text = text[:match.start()] + text[match.end():]
    if debris_matches:
        text = _strip_leading_blobs(text.strip(), known_tool_names, calls)
    pattern = re.compile(
        r"(?:<\|[a-z_]+\|>[ \t]*)*"
        r"(?:(?P<channel>commentary|analysis|final)\b[ \t]*)?"
        r"(?P<prefix>to=(?:functions\.)?)(?P<name>" + name_alt + r")\b"
        r"(?:[ \t]*<\|[a-z_]+\|>)*"
        r"(?P<constrain>[ \t]*(?:json|code|python|plain|text|markdown|yaml))?"  # may be glued to <|constrain|>
        r"(?:[ \t]+[^\x00-\x7F]{1,12})?"           # a mis-decoded constrain token, e.g. Cyrillic
        r"(?:[ \t]*<\|[a-z_]+\|>)*"
        r"(?:[ \t\r\n]*(?P<args>" + _HARMONY_JSON + r"))?",
        re.IGNORECASE,
    )
    spans: List[tuple[int, int]] = []
    for index, match in enumerate(pattern.finditer(text), start=1):
        looks_like_header = bool(
            match.group("args")
            or match.group("constrain")
            or match.group("channel")
            or "functions." in match.group("prefix")
            or "<|" in match.group(0)
        )
        if not looks_like_header:
            continue
        spans.append(match.span())
        raw = match.group("args")
        if not raw:
            continue
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and match.group("name") in known_tool_names:
            calls.append(
                ToolCall(id=f"harmony_tool_call_{index}", name=match.group("name"), input=decoded)
            )

    if not spans:
        return text, calls
    pieces: List[str] = []
    cursor = 0
    for start, end in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    # Removing a `to=NAME code:` header can uncover a ":\n{...}" blob at the front.
    return _strip_leading_blobs("".join(pieces).strip(), known_tool_names, calls), calls


def _to_responses_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Responses API tools are flat (name/description/parameters at top level), unlike Chat
    # Completions' nested "function" key.
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        for t in anthropic_tools
    ]


def _to_responses_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic-style messages into the Responses API's flat `input` items.

    The system prompt goes separately as `instructions` (see `next_turn`). Tool calls and results are
    their own item types (`function_call` / `function_call_output`), linked by `call_id`.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue

        if msg["role"] == "user":
            # A user turn's content list is either plain text blocks or tool_result blocks (never
            # both, in this agent's usage).
            for block in content:
                if block["type"] == "tool_result":
                    out.append({
                        "type": "function_call_output",
                        "call_id": block["tool_use_id"],
                        "output": block["content"],
                    })
                elif block["type"] == "text":
                    out.append({"role": "user", "content": block["text"]})
        else:  # assistant
            text = next((b["text"] for b in content if b["type"] == "text"), None)
            if text:
                out.append({"role": "assistant", "content": text})
            for b in content:
                if b["type"] == "tool_use":
                    out.append({
                        "type": "function_call",
                        "call_id": b["id"],
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    })
    return out


class OpenAICompatibleClient:
    def __init__(
        self, base_url: str, api_key: str, model: str, max_output_tokens: int,
        request_timeout_seconds: float = 300.0, reasoning_effort: str = "low",
        response_cache: bool = False,
    ):
        self._model = model
        self._max_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        # Opt in to the gateway's exact-match response cache (litellm-gateway/config.yaml).
        self._response_cache = response_cache
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=request_timeout_seconds,
            # Retries and provider fallback belong to the LiteLLM gateway, the one layer that knows
            # which providers are cooling down. See `is_upstream_failure`.
            max_retries=0,
        )

    async def next_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | None = None,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> LLMTurn:
        # `on_text_delta` turns on streaming and receives each answer-text fragment. The returned
        # LLMTurn is the same either way (rebuilt from `response.completed`). Only /ask/stream's final
        # answer passes it. Tool-call text leaks are held back while streaming (below) and stripped
        # from the final text.
        payload = {
            "model": self._model,
            "instructions": system,
            "input": _to_responses_input(messages),
            "tools": _to_responses_tools(tools),
            # Responses API's field for the output token budget — "max_output_tokens", not Chat
            # Completions' "max_tokens"/"max_completion_tokens".
            "max_output_tokens": self._max_tokens,
            # Light reasoning by default: tool selection benefits from some, without spending a lot
            # on narrow calls.
            "reasoning": {"effort": self._reasoning_effort},
        }
        # The end user, in LiteLLM's end-user header. All calls share one gateway key, so this is the
        # only way LiteLLM can track cost, budgets and rate limits per person. Not OpenAI's `user`
        # body field: LiteLLM puts that in the response-cache key, so no one could reuse another
        # person's identical call. Omitted (not sent empty) when unknown — eval runs have no user,
        # and "" would lump them together.
        billing_user = get_billing_user()
        if billing_user:
            payload["extra_headers"] = {"x-litellm-end-user-id": billing_user}
        if self._response_cache:
            payload["extra_body"] = {"cache": {"use-cache": True}}
        if tool_choice:
            # Offer only the chosen tool and require a call (string form is the most widely supported).
            payload["tools"] = [
                tool for tool in payload["tools"]
                if tool["name"] == tool_choice
            ]
            payload["tool_choice"] = "required"

        cache_hit = [False]

        async def _create(**kwargs: Any) -> Any:
            if not self._response_cache:
                return await self._client.responses.create(**payload, **kwargs)
            # The raw form only to read the hit header; the parsed body/stream is the same object.
            raw = await self._client.responses.with_raw_response.create(**payload, **kwargs)
            cache_hit[0] = _CACHE_HIT_HEADER in raw.headers
            return raw.parse()

        async def _call() -> Dict[str, Any]:
            response = await _create()
            return response.model_dump()

        # [n_forwarded] so the fallback below can tell "failed before any delta" (safe to retry
        # non-streamed) from "failed mid-answer" (a retry would replay text the user already saw).
        async def _call_streaming(callback: Callable[[str], None], n_forwarded: List[int]) -> Dict[str, Any]:
            # Hold back a short prefix before showing anything: once garbage is painted in the UI it
            # can't be taken back. Leaked tool headers show up near the start, so normal answers only
            # pay a tiny delay.
            preview_started = False
            preview_blocked = False
            preview_buffer = ""
            seen = ""
            stream = await _create(stream=True)
            final = None
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    seen += delta
                    if not preview_started and not preview_blocked:
                        preview_buffer += delta
                        if _looks_like_harmony_stream_leak(seen):
                            preview_blocked = True
                            preview_buffer = ""
                        elif len(preview_buffer) >= 64:
                            callback(preview_buffer)
                            n_forwarded[0] += 1
                            preview_started = True
                            preview_buffer = ""
                    elif preview_started and delta:
                        callback(delta)
                        n_forwarded[0] += 1
                elif etype == "response.completed":
                    final = event.response
            if final is None:
                raise RuntimeError("Responses stream ended without a response.completed event")
            return final.model_dump()

        forwarded = [0]

        async def _send() -> Dict[str, Any]:
            if on_text_delta is None:
                return await _call()
            try:
                return await _call_streaming(on_text_delta, forwarded)
            except Exception as exc:  # noqa: BLE001 - a backend that can't stream still answers non-streamed
                if forwarded[0] or is_upstream_failure(exc):
                    raise
                logger.warning("streaming Responses call failed before first token; retrying non-streamed", exc_info=True)
                return await _call()

        breaker = breaker_for(self._model)
        breaker.check()
        try:
            body = await _send()
        except Exception as exc:
            # Any other error (a 400, a parse failure) still means the gateway answered.
            if is_upstream_failure(exc):
                breaker.record_failure()
            else:
                breaker.record_success()
            raise
        except BaseException:  # cancelled mid-call: says nothing about the upstream
            breaker.release_probe()
            raise
        breaker.record_success()

        blocks: List[Dict[str, Any]] = []
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []
        # `response.output_text` is a client-side property that `.model_dump()` drops (it came back ""
        # on a real answer), so walk `output` ourselves.
        for item in body.get("output", []):
            item_type = item.get("type")
            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and part.get("text"):
                        text_parts.append(part["text"])
            elif item_type == "function_call":
                args = json.loads(item.get("arguments") or "{}")
                call_id = item["call_id"]
                blocks.append({"type": "tool_use", "id": call_id, "name": item["name"], "input": args})
                tool_calls.append(ToolCall(id=call_id, name=item["name"], input=args))
            # "reasoning" items carry no visible content the agent needs — skipped.
        text = "".join(text_parts)

        recovered_text, recovered_calls = _parse_text_tool_calls(text)
        if recovered_calls:
            text = recovered_text
        # Harmony leaks also appear on the final turn (no tools offered); there it's just noise to strip.
        harmony_text, harmony_calls = _parse_harmony_tool_calls(
            # Use the tools actually sent: with a named tool_choice that's exactly one, which lets a
            # bare-JSON call be recovered instead of dropped as noise.
            text, frozenset(t["name"] for t in payload["tools"])
        )
        if harmony_text != text:
            text = harmony_text
            recovered_calls = [*recovered_calls, *harmony_calls]
        # Nothing previewed yet (short answer, or prefix held back): send the cleaned text now, before
        # verification, so raw protocol text never reaches the UI.
        if (
            on_text_delta is not None
            and forwarded[0] == 0
            and text
            and not tool_calls
            and not recovered_calls
        ):
            on_text_delta(text)
            forwarded[0] += 1
        if text:
            blocks.insert(0, {"type": "text", "text": text})
        for tc in recovered_calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            tool_calls.append(tc)

        # Not every OpenAI-compatible server reports usage — default to 0 rather than KeyError so a
        # missing field never breaks the agent loop over a cost-visibility nicety.
        if cache_hit[0]:
            # A replay carries the original call's usage, but the gateway didn't bill it again.
            LLM_RESPONSE_CACHE_HITS_TOTAL.labels(self._model).inc()
            usage = Usage()
        else:
            raw_usage = body.get("usage") or {}
            usage = Usage(
                input_tokens=raw_usage.get("input_tokens", 0),
                output_tokens=raw_usage.get("output_tokens", 0),
            )
            answered_by = served_model(self._model, body.get("model"))
            if answered_by != self._model:
                usage.fallback = {answered_by: Usage(usage.input_tokens, usage.output_tokens)}
        return LLMTurn(
            stop_reason="tool_use" if tool_calls else "end_turn",
            text=text,
            tool_calls=tool_calls,
            assistant_message={"role": "assistant", "content": blocks},
            usage=usage,
        )

    async def aclose(self) -> None:
        await self._client.close()
