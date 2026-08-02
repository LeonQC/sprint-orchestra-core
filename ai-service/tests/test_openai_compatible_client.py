"""OpenAI-compatible adapter regressions — no server or network required."""

from unittest.mock import AsyncMock

from app.llm.openai_compatible_client import (
    _looks_like_harmony_stream_leak, _parse_harmony_tool_calls, _parse_text_tool_calls,
)
from app.llm.openai_compatible_client import OpenAICompatibleClient
from app.llm.types import ToolCall


def test_recovers_xml_tool_call_instead_of_leaking_it_as_answer_text():
    text = """<tool_call>
<function=search_knowledge_base>
<parameter=query>
Sprint 1 Sprint 2 Sprint 3
</parameter>
<parameter=mode>
lexical
</parameter>
</function>
</tool_call>"""

    cleaned, calls = _parse_text_tool_calls(text)

    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].name == "search_knowledge_base"
    assert calls[0].input == {
        "query": "Sprint 1 Sprint 2 Sprint 3",
        "mode": "lexical",
    }


def test_recovers_json_tool_call_and_preserves_surrounding_text():
    text = (
        'I will check. <tool_call>{"name":"query_sprints","arguments":{}}</tool_call> '
        "Please wait."
    )

    cleaned, calls = _parse_text_tool_calls(text)

    assert cleaned == "I will check.  Please wait."
    assert calls[0].name == "query_sprints"
    assert calls[0].input == {}


def test_malformed_tool_markup_is_not_silently_removed():
    text = "<tool_call><function=query_sprints>"

    cleaned, calls = _parse_text_tool_calls(text)

    assert cleaned == text
    assert calls == []


def test_harmony_channel_leak_is_stripped_from_the_answer_and_recovered_as_a_call():
    # The exact shape observed in the UI: a mangled harmony header (constrain marker mis-decoded to a
    # Cyrillic token), the JSON args, then the header repeated glued to the real answer.
    text = (
        'to=query_issues code еит {"issue_keys":["ATC-3","ATC-33","ATC-35"]} '
        "to=query_issues codeYes—several issues are related to checkout:\n\n- **ATC-3**"
    )

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset({"query_issues", "query_sprints"}))

    assert cleaned == "Yes—several issues are related to checkout:\n\n- **ATC-3**"
    assert len(calls) == 1
    assert calls[0].name == "query_issues"
    assert calls[0].input == {"issue_keys": ["ATC-3", "ATC-33", "ATC-35"]}


def test_harmony_leak_on_synthesis_turn_is_stripped_without_recovering_a_call():
    # No tools offered (the final answer turn) — the header is pure noise, nothing to execute.
    text = 'to=functions.query_issues <|constrain|>json {"issue_keys":["ATC-9"]} The answer is ATC-9.'

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset())

    assert cleaned == "The answer is ATC-9."
    assert calls == []


def test_bare_to_equals_in_prose_is_left_alone():
    text = "Set the assignee to=nobody and move on."

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset())

    assert cleaned == text
    assert calls == []


def test_fully_stripped_leak_is_a_bare_json_blob_glued_to_the_answer():
    # The `to=functions.NAME` header gone entirely, leaving `:\n{args}` welded to the first word --
    # the exact form found poisoning a cached answer.
    text = (
        ':\n{"issue_keys":["ATC-3","ATC-33","ATC-45"]}Yes. Checkout-related issues include:\n\n'
        "- **ATC-45** — Disable place order"
    )

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset())

    assert cleaned == "Yes. Checkout-related issues include:\n\n- **ATC-45** — Disable place order"
    assert calls == []


def test_mangled_constrain_marker_and_html_space_are_stripped_from_answer_start():
    # Exact production shape captured from Ask AI: the Harmony constrain marker was mojibake before
    # ``json``, and an HTML-space entity survived in front of it.
    text = (
        '&#x20;\n\nოციjson {"issue_keys":["ATC-30","ATC-68"]}'
        '-\u00a0**ATC-68** is currently **done**.'
    )

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset())

    assert cleaned == '-\u00a0**ATC-68** is currently **done**.'
    assert calls == []


def test_several_junk_tokens_before_json_and_a_welded_word_are_stripped():
    # Both captured from the 2026-11-01 RAGAS run.
    first = (
        'Zusammen 代ന്തjson\n{"issue_keys":["ATC-22","ATC-10"]}-offsetof'
        'The estimate increased from **5 to 8 points**.'
    )
    second = (
        'Japgolly 代 Аԥс json\n{"issue_keys":["ATC-43","ATC-44"]}'
        'The incident produced three true subtasks.'
    )

    assert _parse_harmony_tool_calls(first, frozenset()) == ("The estimate increased from **5 to 8 points**.", [])
    assert _parse_harmony_tool_calls(second, frozenset()) == ("The incident produced three true subtasks.", [])
    assert _looks_like_harmony_stream_leak('Zusammen 代ന്തjson\n{"issue_')
    assert _looks_like_harmony_stream_leak('Japgolly 代 Аԥс json\n{')


def test_the_same_blob_leaked_twice_is_stripped_both_times():
    # From the 2026-10-30 RAGAS run: one pass left the second copy at the top of the answer.
    blob = '{"issue_keys":["ATC-10","ATC-22"]}'
    text = f":\n{blob} :\n{blob}### Sprint 6 → Sprint 7\n- **ATC-72** carried over."

    assert _parse_harmony_tool_calls(text, frozenset()) == ("### Sprint 6 → Sprint 7\n- **ATC-72** carried over.", [])


def test_a_header_with_debris_before_its_json_is_stripped():
    # Shapes saved by earlier RAGAS runs that the stricter header pattern missed.
    twice = (
        ' to=query_issues  (JSON): {"issue_keys":["ATC-68"]} to=query_issues  (JSON): '
        '{"issue_keys":["ATC-68"],"limit":1}ATC-68 was reopened on July 23.'
    )
    junk = ' to=query_issues 代 geschniegeltասպանjson  code\n{"issue_keys":["ATC-43"]}ATC-43 is about a checkout bug.'

    assert _parse_harmony_tool_calls(twice, frozenset()) == ("ATC-68 was reopened on July 23.", [])
    assert _parse_harmony_tool_calls(junk, frozenset()) == ("ATC-43 is about a checkout bug.", [])


def test_to_equals_in_prose_without_json_is_untouched():
    text = "Set the retry policy to=exponential in ATC-22, then rerun the import."

    assert _parse_harmony_tool_calls(text, frozenset({"query_issues"})) == (text, [])


def test_an_english_answer_opening_with_json_is_untouched():
    text = 'Set this json {"retries": 3} in the importer config, per ATC-22.'

    assert _parse_harmony_tool_calls(text, frozenset({"query_issues"})) == (text, [])
    assert not _looks_like_harmony_stream_leak(text)


def test_answer_that_merely_contains_json_mid_sentence_is_untouched():
    text = 'The webhook payload looks like {"event": "checkout.completed"} in the logs.'

    cleaned, calls = _parse_harmony_tool_calls(text, frozenset({"query_issues"}))

    assert cleaned == text
    assert calls == []


async def test_stream_holds_and_removes_mangled_harmony_prefix_before_preview():
    dirty = (
        '&#x20;\n\nოციjson {"issue_keys":["ATC-30","ATC-68"]}'
        '- **ATC-68** is currently **done**.'
    )
    clean = '- **ATC-68** is currently **done**.'

    class FakeResponse:
        def model_dump(self):
            return {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": dirty}],
                }],
                "usage": {},
            }

    class FakeEvent:
        def __init__(self, event_type, *, delta="", response=None):
            self.type = event_type
            self.delta = delta
            self.response = response

    class FakeStream:
        def __init__(self, events):
            self.events = events

        async def __aiter__(self):
            for event in self.events:
                yield event

    response = FakeResponse()
    stream = FakeStream([
        FakeEvent("response.output_text.delta", delta="&#x20;\n\nოცი"),
        FakeEvent("response.output_text.delta", delta='json {"issue_keys":'),
        FakeEvent("response.output_text.delta", delta='["ATC-30","ATC-68"]}- **ATC-68**'),
        FakeEvent("response.output_text.delta", delta=" is currently **done**."),
        FakeEvent("response.completed", response=response),
    ])
    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=stream)
    preview = []

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "why?"}], [], on_text_delta=preview.append,
    )

    assert turn.text == clean
    assert "".join(preview) == clean
    await client.aclose()


async def test_stream_recovers_forced_tool_from_mangled_bare_json_without_previewing_prose():
    dirty = (
        'ოციjson {"issue_keys":["ATC-30","ATC-68"]}'
        '- **ATC-68** is currently **done**.'
    )

    class FakeResponse:
        def model_dump(self):
            return {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": dirty}],
                }],
                "usage": {},
            }

    class FakeEvent:
        def __init__(self, event_type, *, delta="", response=None):
            self.type = event_type
            self.delta = delta
            self.response = response

    class FakeStream:
        async def __aiter__(self):
            yield FakeEvent("response.output_text.delta", delta=dirty)
            yield FakeEvent("response.completed", response=FakeResponse())

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeStream())
    preview = []
    tools = [
        {"name": "query_issues", "description": "query", "input_schema": {"type": "object"}},
        {"name": "get_issue_history", "description": "history", "input_schema": {"type": "object"}},
    ]

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "why?"}], tools,
        tool_choice="get_issue_history", on_text_delta=preview.append,
    )

    assert turn.stop_reason == "tool_use"
    assert turn.tool_calls == [
        ToolCall(
            id="harmony_tool_call_0",
            name="get_issue_history",
            input={"issue_keys": ["ATC-30", "ATC-68"]},
        )
    ]
    assert preview == []
    await client.aclose()


async def test_stream_prefix_buffer_preserves_normal_answer_exactly():
    answer = (
        "ATC-68 was reopened because the resend-email path used an older saved order value, "
        "which opened the wrong order."
    )

    class FakeResponse:
        def model_dump(self):
            return {
                "output": [{
                    "type": "message",
                    "content": [{"type": "output_text", "text": answer}],
                }],
                "usage": {},
            }

    class FakeEvent:
        def __init__(self, event_type, *, delta="", response=None):
            self.type = event_type
            self.delta = delta
            self.response = response

    class FakeStream:
        async def __aiter__(self):
            yield FakeEvent("response.output_text.delta", delta=answer[:15])
            yield FakeEvent("response.output_text.delta", delta=answer[15:75])
            yield FakeEvent("response.output_text.delta", delta=answer[75:])
            yield FakeEvent("response.completed", response=FakeResponse())

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeStream())
    preview = []

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "why?"}], [], on_text_delta=preview.append,
    )

    assert turn.text == answer
    assert "".join(preview) == answer
    assert len(preview) == 2  # buffered prefix, then the remaining live delta
    await client.aclose()


async def test_named_tool_choice_is_forwarded_to_openai_compatible_server():
    class FakeResponse:
        def model_dump(self):
            return {"output": [], "usage": {}}

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1",
        api_key="test",
        model="local-model",
        max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeResponse())

    await client.next_turn(
        "system",
        [{"role": "user", "content": "how many sprints?"}],
        [{"name": "query_sprints", "description": "query", "input_schema": {"type": "object"}}],
        tool_choice="query_sprints",
    )

    payload = client._client.responses.create.call_args.kwargs
    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"]] == ["query_sprints"]
    assert payload["reasoning"] == {"effort": "low"}
    await client.aclose()


async def test_reasoning_effort_can_be_disabled_for_small_router_models():
    class FakeResponse:
        def model_dump(self):
            return {"output": [], "usage": {}}

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1",
        api_key="test",
        model="local-router",
        max_output_tokens=128,
        reasoning_effort="none",
    )
    client._client.responses.create = AsyncMock(return_value=FakeResponse())

    await client.next_turn("route", [{"role": "user", "content": "question"}], [])

    payload = client._client.responses.create.call_args.kwargs
    assert payload["reasoning"] == {"effort": "none"}
    await client.aclose()


async def test_function_call_output_parsed_from_responses_api_output_array():
    class FakeResponse:
        def model_dump(self):
            return {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "query_sprints",
                        "arguments": '{"sprint_ids": [7]}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeResponse())

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "which sprint?"}],
        [{"name": "query_sprints", "description": "query", "input_schema": {"type": "object"}}],
    )

    assert turn.stop_reason == "tool_use"
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0] == ToolCall(id="call_1", name="query_sprints", input={"sprint_ids": [7]})
    assert turn.usage.input_tokens == 10
    assert turn.usage.output_tokens == 5
    await client.aclose()


async def test_message_text_parsed_from_responses_api_output_array_not_sdk_output_text_property():
    # Regression test for a real bug caught live: `response.output_text` is a client-side SDK
    # *property* (openai.types.responses.Response.output_text) that aggregates message text — it does
    # NOT survive `.model_dump()`, which only serializes actual fields. A first version of this parser
    # read `body.get("output_text")` and silently got "" back on every real multi-turn call that had
    # a genuine text answer. Text must be read from the `output` array's `message` items instead.
    class FakeResponse:
        def model_dump(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "There are 3 sprints."}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeResponse())

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "how many sprints?"}], [],
    )

    assert turn.stop_reason == "end_turn"
    assert turn.text == "There are 3 sprints."
    await client.aclose()


async def test_assistant_tool_use_and_tool_result_translate_to_responses_input_items():
    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    fake = AsyncMock(return_value=type("R", (), {"model_dump": lambda self: {"output": [], "usage": {}}})())
    client._client.responses.create = fake

    await client.next_turn(
        "system",
        [
            {"role": "user", "content": "how many sprints?"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "query_sprints", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "3 sprints"}],
            },
        ],
        [],
    )

    input_items = fake.call_args.kwargs["input"]
    assert {"role": "user", "content": "how many sprints?"} in input_items
    assert {"type": "function_call", "call_id": "call_1", "name": "query_sprints", "arguments": "{}"} in input_items
    assert {"type": "function_call_output", "call_id": "call_1", "output": "3 sprints"} in input_items
    # System prompt goes in "instructions", never inside the input item list.
    assert all(item.get("role") != "system" for item in input_items)
    assert fake.call_args.kwargs["instructions"] == "system"
    await client.aclose()


class _FakeRaw:
    """What `responses.with_raw_response.create` returns: headers plus the parsed body or stream."""

    def __init__(self, parsed, headers):
        self._parsed = parsed
        self.headers = headers

    def parse(self):
        return self._parsed


def _response_cache_client(parsed, headers):
    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
        response_cache=True,
    )
    create = AsyncMock(return_value=_FakeRaw(parsed, headers))
    client._client.responses.with_raw_response = type("W", (), {"create": create})()
    return client, create


class _AnswerResponse:
    def model_dump(self):
        return {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Blue."}]}],
            "usage": {"input_tokens": 33, "output_tokens": 6},
        }


async def test_response_cache_off_sends_no_cache_flag():
    client = OpenAICompatibleClient(
        base_url="http://local.invalid/v1", api_key="test", model="local-model", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=_AnswerResponse())

    await client.next_turn("system", [{"role": "user", "content": "sky?"}], [])

    assert "extra_body" not in client._client.responses.create.call_args.kwargs
    await client.aclose()


async def test_response_cache_miss_opts_in_and_reports_real_usage():
    client, create = _response_cache_client(_AnswerResponse(), headers={})

    turn = await client.next_turn("system", [{"role": "user", "content": "sky?"}], [])

    assert create.call_args.kwargs["extra_body"] == {"cache": {"use-cache": True}}
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (33, 6)
    await client.aclose()


async def test_response_cache_hit_is_not_billed_again():
    # A replay carries the original call's usage; counting it would charge the same answer twice.
    client, _ = _response_cache_client(
        _AnswerResponse(), headers={"x-litellm-cache-key": "litellm.response_cache:abc"},
    )

    turn = await client.next_turn("system", [{"role": "user", "content": "sky?"}], [])

    assert turn.text == "Blue."
    assert (turn.usage.input_tokens, turn.usage.output_tokens) == (0, 0)
    assert turn.usage.fallback == {}
    await client.aclose()


async def test_response_cache_hit_also_works_when_streaming():
    class FakeEvent:
        def __init__(self, event_type, *, delta="", response=None):
            self.type = event_type
            self.delta = delta
            self.response = response

    class FakeStream:
        async def __aiter__(self):
            yield FakeEvent("response.output_text.delta", delta="Blue.")
            yield FakeEvent("response.completed", response=_AnswerResponse())

    client, create = _response_cache_client(
        FakeStream(), headers={"x-litellm-cache-key": "litellm.response_cache:abc"},
    )
    preview = []

    turn = await client.next_turn(
        "system", [{"role": "user", "content": "sky?"}], [], on_text_delta=preview.append,
    )

    assert create.call_args.kwargs["stream"] is True
    assert "".join(preview) == "Blue."
    assert turn.usage.input_tokens == 0
    await client.aclose()
