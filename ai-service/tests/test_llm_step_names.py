"""LLM spans are named by step ("ask.turn_1 → search_knowledge_base"), not the SDK's generic "Response".

Runs the real OpenAI SDK under the real OpenInference instrumentor, with only the network replaced by
an httpx MockTransport — the attribute names the outcome label reads come from that library, so a
hand-built span would only prove this file's own assumptions about them.
"""
import httpx
import pytest
from openai import AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.tracing import (
    LLM_STEP_ATTRIBUTE,
    TRACE_NAME_ATTRIBUTE,
    _CostRelevantSpans,
    _span_namer,
    _OutcomeLabels,
    _reopen_phoenix_export,
    llm_step,
)

_TOOL_CALL = {
    "type": "function_call", "id": "fc_1", "call_id": "call_1", "status": "completed",
    "name": "search_knowledge_base", "arguments": '{"query": "rate limit"}',
}
_MESSAGE = {
    "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
    "content": [{"type": "output_text", "text": "The limit is 10/min.", "annotations": []}],
}


def _response_body(output_item):
    return {
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-test",
        "status": "completed", "output": [output_item], "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }


@pytest.fixture
def traced():
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(_span_namer())
    provider.add_span_processor(SimpleSpanProcessor(_OutcomeLabels(memory)))
    OpenAIInstrumentor().instrument(tracer_provider=provider)
    try:
        yield memory
    finally:
        OpenAIInstrumentor().uninstrument()


def _client(output_item) -> AsyncOpenAI:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_response_body(output_item)))
    return AsyncOpenAI(
        api_key="test", base_url="http://llm.test/v1", http_client=httpx.AsyncClient(transport=transport)
    )


async def _call(client: AsyncOpenAI):
    await client.responses.create(model="gpt-test", input="what is the rate limit?")


async def test_agent_turn_is_named_by_step_and_the_tool_it_chose(traced):
    with llm_step("ask.turn_1", label_outcome=True):
        await _call(_client(_TOOL_CALL))

    (span,) = traced.get_finished_spans()
    assert span.name == "ask.turn_1 → search_knowledge_base"
    assert span.attributes[LLM_STEP_ATTRIBUTE] == "ask.turn_1"


async def test_agent_turn_that_writes_text_is_labelled_answer(traced):
    with llm_step("ask.turn_3", label_outcome=True):
        await _call(_client(_MESSAGE))

    (span,) = traced.get_finished_spans()
    assert span.name == "ask.turn_3 → answer"


async def test_step_without_outcome_label_keeps_the_plain_name(traced):
    with llm_step("ask.router"):
        await _call(_client(_TOOL_CALL))

    (span,) = traced.get_finished_spans()
    assert span.name == "ask.router"


async def test_call_outside_any_step_keeps_the_sdk_name(traced):
    await _call(_client(_MESSAGE))

    (span,) = traced.get_finished_spans()
    assert span.name == "Response"
    assert LLM_STEP_ATTRIBUTE not in span.attributes


async def test_step_name_does_not_leak_past_its_block(traced):
    with llm_step("ask.router"):
        pass
    await _call(_client(_MESSAGE))

    (span,) = traced.get_finished_spans()
    assert span.name == "Response"


async def test_enclosing_step_wins_over_the_inner_default(traced):
    # plan-epic's graph node knows the revision round; the service it calls only knows "critique".
    with llm_step("plan_epic.critique_round_2"):
        with llm_step("plan_epic.critique"):
            await _call(_client(_MESSAGE))

    (span,) = traced.get_finished_spans()
    assert span.name == "plan_epic.critique_round_2"


async def test_langfuse_export_applies_the_outcome_label_too(traced):
    with llm_step("ask.turn_2", label_outcome=True):
        await _call(_client(_TOOL_CALL))
    (span,) = traced.get_finished_spans()
    # The memory exporter saw the labelled copy; hand Langfuse's wrapper the raw span instead.
    raw = span.__class__(
        name="ask.turn_2", context=span.context, parent=span.parent, resource=span.resource,
        attributes=span.attributes, kind=span.kind, status=span.status,
        start_time=span.start_time, end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
    seen = InMemorySpanExporter()

    _CostRelevantSpans(seen, feature_attribute="ai.feature").export([raw])

    (exported,) = seen.get_finished_spans()
    assert exported.name == "ask.turn_2 → search_knowledge_base"


def test_every_span_carries_the_request_name_as_its_trace_name():
    # Langfuse's observation list interleaves concurrent requests; without this only the root row
    # says which request it belongs to.
    tracer = TracerProvider()
    memory = InMemorySpanExporter()
    tracer.add_span_processor(_span_namer())
    tracer.add_span_processor(SimpleSpanProcessor(memory))
    t = tracer.get_tracer("test")
    with t.start_as_current_span("POST /ask"):
        with t.start_as_current_span("CragAgent.ask"):
            with t.start_as_current_span("ask.router"):
                pass

    assert {s.name: s.attributes[TRACE_NAME_ATTRIBUTE] for s in memory.get_finished_spans()} == {
        "POST /ask": "POST /ask", "CragAgent.ask": "POST /ask", "ask.router": "POST /ask",
    }


def test_request_continued_from_another_service_is_named_after_its_own_entry_span():
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    tracer = TracerProvider()
    memory = InMemorySpanExporter()
    tracer.add_span_processor(_span_namer())
    tracer.add_span_processor(SimpleSpanProcessor(memory))
    remote = SpanContext(trace_id=1, span_id=2, is_remote=True, trace_flags=TraceFlags(1))
    with tracer.get_tracer("test").start_as_current_span(
        "POST /ask", context=trace.set_span_in_context(NonRecordingSpan(remote))
    ):
        pass

    (span,) = memory.get_finished_spans()
    assert span.attributes[TRACE_NAME_ATTRIBUTE] == "POST /ask"


def test_namer_first_then_phoenix_keeps_phoenix_export():
    """configure_tracing's order: the namer spends Phoenix's one-time default-processor wipe."""
    from phoenix.otel.otel import TracerProvider as PhoenixTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from types import SimpleNamespace

    provider = PhoenixTracerProvider(endpoint="http://localhost:4317", protocol="grpc", verbose=False)
    provider.add_span_processor(_span_namer())
    _reopen_phoenix_export(provider, SimpleNamespace(phoenix_collector_endpoint="http://localhost:4317"))

    processors = provider._active_span_processor._span_processors
    assert len(processors) == 2
    assert type(processors[0]).__name__ == "_SpanNamer"
    assert isinstance(processors[1], BatchSpanProcessor)
    for processor in processors:
        processor.shutdown()
