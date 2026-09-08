"""`_CostRelevantSpans` (app/tracing.py) — which spans reach Langfuse vs. Phoenix-only.

Phoenix gets every span, unfiltered, straight from `register()`; this filter sits only in front of
the Langfuse exporter. What's worth testing here is the predicate's three branches, verified against
real span kinds rather than assumed — see the class docstring for the live measurement (one real
request: one SpanKind.SERVER, one SpanKind.CLIENT per httpx call, two SpanKind.INTERNAL) that this
suite's SpanKind fixtures reproduce directly.
"""
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from app.tracing import _CostRelevantSpans

FEATURE = "ai.feature"


def _spans(build):
    """Run `build(tracer)` against a real TracerProvider and return the finished spans it made —
    real ReadableSpan objects, not mocks, so `.kind` and `.attributes` behave exactly as they would
    against a genuine FastAPI/httpx/OpenInference-instrumented request."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    build(provider.get_tracer("test"))
    return exporter.get_finished_spans()


def _kept_names(spans):
    captured = []

    class _Capture:
        def export(self, spans):
            captured.extend(s.name for s in spans)
            from opentelemetry.sdk.trace.export import SpanExportResult
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=30_000):
            pass

    _CostRelevantSpans(_Capture(), FEATURE).export(spans)
    return captured


def test_an_llm_call_span_is_kept():
    """OpenInference's own instrumentors set this kind automatically on every real LLM call."""
    spans = _spans(lambda t: t.start_span(
        "ChatCompletion",
        attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value},
    ).end())

    assert _kept_names(spans) == ["ChatCompletion"]


def test_a_span_carrying_the_usage_ledger_is_kept():
    spans = _spans(lambda t: t.start_span("POST /comment-digest", attributes={FEATURE: "summarize_comments"}).end())

    assert _kept_names(spans) == ["POST /comment-digest"]


def test_an_inbound_http_entrypoint_span_is_kept_even_with_neither_marker():
    """The fix for `/ask`'s orphaned `CragAgent.ask`: its usage ledger lands on the agent span, not
    this one, deliberately (see app/api/routes.py) — so this span carries neither OpenInference kind
    nor the ledger attribute, and needs keeping on `SpanKind.SERVER` alone or the request's own
    FastAPI-instrumented root never reaches Langfuse, leaving every child an orphan with an
    unresolvable trace name."""
    spans = _spans(lambda t: t.start_span("POST /ask", kind=SpanKind.SERVER).end())

    assert _kept_names(spans) == ["POST /ask"]


def test_an_outgoing_httpx_call_is_dropped():
    """The exact noise this filter exists to keep out of Langfuse — verified live: 22 of 33 spans in
    one real two-turn conversation were 0-token, $0 httpx client spans like this one. Confirms adding
    the SpanKind.SERVER branch didn't reopen that hole: a CLIENT-kind span with neither other marker
    still drops."""
    spans = _spans(lambda t: t.start_span("GET", kind=SpanKind.CLIENT).end())

    assert _kept_names(spans) == []


def test_an_internal_asgi_span_is_dropped():
    """ASGI's own `http send`/`http receive` events — already excluded at the instrumentation level
    via `instrument_app`'s `exclude_spans`, but confirmed dropped here too as a second line of
    defense in case that ever lapses."""
    spans = _spans(lambda t: t.start_span("http send", kind=SpanKind.INTERNAL).end())

    assert _kept_names(spans) == []


def test_an_empty_batch_does_not_error():
    """`export([])` for a batch where nothing survived the filter — must succeed, not raise, since a
    real BatchSpanProcessor calls this on every flush interval regardless of whether anything is
    pending."""
    from opentelemetry.sdk.trace.export import SpanExportResult

    assert _CostRelevantSpans(None, FEATURE).export([]) == SpanExportResult.SUCCESS
