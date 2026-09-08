"""Regression: adding the Langfuse exporter must not silently discard Phoenix's own.

**Found live, not in a test — that's the point of this file.** `phoenix.otel.register()` installs its
own SimpleSpanProcessor and flags it `_default_processor = True`. Its `TracerProvider` overrides
`add_span_processor` to check that flag on every call and, when set, shut down and discard *every*
existing processor before adding the new one — documented in its own docstring ("If this
TracerProvider has a default processor, it will be removed"), easy to miss because nothing raises.
`_add_langfuse_exporter`'s own `add_span_processor` call was the first one this project ever made
after `register()`, so it silently wiped Phoenix's export the moment Langfuse was wired in: every
ai-service restart for a full day re-ran `register()`, then immediately re-triggered the same wipe.
Confirmed directly against real data — a live trace landed complete in Langfuse and nowhere in
Phoenix. The fix (`_reopen_phoenix_export`) spends that one-time wipe on a processor built to be
replaced, so the *second* `add_span_processor` call (Langfuse's) lands on an already-clear flag and
appends instead of clobbering.

Built against `phoenix.otel`'s real `TracerProvider` class, not a mock of it — a mock would only ever
prove this test's own understanding of the bug, not that the actual library still behaves this way.
"""
from phoenix.otel.otel import TracerProvider
from types import SimpleNamespace

import json

import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

from app.tracing import _CostRelevantSpans, _add_langfuse_exporter, _reopen_phoenix_export


def _phoenix_provider() -> TracerProvider:
    """A real `phoenix.otel` TracerProvider, in exactly the state `register()` leaves it: one
    default-flagged processor already installed, primed to be wiped by the next `add_span_processor`
    call — no network calls, since span exporters are lazy until something actually flushes."""
    return TracerProvider(endpoint="http://localhost:4317", protocol="grpc", verbose=False)


def _settings(**overrides):
    defaults = dict(
        phoenix_collector_endpoint="http://localhost:4317",
        langfuse_base_url="http://localhost:3300",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _processor_count(provider: TracerProvider) -> int:
    return len(provider._active_span_processor._span_processors)


def test_register_alone_leaves_exactly_one_processor():
    """Baseline: confirms the fixture matches `register()`'s real starting state before either
    fix function touches it."""
    provider = _phoenix_provider()

    assert _processor_count(provider) == 1


def test_adding_langfuse_without_the_fix_would_wipe_phoenix():
    """Pins the bug itself, not just the fix — if `phoenix.otel` ever changes this behavior, this
    test (not just the one below) should be the one that starts failing."""
    provider = _phoenix_provider()

    _add_langfuse_exporter(provider, _settings())

    assert _processor_count(provider) == 1  # Phoenix's default was discarded, not added-to


def test_reopening_phoenix_first_leaves_both_exporters_registered():
    """The actual fix: both processors survive when Phoenix's is explicitly rebuilt before
    Langfuse's is added — the exact sequence `configure_tracing` now runs."""
    provider = _phoenix_provider()

    _reopen_phoenix_export(provider, _settings())
    _add_langfuse_exporter(provider, _settings())

    assert _processor_count(provider) == 2


def test_reopened_phoenix_processor_is_a_batch_processor():
    """Also resolves Phoenix's own startup warning ("strongly advised to use a BatchSpanProcessor
    in production") as a side effect of the fix, not a second unrelated change."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = _phoenix_provider()
    _reopen_phoenix_export(provider, _settings())

    assert isinstance(provider._active_span_processor._span_processors[0], BatchSpanProcessor)


class _CaptureExporter:
    def __init__(self):
        self.spans = []

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


@pytest.mark.parametrize(
    ("provided_model", "invocation_model", "expected_model", "expected_alias"),
    [
        # LiteLLM non-streaming response: response.model echoes the route alias.
        ("openai-production", "openai-production", "gpt-5.6-luna", "openai-production"),
        # LiteLLM streaming response: response.completed.model already contains the real model.
        ("gpt-5.6-luna", "openai-production", "gpt-5.6-luna", "openai-production"),
        # Preserve a real model selected by a future gateway fallback; only the alias is metadata.
        ("fallback-model", "openai-production", "fallback-model", "openai-production"),
        # A real provider model with no gateway alias must pass through unchanged.
        ("gpt-5.6-luna", "gpt-5.6-luna", "gpt-5.6-luna", None),
    ],
)
def test_langfuse_export_normalizes_route_alias_without_hiding_real_models(
    provided_model, invocation_model, expected_model, expected_alias,
):
    capture = _CaptureExporter()
    exporter = _CostRelevantSpans(capture, feature_attribute="ai.usage.feature")
    span = ReadableSpan(
        name="Response",
        attributes={
            "ai.usage.feature": "ask_ai",
            SpanAttributes.LLM_MODEL_NAME: provided_model,
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps({"model": invocation_model}),
        },
    )

    assert exporter.export([span]) == SpanExportResult.SUCCESS
    exported = capture.spans[0]
    assert exported.attributes[SpanAttributes.LLM_MODEL_NAME] == expected_model
    assert exported.attributes.get("ai.model.route_alias") == expected_alias


def test_langfuse_model_normalization_tolerates_malformed_invocation_parameters():
    capture = _CaptureExporter()
    exporter = _CostRelevantSpans(capture, feature_attribute="ai.usage.feature")
    span = ReadableSpan(
        name="Response",
        attributes={
            "ai.usage.feature": "ask_ai",
            SpanAttributes.LLM_MODEL_NAME: "openai-production",
            SpanAttributes.LLM_INVOCATION_PARAMETERS: "not-json",
        },
    )

    assert exporter.export([span]) == SpanExportResult.SUCCESS
    assert capture.spans[0].attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-5.6-luna"
    assert capture.spans[0].attributes["ai.model.route_alias"] == "openai-production"
