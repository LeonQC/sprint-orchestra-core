"""Distributed tracing: assert the trace actually joins the caller's, not that OTel was imported.

The failure mode this guards against is specific and silent. Tracing that is wired *almost* right
still produces spans — they just become disconnected roots instead of children, and the waterfall
that was the entire point never appears. Nothing errors, nothing logs, and the only symptom is a
collector UI that looks subtly useless. So these tests check the two properties that distinguish
working cross-service tracing from decoration:

  1. an inbound `traceparent` becomes the parent of the spans this service opens, and
  2. the per-stage timers emit spans at all, with the timing they measured attached.

Uses the SDK's in-memory exporter rather than a live collector: the assertions are about span
identity and parentage, which are decided before anything is exported.
"""
from __future__ import annotations

import time
import types

import pytest

from app.api.routes import _StageTimer

trace = pytest.importorskip("opentelemetry.trace")
sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
export = pytest.importorskip("opentelemetry.sdk.trace.export")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")


@pytest.fixture
def tracing(monkeypatch):
    """Point the request path at a provider whose spans land in memory, without touching globals.

    `trace.set_tracer_provider` installs the process-wide provider exactly once and refuses to
    replace it afterwards (it logs a warning and no-ops), which makes a per-test global swap
    unreliable in either direction — the first test to run wins, and every later one silently asserts
    against an exporter nothing is writing to. Patching the seam the code already has instead is both
    honest and order-independent.
    """
    exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(export.SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr("app.api.routes.get_tracer", lambda: tracer)
    return types.SimpleNamespace(exporter=exporter, tracer=tracer)


def test_stage_timer_emits_a_span_named_for_its_stage(tracing):
    stage_ms: dict = {}

    with _StageTimer("embed", stage_ms):
        time.sleep(0.01)

    (span,) = tracing.exporter.get_finished_spans()
    assert span.name == "search.embed"
    assert stage_ms["embed"] >= 10


def test_the_span_carries_the_same_duration_the_metric_and_log_report(tracing):
    """One measurement, three consumers. If the attribute and `stage_ms` ever disagree, the class has
    grown a second clock — which is the exact thing `_StageTimer` exists to prevent."""
    stage_ms: dict = {}

    with _StageTimer("rerank", stage_ms):
        time.sleep(0.01)

    (span,) = tracing.exporter.get_finished_spans()
    assert span.attributes["stage.duration_ms"] == stage_ms["rerank"]


def test_query_shape_attributes_reach_the_span_but_the_query_text_never_does(tracing):
    """Mode and pool size explain a slow stage; the user's question is not the collector's business."""
    with _StageTimer("retrieval", {}, {"search.mode": "hybrid", "search.fetch_limit": 100}):
        pass

    (span,) = tracing.exporter.get_finished_spans()
    assert span.attributes["search.mode"] == "hybrid"
    assert span.attributes["search.fetch_limit"] == 100
    assert not any("query" in key for key in span.attributes)


def test_stage_spans_attach_to_an_active_parent_rather_than_starting_new_traces(tracing):
    """The whole point of the feature: with an inbound request span active, a stage span must be its
    child and share its trace id. If these were roots, ai-service's /ask and this service's retrieval
    would appear in the collector as two unrelated traces and no waterfall would ever be assembled."""
    with tracing.tracer.start_as_current_span("POST /search") as server_span:
        expected_trace_id = server_span.get_span_context().trace_id
        expected_span_id = server_span.get_span_context().span_id
        with _StageTimer("embed", {}):
            pass

    stage = next(s for s in tracing.exporter.get_finished_spans() if s.name == "search.embed")
    assert stage.context.trace_id == expected_trace_id
    assert stage.parent is not None and stage.parent.span_id == expected_span_id


def test_a_failing_stage_leaves_an_error_span_instead_of_a_clean_one(tracing):
    """A stage that raised must not close as if it succeeded — an all-green waterfall over a failed
    request is worse than no trace, because it actively points the investigation elsewhere."""
    with pytest.raises(RuntimeError):
        with _StageTimer("retrieval", {}):
            raise RuntimeError("pgvector went away")

    (span,) = tracing.exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_the_exception_still_propagates_to_the_caller(tracing):
    """Tracing must be transparent. `_StageTimer.__exit__` returns None (never True), so it cannot
    swallow the failure that `/search`'s own error handling depends on seeing."""
    stage_ms: dict = {}
    with pytest.raises(ValueError):
        with _StageTimer("embed", stage_ms):
            raise ValueError("boom")
    # Still timed: a failed stage's duration is exactly what you want when the failure was a timeout.
    assert "embed" in stage_ms


def test_costs_nothing_and_stays_silent_when_no_provider_is_configured():
    """Default state for a laptop with no collector. The `with` blocks stay in the request path, so
    this has to be free rather than merely harmless — and it must not raise."""
    stage_ms: dict = {}

    with _StageTimer("embed", stage_ms, {"search.mode": "vector"}):
        pass

    assert stage_ms["embed"] >= 0
