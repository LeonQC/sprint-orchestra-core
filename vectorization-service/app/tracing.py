"""Distributed tracing: this service's spans join the caller's trace.

A request id let logs be grepped together, but had no parent/child links or durations, so "which
service and stage made this 8s answer slow?" meant reading two services' logs by hand. With W3C trace
context, ai-service sends `traceparent`, FastAPIInstrumentor reads it, and our spans become children
of the `/ask` span — one waterfall across both services.

Stage spans come from `_StageTimer` (api/routes.py), which already measures each stage once for the
metric and the log; the trace is a third consumer of the same number.

Off by default. When off, `get_tracer` returns a no-op tracer, so the code needs no
`if tracing_enabled` branches.
"""
from __future__ import annotations

import logging

from opentelemetry import trace

from app.config import Settings

logger = logging.getLogger(__name__)

_SERVICE_NAME = "vectorization-service"


def configure_tracing(settings: Settings) -> None:
    """Install a real TracerProvider exporting via OTLP, if tracing is on. The SDK and exporter are
    imported lazily so a deployment with tracing off doesn't pay for them.
    """
    if not settings.tracing_enabled:
        return

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({
            "service.name": settings.tracing_service_name or _SERVICE_NAME,
            # Without this, both services' spans land in one undifferentiated pile in the collector
            # and "which service was slow" is back to being a guess.
            "service.namespace": "sprint-orchestra",
        })
    )
    # Batched: a simple processor would put a network call on every request's critical path.
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.tracing_collector_endpoint))
    )
    trace.set_tracer_provider(provider)

    # Also trace outgoing httpx calls (embedder, contextualizer, VLM via the gateway), so the
    # gateway's spans join this trace and each call gets connect/read timing.
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()
    logger.info(
        "tracing enabled",
        extra={"endpoint": settings.tracing_collector_endpoint,
               "service_name": settings.tracing_service_name or _SERVICE_NAME},
    )


def instrument_app(app, settings: Settings) -> None:
    """Read `traceparent` from incoming requests so our spans join the caller's trace.

    Must be called at import time, not in `lifespan`: FastAPIInstrumentor wraps the middleware stack,
    which Starlette builds before the lifespan body runs — too late there, with no error. /healthz
    and /metrics are excluded (probe polling would add ~5,700 junk traces a day), and per-ASGI-event
    send/receive spans are dropped as clutter.
    """
    if not settings.tracing_enabled:
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="healthz,metrics", exclude_spans=["send", "receive"]
    )


def get_tracer() -> trace.Tracer:
    """A real tracer when configured, a no-op otherwise. Looked up per call: a tracer captured at
    import would bind to the default no-op provider and silently never emit anything.
    """
    return trace.get_tracer(__name__)
