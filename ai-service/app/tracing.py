"""LLM-call tracing to Arize Phoenix (and optionally Langfuse) via OpenTelemetry + OpenInference.

Prometheus (app/observability.py) answers "is it slow, and which stage". Tracing answers "show me the
actual LLM calls and tool calls for this one request". They complement each other.

Coverage: OpenInference patches the provider SDK clients (`anthropic.AsyncAnthropic`,
`openai.AsyncOpenAI`), so every LLM call made through those is traced — the drafting feature, the
Anthropic agent path, and the OpenAI-compatible path (which uses the real `openai` client for exactly
this reason; a raw httpx call would be invisible).

Cross-service: HTTPXClientInstrumentor puts W3C `traceparent` on every outgoing httpx request, and
vectorization-service reads it, so its search spans show up as children of the `/ask` that caused
them. Instrumenting the client library (not each call site) means a new RetrievalClient call is
traced automatically.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
from contextlib import contextmanager

from app.config import Settings

logger = logging.getLogger(__name__)

LLM_STEP_ATTRIBUTE = "ai.llm.step"
TRACE_NAME_ATTRIBUTE = "langfuse.trace.name"
_LABEL_OUTCOME_ATTRIBUTE = "ai.llm.step.label_outcome"
_LLM_STEP: contextvars.ContextVar[tuple[str, bool] | None] = contextvars.ContextVar(
    "llm_step", default=None
)
_TOOL_CALL_NAME = re.compile(
    r"^llm\.output_messages\.\d+\.message\.tool_calls\.\d+\.tool_call\.function\.name$"
)
_OUTPUT_TEXT = re.compile(r"^llm\.output_messages\.\d+\.message\.content")


@contextmanager
def llm_step(name: str, label_outcome: bool = False):
    """Name the LLM calls made in this block ("ask.router"), instead of the SDK's generic "Response".

    label_outcome=True also appends what the call decided ("→ search_knowledge_base", "→ answer"),
    for agent turns, where that decision is what you are looking for in a trace.

    An enclosing step wins: the caller knows more (plan_epic.critique_round_2 vs plan_epic.critique).
    """
    if _LLM_STEP.get() is not None:
        yield
        return
    token = _LLM_STEP.set((name, label_outcome))
    try:
        yield
    finally:
        _LLM_STEP.reset(token)


def _span_namer():
    """Span processor that names spans at start, while the caller's context is still current.

    - Copies the request's name (`POST /ask`) onto every span as the Langfuse trace name. Langfuse
      takes it from the root only, so in its flat observation list concurrent requests' rows
      interleave with no way to tell which request a row belongs to.
    - Applies `llm_step` names. The outcome suffix is only known at export (`_label_outcome`).
    """
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry import trace
    from opentelemetry.sdk.trace import SpanProcessor

    class _SpanNamer(SpanProcessor):
        def on_start(self, span, parent_context=None):
            # No parent here (or one from another service, which Langfuse never receives): this span
            # is the trace's root as Langfuse sees it.
            if span.parent is None or span.parent.is_remote:
                span.set_attribute(TRACE_NAME_ATTRIBUTE, span.name)
            else:
                parent = trace.get_current_span(parent_context)
                inherited = (getattr(parent, "attributes", None) or {}).get(TRACE_NAME_ATTRIBUTE)
                if inherited:
                    span.set_attribute(TRACE_NAME_ATTRIBUTE, inherited)

            step = _LLM_STEP.get()
            kind = (span.attributes or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            # The SDK call's own httpx child span starts in the same context; leave it alone.
            if step is None or kind != OpenInferenceSpanKindValues.LLM.value:
                return
            name, label_outcome = step
            span.update_name(name)
            span.set_attribute(LLM_STEP_ATTRIBUTE, name)
            if label_outcome:
                span.set_attribute(_LABEL_OUTCOME_ATTRIBUTE, True)

    return _SpanNamer()


def _label_outcome(span):
    attributes = span.attributes or {}
    if not attributes.get(_LABEL_OUTCOME_ATTRIBUTE):
        return span
    tools = list(dict.fromkeys(v for k, v in attributes.items() if _TOOL_CALL_NAME.match(k)))
    if tools:
        outcome = ", ".join(tools)
    elif any(_OUTPUT_TEXT.match(k) for k in attributes):
        outcome = "answer"
    else:
        return span  # failed call or no recorded output: don't guess
    return _replace_span(span, name=f"{span.name} → {outcome}")


def _replace_span(span, name=None, attributes=None):
    from opentelemetry.sdk.trace import ReadableSpan

    return ReadableSpan(
        name=span.name if name is None else name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes if attributes is None else attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class _OutcomeLabels:
    """Span exporter wrapper that applies `_label_outcome` (for Phoenix; Langfuse's wrapper does it too)."""

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def export(self, spans):
        return self._wrapped.export([_label_outcome(span) for span in spans])

    def shutdown(self):
        return self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30_000):
        return self._wrapped.force_flush(timeout_millis)


class CallerAttributeMiddleware:
    """Put the caller's user/session id on every span the request creates, not just the ledger span.

    Langfuse prices per LLM call, and token counts live on the provider-SDK child spans. With the
    identity only on the parent span, "cost by user" in Langfuse came back empty. `using_attributes`
    puts the ids in the OTel context so every downstream span inherits them, including ones created
    inside the SDK. Only the header session is known this early; a session derived from conversation
    history is still added later on the ledger span.
    """

    def __init__(self, app):
        self._app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        from openinference.instrumentation import using_attributes

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        user_id = headers.get("x-user-id") or ""
        session_id = (headers.get("x-session-id") or "").strip()
        if not user_id and not session_id:
            await self._app(scope, receive, send)
            return
        with using_attributes(user_id=user_id, session_id=session_id):
            await self._app(scope, receive, send)


class _CostRelevantSpans:
    """Span exporter wrapper that forwards only spans a cost view can use (for Langfuse).

    Phoenix wants the full waterfall, including raw httpx spans (connect/TLS timing). Langfuse is read
    as a cost table, where those spans are $0 noise — 22 of 33 spans in one measured conversation.

    Kept: anything OpenInference produced (LLM, agent, tool, retriever), anything carrying our usage
    ledger, and every inbound request span (SpanKind.SERVER). The last one matters for /ask: its
    ledger sits on the `CragAgent.ask` span, so without keeping `POST /ask` itself that span became an
    orphan whose parent Langfuse never received, and the trace had no name.
    """

    def __init__(self, wrapped, feature_attribute: str):
        self._wrapped = wrapped
        self._feature_attribute = feature_attribute

    def _keep(self, span) -> bool:
        attributes = span.attributes or {}
        from opentelemetry.trace import SpanKind

        from openinference.semconv.trace import SpanAttributes

        return (
            SpanAttributes.OPENINFERENCE_SPAN_KIND in attributes
            or self._feature_attribute in attributes
            or span.kind == SpanKind.SERVER
        )

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        kept = [_normalize_langfuse_model(_label_outcome(span)) for span in spans if self._keep(span)]
        return self._wrapped.export(kept) if kept else SpanExportResult.SUCCESS

    def shutdown(self):
        return self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30_000):
        return self._wrapped.force_flush(timeout_millis)


def _normalize_langfuse_model(span):
    """Report the real model name to Langfuse instead of the gateway route alias.

    LiteLLM returns the alias for non-streaming calls but the resolved model for streaming ones, which
    split one model into two Langfuse buckets. Only known aliases are rewritten (a different real
    model may be a legitimate fallback). The alias stays queryable as `ai.model.route_alias`.
    Langfuse exporter only; Phoenix keeps the raw value.
    """
    from openinference.semconv.trace import SpanAttributes

    from app.llm.pricing import resolve_model

    attributes = dict(span.attributes or {})
    observed = attributes.get(SpanAttributes.LLM_MODEL_NAME)
    invocation = attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    requested = None
    if isinstance(invocation, str):
        try:
            parameters = json.loads(invocation)
            if isinstance(parameters, dict) and isinstance(parameters.get("model"), str):
                requested = parameters["model"]
        except (TypeError, ValueError):
            pass

    requested_resolved = resolve_model(requested) if requested else requested
    observed_resolved = resolve_model(observed) if isinstance(observed, str) else observed
    route_alias = requested if requested and requested_resolved != requested else None
    if route_alias is None and isinstance(observed, str) and observed_resolved != observed:
        route_alias = observed

    changed = False
    if route_alias is not None:
        attributes["ai.model.route_alias"] = route_alias
        changed = True
    # Preserve an already-resolved response name. This keeps real gateway fallback decisions visible.
    if isinstance(observed, str) and observed_resolved != observed:
        attributes[SpanAttributes.LLM_MODEL_NAME] = observed_resolved
        changed = True
    if not changed:
        return span
    return _replace_span(span, attributes=attributes)


def _reopen_phoenix_export(tracer_provider, settings: Settings) -> None:
    """Re-add Phoenix's exporter after the first processor we add.

    Phoenix's `register()` marks its processor as a default, and the first `add_span_processor` call
    after that silently removes it. Adding Langfuse alone once meant spans went only to Langfuse —
    this ran for a day before anyone noticed. So this runs right after that first call (the span
    namer's), and every later call appends normally. Batch processor, as Phoenix itself recommends.
    """
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            _OutcomeLabels(OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint))
        )
    )


def _add_langfuse_exporter(tracer_provider, settings: Settings) -> None:
    """Send the same spans to Langfuse as well as Phoenix.

    Langfuse reads OpenInference attributes (user, session, tags, model, token counts) natively.
    One trap: Langfuse treats `llm.token_count.prompt` as including cached tokens (OpenAI's
    convention), while our ledger stores fresh and cached input separately (Anthropic's). So never
    copy the ledger's numbers onto `llm.token_count.*` — that double-subtracts the cache (seen live:
    input reported as 0). The ledger keeps its own `ai.usage.*` names for this reason.

    Separate batch processor, so a Langfuse outage only drops its own spans.
    """
    import base64

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from app.llm.usage_ledger import FEATURE

    auth = base64.b64encode(
        f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
    ).decode()
    exporter = _CostRelevantSpans(
        OTLPSpanExporter(
            endpoint=f"{settings.langfuse_base_url.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}"},
        ),
        feature_attribute=FEATURE,
    )
    tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    logger.info("Langfuse span export enabled", extra={"endpoint": settings.langfuse_base_url})


def configure_tracing(settings: Settings) -> None:
    if not (settings.phoenix_enabled or settings.langfuse_enabled):
        return

    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from phoenix.otel import register

    # Used even when only Langfuse is on: same provider setup either way. If Phoenix isn't running,
    # that only costs a logged export failure off the request path.
    tracer_provider = register(
        endpoint=settings.phoenix_collector_endpoint,
        project_name=settings.phoenix_project_name,
        protocol="grpc",
    )
    # This first add wipes register()'s own Phoenix processor; the next line puts Phoenix back.
    tracer_provider.add_span_processor(_span_namer())
    _reopen_phoenix_export(tracer_provider, settings)
    if settings.langfuse_enabled:
        _add_langfuse_exporter(tracer_provider, settings)
    # Global and idempotent, so enabling both is harmless.
    AnthropicInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    # Propagates traceparent to vectorization-service. SDK calls also get an httpx span here; that
    # duplication is useful (LLM span = tokens/cost, httpx span = connect/TLS timing).
    HTTPXClientInstrumentor().instrument()
    logger.info(
        "Phoenix tracing enabled",
        extra={"endpoint": settings.phoenix_collector_endpoint, "project": settings.phoenix_project_name},
    )


def instrument_app(app, settings: Settings) -> None:
    """Give every inbound request a server span, so its downstream calls share one trace.

    Must be called at import time, not in `lifespan`: FastAPIInstrumentor wraps the middleware
    stack, which Starlette builds before the lifespan body runs — too late there, and it fails
    silently. /healthz and /metrics are excluded (probe polling would add ~5,700 junk traces a day),
    and per-ASGI-event send/receive spans are dropped as noise.
    """
    if not (settings.phoenix_enabled or settings.langfuse_enabled):
        return

    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(
        app, excluded_urls="healthz,metrics", exclude_spans=["send", "receive"]
    )
