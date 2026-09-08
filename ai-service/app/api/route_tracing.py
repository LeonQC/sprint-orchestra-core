"""Record each endpoint's request and response body on its server span.

Langfuse shows a trace's Input/Output from its root span, which is the FastAPI server span. Without
this every trace read "undefined" there, and finding out what was asked meant opening an LLM call.
"""
from __future__ import annotations

from typing import AsyncIterator, Callable

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace

# Enough for a full epic plan; anything longer is cut, not dropped.
_MAX_CHARS = 32_000
_JSON = "application/json"
# The SSE event the UI renders last: the answer, or why there isn't one.
_FINAL_EVENTS = ("event: result\n", "event: error\n")


def _text(raw: bytes | str) -> str:
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "…[truncated]"


async def _record_final_event(chunks: AsyncIterator, span) -> AsyncIterator:
    # The server span stays open until the stream finishes, so the output can be set at the end.
    final = None
    async for chunk in chunks:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for event in text.split("\n\n"):
            if event.startswith(_FINAL_EVENTS) and "data: " in event:
                final = event.split("data: ", 1)[1]
        yield chunk
    if final is not None and span.is_recording():
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, _text(final))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, _JSON)


class TracedRoute(APIRoute):
    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def traced(request: Request) -> Response:
            span = trace.get_current_span()
            if not span.is_recording():
                return await handler(request)
            body = await request.body()  # cached on the request; the handler reads the same bytes
            if body:
                span.set_attribute(SpanAttributes.INPUT_VALUE, _text(body))
                span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, _JSON)
            response = await handler(request)
            if isinstance(response, StreamingResponse):
                response.body_iterator = _record_final_event(response.body_iterator, span)
            elif getattr(response, "body", None):
                span.set_attribute(SpanAttributes.OUTPUT_VALUE, _text(response.body))
                span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, _JSON)
            return response

        return traced
