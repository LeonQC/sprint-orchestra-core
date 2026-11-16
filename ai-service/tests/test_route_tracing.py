"""The server span carries the request and response body, so Langfuse's trace Input/Output aren't empty.

Real FastAPI + the real FastAPI instrumentor: the point is that the span TracedRoute writes to is the
server span the instrumentor creates, which a hand-made span would not prove.
"""
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi.testclient import TestClient
from openinference.semconv.trace import SpanAttributes
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from app.api.route_tracing import TracedRoute


class Proposal(BaseModel):
    proposal: str


def _app_and_spans():
    router = APIRouter(route_class=TracedRoute)

    @router.post("/plan-epic")
    async def plan_epic(req: Proposal) -> dict:
        return {"epic": {"title": req.proposal.title()}}

    @router.post("/ask/stream")
    async def ask_stream(req: Proposal):
        async def events():
            yield 'event: stage\ndata: {"stage": "searching"}\n\n'
            yield 'event: token\ndata: {"text": "Sav"}\n\n'
            yield 'event: result\ndata: {"answer": "Saved cards ship in Sprint 2."}\n\n'
        return StreamingResponse(events(), media_type="text/event-stream")

    app = FastAPI()
    app.include_router(router)
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider, exclude_spans=["send", "receive"])
    return app, memory


def _server_span(memory):
    (span,) = [s for s in memory.get_finished_spans() if s.kind == SpanKind.SERVER]
    return span


def test_json_endpoint_records_request_and_response_on_the_server_span():
    app, memory = _app_and_spans()

    response = TestClient(app).post("/plan-epic", json={"proposal": "save cards"})

    assert response.json() == {"epic": {"title": "Save Cards"}}  # the handler still read the body
    span = _server_span(memory)
    assert json.loads(span.attributes[SpanAttributes.INPUT_VALUE]) == {"proposal": "save cards"}
    assert json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE]) == {"epic": {"title": "Save Cards"}}


def test_streaming_endpoint_records_the_final_result_event_as_output():
    app, memory = _app_and_spans()

    with TestClient(app).stream("POST", "/ask/stream", json={"proposal": "when do saved cards ship?"}) as r:
        body = "".join(r.iter_text())

    assert "event: token" in body  # the client still gets every event, untouched
    span = _server_span(memory)
    assert json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE]) == {
        "answer": "Saved cards ship in Sprint 2."
    }
