"""Each JSON log line written inside a request carries that request's trace id."""
import json
import logging

from opentelemetry.sdk.trace import TracerProvider

from app.logging_config import JsonFormatter


def _line(message: str) -> dict:
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, message, (), None)
    return json.loads(JsonFormatter().format(record))


def test_log_line_inside_a_span_carries_its_trace_id():
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("POST /ask") as span:
        line = _line("route selected")

    assert line["trace_id"] == format(span.get_span_context().trace_id, "032x")


def test_log_line_outside_any_span_has_no_trace_id():
    assert "trace_id" not in _line("startup")
