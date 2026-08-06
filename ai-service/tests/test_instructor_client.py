"""_RequestRewriteTransport tests — fake wrapped transport, no network."""
import json

import httpx

from app.drafting.instructor_client import _RequestRewriteTransport


class _FakeWrappedTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response):
        self._response = response
        self.seen_requests = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.seen_requests.append(request)
        return self._response


def _completion_response(message: dict, status_code: int = 200) -> httpx.Response:
    body = {"choices": [{"message": message}]}
    return httpx.Response(status_code, json=body)


async def test_passes_the_response_through_untouched():
    inner = _completion_response({"content": '{"title": "Already fine"}'})
    transport = _RequestRewriteTransport(_FakeWrappedTransport(inner))

    response = await transport.handle_async_request(
        httpx.Request("POST", "http://localhost:4000/chat/completions", json={"messages": []})
    )

    assert response is inner


async def test_ignores_non_chat_completions_requests():
    wrapped = _FakeWrappedTransport(httpx.Response(200, json={"unrelated": "payload"}))
    transport = _RequestRewriteTransport(wrapped)
    original = httpx.Request("GET", "http://localhost:4000/v1/models")

    await transport.handle_async_request(original)

    assert wrapped.seen_requests[0] is original


async def test_renames_max_tokens_to_max_completion_tokens_on_the_outgoing_request():
    """Regression test for a live-found total outage: against this project's configured production
    model (hosted OpenAI gpt-5.6-luna), every structured-output call 400'd on `max_tokens` and each
    caller's own graceful-degradation path swallowed it, so Plan Epic / Draft Task / Sprint Pace all
    silently served fallbacks with `degraded=True` instead of failing loudly.
    """
    wrapped = _FakeWrappedTransport(_completion_response({"content": "{}"}))
    transport = _RequestRewriteTransport(wrapped)

    await transport.handle_async_request(
        httpx.Request(
            "POST", "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-5.6-luna", "max_tokens": 2000, "messages": []},
        )
    )

    sent = json.loads(wrapped.seen_requests[0].content)
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 2000
    assert sent["model"] == "gpt-5.6-luna"  # everything else survives the rewrite


async def test_leaves_a_request_without_max_tokens_alone():
    wrapped = _FakeWrappedTransport(_completion_response({"content": "{}"}))
    transport = _RequestRewriteTransport(wrapped)
    original = httpx.Request(
        "POST", "https://api.openai.com/v1/chat/completions", json={"model": "x", "messages": []}
    )

    await transport.handle_async_request(original)

    assert wrapped.seen_requests[0] is original
