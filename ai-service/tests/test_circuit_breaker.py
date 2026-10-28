"""Fail fast while every provider behind a gateway model is down."""
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from app.llm.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.llm.openai_compatible_client import OpenAICompatibleClient


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _breaker(clock) -> CircuitBreaker:
    return CircuitBreaker("test-model", failure_threshold=3, reset_after_seconds=30, clock=clock)


def test_opens_after_consecutive_failures_and_fails_fast():
    breaker = _breaker(_Clock())
    for _ in range(3):
        breaker.check()
        breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.check()


def test_a_success_resets_the_failure_count():
    breaker = _breaker(_Clock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()

    breaker.check()  # still closed: failures must be consecutive


def test_after_the_wait_one_probe_goes_through_and_its_result_decides():
    clock = _Clock()
    breaker = _breaker(clock)
    for _ in range(3):
        breaker.record_failure()

    clock.now += 31
    breaker.check()  # the probe
    with pytest.raises(CircuitOpenError):
        breaker.check()  # everyone else waits for it

    breaker.record_failure()  # probe failed: wait again
    clock.now += 10
    with pytest.raises(CircuitOpenError):
        breaker.check()

    clock.now += 21
    breaker.check()
    breaker.record_success()
    assert not breaker.is_open
    breaker.check()


def _server_error() -> openai.InternalServerError:
    request = httpx.Request("POST", "http://gateway.invalid/v1/responses")
    return openai.InternalServerError("upstream down", response=httpx.Response(500, request=request), body=None)


def _client(model: str) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(base_url="http://gateway.invalid", api_key="k", model=model, max_output_tokens=10)


async def test_client_stops_calling_the_gateway_once_the_breaker_opens():
    client = _client("breaker-test-open")
    create = AsyncMock(side_effect=_server_error())
    client._client.responses.create = create

    for _ in range(3):
        with pytest.raises(openai.InternalServerError):
            await client.next_turn("s", [{"role": "user", "content": "hi"}], [])
    with pytest.raises(CircuitOpenError):
        await client.next_turn("s", [{"role": "user", "content": "hi"}], [])

    assert create.await_count == 3
    await client.aclose()


async def test_a_bad_request_does_not_count_as_an_outage():
    client = _client("breaker-test-4xx")
    request = httpx.Request("POST", "http://gateway.invalid/v1/responses")
    create = AsyncMock(side_effect=openai.BadRequestError(
        "bad", response=httpx.Response(400, request=request), body=None,
    ))
    client._client.responses.create = create

    for _ in range(5):
        with pytest.raises(openai.BadRequestError):
            await client.next_turn("s", [{"role": "user", "content": "hi"}], [])

    assert create.await_count == 5
    await client.aclose()
