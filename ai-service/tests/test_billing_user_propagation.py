"""The authenticated end user must reach the LLM provider, not stop at this service.

The ledger (app/llm/usage_ledger.py) answers "who spent what" from *our* token counts. This is the
other half: OpenAI's standard `user` field on the outgoing call, which is what gives the LiteLLM
gateway an end-user dimension at all. Every call from this service arrives on one virtual key, so
without it the gateway's own provider-reported cost tracking can only say "ai-service spent $X" —
never which person did — and its per-end-user budgets and rate limits have nothing to key on.

Two independent paths carry it, so both are covered here: the CragAgent's Responses API client, and
the httpx transport every `instructor` structured-output feature shares. The agent sends it as
LiteLLM's `x-litellm-end-user-id` header instead, because the body's `user` is part of the gateway's
response-cache key.
"""
import json

import httpx
import pytest

from app.drafting.instructor_client import _rewrite_request_body
from app.observability import _billing_user


@pytest.fixture
def as_user():
    """Set the request-scoped billing user the way ObservabilityMiddleware does.

    Cleared by assignment rather than by `reset(token)`: an async test body runs in a child Context,
    so a token minted there cannot be reset from the fixture's own Context.
    """
    yield _billing_user.set
    _billing_user.set("")


def _chat_request(**body):
    return httpx.Request(
        "POST", "https://api.openai.com/v1/chat/completions",
        json={"model": "gpt-5.6-luna", "messages": [], **body},
    )


def _rewritten(request):
    return json.loads(_rewrite_request_body(request).content)


# --- the instructor path (Draft Task, Plan Epic, Summarize Comments, ...) --------------------


def test_the_authenticated_user_is_stamped_onto_a_structured_output_call(as_user):
    as_user("42")

    assert _rewritten(_chat_request())["user"] == "42"


def test_no_user_field_is_sent_when_there_is_no_end_user(as_user):
    """An eval script or a Kafka-triggered workflow has no requester. Sending a blank identifier
    would merge every unattended call into one fake person at the gateway."""
    as_user("")

    assert "user" not in _rewritten(_chat_request())


def test_a_deliberately_set_user_is_never_overwritten(as_user):
    as_user("42")

    assert _rewritten(_chat_request(user="service-account"))["user"] == "service-account"


def test_both_body_rewrites_apply_to_the_same_request(as_user):
    """They were separate passes over the body; merging them must not drop either. `max_tokens` ->
    `max_completion_tokens` is the fix that stopped every structured-output feature 400ing."""
    as_user("42")

    body = _rewritten(_chat_request(max_tokens=500))

    assert body["max_completion_tokens"] == 500 and "max_tokens" not in body
    assert body["user"] == "42"


def test_a_request_needing_no_rewrite_is_passed_through_untouched(as_user):
    as_user("")
    request = _chat_request()

    assert _rewrite_request_body(request) is request


def test_only_chat_completions_bodies_are_rewritten(as_user):
    """The transport sees every request the client makes, including model listings."""
    as_user("42")
    request = httpx.Request("GET", "https://api.openai.com/v1/models")

    assert _rewrite_request_body(request) is request


def test_an_unparseable_body_is_left_alone_rather_than_raising(as_user):
    as_user("42")
    request = httpx.Request(
        "POST", "https://api.openai.com/v1/chat/completions", content=b"not json",
    )

    assert _rewrite_request_body(request) is request


# --- the CragAgent path (/ask) ---------------------------------------------------------------


class _CapturingResponses:
    def __init__(self):
        self.payload = None

    async def create(self, **payload):
        self.payload = payload
        raise RuntimeError("stop here — the payload is what this test is about")


def _agent_client(monkeypatch):
    from app.llm.openai_compatible_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(
        base_url="http://localhost:1234/v1", api_key="x", model="m", max_output_tokens=256,
    )
    captured = _CapturingResponses()
    monkeypatch.setattr(client._client, "responses", captured)
    return client, captured


async def test_the_agent_call_carries_the_user_it_is_answering_for(monkeypatch, as_user):
    as_user("42")
    client, captured = _agent_client(monkeypatch)

    with pytest.raises(RuntimeError):
        await client.next_turn("system", [{"role": "user", "content": "hi"}], [])

    assert captured.payload["extra_headers"] == {"x-litellm-end-user-id": "42"}
    assert "user" not in captured.payload


async def test_the_agent_call_omits_the_field_for_an_internal_caller(monkeypatch, as_user):
    as_user("")
    client, captured = _agent_client(monkeypatch)

    with pytest.raises(RuntimeError):
        await client.next_turn("system", [{"role": "user", "content": "hi"}], [])

    assert "user" not in captured.payload
    assert "extra_headers" not in captured.payload


# --- the middleware that sets it -------------------------------------------------------------


async def test_the_middleware_scopes_the_user_to_one_request():
    """A ContextVar that leaked between requests would bill one person's calls to whoever came
    before them — the worst possible failure for this particular field."""
    from types import SimpleNamespace

    from app.observability import ObservabilityMiddleware, get_billing_user

    seen = {}

    async def _call_next(request):
        seen["during"] = get_billing_user()
        return SimpleNamespace(status_code=200, headers={})

    request = SimpleNamespace(
        headers={"x-user-id": "42"}, url=SimpleNamespace(path="/ask"), method="POST",
    )
    await ObservabilityMiddleware(app=None).dispatch(request, _call_next)

    assert seen["during"] == "42"
    assert get_billing_user() == ""  # reset once the request is done
