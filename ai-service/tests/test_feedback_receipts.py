"""Check answer identity across HTTP/SSE, users and shared-cache reuse without a live LLM."""
import base64
import hashlib
import hmac
import json
from types import SimpleNamespace

import pytest

from app import feedback
from app.api import routes
from tests.test_usage_ledger_routes import _ask_request

SECRET = "feedback-test-secret-not-a-production-key"


def unpack(receipt):
    payload, signature = receipt.split(".")
    assert hmac.compare_digest(signature, hmac.new(SECRET.encode(), feedback.DOMAIN + payload.encode(), hashlib.sha256).hexdigest())
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


class Cache:
    value = None

    async def get(self, *args, **kwargs):
        return self.value

    async def put(self, question, space_ids, response, watermark, execution_mode="single"):
        assert "feedback_receipt" not in response and "answer_id" not in response
        self.value = response


async def watermark(space_ids):
    return "fixture-data-v1"


async def run(request, ask, streaming):
    if not streaming:
        return await routes.ask(ask, request)
    response = await routes.ask_stream(ask, request)
    async for event in response.body_iterator:
        if event.startswith("event: result\n"):
            return routes.AskResponse.model_validate(json.loads(event.split("data: ", 1)[1]))
    raise AssertionError("No SSE result received")


@pytest.mark.parametrize("streaming", [False, True])
async def test_each_answer_has_its_own_owner_bound_receipt_even_when_cached(streaming):
    request = _ask_request({"x-user-id": "42", "x-username": "alice"}, [])
    request.app.state.settings = SimpleNamespace(internal_service_token=SECRET)
    request.app.state.cache = Cache()
    request.app.state.index_watermark = SimpleNamespace(watermark=watermark)
    ask = routes.AskRequest(question="Which issues are blocked?", spaceIds=[1])
    first = await run(request, ask, streaming)
    snapshot = unpack(first.feedback_receipt)
    assert snapshot["userId"] == 42
    assert snapshot["answerId"] == first.answer_id
    assert snapshot["request"]["spaceIds"] == [1]
    assert snapshot["dataVersion"] == "fixture-data-v1"
    assert snapshot["expiresAt"] - snapshot["issuedAt"] == 604800
    assert snapshot["response"]["cacheHit"] is False
    request.headers["x-user-id"] = "43"
    second = await run(request, ask, streaming)
    cached = unpack(second.feedback_receipt)
    assert second.cache_hit is True
    assert first.answer_id != second.answer_id
    assert cached["userId"] == 43
    assert cached["response"]["cacheHit"] is True
    assert "feedbackReceipt" not in cached["response"]


async def test_snapshot_keeps_followup_context_and_original_answer():
    request = _ask_request({"x-user-id": "42"}, [])
    request.app.state.settings = SimpleNamespace(internal_service_token=SECRET)
    ask = routes.AskRequest(question="Why?", spaceIds=[1], history=[{"role": "user", "content": "What about ATC-77?"}])
    result = await run(request, ask, False)
    snapshot = unpack(result.feedback_receipt)
    assert snapshot["request"]["history"][0]["content"] == "What about ATC-77?"
    assert snapshot["response"]["answer"] == result.answer


@pytest.mark.parametrize("user", ["", "invalid", "0"])
async def test_internal_or_invalid_identity_gets_no_receipt(user):
    request = _ask_request({"x-user-id": user}, [])
    request.app.state.settings = SimpleNamespace(internal_service_token=SECRET)
    result = await run(request, routes.AskRequest(question="Why?", spaceIds=[1]), False)
    assert result.feedback_receipt is None


async def test_large_receipt_does_not_break_answer(monkeypatch):
    monkeypatch.setattr(feedback, "MAX_RECEIPT_BYTES", 10)
    request = _ask_request({"x-user-id": "42"}, [])
    request.app.state.settings = SimpleNamespace(internal_service_token=SECRET)
    result = await run(request, routes.AskRequest(question="Why?", spaceIds=[1]), False)
    assert result.answer
    assert result.feedback_receipt is None
