"""Per-user AI limits (app/auth/user_quota.py): requests per minute and dollars per day."""
from __future__ import annotations

import ast
import asyncio
import pathlib
import time

import fakeredis
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.auth import user_quota
from app.auth.request_auth import JwtVerifier, RequestAuthMiddleware
from app.auth.user_quota import UserQuota, enforce_user_quota
from app.llm.usage_ledger import Caller, UsageLedger, charge_for
from app.llm.types import Usage

# 2026-10-29 12:00:30 UTC
NOON = 1_793_275_230.0


class _Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


def _quota(rpm=3, budget=1.0, clock=None, redis_client=None) -> UserQuota:
    return UserQuota(redis_client or fakeredis.FakeAsyncRedis(), rpm, budget, clock=clock or _Clock(NOON))


async def test_requests_over_the_per_minute_limit_get_429():
    clock = _Clock(NOON)
    quota = _quota(rpm=3, clock=clock)
    for _ in range(3):
        await quota.admit("42")
    with pytest.raises(HTTPException) as exc:
        await quota.admit("42")
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == "30"

    await quota.admit("7")  # another user has their own counter
    clock.now += 60
    await quota.admit("42")  # next minute


async def test_daily_budget_blocks_until_utc_midnight():
    clock = _Clock(NOON)
    quota = _quota(rpm=100, budget=1.0, clock=clock)
    await quota.add_spend("42", 0.6)
    await quota.admit("42")
    await quota.add_spend("42", 0.5)
    with pytest.raises(HTTPException) as exc:
        await quota.admit("42")
    assert exc.value.status_code == 429
    assert "Daily AI budget" in exc.value.detail
    assert exc.value.headers["Retry-After"] == str(12 * 3600 - 30)

    status = await quota.status("42")
    assert status.spent_today_usd == pytest.approx(1.1)
    assert status.resets_at == "2026-10-30T00:00:00+00:00"

    clock.now += 12 * 3600
    await quota.admit("42")
    assert (await quota.status("42")).spent_today_usd == 0


async def test_over_budget_does_not_also_use_up_the_minute_window():
    quota = _quota(rpm=1, budget=0.1)
    await quota.add_spend("42", 0.2)
    for _ in range(3):
        with pytest.raises(HTTPException) as exc:
            await quota.admit("42")
        assert "Daily" in exc.value.detail


class _BrokenRedis:
    def __getattr__(self, name):
        raise ConnectionError("redis down")


async def test_redis_down_still_limits_within_this_process():
    clock = _Clock(NOON)
    quota = _quota(rpm=2, budget=1.0, clock=clock, redis_client=_BrokenRedis())
    await quota.admit("42")
    await quota.admit("42")
    with pytest.raises(HTTPException) as exc:
        await quota.admit("42")
    assert "per minute" in exc.value.detail

    clock.now += 60
    await quota.add_spend("42", 1.5)
    with pytest.raises(HTTPException) as exc:
        await quota.admit("42")
    assert "Daily" in exc.value.detail

    clock.now += 12 * 3600  # next UTC day
    await quota.admit("42")
    assert (await quota.status("42")).spent_today_usd == 0


async def test_a_ledger_with_a_user_is_charged_to_that_user():
    quota = _quota()
    user_quota.install(quota)
    try:
        charge = charge_for(
            "structured_output", "openai_compatible", "claude-haiku-4-5", "hosted",
            Usage(input_tokens=1_000_000, output_tokens=0),
        )
        assert charge.cost_usd > 0
        UsageLedger("draft_task", Caller("42", "maya", "s"), "r", [charge]).attach()
        UsageLedger("draft_task", Caller("", "", "s"), "r", [charge]).attach()  # internal: no user
        await asyncio.gather(*user_quota._pending)
    finally:
        user_quota.install(None)
    assert (await quota.status("42")).spent_today_usd == pytest.approx(charge.cost_usd)


# --- Through the real auth middleware ----------------------------------------------------------

INTERNAL = "test-internal-token"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _FakeJwks:
    def get_signing_key_from_jwt(self, token):
        return jwt.PyJWK.from_dict(jwt.algorithms.RSAAlgorithm.to_jwk(KEY.public_key(), as_dict=True))


def _client(quota: UserQuota) -> TestClient:
    router = APIRouter()

    @router.post("/ask", dependencies=[Depends(enforce_user_quota)])
    async def ask():
        return {"ok": True}

    app = FastAPI()
    app.include_router(router)
    app.state.user_quota = quota
    verifier = JwtVerifier("http://jwks.invalid/jwks.json", "jira-agentic-ai")
    verifier._jwks = _FakeJwks()
    return TestClient(RequestAuthMiddleware(app, verifier=verifier, internal_token=INTERNAL))


def _bearer(uid: int) -> dict:
    token = jwt.encode(
        {"iss": "jira-agentic-ai", "sub": "maya", "uid": uid, "exp": int(time.time()) + 60},
        KEY, algorithm="RS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_signed_in_user_is_limited_and_gets_the_reason():
    client = _client(_quota(rpm=2, clock=time.time))
    assert client.post("/ask", headers=_bearer(42)).status_code == 200
    assert client.post("/ask", headers=_bearer(42)).status_code == 200
    resp = client.post("/ask", headers=_bearer(42))
    assert resp.status_code == 429
    assert "per minute" in resp.json()["detail"]
    assert "retry-after" in resp.headers


def test_internal_callers_are_not_limited_even_with_a_user_header():
    client = _client(_quota(rpm=1, clock=time.time))
    headers = {"X-Internal-Token": INTERNAL, "X-User-Id": "42"}
    for _ in range(3):
        assert client.post("/ask", headers=headers).status_code == 200


# --- Every POST endpoint made a choice -----------------------------------------------------------

# Continue a workflow that already exists (never blocked, so a user can finish what they started),
# or make no LLM call.
NOT_LIMITED = {
    "/search",  # retrieval only
    "/plan-epic/rollout/{thread_id}/decision",
    "/plan-epic/rollout/{thread_id}/retry",
    "/{thread_id}/clarify",
    "/{thread_id}/clarify/stream",
    "/{thread_id}/decision",
    "/{thread_id}/decision/stream",
    "/{thread_id}/retry",
    "/{thread_id}/time-travel",
    "/by-sprint/stop",  # ends a check; no LLM call
}


def _post_routes():
    root = pathlib.Path(__file__).resolve().parents[1]
    for relative in ("app/api/routes.py", "app/api/sprint_recovery_routes.py"):
        for node in ast.parse((root / relative).read_text()).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "post" and getattr(dec.func.value, "id", None) == "router"
                ):
                    limited = any(
                        kw.arg == "dependencies" and getattr(kw.value, "id", None) == "_NEW_AI_WORK"
                        for kw in dec.keywords
                    )
                    yield dec.args[0].value, limited


def test_every_post_endpoint_is_either_limited_or_listed():
    routes = list(_post_routes())
    assert len(routes) > 20
    for path, limited in routes:
        assert limited != (path in NOT_LIMITED), (
            f"{path}: add dependencies=_NEW_AI_WORK if it starts new AI work, else list it in NOT_LIMITED"
        )
