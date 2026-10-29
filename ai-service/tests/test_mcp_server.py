"""MCP server over the real protocol: an MCP client session talks to /mcp through httpx's ASGI
transport, so tool discovery, argument validation and error results go through the same JSON-RPC path
Claude Code or Cursor would use. Retrieval and membership are fakes; no network.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import fakeredis
import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.agent.retrieval_tool import (
    IssueAttachmentsResult,
    IssueCommentsResult,
    IssueDetailsResult,
    IssueHistoryResult,
    IssueQueryResult,
    SprintQueryResult,
)
from app.api import routes as api_routes
from app.auth.request_auth import JwtVerifier, RequestAuthMiddleware
from app.auth.space_membership import NoopSpaceMembershipChecker
from app.auth.user_quota import UserQuota
from app.mcp_server.server import install_mcp_route, run_mcp_server

ATC, OTHER = 101, 202


class FakeMembership:
    def __init__(self, spaces_by_user):
        self.spaces_by_user = spaces_by_user

    async def authorized_spaces(self, user_id, username):
        return self.spaces_by_user.get(user_id, [])


class FakeRetrieval:
    """Records the space_ids every call was scoped to."""

    def __init__(self):
        self.scopes: list[tuple[str, list[int]]] = []
        self.issues = {
            "ATC-77": {
                "issue_key": "ATC-77", "space_id": ATC, "title": "Checkout times out",
                "status": "In Progress", "issue_type": "bug", "story_points": 3,
            },
        }

    def _visible(self, space_ids, keys):
        return [self.issues[k] for k in keys if k in self.issues and self.issues[k]["space_id"] in space_ids]

    async def query_issues(self, space_ids, filters):
        self.scopes.append(("query_issues", list(space_ids)))
        if "issue_keys" in filters:
            rows = self._visible(space_ids, filters["issue_keys"])
        else:
            rows = [
                {"issue_key": "ATC-1", "title": "Blocked thing", "status": "Blocked", "story_points": 5,
                 "assignee_name": "Maya", "updated_at": "2026-01-01T00:00:00Z"},
                {"issue_key": "ATC-2", "title": "Done thing", "status": "Done", "story_points": 3,
                 "assignee_name": "Maya", "updated_at": "2026-01-01T00:00:00Z"},
            ]
        return IssueQueryResult(total_count=len(rows), counts_by_type={}, counts_by_status={}, issues=rows)

    async def get_issue_details(self, space_ids, issue_keys, limit=200):
        self.scopes.append(("get_issue_details", list(space_ids)))
        rows = [{"issue_key": i["issue_key"], "content": "Checkout times out under load."}
                for i in self._visible(space_ids, issue_keys)]
        return IssueDetailsResult(total_count=len(rows), details=rows)

    async def get_issue_comments(self, space_ids, issue_keys, limit=200):
        self.scopes.append(("get_issue_comments", list(space_ids)))
        rows = [{"content": "Pool exhausted.", "provenance": {"author": "Maya"}}] if self._visible(space_ids, issue_keys) else []
        return IssueCommentsResult(total_count=len(rows), comments=rows)

    async def get_issue_attachments(self, space_ids, issue_keys, limit=200):
        self.scopes.append(("get_issue_attachments", list(space_ids)))
        rows = [{"source_id": 9, "page_number": 2, "content": "x" * 20_000, "provenance": {}}] if self._visible(space_ids, issue_keys) else []
        return IssueAttachmentsResult(total_count=len(rows), attachments=rows)

    async def query_issue_history(self, space_ids, filters):
        self.scopes.append(("query_issue_history", list(space_ids)))
        return IssueHistoryResult(total_count=0, changes=[])

    async def query_sprints(self, space_ids, filters):
        self.scopes.append(("query_sprints", list(space_ids)))
        rows = [{"sprint_id": 7, "sprint_name": "Sprint 12", "status": "active",
                 "start_date": "2026-01-01", "end_date": "2026-01-14", "goal": "Ship checkout v2"}]
        return SprintQueryResult(total_count=1, counts_by_status={}, sprints=rows)


def _app(membership) -> FastAPI:
    app = FastAPI()
    app.state.space_membership = membership
    app.state.retrieval = FakeRetrieval()
    install_mcp_route(app)
    return app


async def _call(app, tool, args, headers, asgi_app=None):
    def client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app or app), base_url="http://ai-service.test",
            headers=headers, timeout=timeout, auth=auth,
        )

    async with run_mcp_server(app):
        async with streamablehttp_client(
            "http://ai-service.test/mcp", headers=headers, httpx_client_factory=client_factory
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                if tool is None:
                    return session, await session.list_tools(), await session.list_prompts()
                return await session.call_tool(tool, args)


def _text(result) -> str:
    return result.content[0].text


def _json(result) -> dict:
    assert not result.isError, _text(result)
    return json.loads(_text(result))


MAYA = {"x-user-id": "42", "x-username": "maya"}
SPACES = {"42": [{"id": ATC, "key": "ATC", "name": "AtlasCart"}, {"id": OTHER, "key": "OPS", "name": "Ops"}]}


async def test_no_tool_lets_the_model_choose_space_ids():
    _, tools, prompts = await _call(_app(FakeMembership(SPACES)), None, None, MAYA)
    names = {t.name for t in tools.tools}
    assert names == {"ask_project", "get_ticket_context", "find_similar_work", "check_sprint_risk"}
    for t in tools.tools:
        assert "space_ids" not in json.dumps(t.inputSchema), t.name
        assert "ctx" not in t.inputSchema.get("properties", {}), t.name
    assert [p.name for p in prompts.prompts] == ["start_ticket"]


async def test_calls_without_user_identity_are_refused():
    app = _app(FakeMembership(SPACES))
    result = await _call(app, "get_ticket_context", {"issue_key": "ATC-77"}, {})
    assert result.isError
    assert "Not signed in" in _text(result)
    assert app.state.retrieval.scopes == []


async def test_disabled_membership_check_fails_closed():
    app = _app(NoopSpaceMembershipChecker())
    result = await _call(app, "get_ticket_context", {"issue_key": "ATC-77"}, MAYA)
    assert result.isError
    assert "AI_SPACE_MEMBERSHIP_CHECK_ENABLED" in _text(result)


async def test_space_argument_narrows_but_cannot_widen(monkeypatch):
    seen = []

    async def fake_ask(req, request):
        seen.append(req.space_ids)
        return SimpleNamespace(
            answer="Because the pool was exhausted [ATC-77].", abstained=False, needs_clarification=False,
            clarification_options=[], route_tool="search_knowledge_base", cache_hit=False,
            estimated_cost_usd=0.001,
            citations=[SimpleNamespace(issue_key="ATC-77", chunk_type="comment", page_number=None, content="Pool exhausted.")],
        )

    monkeypatch.setattr(api_routes, "ask", fake_ask)
    app = _app(FakeMembership(SPACES))

    body = _json(await _call(app, "ask_project", {"question": "why?", "space": "atc"}, MAYA))
    assert seen == [[ATC]]
    assert body["citations"][0]["issue_key"] == "ATC-77"

    result = await _call(app, "ask_project", {"question": "why?", "space": "SECRET"}, MAYA)
    assert result.isError
    assert "Your spaces: ATC, OPS" in _text(result)
    assert seen == [[ATC]]  # the refused call never reached the agent


async def test_ticket_in_another_users_space_reads_as_not_found():
    app = _app(FakeMembership({"7": [{"id": OTHER, "key": "OPS", "name": "Ops"}]}))
    result = await _call(app, "get_ticket_context", {"issue_key": "atc-77"}, {"x-user-id": "7", "x-username": "x"})
    assert result.isError
    assert "ATC-77 was not found in your spaces" in _text(result)
    assert all(scope == [OTHER] for _, scope in app.state.retrieval.scopes)


async def test_ticket_context_bundles_everything_and_caps_attachment_text():
    app = _app(FakeMembership(SPACES))
    body = _json(await _call(app, "get_ticket_context", {"issue_key": "ATC-77"}, MAYA))
    assert body["issue"]["title"] == "Checkout times out"
    assert body["comments"][0]["author"] == "Maya"
    assert body["attachments"][0]["page_number"] == 2
    assert len(body["attachments"][0]["content"]) < 12_100
    assert all(scope == [ATC, OTHER] for _, scope in app.state.retrieval.scopes)


async def test_sprint_risk_needs_a_space_when_the_user_has_several():
    result = await _call(_app(FakeMembership(SPACES)), "check_sprint_risk", {}, MAYA)
    assert result.isError
    assert "pass `space`" in _text(result)


async def test_sprint_risk_reports_code_computed_progress():
    app = _app(FakeMembership(SPACES))
    body = _json(await _call(app, "check_sprint_risk", {"space": "ATC"}, MAYA))
    assert body["sprint"]["name"] == "Sprint 12"
    assert body["progress"] == {
        "issues_total": 2, "issues_done": 1, "points_total": 8, "points_done": 3, "unestimated_issues": 0,
    }
    assert any(s["issue_key"] == "ATC-1" for s in body["risk_signals"])
    assert all(scope == [ATC] for _, scope in app.state.retrieval.scopes)


# --- Per-user AI limits, through the real JWT middleware ------------------------------------------

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class _FakeJwks:
    def get_signing_key_from_jwt(self, token):
        return jwt.PyJWK.from_dict(jwt.algorithms.RSAAlgorithm.to_jwk(_KEY.public_key(), as_dict=True))


def _signed_in(app: FastAPI):
    verifier = JwtVerifier("http://jwks.invalid/jwks.json", "jira-agentic-ai")
    verifier._jwks = _FakeJwks()
    token = jwt.encode(
        {"iss": "jira-agentic-ai", "sub": "maya", "uid": 42, "exp": int(time.time()) + 60},
        _KEY, algorithm="RS256",
    )
    return RequestAuthMiddleware(app, verifier=verifier, internal_token="internal"), {"Authorization": f"Bearer {token}"}


@pytest.mark.parametrize("tool, args", [
    ("ask_project", {"question": "why?"}),
    ("find_similar_work", {"issue_key": "ATC-77"}),
])
async def test_tools_that_start_ai_work_count_against_the_users_limit(monkeypatch, tool, args):
    calls = []

    async def fake_ask(req, request):
        calls.append(tool)
        return SimpleNamespace(
            answer="ok", abstained=False, needs_clarification=False, clarification_options=[],
            route_tool="search_knowledge_base", cache_hit=False, estimated_cost_usd=0.0, citations=[],
        )

    async def fake_precedents(req, request):
        calls.append(tool)
        return SimpleNamespace(verdict="none_relevant", precedents=[], candidates_considered=0, degraded=False)

    monkeypatch.setattr(api_routes, "ask", fake_ask)
    monkeypatch.setattr(api_routes, "precedents_endpoint", fake_precedents)
    app = _app(FakeMembership(SPACES))
    app.state.user_quota = UserQuota(fakeredis.FakeAsyncRedis(), requests_per_minute=2, daily_budget_usd=2.0)
    asgi_app, headers = _signed_in(app)

    for _ in range(2):
        assert not (await _call(app, tool, args, headers, asgi_app)).isError
    result = await _call(app, tool, args, headers, asgi_app)
    assert result.isError
    assert "per minute" in _text(result)
    assert len(calls) == 2  # the refused call never reached the LLM
