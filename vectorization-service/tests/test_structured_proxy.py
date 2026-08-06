"""Existing AI tool contracts read authoritative facts and fail closed on upstream errors."""
import json

import httpx
import pytest
from fastapi import FastAPI

from app.api.routes import router
from app.structured_client import StructuredRetrievalClient


@pytest.mark.parametrize("path,payload,response", [
    ("issues/query", {"space_ids": [7], "priorities": ["high"], "parent_keys": ["ATC-52"],
                      "updated_after": "2026-01-01T00:00:00Z"},
     {"total_count": 2, "counts_by_type": {"bug": 2}, "counts_by_status": {"done": 2},
      "counts_by_assignee": {"(unassigned)": 2}, "issues": []}),
    ("issues/history", {"space_ids": [7], "reopened_only": True},
     {"total_count": 1, "changes": [{"issue_key": "ATC-1", "event_type": "field_change",
      "field_name": "status", "from_value": "done", "to_value": "in_progress",
      "description": "Updated status", "actor_name": "Alex", "changed_at": "2026-01-01T00:00:00Z"}]}),
    ("sprints/query", {"space_ids": [7], "statuses": ["active"], "order": "asc"},
     {"total_count": 3, "counts_by_status": {"active": 3}, "sprints": []}),
])
async def test_existing_routes_forward_scope_filters_and_preserve_response(path, payload, response):
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json=response)

    source = StructuredRetrievalClient("http://jira", "test-token")
    await source._client.aclose()
    source._client = httpx.AsyncClient(
        base_url="http://jira", headers={"X-Internal-Token": "test-token"},
        transport=httpx.MockTransport(upstream),
    )
    app = FastAPI()
    # Deliberately no VectorStore: these routes must never read the legacy tables.
    app.state.structured = source
    app.include_router(router)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://vec") as client:
            result = await client.post(f"/{path}", json=payload)
        assert result.status_code == 200
        assert result.json()["total_count"] == response["total_count"]
        assert calls[0].url.path == f"/internal/retrieval/{path}"
        assert calls[0].headers["X-Internal-Token"] == "test-token"
        forwarded = json.loads(calls[0].content)
        assert forwarded["space_ids"] == [7]
        for key, value in payload.items():
            if key != "updated_after":
                assert forwarded[key] == value
    finally:
        await source.aclose()


@pytest.mark.parametrize("path", ["issues/query", "issues/history", "sprints/query"])
async def test_scope_and_upstream_failures_cannot_return_stale_replica_results(path):
    class Unavailable:
        async def query(self, *_args):
            raise httpx.ConnectError("jira unavailable")

    app = FastAPI()
    app.state.structured = Unavailable()
    app.include_router(router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://vec") as client:
        assert (await client.post(f"/{path}", json={"space_ids": []})).status_code == 422
        result = await client.post(f"/{path}", json={"space_ids": [7]})
    assert result.status_code == 502


async def test_backend_failure_is_propagated_by_client():
    source = StructuredRetrievalClient("http://jira", "test-token")
    await source._client.aclose()
    source._client = httpx.AsyncClient(base_url="http://jira", transport=httpx.MockTransport(
        lambda _request: httpx.Response(503, json={"error": "database unavailable"}),
    ))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await source.query("issues/query", {"space_ids": [7]})
    finally:
        await source.aclose()
