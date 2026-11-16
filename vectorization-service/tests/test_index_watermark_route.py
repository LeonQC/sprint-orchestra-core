"""GET /index/watermark — the freshness token ai-service's answer cache keys on.

The token's only contract is *different content ⇒ different token*, and everything below tests that
contract rather than the token's format, which callers must treat as opaque.

Why the count is in there at all is the least obvious part and is asserted explicitly: ``max()`` on
its own is not monotonic under row removal. Re-seed a workspace and the max can move *backwards* to a
timestamp the cache has already seen and stored answers under, which would resurrect stale entries
rather than invalidate them. The pair cannot collide that way.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes import index_watermark


class FakeStore:
    def __init__(self, result=(datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc), 100)):
        self.result = result
        self.calls = []

    async def index_watermark(self, space_ids=None):
        self.calls.append(space_ids)
        return self.result


class FakeSource:
    """jira-backend's structured API. The token must never call it (see the last test)."""

    def __init__(self):
        self.calls = []

    async def query(self, path, payload):
        self.calls.append((path, payload))
        return {}


def _request(store, source=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        store=store, structured=source or FakeSource(),
    )))


async def test_returns_a_token_and_scopes_the_query_to_the_requested_spaces():
    store = FakeStore()
    resp = await index_watermark(_request(store), space_ids="5,3")

    assert store.calls == [[5, 3]]
    assert resp.last_change == "2026-09-03T10:00:00+00:00"
    assert resp.document_count == 100
    assert resp.watermark == "2026-09-03T10:00:00+00:00:100"


async def test_an_unindexed_space_gets_a_stable_token_rather_than_an_empty_one():
    """"Nothing indexed yet" is a real state a cache may key on, not a missing value.

    An empty token would collide with any other empty-ish result and, worse, read as "no answer" to
    a caller checking truthiness — which is the code path reserved for *the check failed*, where
    reuse must be skipped entirely.
    """
    resp = await index_watermark(_request(FakeStore((None, 0))), space_ids="9")
    assert resp.last_change == "never"
    assert resp.watermark == "never:0"


async def test_the_token_changes_when_only_the_document_count_changes():
    """The reason ``max(updated_at)`` alone is not enough — see this module's docstring."""
    same_timestamp = datetime(2026, 9, 3, 10, 0, tzinfo=timezone.utc)
    a = await index_watermark(_request(FakeStore((same_timestamp, 100))), space_ids="1")
    b = await index_watermark(_request(FakeStore((same_timestamp, 99))), space_ids="1")
    assert a.watermark != b.watermark


async def test_no_space_ids_means_the_whole_index_not_an_empty_filter():
    """An empty/omitted filter must not be passed through as ``space_id = ANY('{}')``, which matches
    nothing and would report a permanently frozen "never" watermark for every space."""
    store = FakeStore()
    await index_watermark(_request(store), space_ids="")
    await index_watermark(_request(store), space_ids=" , ")
    assert store.calls == [None, None]


async def test_a_non_integer_space_id_is_rejected_rather_than_silently_dropped():
    """Dropping it would answer with a token scoped to *fewer* spaces than asked for — a token that
    stops changing when content in the dropped space changes, i.e. a silently stale cache."""
    with pytest.raises(HTTPException) as exc:
        await index_watermark(_request(FakeStore()), space_ids="1,abc")
    assert exc.value.status_code == 422


async def test_the_token_ignores_business_metadata():
    """A status drag must not empty the space's answer cache. Answers that used live status, sprint
    or history data are never cached (ai-service is_answer_cacheable), so the token only needs the
    searchable text, and it no longer reads every issue row on every question."""
    source = FakeSource()
    resp = await index_watermark(_request(FakeStore(), source), space_ids="5,3")
    assert resp.watermark == "2026-09-03T10:00:00+00:00:100"
    assert source.calls == []
