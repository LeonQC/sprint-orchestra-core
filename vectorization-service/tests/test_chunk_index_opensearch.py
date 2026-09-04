"""Integration tests for the OpenSearch chunk index, run against a real cluster.

Not mocked, on purpose. Everything worth testing here is behavior of OpenSearch itself — that the
analyzer strips HTML, that separator normalization makes "ATLAS-6" findable, that BM25's length
normalization stops a long repetitive chunk from beating a short on-topic one, that a `terms` filter
actually scopes results. A fake client would only assert that we build the request body we think we
build, which is the part that was never in doubt.

Skipped (not failed) when no cluster is reachable, so `pytest` stays runnable without Docker; CI and
local runs with `docker compose up` get the real coverage. Each test gets its own throwaway index so
they neither collide nor depend on ordering.
"""
from __future__ import annotations

import os
import uuid

import pytest

from app.models import Chunk

pytestmark = pytest.mark.asyncio

OS_URL = os.environ.get("VEC_OPENSEARCH_URL", "http://localhost:9200")


@pytest.fixture
async def index():
    opensearchpy = pytest.importorskip("opensearchpy")
    from app.search.chunk_index import ChunkSearchIndex

    client = opensearchpy.AsyncOpenSearch(hosts=[OS_URL], timeout=5)
    try:
        if not await client.ping():
            pytest.skip(f"no OpenSearch at {OS_URL}")
    except Exception:  # noqa: BLE001 - connection refused, DNS failure, TLS mismatch, ...
        await client.close()
        pytest.skip(f"no OpenSearch at {OS_URL}")

    idx = ChunkSearchIndex(client, f"test-chunks-{uuid.uuid4().hex[:8]}")
    await idx.ensure_index()
    try:
        yield idx
    finally:
        await client.indices.delete(index=idx.index_name, ignore_unavailable=True)
        await client.close()


def _chunk(chunk_id: str, content: str, *, space_id: int = 1, issue_id: int = 1,
           issue_key: str = "ATLAS-1", chunk_type: str = "issue", source_id: int = 1) -> Chunk:
    return Chunk(
        id=chunk_id, chunk_type=chunk_type, issue_id=issue_id, issue_key=issue_key,
        space_id=space_id, source_id=source_id, chunk_index=0, content=content,
    )


async def test_bm25_ranks_the_on_topic_chunk_first(index):
    await index.index_chunks([
        _chunk("a", "The checkout service times out under load when the connection pool saturates."),
        _chunk("b", "Weekly team sync notes: attendance, coffee order, parking arrangements."),
    ])
    await index.refresh()

    hits = await index.search("checkout timeout connection pool", [1], limit=5)

    assert [h.id for h in hits] == ["a"]
    assert hits[0].retrievers == ["bm25"]
    assert hits[0].score > 0


async def test_space_filter_is_an_authorization_boundary(index):
    """A caller must never see a chunk from a space they are not scoped to, however well it matches."""
    await index.index_chunks([
        _chunk("mine", "checkout timeout", space_id=1),
        _chunk("theirs", "checkout timeout", space_id=99),
    ])
    await index.refresh()

    hits = await index.search("checkout timeout", [1], limit=10)

    assert [h.id for h in hits] == ["mine"]


async def test_html_markup_is_not_searchable_content(index):
    """Issue text is stored as HTML. Without html_strip, a query for a tag name matches every
    document, which is both wrong and unrankable."""
    await index.index_chunks([
        _chunk("a", "<p>The <strong>payment</strong> gateway rejects expired cards.</p>"),
    ])
    await index.refresh()

    assert [h.id for h in await index.search("payment gateway", [1], limit=5)] == ["a"]
    assert await index.search("strong", [1], limit=5) == []


async def test_identifier_separators_are_normalized(index):
    """Migration V24's problem, ported: a hyphenated identifier must tokenize the same way whether it
    arrives alone (as a query) or inside a sentence (as content)."""
    await index.index_chunks([
        _chunk("a", "Rolled back after the incident.", issue_key="ATLAS-6"),
        _chunk("b", "The attachment metrics-wire-dlq-depth.csv shows the spike.", issue_key="ATLAS-7"),
    ])
    await index.refresh()

    # Migration 003's fix: the issue key is searchable even though it appears nowhere in the prose,
    # and an AND-matched key field keeps the named issue on top rather than every ATLAS-* issue
    # tying on the shared "atlas" token.
    assert (await index.search("ATLAS-6", [1], limit=5))[0].id == "a"
    # A filename queried as a bare phrase still matches the sentence it is embedded in.
    assert "b" in [h.id for h in await index.search("metrics wire dlq depth", [1], limit=5)]


async def test_bm25_length_normalization_beats_raw_term_frequency(index):
    """The concrete reason ts_rank_cd was not good enough: it has no length normalization, so a long
    chunk that merely repeats a term outranks a short chunk that is actually about it. BM25's `b`
    parameter is what fixes that, and this asserts the fixed behavior rather than the mechanism."""
    padding = " ".join(f"unrelated{i}" for i in range(400))
    await index.index_chunks([
        _chunk("short", "Redis eviction policy tuning."),
        _chunk("long", f"redis {padding} redis {padding} redis {padding} redis"),
    ])
    await index.refresh()

    hits = await index.search("redis eviction policy", [1], limit=5)

    assert hits[0].id == "short"


async def test_replace_source_drops_chunks_that_no_longer_exist(index):
    """A re-parse can yield fewer chunks than last time. Keyed upserts alone would leave the vanished
    tail behind as stale hits — which reads to a user as a deleted page still being cited."""
    await index.replace_source("attachment", 5, [
        _chunk("attachment:5#0", "page one about caching", chunk_type="attachment", source_id=5),
        _chunk("attachment:5#1", "page two about caching", chunk_type="attachment", source_id=5),
    ])
    assert len(await index.search("caching", [1], limit=10)) == 2

    await index.replace_source("attachment", 5, [
        _chunk("attachment:5#0", "page one about caching", chunk_type="attachment", source_id=5),
    ])

    assert [h.id for h in await index.search("caching", [1], limit=10)] == ["attachment:5#0"]


async def test_delete_issue_removes_its_comments_and_attachments_too(index):
    await index.index_chunks([
        _chunk("issue:1", "deployment rollback", issue_id=1, chunk_type="issue", source_id=1),
        _chunk("comment:2", "deployment rollback discussion", issue_id=1, chunk_type="comment", source_id=2),
        _chunk("issue:9", "deployment rollback elsewhere", issue_id=9, chunk_type="issue", source_id=9),
    ])
    await index.refresh()

    removed = await index.delete_issue(1)

    assert removed == 2
    assert [h.id for h in await index.search("deployment rollback", [1], limit=10)] == ["issue:9"]


async def test_reindexing_the_same_chunk_does_not_duplicate_it(index):
    """`POST /search/reindex` is the repair path for Postgres/OpenSearch drift, so running it
    twice has to converge. Deterministic document ids are what make that true."""
    chunk = _chunk("issue:1", "idempotent write path")
    await index.index_chunks([chunk])
    await index.index_chunks([chunk])
    await index.refresh()

    assert await index.count() == 1


async def test_empty_query_returns_nothing_rather_than_everything(index):
    await index.index_chunks([_chunk("a", "anything at all")])
    await index.refresh()

    assert await index.search("   ", [1], limit=5) == []
    assert await index.search("anything", [], limit=5) == []


async def test_deleting_issue_keeps_sprint_with_same_numeric_id(index):
    await index.index_chunks([
        _chunk("issue:42", "shared identifier", issue_id=42, chunk_type="issue", source_id=42),
        _chunk("sprint:42", "shared identifier", issue_id=42, issue_key="Sprint 42",
               chunk_type="sprint", source_id=42),
    ])
    await index.refresh()
    assert await index.delete_issue(42) == 1
    assert [h.id for h in await index.search("shared identifier", [1], limit=10)] == ["sprint:42"]
    assert await index.delete_source("sprint", 42) == 1
    assert await index.search("shared identifier", [1], limit=10) == []
