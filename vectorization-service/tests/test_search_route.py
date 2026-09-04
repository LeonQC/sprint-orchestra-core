"""Regression: the /search route must thread VEC_RERANK_CANDIDATE_POOL into search_hybrid's own
``candidate_pool`` argument, not just its ``limit`` argument.

Found live: raising VEC_RERANK_CANDIDATE_POOL from 20 to 100 had zero effect on a hybrid-mode query
whose correct chunk ranked #31 by raw vector similarity — still missing, unchanged. Root cause:
``VectorStore.search_hybrid`` takes its OWN ``candidate_pool`` parameter (how many vector/lexical rows
feed RRF fusion before reranking), independently defaulted to 20, completely separate from the `limit`
parameter (how many FUSED rows come out). The route was calling
``store.search_hybrid(embedding, query, space_ids, fetch_limit)`` — filling only `limit` — so the
first-stage pool feeding fusion silently stayed at 20 no matter how wide `fetch_limit` (derived from
the raised setting) was. A chunk outside the top 20 of vector OR lexical search alone could never
reach RRF fusion, let alone the reranker, regardless of the setting.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.api.routes import SearchRequest, search
from app.db.reranker import NoopReranker
from app.models import SearchHit


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    """Records every call it receives; returns one hit so the route completes normally."""

    def __init__(self):
        self.search_hybrid_calls = []

    async def search_hybrid(self, embedding, query, space_ids, limit, candidate_pool=20):
        self.search_hybrid_calls.append(
            {"limit": limit, "candidate_pool": candidate_pool, "space_ids": list(space_ids)}
        )
        return [
            SearchHit(
                id="issue:1", chunk_type="issue", issue_id=1, issue_key="X-1", space_id=space_ids[0],
                source_id=1, content="c", score=1.0, retrievers=["vector"],
            )
        ]


def _fake_request(settings):
    app_state = SimpleNamespace(
        store=FakeStore(), embedder=FakeEmbedder(), reranker=NoopReranker(), settings=settings,
    )
    return SimpleNamespace(app=SimpleNamespace(state=app_state))


async def test_hybrid_search_passes_the_raised_setting_as_search_hybrid_candidate_pool():
    settings = SimpleNamespace(rerank_enabled=True, rerank_candidate_pool=100)
    request = _fake_request(settings)
    req = SearchRequest(query="ATC-432 LINGYI flight information", space_ids=[5000018], limit=5, mode="hybrid")

    await search(req, request)

    store = request.app.state.store
    assert len(store.search_hybrid_calls) == 1
    call = store.search_hybrid_calls[0]
    # This is the exact assertion that would have failed before the fix: candidate_pool silently
    # stayed at search_hybrid's own default (20) instead of picking up the raised setting.
    assert call["candidate_pool"] == 100
    assert call["limit"] == 100


async def test_hybrid_search_candidate_pool_tracks_a_different_raised_setting_value():
    """Not a coincidence with the default — a second, distinct value proves the setting is actually
    threaded through, not hardcoded to a lucky-matching constant."""
    settings = SimpleNamespace(rerank_enabled=True, rerank_candidate_pool=37)
    request = _fake_request(settings)
    req = SearchRequest(query="anything", space_ids=[7], limit=5, mode="hybrid")

    await search(req, request)

    call = request.app.state.store.search_hybrid_calls[0]
    assert call["candidate_pool"] == 37


class FailingEmbedder:
    async def embed(self, texts):
        raise ConnectionError("embedding provider unreachable")


class Bm25Store(FakeStore):
    def __init__(self):
        super().__init__()
        self.search_bm25_calls = 0

    async def search_bm25(self, query, space_ids, limit):
        self.search_bm25_calls += 1
        return [
            SearchHit(
                id="issue:2", chunk_type="issue", issue_id=2, issue_key="X-2", space_id=space_ids[0],
                source_id=2, content="keyword hit", score=3.2, retrievers=["bm25"],
            )
        ]


async def test_embedding_outage_degrades_hybrid_search_to_bm25_instead_of_failing():
    settings = SimpleNamespace(rerank_enabled=False, rerank_candidate_pool=20)
    request = _fake_request(settings)
    request.app.state.embedder = FailingEmbedder()
    request.app.state.store = Bm25Store()
    req = SearchRequest(query="payment outage", space_ids=[7], limit=5, mode="hybrid")

    response = await search(req, request)

    assert [h.issue_key for h in response.hits] == ["X-2"]
    assert (response.mode, response.degraded) == ("bm25", True)
    assert request.app.state.store.search_hybrid_calls == []


async def test_a_healthy_hybrid_search_is_not_marked_degraded():
    settings = SimpleNamespace(rerank_enabled=False, rerank_candidate_pool=20)
    request = _fake_request(settings)
    req = SearchRequest(query="payment outage", space_ids=[7], limit=5, mode="hybrid")

    response = await search(req, request)

    assert (response.mode, response.degraded) == ("hybrid", False)
