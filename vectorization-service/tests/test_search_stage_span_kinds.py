"""Each `/search` stage span must carry the OpenInference kind it actually is.

Before this, `search.embed`/`search.retrieval`/`search.rerank` carried no `openinference.span.kind`
attribute at all, so Phoenix and Langfuse showed all three as a kind-less generic `SPAN` —
indistinguishable from a plain HTTP hop, and (in ai-service's Langfuse exporter, which forwards only
spans carrying an OpenInference kind or its own usage-ledger attribute) not even exported to Langfuse
at all. Verified against the real enum values rather than assumed strings, since a typo here fails
silently — Langfuse falls back to the same generic `SPAN` display for an unrecognized kind string as
it does for no kind at all.
"""
from __future__ import annotations

from types import SimpleNamespace

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.api import routes
from app.db.reranker import NoopReranker
from app.models import SearchHit


class FakeEmbedder:
    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    async def search_vector(self, embedding, space_ids, limit):
        return [_hit(space_ids[0])]

    async def search_hybrid(self, embedding, query, space_ids, limit, candidate_pool=20):
        return [_hit(space_ids[0])]

    async def search_bm25(self, query, space_ids, limit):
        return [_hit(space_ids[0])]


def _hit(space_id):
    return SearchHit(
        id="issue:1", chunk_type="issue", issue_id=1, issue_key="X-1", space_id=space_id,
        source_id=1, content="c", score=1.0, retrievers=["vector"],
    )


def _request(monkeypatch, *, rerank_enabled: bool):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(routes, "get_tracer", lambda: provider.get_tracer("test"))

    settings = SimpleNamespace(rerank_enabled=rerank_enabled, rerank_candidate_pool=20)
    app_state = SimpleNamespace(
        store=FakeStore(), embedder=FakeEmbedder(),
        reranker=NoopReranker() if rerank_enabled else None, settings=settings,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=app_state))
    return request, exporter


def _kinds_by_stage(exporter):
    return {
        span.name.removeprefix("search."): span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        for span in exporter.get_finished_spans()
    }


async def test_embed_and_retrieval_stages_carry_their_real_kinds(monkeypatch):
    request, exporter = _request(monkeypatch, rerank_enabled=False)

    await routes.search(
        routes.SearchRequest(query="q", space_ids=[5000018], limit=5, mode="vector"), request
    )

    kinds = _kinds_by_stage(exporter)
    assert kinds["embed"] == OpenInferenceSpanKindValues.EMBEDDING.value
    assert kinds["retrieval"] == OpenInferenceSpanKindValues.RETRIEVER.value


async def test_the_rerank_stage_carries_the_reranker_kind(monkeypatch):
    request, exporter = _request(monkeypatch, rerank_enabled=True)

    await routes.search(
        routes.SearchRequest(query="q", space_ids=[5000018], limit=5, mode="hybrid"), request
    )

    assert _kinds_by_stage(exporter)["rerank"] == OpenInferenceSpanKindValues.RERANKER.value


async def test_bm25_mode_skips_the_embedding_call_and_its_span_entirely(monkeypatch):
    """No vector, nothing to embed — see the route's own comment on why paying for an embedding
    here would be latency and spend for a vector nothing reads. A span that never opens can't be
    mis-kinded, but it's worth pinning that the bm25 path really does skip it rather than opening an
    embed span with zero work inside it."""
    request, exporter = _request(monkeypatch, rerank_enabled=False)

    await routes.search(
        routes.SearchRequest(query="q", space_ids=[5000018], limit=5, mode="bm25"), request
    )

    kinds = _kinds_by_stage(exporter)
    assert "embed" not in kinds
    assert kinds["retrieval"] == OpenInferenceSpanKindValues.RETRIEVER.value
