"""Tests for app/ingest/embedder.py's MRL truncation path — same mocked-httpx-transport approach
as tests/test_vlm_describer.py uses for the analogous VLM client.
"""
from __future__ import annotations

import json
import math

import httpx

from app.config import Settings
from app.ingest.embedder import OpenAIEmbedder, _truncate_and_renormalize


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_body: dict):
        self._body = response_body
        self.sent_requests = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.sent_requests.append(request)
        return httpx.Response(200, json=self._body)


def _unit_vector(dim: int, seed: float) -> list[float]:
    raw = [seed + i for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def test_truncate_and_renormalize_slices_then_returns_to_the_unit_sphere():
    # A native 4096-dim unit vector, truncated to 1024: the first 1024 values alone are not unit
    # norm any more (dropping the tail changes the magnitude) — renormalizing afterward is what
    # makes the result comparable via cosine similarity to any other 1024-dim vector in the index.
    native = _unit_vector(4096, seed=1.0)
    truncated = _truncate_and_renormalize(native, 1024)

    assert len(truncated) == 1024
    # Sanity: the raw slice, before renormalizing, is NOT unit norm — this is exactly why skipping
    # the renormalize step would silently corrupt every cosine-similarity comparison downstream.
    raw_slice_norm = math.sqrt(sum(x * x for x in native[:1024]))
    assert not math.isclose(raw_slice_norm, 1.0, rel_tol=1e-9)
    norm = math.sqrt(sum(x * x for x in truncated))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)
    # Renormalizing preserves direction (proportional to the raw slice), just rescales it.
    ratio = truncated[0] / native[0]
    assert all(math.isclose(t / n, ratio, rel_tol=1e-9) for t, n in zip(truncated, native[:1024]))


def test_truncate_and_renormalize_leaves_a_zero_vector_unchanged_rather_than_dividing_by_zero():
    assert _truncate_and_renormalize([0.0] * 10, 4) == [0.0, 0.0, 0.0, 0.0]


async def test_openai_embedder_passes_vectors_through_unchanged_when_truncate_dim_is_unset():
    fake = _FakeTransport({"data": [{"embedding": [0.6, 0.8]}]})
    settings = Settings(embedding_provider="openai", embedding_model="m", embedding_dim=2)
    embedder = OpenAIEmbedder(settings, "local", endpoint="http://x/embeddings")
    embedder._client = httpx.AsyncClient(transport=fake)

    vectors = await embedder.embed(["hello"])

    assert vectors == [[0.6, 0.8]]
    await embedder.aclose()


async def test_openai_embedder_truncates_and_renormalizes_when_configured():
    # Simulates the real Qwen3-Embedding-8B case: the provider returns its native 4-dim vector (a
    # small stand-in for 4096), and the embedder must hand back a renormalized 2-dim (stand-in for
    # 1024) vector matching what the pgvector column actually expects.
    native = [3.0, 4.0, 0.0, 0.0]  # norm 5; first two values (3,4) are NOT unit norm on their own
    fake = _FakeTransport({"data": [{"embedding": native}]})
    settings = Settings(
        embedding_provider="openai", embedding_model="m", embedding_dim=2, embedding_truncate_dim=2,
    )
    embedder = OpenAIEmbedder(settings, "local", endpoint="http://x/embeddings")
    embedder._client = httpx.AsyncClient(transport=fake)

    vectors = await embedder.embed(["hello"])

    assert len(vectors[0]) == 2
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-9)
    assert vectors[0] == [0.6, 0.8]  # (3,4) renormalized to unit length
    await embedder.aclose()


async def test_openai_embedder_sends_the_configured_model_regardless_of_truncation():
    fake = _FakeTransport({"data": [{"embedding": [1.0, 0.0]}]})
    settings = Settings(
        embedding_provider="openai", embedding_model="text-embedding-qwen3-8b-text-embedding",
        embedding_dim=2, embedding_truncate_dim=2,
    )
    embedder = OpenAIEmbedder(settings, "local", endpoint="http://x/embeddings")
    embedder._client = httpx.AsyncClient(transport=fake)

    await embedder.embed(["hello"])

    sent = json.loads(fake.sent_requests[0].content)
    assert sent["model"] == "text-embedding-qwen3-8b-text-embedding"
    await embedder.aclose()
