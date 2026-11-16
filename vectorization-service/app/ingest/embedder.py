"""Embedding providers.

Production: the `openai` provider pointed at the LiteLLM gateway, whose `openai-embedding-3-small`
route is OpenAI text-embedding-3-small (1536 dims, MRL-truncated to 1024 by `_truncate_and_renormalize`
for the vector(1024) column). It works with any OpenAI-compatible server via `openai_base_url`. A deterministic fake embedder runs without a key or network (tests).

Providers share `embed(texts) -> list[vector]`, batch internally and retry transient failures —
the slow, flaky part of ingestion, which is why it runs behind Kafka in its own service.
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import List, Protocol

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import Settings

logger = logging.getLogger(__name__)

Vector = List[float]


def _truncate_and_renormalize(vector: Vector, dim: int) -> Vector:
    """Keep the first `dim` values of an MRL embedding and re-normalize to unit length. Normalize after
    slicing — the prefix's length drifts from 1, and cosine similarity is sensitive to that. A zero
    vector is left as is.
    """
    head = vector[:dim]
    norm = math.sqrt(sum(x * x for x in head))
    if norm == 0.0:
        return head
    return [x / norm for x in head]


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: List[str]) -> List[Vector]: ...

    async def aclose(self) -> None: ...


def _batched(items: List[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class _HttpEmbedder:
    """Shared batching + retry wrapper for REST-based (OpenAI-compatible) providers."""

    endpoint: str
    dim: int

    def __init__(self, settings: Settings, api_key: str, endpoint: str | None = None):
        self._model = settings.embedding_model
        self._batch_size = settings.embedding_batch_size
        self._max_retries = settings.embedding_max_retries
        self.dim = settings.embedding_dim
        self._truncate_dim = settings.embedding_truncate_dim
        if endpoint is not None:
            self.endpoint = endpoint
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    async def embed(self, texts: List[str]) -> List[Vector]:
        vectors: List[Vector] = []
        for batch in _batched(texts, self._batch_size):
            vectors.extend(await self._embed_batch(batch))
        if self._truncate_dim is not None:
            vectors = [_truncate_and_renormalize(v, self._truncate_dim) for v in vectors]
        return vectors

    async def _embed_batch(self, batch: List[str]) -> List[Vector]:
        # tenacity retries transient API failures (429/5xx/timeouts) with backoff. A batch that
        # still fails after retries raises, so the Kafka consumer does not commit its offset and the
        # message is redelivered — at-least-once, which the deterministic upsert makes safe.
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            reraise=True,
        )
        async def _call() -> List[Vector]:
            resp = await self._client.post(self.endpoint, json=self._payload(batch))
            resp.raise_for_status()
            return self._parse(resp.json())

        return await _call()

    def _payload(self, batch: List[str]) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def _parse(self, data: dict) -> List[Vector]:
        return [item["embedding"] for item in data["data"]]

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAIEmbedder(_HttpEmbedder):
    """OpenAI or any OpenAI-compatible server (the LiteLLM gateway by default), via ``openai_base_url``."""

    def _payload(self, batch: List[str]) -> dict:
        return {"model": self._model, "input": batch}


class FakeEmbedder:
    """Deterministic hash-based vectors — no network, no key. For local runs and tests only."""

    def __init__(self, dim: int):
        self.dim = dim

    async def embed(self, texts: List[str]) -> List[Vector]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> Vector:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        # Cycle the digest bytes into `dim` floats in [0, 1). Deterministic: same text -> same vector.
        return [seed[i % len(seed)] / 255.0 for i in range(self.dim)]

    async def aclose(self) -> None:  # pragma: no cover - nothing to close
        return None


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        base_url = settings.openai_base_url.rstrip("/")
        is_local = "api.openai.com" not in base_url
        # A self-hosted OpenAI-compatible server may need no real key — any string works.
        if not settings.openai_api_key and not is_local:
            logger.warning("VEC_OPENAI_API_KEY is empty; falling back to the fake embedder")
            return FakeEmbedder(settings.embedding_dim)
        api_key = settings.openai_api_key or "local"
        return OpenAIEmbedder(settings, api_key, endpoint=f"{base_url}/embeddings")
    if provider == "fake":
        return FakeEmbedder(settings.embedding_dim)
    raise ValueError(
        f"unknown embedding_provider: {settings.embedding_provider!r} (expected 'openai' or 'fake')"
    )
