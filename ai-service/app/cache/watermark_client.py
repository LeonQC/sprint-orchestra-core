"""Client for vectorization-service's GET /index/watermark — the answer cache's freshness token.

The cache used to be TTL-only and was turned off, because a TTL can't promise an answer is still true
after an upload. With this token in the cache key, any indexed change makes older answers unreachable
at once. It combines index progress with authoritative issue/sprint/history fingerprints,
so business edits invalidate answers even before ingestion catches up.

Costs one cheap GET per lookup. `ttl_seconds` can memoize it, but defaults to 0 (always fetch) to
avoid answering from content that changed seconds ago. No token (cache off, or an error) means skip
the cache and compute — never reuse unchecked.
"""
from __future__ import annotations

import time
from typing import Sequence

import httpx

from app.observability import get_request_id


class IndexWatermarkClient:
    def __init__(self, base_url: str, ttl_seconds: float = 0.0, timeout: float = 5.0):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout)
        self._ttl = ttl_seconds
        self._memo: dict[str, tuple[float, str]] = {}

    async def watermark(self, space_ids: Sequence[int]) -> str:
        key = ",".join(str(s) for s in sorted(space_ids))
        if self._ttl > 0:
            hit = self._memo.get(key)
            if hit is not None and (time.monotonic() - hit[0]) < self._ttl:
                return hit[1]

        resp = await self._client.get(
            "/index/watermark",
            params={"space_ids": key},
            headers={"x-request-id": get_request_id()},
        )
        resp.raise_for_status()
        token = resp.json()["watermark"]

        if self._ttl > 0:
            self._memo[key] = (time.monotonic(), token)
        return token

    async def aclose(self) -> None:
        await self._client.aclose()


class NoopWatermarkClient:
    """Used when the cache is disabled, and as the hermetic default in tests."""

    async def watermark(self, space_ids: Sequence[int]) -> str:
        return "noop"

    async def aclose(self) -> None:
        return None
