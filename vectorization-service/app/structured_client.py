"""Read authoritative facts through jira-backend; never fall back to stale local business tables."""
from __future__ import annotations

import httpx

from app.observability import get_request_id


class StructuredRetrievalClient:
    def __init__(self, base_url: str, internal_service_token: str, timeout: float = 15.0):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-Internal-Token": internal_service_token},
            timeout=timeout,
        )

    async def query(self, path: str, payload: dict) -> dict:
        response = await self._client.post(
            f"/internal/retrieval/{path}", json=payload,
            headers={"x-request-id": get_request_id()},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
