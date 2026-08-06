"""Check that a caller's requested space_ids are within their real space memberships.

space_ids used to be trusted from the request body. Now the caller's X-User-Id (rewritten from the
verified JWT by app/auth/request_auth.py) is checked against jira-backend's membership list
(GET /api/spaces?userId=X, the same one the frontend uses).

Only enforced when X-User-Id is present, which is always the case for users. Without it the caller
proved it holds the internal service token (eval/smoke scripts), so it's trusted.
"""
from __future__ import annotations

import time
from typing import Optional

import httpx


class SpaceMembershipError(Exception):
    """Raised when a caller requests a space_id they are not a member of."""

    def __init__(self, user_id: str, unauthorized_space_ids: list[int]):
        self.user_id = user_id
        self.unauthorized_space_ids = unauthorized_space_ids
        super().__init__(
            f"user {user_id} is not a member of space(s) {unauthorized_space_ids}"
        )


class SpaceMembershipChecker:
    def __init__(self, jira_backend_url: str, internal_service_token: str, cache_ttl_seconds: float = 60.0):
        self._client = httpx.AsyncClient(
            base_url=jira_backend_url.rstrip("/"),
            headers={"X-Internal-Token": internal_service_token},
            timeout=5.0,
        )
        self._cache_ttl = cache_ttl_seconds
        # user_id -> (cached_at, the user's spaces). Memberships rarely change and /ask is
        # latency-sensitive, so cache briefly instead of calling jira-backend every time.
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    async def authorized_spaces(self, user_id: str, username: str) -> list[dict]:
        """The user's spaces as {id, key, name}. MCP tools take no space_ids, so they need the full list."""
        now = time.monotonic()
        cached = self._cache.get(user_id)
        if cached is not None and now - cached[0] < self._cache_ttl:
            return cached[1]
        # jira-backend needs BOTH X-User-Id and X-Username with the internal token, or it 401s.
        resp = await self._client.get(
            "/api/spaces", params={"userId": user_id},
            headers={"X-User-Id": user_id, "X-Username": username},
        )
        resp.raise_for_status()
        spaces = [
            {"id": space["id"], "key": space.get("key"), "name": space.get("name")}
            for space in resp.json()
        ]
        self._cache[user_id] = (now, spaces)
        return spaces

    async def _authorized_space_ids(self, user_id: str, username: str) -> set[int]:
        return {space["id"] for space in await self.authorized_spaces(user_id, username)}

    async def validate(
        self, user_id: Optional[str], username: Optional[str], requested_space_ids: list[int]
    ) -> None:
        """Raise SpaceMembershipError if user_id is set and any space isn't theirs. No-op (and no call)
        when user_id is None.
        """
        if not user_id:
            return
        authorized = await self._authorized_space_ids(user_id, username or "")
        unauthorized = [s for s in requested_space_ids if s not in authorized]
        if unauthorized:
            raise SpaceMembershipError(user_id, unauthorized)

    async def aclose(self) -> None:
        await self._client.aclose()


class NoopSpaceMembershipChecker:
    """Used when space_membership_check_enabled=false — always permits, never calls jira-backend."""

    async def validate(
        self, user_id: Optional[str], username: Optional[str], requested_space_ids: list[int]
    ) -> None:
        return None

    async def authorized_spaces(self, user_id: str, username: str) -> Optional[list[dict]]:
        """None = unknown. Callers that derive scope from membership (MCP) must fail closed on it."""
        return None

    async def aclose(self) -> None:
        return None


def build_space_membership_checker(settings):
    if not settings.space_membership_check_enabled:
        return NoopSpaceMembershipChecker()
    return SpaceMembershipChecker(
        settings.jira_backend_url,
        settings.internal_service_token,
        settings.space_membership_cache_ttl_seconds,
    )
