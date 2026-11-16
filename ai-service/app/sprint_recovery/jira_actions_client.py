"""Runs the 4 recovery-action types against jira-backend: link_dependency, change_priority,
move_out_of_sprint, add_comment.

`commit_one_action_node` never repeats an action already in `committed_actions`. That doesn't cover a
write that succeeded while its HTTP response was lost (the retry would post a second comment), so
each request carries an `Idempotency-Key` and jira-backend's IdempotencyFilter replays the original
response instead of running it again.

Uses the internal service token + X-User-Id/X-Username headers, like jira_commit_client.
"""
from __future__ import annotations

from typing import Optional

import httpx

from app.sprint_recovery.schemas import RecoveryAction


class JiraActionError(Exception):
    """An action call to jira-backend failed. Callers must not guess at partial success."""


class JiraActionsClient:
    def __init__(self, jira_backend_url: str, internal_service_token: str, timeout: float = 10.0):
        self._client = httpx.AsyncClient(
            base_url=jira_backend_url.rstrip("/"),
            headers={"X-Internal-Token": internal_service_token},
            timeout=timeout,
        )

    def _headers(self, user_id: str, username: str, idempotency_key: Optional[str] = None) -> dict:
        headers = {"X-User-Id": user_id, "X-Username": username} if user_id else {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def _resolve_issue_id(self, space_id: int, issue_key: str, user_id: str, username: str) -> int:
        resp = await self._client.get(
            f"/api/spaces/{space_id}/issues/{issue_key}", headers=self._headers(user_id, username),
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def issue_updated_at(
        self, space_id: int, issue_key: str, user_id: str, username: str,
    ) -> Optional[str]:
        """The issue's `updatedAt` from jira-backend (the write side, not the index), recorded after a
        write so reevaluate can wait for the index. Returns None on failure — that only means "don't
        wait", which beats failing a successful commit.
        """
        try:
            resp = await self._client.get(
                f"/api/spaces/{space_id}/issues/{issue_key}", headers=self._headers(user_id, username),
            )
            resp.raise_for_status()
            return resp.json().get("updatedAt")
        except (httpx.HTTPError, ValueError, KeyError):
            return None

    async def execute(
        self, space_id: int, action: RecoveryAction, user_id: str, username: str,
        idempotency_key: Optional[str] = None,
    ) -> str:
        """Run exactly one action and return a short summary. Raises JiraActionError on failure; the
        retry then repeats the same action, and `idempotency_key` makes jira-backend replay rather
        than re-run it.
        """
        try:
            if action.action_type == "link_dependency":
                return await self._link_dependency(space_id, action, user_id, username, idempotency_key)
            if action.action_type == "change_priority":
                return await self._change_priority(space_id, action, user_id, username, idempotency_key)
            if action.action_type == "move_out_of_sprint":
                return await self._move_out_of_sprint(space_id, action, user_id, username, idempotency_key)
            if action.action_type == "add_comment":
                return await self._add_comment(space_id, action, user_id, username, idempotency_key)
            raise JiraActionError(f"unknown action_type: {action.action_type}")
        except httpx.HTTPError as exc:
            raise JiraActionError(f"{action.action_type} on {action.target_issue_key} failed: {exc}") from exc

    async def _link_dependency(self, space_id, action: RecoveryAction, user_id, username, idempotency_key=None) -> str:
        if not action.depends_on_issue_key:
            raise JiraActionError("link_dependency requires depends_on_issue_key")
        issue_id = await self._resolve_issue_id(space_id, action.target_issue_key, user_id, username)
        resp = await self._client.post(
            f"/api/issues/{issue_id}/links",
            json={"relation": "is blocked by", "targetIssueKey": action.depends_on_issue_key},
            headers=self._headers(user_id, username, idempotency_key),
        )
        resp.raise_for_status()
        return f"{action.target_issue_key} is blocked by {action.depends_on_issue_key}"

    async def _change_priority(self, space_id, action: RecoveryAction, user_id, username, idempotency_key=None) -> str:
        if not action.new_priority:
            raise JiraActionError("change_priority requires new_priority")
        resp = await self._client.put(
            f"/api/spaces/{space_id}/issues/{action.target_issue_key}",
            json={"priority": action.new_priority},
            headers=self._headers(user_id, username, idempotency_key),
        )
        resp.raise_for_status()
        return f"{action.target_issue_key} priority -> {action.new_priority}"

    async def _move_out_of_sprint(self, space_id, action: RecoveryAction, user_id, username, idempotency_key=None) -> str:
        resp = await self._client.put(
            f"/api/spaces/{space_id}/issues/{action.target_issue_key}",
            json={"clearSprint": True},
            headers=self._headers(user_id, username, idempotency_key),
        )
        resp.raise_for_status()
        return f"{action.target_issue_key} moved out of sprint"

    async def _add_comment(self, space_id, action: RecoveryAction, user_id, username, idempotency_key=None) -> str:
        """jira-backend's comment create needs `authorId` in the body (unlike issue create/update, which
        use headers) — it 500'd without it. Use the caller's numeric id (same value as X-User-Id).
        """
        if not action.comment_body:
            raise JiraActionError("add_comment requires comment_body")
        if not user_id:
            raise JiraActionError("add_comment requires a real user_id to attribute authorship to")
        issue_id = await self._resolve_issue_id(space_id, action.target_issue_key, user_id, username)
        resp = await self._client.post(
            f"/api/issues/{issue_id}/comments",
            json={"content": action.comment_body, "authorId": int(user_id)},
            headers=self._headers(user_id, username, idempotency_key),
        )
        resp.raise_for_status()
        return f"comment added to {action.target_issue_key}"

    async def aclose(self) -> None:
        await self._client.aclose()
