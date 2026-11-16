"""Publishes an approved AI epic plan to jira-backend.

The durable graph calls one small operation per side effect (create/reuse sprints, create the epic,
create each child issue, create each dependency link), so progress is checkpointed between them.
Uses the internal service token + X-User-Id/X-Username headers, like app/auth/space_membership.py.

Creates a real epic issue (`issueType: "epic"`) and links children via `parentId`. (An early version
wrongly assumed jira-backend had no epics and used a label — that limit was only on what the LLM may
propose.)
"""
from __future__ import annotations

from typing import Optional

import httpx

from app.planning.schemas import EpicDraft, IssueDraft


class JiraCommitError(Exception):
    """A Jira publishing call failed or returned an unusable response.

    The graph does not record the step as completed and exposes an explicit resume path. Every
    mutation also carries a stable workflow/action ``Idempotency-Key``; jira-backend can therefore
    replay the original 2xx response if that response was lost after the remote transaction committed.
    """


class JiraCommitClient:
    def __init__(self, jira_backend_url: str, internal_service_token: str, timeout: float = 10.0):
        self._client = httpx.AsyncClient(
            base_url=jira_backend_url.rstrip("/"),
            headers={"X-Internal-Token": internal_service_token},
            timeout=timeout,
        )

    def _headers(
        self, user_id: str, username: str, idempotency_key: Optional[str] = None,
    ) -> dict:
        # Identity headers only when real ones were forwarded; a placeholder would misattribute the
        # reporter.
        headers = {"X-User-Id": user_id, "X-Username": username} if user_id else {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def validate_future_sprint(
        self, space_id: int, sprint_id: int, user_id: str, username: str,
    ) -> None:
        """Fail before publishing if an existing destination is stale or no longer selectable."""
        try:
            resp = await self._client.get(
                f"/api/spaces/{space_id}/sprints/{sprint_id}",
                headers=self._headers(user_id, username),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JiraCommitError(
                f"existing sprint {sprint_id} is unavailable in space {space_id}: {exc}"
            ) from exc
        try:
            data = resp.json()
        except ValueError as exc:
            raise JiraCommitError(
                f"existing sprint {sprint_id} returned an unreadable response"
            ) from exc
        if not isinstance(data, dict):
            raise JiraCommitError(f"existing sprint {sprint_id} returned an unusable response")
        if data.get("status") != "future":
            raise JiraCommitError(
                f"existing sprint {sprint_id} must still be a future sprint "
                f"(current status={data.get('status')!r})"
            )

    async def create_epic_issue(
        self,
        space_id: int,
        epic: EpicDraft,
        user_id: str,
        username: str,
        idempotency_key: Optional[str] = None,
    ) -> tuple[int, str]:
        """Create the epic issue. Returns `(issue_id, issue_key)` — children link via the internal id
        (`parentId`), not the key.
        """
        body = {"title": epic.title, "description": epic.description, "issueType": "epic"}
        try:
            resp = await self._client.post(
                f"/api/spaces/{space_id}/issues",
                json=body,
                headers=self._headers(user_id, username, idempotency_key),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JiraCommitError(f"failed creating epic issue '{epic.title}': {exc}") from exc
        data = resp.json()
        issue_id, issue_key = data.get("id"), data.get("issueKey")
        if not issue_id or not issue_key:
            raise JiraCommitError(f"jira-backend response for epic '{epic.title}' missing id/issueKey: {data}")
        return issue_id, issue_key

    async def create_sprint(
        self,
        space_id: int,
        name: str,
        user_id: str,
        username: str,
        idempotency_key: Optional[str] = None,
    ) -> int:
        try:
            resp = await self._client.post(
                f"/api/spaces/{space_id}/sprints",
                json={"name": name},
                headers=self._headers(user_id, username, idempotency_key),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JiraCommitError(f"failed creating sprint '{name}': {exc}") from exc
        sprint_id = resp.json().get("id")
        if not sprint_id:
            raise JiraCommitError(f"jira-backend response for sprint '{name}' missing id")
        return int(sprint_id)

    async def create_issue(
        self,
        space_id: int,
        issue: IssueDraft,
        parent_id: int,
        sprint_id: Optional[int],
        user_id: str,
        username: str,
        idempotency_key: Optional[str] = None,
    ) -> tuple[int, str]:
        """Create one issue under `parent_id`; returns its id and key (ids are kept for dependency
        links later). Raises JiraCommitError on any failure — don't guess at partial success.
        """
        description = issue.description.strip()
        # Skip if the description already has an estimate-reasoning section (house format), or it'd
        # appear twice.
        if issue.estimate_rationale and issue.estimate_rationale.strip() not in description:
            description = f"{description}\n\nEstimate rationale: {issue.estimate_rationale}".strip()
        if issue.estimate_reference:
            # Also publish the verified keys: the prose rationale isn't checked, this list is, and
            # Jira is where the reviewer looks a week later.
            description = (
                f"{description}\n\nSized against: {', '.join(issue.estimate_reference)}".strip()
            )
        body = {
            "title": issue.title,
            "description": description,
            "issueType": issue.issue_type,
            "storyPoints": issue.estimate_story_points,
            "labels": issue.labels,
            "parentId": parent_id,
            "sprintId": sprint_id,
        }
        try:
            resp = await self._client.post(
                f"/api/spaces/{space_id}/issues",
                json=body,
                headers=self._headers(user_id, username, idempotency_key),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JiraCommitError(f"failed creating issue '{issue.title}': {exc}") from exc
        data = resp.json()
        issue_id, issue_key = data.get("id"), data.get("issueKey")
        if not issue_id or not issue_key:
            raise JiraCommitError(f"jira-backend response for '{issue.title}' missing id/issueKey: {data}")
        return int(issue_id), issue_key

    async def create_issue_link(
        self,
        source_issue_id: int,
        target_issue_key: str,
        user_id: str,
        username: str,
        idempotency_key: Optional[str] = None,
    ) -> int:
        try:
            resp = await self._client.post(
                f"/api/issues/{source_issue_id}/links",
                json={"relation": "is blocked by", "targetIssueKey": target_issue_key},
                headers=self._headers(user_id, username, idempotency_key),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise JiraCommitError(
                f"failed linking issue id {source_issue_id} to '{target_issue_key}': {exc}"
            ) from exc
        link_id = resp.json().get("id")
        if not link_id:
            raise JiraCommitError(
                f"jira-backend link response for issue id {source_issue_id} missing id"
            )
        return int(link_id)

    async def aclose(self) -> None:
        await self._client.aclose()
