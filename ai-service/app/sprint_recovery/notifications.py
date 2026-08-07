"""Notify someone when a workflow escalates — before, it was just a badge in a modal nobody reopened.

Always: a WARNING log line and a Prometheus counter. Optional: a webhook
(`AI_SPRINT_RECOVERY_ESCALATION_WEBHOOK_URL`, off by default). Called from the API layer
after seeing status "escalated", not from inside the graph node: the node has no settings or
compiled graph, and a side effect inside a node could repeat if the node re-runs.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.observability import SPRINT_RECOVERY_ESCALATIONS_TOTAL

logger = logging.getLogger(__name__)


async def notify_escalation(
    webhook_url: Optional[str], thread_id: str, space_id: int, sprint_id: int, sprint_name: str, summary: str,
) -> None:
    SPRINT_RECOVERY_ESCALATIONS_TOTAL.labels(str(space_id)).inc()
    logger.warning(
        "sprint recovery escalated — automated remediation exhausted, human intervention needed",
        extra={
            "thread_id": thread_id, "space_id": space_id, "sprint_id": sprint_id,
            "sprint_name": sprint_name, "summary": summary,
        },
    )
    if not webhook_url:
        return
    # "text" is a real Slack incoming-webhook's expected top-level field — pointing this at a real
    # Slack webhook URL works with zero payload translation, not just a hypothetical shape.
    payload = {
        "text": f"Sprint recovery escalated for \"{sprint_name}\" (thread {thread_id[:8]}): {summary}",
        "thread_id": thread_id, "space_id": space_id, "sprint_id": sprint_id, "sprint_name": sprint_name,
        "summary": summary,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a failed notification must never break the actual status transition
        logger.warning(
            "sprint recovery escalation webhook call failed", extra={"thread_id": thread_id, "error": str(exc)},
        )
