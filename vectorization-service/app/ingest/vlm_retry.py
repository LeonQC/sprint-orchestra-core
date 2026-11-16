"""Re-run the VLM pass for attachments that were indexed while the VLM was failing.

When a VLM call raises, the parser keeps the OCR/Docling text so the document is searchable at once,
and the pipeline queues the attachment. This worker re-ingests queued attachments
once the VLM answers again. Re-ingestion is the normal attachment path, so it reuses everything that
makes a Kafka redelivery safe: deterministic chunk ids replace the OCR chunks, and VLM results that
did succeed come from the result cache instead of being paid for twice.

Each round treats the first due attachment as a probe: if its VLM calls still fail, the outage isn't
over and the rest wait for the next round, instead of every queued document burning an attempt.

Single-instance by design: `IngestPipeline.lock` serializes retries with the consumer in this
process. With several replicas, re-drive through the attachment topic instead, so partition order
does that job.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from app.config import Settings
from app.models import AttachmentIngestionMessage, VlmRetryItem
from app.observability import VLM_RETRIES_TOTAL

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 6 * 3600


class VlmRetryQueue(Protocol):
    async def queue_vlm_retry(
        self, *, attachment_id: int, issue_id: int, space_id: int | None, event_id: str,
        message: dict, failed_calls: int, last_error: str, retry_after_seconds: float,
    ) -> None: ...

    async def clear_vlm_retry(self, attachment_id: int) -> None: ...


class NoopVlmRetryQueue:
    """For pipelines built without a database (tests, evals) or with the retry disabled."""

    async def queue_vlm_retry(self, **_fields) -> None: ...

    async def clear_vlm_retry(self, attachment_id: int) -> None: ...


def backoff_seconds(settings: Settings, attempts: int) -> float:
    return min(settings.vlm_retry_interval_seconds * 2 ** attempts, _MAX_BACKOFF_SECONDS)


class VlmRetryWorker:
    def __init__(self, settings: Settings, pipeline, store):
        self._settings = settings
        self._pipeline = pipeline
        self._store = store
        self._task: asyncio.Task | None = None

    async def run_once(self) -> dict[str, int]:
        """One round over the due queue. Returns a count per outcome."""
        due = await self._store.due_vlm_retries(
            max_attempts=self._settings.vlm_retry_max_attempts,
            limit=self._settings.vlm_retry_batch_size,
        )
        outcomes: dict[str, int] = {}
        for item in due:
            outcome = await self._retry(item)
            VLM_RETRIES_TOTAL.labels(outcome).inc()
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            # The VLM (or the pipeline behind it) is still down; the rest of the queue waits.
            if outcome in ("still_failing", "error"):
                break
        if outcomes:
            logger.info("VLM retry round finished", extra={"outcomes": outcomes, "due": len(due)})
        return outcomes

    async def _retry(self, item: VlmRetryItem) -> str:
        async with self._pipeline.lock:
            # Re-read under the lock: a delete or re-upload handled since the round started removed
            # or replaced the row, and its content must not be overwritten with this older event.
            current = await self._store.get_vlm_retry(item.attachment_id)
            if current is None or current.event_id != item.event_id:
                return "superseded"
            await self._store.record_vlm_retry_attempt(
                item.attachment_id, backoff_seconds(self._settings, current.attempts + 1)
            )
            try:
                await self._pipeline.handle_attachment(
                    AttachmentIngestionMessage.model_validate(item.message)
                )
            except Exception:  # noqa: BLE001 - the row stays queued; the next round tries again
                logger.warning(
                    "VLM retry ingestion failed", exc_info=True,
                    extra={"attachment_id": item.attachment_id},
                )
                return "error"
            # The pipeline clears the row after a pass with no VLM failure, and re-queues it otherwise.
            still_queued = await self._store.get_vlm_retry(item.attachment_id)
            return "still_failing" if still_queued is not None else "recovered"

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._settings.vlm_retry_interval_seconds)
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001 - a DB blip must not end the worker for the process lifetime
                logger.warning("VLM retry round failed", exc_info=True)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="vlm-retry-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
