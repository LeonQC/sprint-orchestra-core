"""aiokafka consumer driving the ingestion pipeline, as a background task for the app's lifetime.

Offsets are committed after a message is handled (at-least-once): if handling fails (e.g. the
embedding API is down) the message is redelivered. Every handler does a keyed upsert/delete, so
reprocessing is harmless. Poison messages (e.g. malformed JSON) are logged and skipped; production
would use a dead-letter topic.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from aiokafka import AIOKafkaConsumer, TopicPartition
from pydantic import ValidationError

from app.config import Settings
from app.ingest.pipeline import IngestPipeline
from app.models import (
    AttachmentIngestionMessage,
    CommentIngestionMessage,
    IssueIngestionMessage,
    SprintIngestionMessage,
)

logger = logging.getLogger(__name__)

# Structural failures that reprocessing can never fix -> skip (poison), don't rewind. pydantic's
# ValidationError is NOT a ValueError subclass in v2, so it must be listed explicitly or a malformed
# payload would be treated as transient and retried forever.
_POISON_ERRORS = (KeyError, TypeError, ValueError, ValidationError)

# On a transient failure, wait this long before rewinding + re-consuming the same message.
_RETRY_BACKOFF_SECONDS = 2.0


@dataclass
class ConsumerStats:
    """Lightweight observability surface exposed at /stats and /healthz."""

    running: bool = False
    processed: int = 0
    skipped_poison: int = 0
    failed: int = 0
    last_error: Optional[str] = None
    topics: List[str] = field(default_factory=list)


class IngestionConsumer:
    def __init__(self, settings: Settings, pipeline: IngestPipeline):
        self._settings = settings
        self._pipeline = pipeline
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._task: Optional[asyncio.Task] = None
        self.stats = ConsumerStats()

    async def start(self) -> None:
        topics = [self._settings.kafka_content_topic]
        if self._settings.kafka_consume_attachments:
            topics.append(self._settings.kafka_attachment_topic)
        self.stats.topics = topics

        self._consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self._settings.kafka_bootstrap_servers,
            group_id=self._settings.kafka_group_id,
            enable_auto_commit=False,
            auto_offset_reset=self._settings.kafka_auto_offset_reset,
            value_deserializer=lambda b: b,  # keep raw bytes; we json.loads ourselves for error control
        )
        await self._consumer.start()
        self.stats.running = True
        self._task = asyncio.create_task(self._run(), name="ingestion-consumer")
        logger.info("consumer started", extra={"topics": topics, "group": self._settings.kafka_group_id})

    async def _run(self) -> None:
        assert self._consumer is not None
        try:
            async for record in self._consumer:
                handled = await self._process(record)
                if handled:
                    await self._consumer.commit()
                else:
                    # Transient failure: seek back and retry after a backoff, instead of committing
                    # (losing the message) or crashing the loop.
                    tp = TopicPartition(record.topic, record.partition)
                    self._consumer.seek(tp, record.offset)
                    await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        except asyncio.CancelledError:  # graceful shutdown
            raise
        except Exception as exc:  # noqa: BLE001 - never let an unexpected error silently kill ingestion
            self.stats.running = False
            self.stats.last_error = str(exc)
            logger.exception("consumer loop crashed")

    async def _process(self, record) -> bool:
        """Return True to commit (success or poison-skip); False to rewind and retry (transient)."""
        try:
            payload = json.loads(record.value)
            event_type = payload.get("eventType", "")
        except (ValueError, AttributeError) as exc:
            self.stats.skipped_poison += 1
            logger.warning("skipping unparseable message", extra={"error": str(exc), "offset": record.offset})
            return True

        try:
            async with self._pipeline.lock:
                dispatched = await self._dispatch(event_type, payload)
            if dispatched:
                self.stats.processed += 1
            else:
                self.stats.skipped_poison += 1  # unknown event type
            return True
        except _POISON_ERRORS as exc:
            # Structural problem with the payload — retrying won't help. Skip so it doesn't wedge us.
            self.stats.skipped_poison += 1
            logger.warning("skipping malformed event", extra={"event_type": event_type, "error": str(exc)})
            return True
        except Exception as exc:  # noqa: BLE001 - transient (embedding API, DB down); rewind + retry
            self.stats.failed += 1
            self.stats.last_error = str(exc)
            logger.exception("event handling failed; will retry", extra={"event_type": event_type})
            return False

    async def _dispatch(self, event_type: str, payload: dict) -> bool:
        """Route by event type. Returns False for an unknown type (counted as poison by the caller)."""
        # Drain legacy metadata-only messages already in Kafka during the rollout. They no longer
        # write anything: history is read directly from jira-backend's authoritative database.
        if event_type == "issue_history_added":
            return True
        elif event_type.startswith("issue_"):
            await self._pipeline.handle_issue(IssueIngestionMessage.model_validate(payload))
        elif event_type.startswith("comment_"):
            await self._pipeline.handle_comment(CommentIngestionMessage.model_validate(payload))
        elif event_type.startswith("attachment_"):
            await self._pipeline.handle_attachment(AttachmentIngestionMessage.model_validate(payload))
        elif event_type.startswith("sprint_"):
            await self._pipeline.handle_sprint(SprintIngestionMessage.model_validate(payload))
        else:
            logger.warning("unknown event type", extra={"event_type": event_type})
            return False
        return True

    async def stop(self) -> None:
        self.stats.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._consumer:
            await self._consumer.stop()
        logger.info("consumer stopped")
