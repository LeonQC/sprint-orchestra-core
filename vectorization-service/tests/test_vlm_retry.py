"""VLM outage handling: index with the OCR text now, queue the attachment, re-run the VLM pass later."""
import asyncio
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.ingest import docling_parser
from app.ingest import pipeline as pipeline_module
from app.ingest.docling_parser import (
    ParsedAttachmentSection, _maybe_escalate_to_vlm, _OcrResult, pop_vlm_failures,
)
from app.ingest.embedder import FakeEmbedder
from app.ingest.pipeline import IngestPipeline
from app.ingest.vlm_retry import VlmRetryWorker, backoff_seconds
from app.models import AttachmentIngestionMessage, VlmRetryItem


class _FailingVlm:
    async def describe(self, image_data, mime_type):
        raise ConnectionError("gateway unreachable")


async def test_a_failed_vlm_call_keeps_the_ocr_text_and_is_recorded():
    pop_vlm_failures()
    ocr = _OcrResult(text="blurry OCR text", dropped_count=3)

    text = await _maybe_escalate_to_vlm(b"img", "image/png", ocr, Settings(vlm_ocr_enabled=True), _FailingVlm())

    assert text == "blurry OCR text"
    assert pop_vlm_failures() == ["ConnectionError: gateway unreachable"]
    assert pop_vlm_failures() == []


def _attachment_msg(event_id="e1", event_type="attachment_uploaded") -> AttachmentIngestionMessage:
    return AttachmentIngestionMessage(
        eventId=event_id, eventType=event_type, emittedAt=datetime(2026, 10, 28, tzinfo=timezone.utc),
        attachmentId=99, issueId=42, issueKey="PROJ-42", spaceId=7, filename="scan.png",
        storageUri="s3://bucket/scan.png",
    )


class _QueueStore:
    """Chunk store plus the vlm_retry_queue methods, in memory."""

    def __init__(self):
        self.rows = {}
        self.queue: dict[int, dict] = {}

    async def replace_source(self, chunk_type, source_id, chunks, vectors, parents=()):
        self.rows = {k: c for k, c in self.rows.items() if c.source_id != source_id}
        self.rows.update({c.id: c for c in chunks})

    async def get_source_chunks(self, chunk_type, source_id):
        return sorted(
            (c for c in self.rows.values() if c.source_id == source_id), key=lambda c: c.chunk_index
        )

    async def refresh_lexical_index(self, chunk_type, source_id, chunks):
        return None

    async def delete_source(self, chunk_type, source_id) -> int:
        before = len(self.rows)
        self.rows = {k: c for k, c in self.rows.items() if c.source_id != source_id}
        return before - len(self.rows)

    async def queue_vlm_retry(self, *, attachment_id, event_id, message, retry_after_seconds, **_):
        existing = self.queue.get(attachment_id)
        same_event = existing is not None and existing["event_id"] == event_id
        self.queue[attachment_id] = {
            "event_id": event_id, "message": message,
            "attempts": existing["attempts"] if same_event else 0,
        }

    async def clear_vlm_retry(self, attachment_id):
        self.queue.pop(attachment_id, None)

    async def get_vlm_retry(self, attachment_id):
        row = self.queue.get(attachment_id)
        return VlmRetryItem(attachment_id, row["event_id"], row["message"], row["attempts"]) if row else None

    async def due_vlm_retries(self, *, max_attempts, limit):
        return [
            VlmRetryItem(aid, r["event_id"], r["message"], r["attempts"])
            for aid, r in self.queue.items() if r["attempts"] < max_attempts
        ][:limit]

    async def record_vlm_retry_attempt(self, attachment_id, next_attempt_after_seconds):
        self.queue[attachment_id]["attempts"] += 1


@pytest.fixture
def vlm_down(monkeypatch):
    """Parse result switchable between "VLM failed, OCR kept" and "VLM answered"."""
    state = {"down": True}

    async def fake_fetch(*_args):
        return b"png-bytes"

    async def fake_parse(data, filename, settings, describer):
        if state["down"]:
            docling_parser._record_vlm_failure(ConnectionError("gateway unreachable"))
            return [ParsedAttachmentSection("blurry OCR text")]
        return [ParsedAttachmentSection("faithful VLM transcription")]

    monkeypatch.setattr(pipeline_module, "fetch_attachment", fake_fetch)
    # The inner function, so the real entry point still resets the per-parse failure list.
    monkeypatch.setattr(docling_parser, "_parse_attachment_sections", fake_parse)
    return state


def _pipeline(store) -> IngestPipeline:
    settings = Settings(embedding_provider="fake", embedding_dim=8)
    return IngestPipeline(settings, FakeEmbedder(8), store, vlm_retry_queue=store)


def _stored_text(store) -> str:
    return " ".join(c.content for c in store.rows.values())


async def test_attachment_is_indexed_with_ocr_text_and_queued_when_the_vlm_fails(vlm_down):
    store = _QueueStore()

    await _pipeline(store).handle_attachment(_attachment_msg())

    assert "blurry OCR text" in _stored_text(store)
    assert store.queue[99]["event_id"] == "e1"
    # The queued message round-trips into the same event the consumer handled.
    assert AttachmentIngestionMessage.model_validate(store.queue[99]["message"]) == _attachment_msg()


async def test_a_clean_pass_and_a_delete_both_clear_the_queue(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())

    vlm_down["down"] = False
    await pipe.handle_attachment(_attachment_msg())
    assert 99 not in store.queue

    vlm_down["down"] = True
    await pipe.handle_attachment(_attachment_msg())
    await pipe.handle_attachment(_attachment_msg(event_id="e2", event_type="attachment_deleted"))
    assert 99 not in store.queue


def _worker(store, pipe, **overrides) -> VlmRetryWorker:
    settings = Settings(embedding_provider="fake", vlm_retry_max_attempts=3, **overrides)
    return VlmRetryWorker(settings, pipe, store)


async def test_retry_replaces_the_ocr_chunks_once_the_vlm_is_back(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())

    vlm_down["down"] = False
    outcomes = await _worker(store, pipe).run_once()

    assert outcomes == {"recovered": 1}
    assert "faithful VLM transcription" in _stored_text(store)
    assert "blurry OCR text" not in _stored_text(store)
    assert store.queue == {}


async def test_a_round_stops_at_the_first_attachment_that_still_fails(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())
    second = _attachment_msg().model_copy(update={"attachment_id": 100, "event_id": "e9"})
    await pipe.handle_attachment(second)

    outcomes = await _worker(store, pipe).run_once()

    # One probe, not one wasted attempt per queued document.
    assert outcomes == {"still_failing": 1}
    assert [store.queue[99]["attempts"], store.queue[100]["attempts"]] == [1, 0]


async def test_gives_up_after_max_attempts(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())
    worker = _worker(store, pipe)

    for _ in range(5):
        await worker.run_once()

    assert store.queue[99]["attempts"] == 3
    assert await worker.run_once() == {}


async def test_a_retry_queued_for_an_older_event_is_dropped(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())
    stale = await store.get_vlm_retry(99)
    # A re-upload was handled after the round read the queue.
    vlm_down["down"] = False
    await pipe.handle_attachment(_attachment_msg(event_id="e2"))
    store.queue[99] = {"event_id": "e2", "message": stale.message, "attempts": 0}

    assert await _worker(store, pipe)._retry(stale) == "superseded"


async def test_retry_waits_for_an_event_the_consumer_is_handling(vlm_down):
    store = _QueueStore()
    pipe = _pipeline(store)
    await pipe.handle_attachment(_attachment_msg())
    item = await store.get_vlm_retry(99)

    async with pipe.lock:
        retry = asyncio.create_task(_worker(store, pipe)._retry(item))
        await asyncio.sleep(0)
        assert not retry.done()
        # The consumer deletes the attachment while the retry is blocked.
        await pipe.handle_attachment(_attachment_msg(event_id="e2", event_type="attachment_deleted"))

    assert await retry == "superseded"
    assert store.rows == {}


def test_backoff_doubles_and_is_capped():
    settings = Settings(embedding_provider="fake", vlm_retry_interval_seconds=600)
    assert [backoff_seconds(settings, n) for n in (1, 2, 3)] == [1200, 2400, 4800]
    assert backoff_seconds(settings, 20) == 6 * 3600
