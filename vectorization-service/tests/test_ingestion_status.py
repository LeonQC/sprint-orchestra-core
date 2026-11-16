"""Per-document ingestion status tracking (app/ingest/status.py).

Hermetic: a recording fake stands in for `VectorStore.upsert_ingestion_status`, so these assert the
*state machine* — which stage each source visits, which terminal state each exit records, and that a
tracking failure cannot break ingestion.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.ingest.pipeline import IngestPipeline
from app.ingest.status import DbIngestionTracker, Stage, State
from app.models import (
    AttachmentIngestionMessage,
    CommentIngestionMessage,
    IssueIngestionMessage,
)


class RecordingStore:
    """Captures status writes and stands in for the parts of VectorStore the pipeline calls."""

    def __init__(self):
        self.writes: list[dict] = []
        self.chunks: dict = {}

    async def upsert_ingestion_status(self, **fields):
        self.writes.append(fields)

    async def replace_source(self, chunk_type, source_id, chunks, vectors, parents=()):
        self.chunks[(chunk_type, source_id)] = chunks

    async def get_source_chunks(self, chunk_type, source_id):
        return self.chunks.get((chunk_type, source_id), [])

    async def refresh_lexical_index(self, chunk_type, source_id, chunks):
        return None

    async def delete_source(self, chunk_type, source_id):
        self.chunks.pop((chunk_type, source_id), None)
        return 1

    async def delete_issue(self, issue_id):
        return 3

    def states(self) -> list[str]:
        return [w["state"] for w in self.writes]

    def stages(self) -> list[str]:
        return [w.get("stage") for w in self.writes if w["state"] == State.IN_PROGRESS]

    def final(self) -> dict:
        return self.writes[-1]


class FakeEmbedder:
    dim = 8

    async def embed(self, texts):
        return [[0.1] * self.dim for _ in texts]

    async def aclose(self):
        return None


def _pipeline(store, **overrides):
    settings = Settings(embedding_provider="fake", embedding_dim=8, **overrides)
    return IngestPipeline(
        settings, FakeEmbedder(), store, tracker=DbIngestionTracker(store)
    )


def _issue(**overrides) -> dict:
    return {
        "eventId": "e1", "eventType": "issue_upserted", "emittedAt": "2026-01-01T00:00:00Z",
        "issueId": 1, "issueKey": "ATC-1", "spaceId": 5,
        "title": "Checkout times out", "description": "The pool saturates under load.",
        **overrides,
    }


def _attachment(**overrides) -> dict:
    return {
        "eventId": "e2", "eventType": "attachment_uploaded", "emittedAt": "2026-01-01T00:00:00Z",
        "attachmentId": 77, "issueId": 1, "issueKey": "ATC-1", "spaceId": 5,
        "filename": "runbook.txt", "mimeType": "text/plain", "byteSize": 40,
        "bucket": "b", "storageKey": "k",
        **overrides,
    }


# --- happy path ---------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_issue_records_every_stage_then_indexed():
    store = RecordingStore()
    await _pipeline(store).handle_issue(IssueIngestionMessage.model_validate(_issue()))

    assert store.stages() == [Stage.PARSE, Stage.CHUNK, Stage.EMBED, Stage.STORE]
    final = store.final()
    assert final["state"] == State.INDEXED
    assert final["chunk_count"] == 1
    assert final["issue_key"] == "ATC-1"
    assert final["space_id"] == 5
    # Every stage that ran is timed, so "which step is slow" is answerable from the row alone.
    assert set(final["stage_ms"]) == {Stage.PARSE, Stage.CHUNK, Stage.EMBED, Stage.STORE}
    assert final["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_attachment_records_the_fetch_stage_issues_do_not_have():
    store = RecordingStore()
    pipeline = _pipeline(store)

    async def fake_fetch(settings, uri, bucket, key):
        return b"# Runbook\n\nDrain the pool before restarting."

    import app.ingest.pipeline as pipeline_module

    pipeline_module.fetch_attachment = fake_fetch
    await pipeline.handle_attachment(AttachmentIngestionMessage.model_validate(_attachment()))

    assert store.stages()[0] == Stage.FETCH
    assert store.final()["state"] == State.INDEXED
    assert store.final()["source_label"] == "runbook.txt"


# --- the distinctions the table exists to make ---------------------------------------------------


@pytest.mark.asyncio
async def test_blank_issue_records_empty_not_indexed():
    """A blank description legitimately produces zero chunks. That is not a failure, and it is not
    success either — conflating it with INDEXED is what makes "absent from chunks" uninterpretable."""
    store = RecordingStore()
    await _pipeline(store).handle_issue(
        IssueIngestionMessage.model_validate(_issue(title="", description=""))
    )
    final = store.final()
    assert final["state"] == State.EMPTY
    assert final["chunk_count"] == 0


@pytest.mark.asyncio
async def test_unfetchable_attachment_records_failed_at_the_fetch_stage():
    """`fetch_attachment` logs and returns None rather than raising, so before this the document
    vanished with no record. This is the single most valuable row in the table."""
    store = RecordingStore()
    pipeline = _pipeline(store)

    async def missing(settings, uri, bucket, key):
        return None

    import app.ingest.pipeline as pipeline_module

    pipeline_module.fetch_attachment = missing
    await pipeline.handle_attachment(AttachmentIngestionMessage.model_validate(_attachment()))

    final = store.final()
    assert final["state"] == State.FAILED
    assert final["stage"] == Stage.FETCH
    assert final["error_type"] == "FileNotFoundError"
    assert "not retrievable" in final["error_message"]


@pytest.mark.asyncio
async def test_oversized_attachment_records_skipped_with_the_reason():
    store = RecordingStore()
    pipeline = _pipeline(store, attachment_max_bytes=10)
    await pipeline.handle_attachment(
        AttachmentIngestionMessage.model_validate(_attachment(byteSize=999_999))
    )
    final = store.final()
    assert final["state"] == State.SKIPPED
    assert "exceeds" in final["error_message"]


@pytest.mark.asyncio
async def test_delete_event_records_deleted_with_the_row_count():
    store = RecordingStore()
    await _pipeline(store).handle_issue(
        IssueIngestionMessage.model_validate(_issue(eventType="issue_deleted"))
    )
    final = store.final()
    assert final["state"] == State.DELETED
    assert final["chunk_count"] == 3


@pytest.mark.asyncio
async def test_embedding_failure_records_the_stage_it_died_in_and_re_raises():
    """The stage matters: an embed failure is a model-server problem, a store failure is a database
    problem, and they are indistinguishable from `chunks` (no rows either way)."""

    class BrokenEmbedder(FakeEmbedder):
        async def embed(self, texts):
            raise RuntimeError("embedding provider unreachable")

    store = RecordingStore()
    settings = Settings(embedding_provider="fake", embedding_dim=8)
    pipeline = IngestPipeline(
        settings, BrokenEmbedder(), store, tracker=DbIngestionTracker(store)
    )

    with pytest.raises(RuntimeError):
        await pipeline.handle_issue(IssueIngestionMessage.model_validate(_issue()))

    final = store.final()
    assert final["state"] == State.FAILED
    assert final["stage"] == Stage.EMBED
    assert final["error_type"] == "RuntimeError"
    assert "unreachable" in final["error_message"]


@pytest.mark.asyncio
async def test_comment_is_tracked_under_its_own_source_id_not_its_issue():
    store = RecordingStore()
    await _pipeline(store).handle_comment(
        CommentIngestionMessage.model_validate(
            {
                "eventId": "e3", "eventType": "comment_upserted",
                "emittedAt": "2026-01-01T00:00:00Z", "commentId": 900, "issueId": 1,
                "issueKey": "ATC-1", "spaceId": 5, "content": "Raising the pool size fixed it.",
            }
        )
    )
    final = store.final()
    assert (final["source_type"], final["source_id"]) == ("comment", 900)
    assert final["issue_key"] == "ATC-1"
    assert final["state"] == State.INDEXED


# --- constraint 1: tracking must never break ingestion -------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_status_write_does_not_fail_the_document():
    """The whole table is an observability aid. Losing a status row is acceptable; losing the
    document because we could not record that it succeeded is not."""

    class BrokenStatusStore(RecordingStore):
        async def upsert_ingestion_status(self, **fields):
            raise RuntimeError("status table is gone")

    store = BrokenStatusStore()
    await _pipeline(store).handle_issue(IssueIngestionMessage.model_validate(_issue()))

    # The document still made it into the index despite every status write throwing.
    assert ("issue", 1) in store.chunks


@pytest.mark.asyncio
async def test_tracking_can_be_disabled_entirely():
    from app.ingest.status import NoopIngestionTracker, build_ingestion_tracker

    store = RecordingStore()
    tracker = build_ingestion_tracker(Settings(ingestion_status_enabled=False), store)
    assert isinstance(tracker, NoopIngestionTracker)

    pipeline = IngestPipeline(
        Settings(embedding_provider="fake", embedding_dim=8, ingestion_status_enabled=False),
        FakeEmbedder(), store, tracker=tracker,
    )
    await pipeline.handle_issue(IssueIngestionMessage.model_validate(_issue()))
    assert store.writes == []
    assert ("issue", 1) in store.chunks


# --- the HTTP surface: /ingestion/health + /ingestion/status ----------------------------------------


class _StatusStore:
    """Minimal store for exercising the three /ingestion/* route handlers directly."""

    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    async def query_ingestion_status(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


def _status_result(total, by_state, rows=()):
    from app.models import IngestionStatusQueryResult

    return IngestionStatusQueryResult(total_count=total, counts_by_state=dict(by_state), rows=list(rows))


@pytest.mark.asyncio
async def test_ingestion_health_reports_needs_attention_as_failed_plus_stuck():
    from types import SimpleNamespace

    from app.api.routes import ingestion_health

    # First call (overall) sees 2 failed; second call (stuck) sees 1.
    class TwoShotStore:
        def __init__(self):
            self.n = 0

        async def query_ingestion_status(self, **kw):
            self.n += 1
            if kw.get("stale_after_seconds"):
                return _status_result(1, {"in_progress": 1})
            return _status_result(50, {"indexed": 45, "empty": 2, "failed": 2, "in_progress": 1})

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=TwoShotStore())))
    body = await ingestion_health(request)

    assert body["total_documents"] == 50
    # empty and skipped are explicitly NOT counted as needing attention.
    assert body["needs_attention"] == 2 + 1


@pytest.mark.asyncio
async def test_ingestion_status_route_forwards_every_filter():
    from types import SimpleNamespace

    from app.api.routes import IngestionStatusRequest, ingestion_status

    store = _StatusStore(_status_result(0, {}))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(store=store)))
    req = IngestionStatusRequest(
        space_ids=[5], source_types=["attachment"], states=["failed"],
        issue_keys=["ATC-46"], filename_contains="invoice", stale_after_seconds=600, limit=10,
    )

    await ingestion_status(req, request)

    call = store.calls[0]
    assert call["space_ids"] == [5]
    assert call["source_types"] == ["attachment"]
    assert call["states"] == ["failed"]
    assert call["issue_keys"] == ["ATC-46"]
    assert call["source_label_contains"] == "invoice"
    assert call["stale_after_seconds"] == 600
    assert call["limit"] == 10


@pytest.mark.asyncio
async def test_corrupt_attachment_records_failed_at_parse_not_empty():
    """A file the parser genuinely could not read (Docling raised, VLM recovered nothing) must be
    `failed at parse`, not `empty` — `empty` is reserved for a file that parsed fine and simply had
    no text. Before parse-error tracking these were the same state."""
    import app.ingest.pipeline as pipeline_module
    from app.ingest import docling_parser

    store = RecordingStore()
    pipeline = _pipeline(store)

    async def fetch(settings, uri, bucket, key):
        return b"%PDF-1.4 not really a pdf"

    async def broken_parse(data, filename, settings, vlm=None):
        # Mimic the real contract: swallow the error, return nothing, but leave a breadcrumb.
        docling_parser._parse_errors.set([])
        docling_parser._record_parse_error("parse (pdf)", ValueError("EOF marker not found"))
        return []

    pipeline_module.fetch_attachment = fetch
    pipeline_module.parse_attachment_sections = broken_parse
    try:
        await pipeline.handle_attachment(
            AttachmentIngestionMessage.model_validate(_attachment(filename="report.pdf"))
        )
    finally:
        pipeline_module.parse_attachment_sections = docling_parser.parse_attachment_sections

    final = store.final()
    assert final["state"] == State.FAILED
    assert final["stage"] == Stage.PARSE
    assert "EOF marker not found" in final["error_message"]


@pytest.mark.asyncio
async def test_genuinely_empty_attachment_still_records_empty_not_failed():
    """The other half of the distinction: parse succeeded, found nothing, no error recorded."""
    import app.ingest.pipeline as pipeline_module
    from app.ingest import docling_parser

    store = RecordingStore()
    pipeline = _pipeline(store)

    async def fetch(settings, uri, bucket, key):
        return b"\x89PNG blank"

    async def empty_parse(data, filename, settings, vlm=None):
        docling_parser._parse_errors.set([])  # no error recorded
        return []

    pipeline_module.fetch_attachment = fetch
    pipeline_module.parse_attachment_sections = empty_parse
    try:
        await pipeline.handle_attachment(
            AttachmentIngestionMessage.model_validate(_attachment(filename="blank.png"))
        )
    finally:
        pipeline_module.parse_attachment_sections = docling_parser.parse_attachment_sections

    assert store.final()["state"] == State.EMPTY
