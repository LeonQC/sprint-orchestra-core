"""Per-document ingestion status: where each source got to, and why it stopped.

Without it, "never arrived", "fetch failed", "parsed to nothing" and "embedding provider down" all
look the same (no rows in `chunks`), and /stats only says how many failed.

Rules:
1. Tracking never breaks ingestion: every write goes through `_safe`, which logs and swallows. (It's
   observability, unlike a permission check, which must fail closed.)
2. A crash leaves evidence: each stage is written as it starts, so a worker killed mid-parse leaves
   `in_progress / parse`.
3. Empty isn't failed: a blank description or an unreadable image legitimately gives zero chunks,
   and gets its own state.

`NoopIngestionTracker` lets tests and evals build an IngestPipeline without a database.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from app.observability import get_request_id

logger = logging.getLogger(__name__)


class Stage:
    """Pipeline stages, in order; stored as-is in `ingestion_status.stage`. Only attachments have
    fetch and parse, so a row stuck at `fetch` must be an attachment.

      fetch — download bytes from object storage (the message only has an s3:// locator).
      parse — bytes to text (Docling, EasyOCR, or the VLM). Failure = corrupt/unsupported file.
      chunk — split into ~800-token chunks (+ optional contextual sentence per chunk).
      embed — call the embedding model. Failure = model server or gateway unreachable.
      store — write chunks, vectors and metadata to pgvector.

    For issues/comments parse and chunk are near-instant; for a 15-page scanned PDF, parse took 40s
    and chunk 150s.
    """

    FETCH = "fetch"
    PARSE = "parse"
    CHUNK = "chunk"
    EMBED = "embed"
    STORE = "store"


class State:
    IN_PROGRESS = "in_progress"
    INDEXED = "indexed"
    EMPTY = "empty"
    DELETED = "deleted"
    SKIPPED = "skipped"
    FAILED = "failed"


class IngestionTracker(Protocol):
    def begin(self, **identity) -> "DocumentTracker": ...


class DocumentTracker:
    """Records one document's progress (from `IngestionTracker.begin`). Not a context manager: the
    handlers have several normal early returns (too large, unfetchable, deleted), each needing its
    own terminal state, so explicit calls keep every exit visible.
    """

    def __init__(self, store, identity: dict):
        self._store = store
        self._identity = identity
        self._started = time.perf_counter()
        self._stage_started = self._started
        self._stage: Optional[str] = None
        self._stage_ms: dict[str, int] = {}

    async def stage(self, stage: str) -> None:
        """Enter a stage, closing the timing on the previous one."""
        self._close_stage()
        self._stage = stage
        self._stage_started = time.perf_counter()
        await self._write(state=State.IN_PROGRESS, stage=stage)

    def _close_stage(self) -> None:
        if self._stage is not None:
            elapsed = int((time.perf_counter() - self._stage_started) * 1000)
            # Add up repeat visits (a multi-section attachment parses once per section).
            self._stage_ms[self._stage] = self._stage_ms.get(self._stage, 0) + elapsed

    async def indexed(self, chunk_count: int) -> None:
        """Terminal success. ``chunk_count == 0`` records EMPTY, not INDEXED — see module docstring."""
        await self._finish(
            State.INDEXED if chunk_count else State.EMPTY, chunk_count=chunk_count
        )

    async def deleted(self, removed: int) -> None:
        await self._finish(State.DELETED, chunk_count=removed)

    async def skipped(self, reason: str) -> None:
        await self._finish(State.SKIPPED, error_message=reason)

    async def failed(self, exc: BaseException) -> None:
        await self._finish(
            State.FAILED, error_type=type(exc).__name__, error_message=str(exc)[:2000]
        )

    async def _finish(self, state: str, **fields) -> None:
        # Keep the last stage on the terminal write: on `failed` it says where it died (setting None
        # here once threw that away; test_embedding_failure_records_the_stage_it_died_in_and_re_raises).
        last_stage = self._stage
        self._close_stage()
        self._stage = None
        await self._write(
            state=state,
            stage=last_stage,
            duration_ms=int((time.perf_counter() - self._started) * 1000),
            **fields,
        )

    async def _write(self, **fields) -> None:
        await _safe(
            self._store.upsert_ingestion_status(
                **self._identity,
                request_id=get_request_id(),
                stage_ms=dict(self._stage_ms),
                **fields,
            )
        )


class DbIngestionTracker:
    def __init__(self, store):
        self._store = store

    def begin(self, **identity) -> DocumentTracker:
        return DocumentTracker(self._store, identity)


class _NoopDocumentTracker:
    async def stage(self, stage: str) -> None: ...
    async def indexed(self, chunk_count: int) -> None: ...
    async def deleted(self, removed: int) -> None: ...
    async def skipped(self, reason: str) -> None: ...
    async def failed(self, exc: BaseException) -> None: ...


class NoopIngestionTracker:
    """Used when status tracking is disabled, and by tests/eval that build a pipeline without a DB."""

    def begin(self, **identity) -> _NoopDocumentTracker:
        return _NoopDocumentTracker()


async def _safe(coro) -> None:
    """Await a status write, swallowing any failure. See constraint 1 in the module docstring."""
    try:
        await coro
    except Exception:  # noqa: BLE001 - status tracking must never fail a document
        logger.warning("ingestion status write failed; ingestion continues", exc_info=True)


def build_ingestion_tracker(settings, store) -> IngestionTracker:
    return DbIngestionTracker(store) if settings.ingestion_status_enabled else NoopIngestionTracker()
