"""Ingestion: Kafka message -> chunks -> embeddings -> vector store.

One entry point per event kind; each chunks one source, embeds it, and does a keyed replace/delete.
Provider- and transport-agnostic, so it's unit-tested with a fake embedder and store
(tests/test_pipeline_idempotency.py).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import List

from app.config import Settings
from app.db.vector_store import VectorStore
from app.ingest.attachment_fetcher import fetch_attachment
from app.ingest.chunker import (
    PagedChunk, ParentSection, chunk_pages_with_parents, chunk_text, chunk_text_with_parents,
)
from app.ingest.contextualizer import ContextGenerator, NoopContextGenerator
from app.ingest.docling_parser import (
    parse_attachment_sections, pop_parse_errors, pop_vlm_failures,
)
from app.ingest.embedder import Embedder
from app.ingest.html_text import html_to_text
from app.ingest.status import IngestionTracker, NoopIngestionTracker, Stage
from app.ingest.vlm_describer import NoopVlmImageDescriber, VlmImageDescriber
from app.ingest.vlm_retry import NoopVlmRetryQueue, VlmRetryQueue
from app.observability import INGEST_UNCHANGED_TOTAL, VLM_DEGRADED_ATTACHMENTS_TOTAL
from app.models import (
    AttachmentIngestionMessage,
    Chunk,
    ChunkParent,
    CommentIngestionMessage,
    IssueIngestionMessage,
    SprintIngestionMessage,
)

logger = logging.getLogger(__name__)


class IngestPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        context_generator: ContextGenerator | None = None,
        vlm_describer: VlmImageDescriber | None = None,
        tracker: IngestionTracker | None = None,
        vlm_retry_queue: VlmRetryQueue | None = None,
    ):
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._context_generator = context_generator or NoopContextGenerator()
        self._vlm_describer = vlm_describer or NoopVlmImageDescriber()
        # Defaults to a no-op for the same reason context_generator/vlm_describer do: a pipeline
        # built without a database (tests) must still run.
        self._tracker = tracker or NoopIngestionTracker()
        self._vlm_retry_queue = vlm_retry_queue or NoopVlmRetryQueue()
        # Held around every handled event by the consumer and the VLM retry worker, so a retry can't
        # interleave with a delete or re-upload of the same attachment.
        self.lock = asyncio.Lock()

    async def _chunks(
        self, text: str, *, chunk_type, issue_id, issue_key, space_id, source_id, key_prefix,
        source_label: str | None = None, page_number: int | None = None,
        provenance: dict | None = None, chunk_index_offset: int = 0, force_numbered_ids: bool = False,
        paged: List[PagedChunk] | None = None,
    ) -> List[Chunk]:
        # `paged`: chunks cut with their parent sections. With no `page_number` it is a PDF chunked as
        # one document, so each chunk names its own pages.
        if paged is not None:
            pieces = [p.text for p in paged]
        else:
            pieces = chunk_text(
                text,
                self._settings.chunk_size_tokens,
                self._settings.chunk_overlap_tokens,
                self._settings.chunk_strategy,
            )
        chunks = []
        for i, piece in enumerate(pieces):
            piece_page, piece_provenance = page_number, provenance
            if paged is not None:
                if page_number is None:
                    piece_page, piece_provenance = paged_location(paged[i], provenance)
                else:
                    piece_provenance = dict(provenance or {})
                if paged[i].parent_indexes:
                    piece_provenance["section_ids"] = [
                        section_id(key_prefix, n) for n in paged[i].parent_indexes
                    ]
            content = piece
            # Only worth contextualizing multi-chunk sources — a single-chunk issue/comment already
            # IS its own context, so there's nothing to situate it within. See contextualizer.py.
            if self._settings.contextual_retrieval_enabled and len(pieces) > 1:
                content = await self._contextualize(text, piece, issue_key, i)
            # Tag chunks with their source (no LLM needed) so each one says where it came from:
            #   - issue: the text starts with the key, but chunks after the first lose it (ATC-43);
            #   - comment/sprint: the raw text never names its issue/sprint;
            #   - attachment: file content rarely names its file or issue.
            # So attachments, comments and sprints are always tagged; issues only after chunk 0.
            if chunk_type == "attachment" and source_label:
                location_suffix = location_label(piece_page, piece_provenance)
                content = f"[Attachment: {source_label} — issue {issue_key}{location_suffix}]\n{content}"
            elif chunk_type == "comment":
                content = f"[Comment on {issue_key}]\n{content}"
            elif chunk_type == "sprint":
                content = f"[Sprint {issue_key}]\n{content}"
            elif chunk_type == "issue" and i > 0:
                content = f"[Issue {issue_key}]\n{content}"
            chunks.append(
                Chunk(
                    id=(f"{key_prefix}#{chunk_index_offset + i}"
                        if force_numbered_ids or len(pieces) > 1 else key_prefix),
                    chunk_type=chunk_type,
                    issue_id=issue_id,
                    issue_key=issue_key,
                    space_id=space_id,
                    source_id=source_id,
                    chunk_index=chunk_index_offset + i,
                    content=content,
                    page_number=piece_page,
                    provenance=piece_provenance or {},
                )
            )
        return chunks

    async def _contextualize(self, document: str, chunk: str, issue_key: str, chunk_index: int) -> str:
        try:
            context = await self._context_generator.generate(document, chunk)
        except Exception as exc:  # noqa: BLE001 - one context-generation failure must not fail ingestion
            logger.warning(
                "context generation failed; using chunk as-is",
                extra={"issue_key": issue_key, "chunk_index": chunk_index, "error": str(exc)},
            )
            return chunk
        return f"{context}\n\n{chunk}" if context else chunk

    async def _replace(
        self, chunk_type: str, source_id: int, chunks: List[Chunk], tracker=None,
        parents: List[ChunkParent] | None = None,
    ) -> None:
        if not chunks:
            # Nothing embeddable (blank title+description, empty comment) — clear any stale rows.
            await self._store.delete_source(chunk_type, source_id)
            return
        if await self._store.get_source_chunks(chunk_type, source_id) == chunks:
            # Nothing searchable changed: a status drag re-sends the whole issue, and Kafka redelivers.
            # Skipping the store also keeps the index watermark still, so ai-service's cached answers
            # survive. OpenSearch is still rewritten: redelivery is how a failed write there is retried.
            # An embedding-model change needs `backfill_jira_backend --truncate-first`, not a replay.
            INGEST_UNCHANGED_TOTAL.labels(chunk_type).inc()
            await self._store.refresh_lexical_index(chunk_type, source_id, chunks)
            return
        if tracker:
            await tracker.stage(Stage.EMBED)
        vectors = await self._embedder.embed([c.content for c in chunks])
        if tracker:
            await tracker.stage(Stage.STORE)
        await self._store.replace_source(chunk_type, source_id, chunks, vectors, parents or [])

    # --- Issues ---
    async def handle_issue(self, msg: IssueIngestionMessage) -> None:
        tracker = self._tracker.begin(
            source_type="issue", source_id=msg.issue_id, space_id=msg.space_id,
            issue_key=msg.issue_key, source_label=msg.issue_key, event_id=msg.event_id,
        )
        try:
            await self._handle_issue(msg, tracker)
        except Exception as exc:
            # Record where it failed, then re-raise unchanged: retry decisions stay with the consumer.
            await tracker.failed(exc)
            raise

    async def _handle_issue(self, msg: IssueIngestionMessage, tracker) -> None:
        if msg.event_type == "issue_deleted":
            removed = await self._store.delete_issue(msg.issue_id)
            await tracker.deleted(removed)
            logger.info("issue deleted", extra={"issue_key": msg.issue_key, "removed": removed})
            return

        await tracker.stage(Stage.PARSE)

        # Embed the key with title + description. The lexical index has the key as a field, but the
        # vector didn't contain it, so an issue's own chunk could rank below its comments for a query
        # that was just its key. (get_issue_details is the real fix; this helps ranking.)
        body = " ".join(filter(None, [msg.title, html_to_text(msg.description)])).strip()
        # Only prefix real content: a blank issue must still produce zero chunks
        # (test_blank_issue_clears_stale_vectors).
        if not body:
            text = ""
        elif msg.parent_key:
            # Embed the parent's key + title into a subtask's text: a subtask like "Reject repeated
            # checkout requests" doesn't say which incident it belongs to, and the agent kept mixing
            # up subtasks and related issues. It's a snapshot, so it can go stale if the parent is
            # renamed (accepted, like sprint_name).
            parent_label = f"{msg.parent_key} ({msg.parent_title})" if msg.parent_title else msg.parent_key
            text = f"{msg.issue_key} Subtask of {parent_label}. {body}".strip()
        else:
            text = f"{msg.issue_key} {body}".strip()
        await tracker.stage(Stage.CHUNK)
        chunks = await self._chunks(
            text,
            chunk_type="issue",
            issue_id=msg.issue_id,
            issue_key=msg.issue_key,
            space_id=msg.space_id,
            source_id=msg.issue_id,
            key_prefix=f"issue:{msg.issue_id}",
        )
        await self._replace("issue", msg.issue_id, chunks, tracker)
        await tracker.indexed(len(chunks))
        logger.info("issue upserted", extra={"issue_key": msg.issue_key, "chunks": len(chunks)})

    # --- Sprints (embeddable goal text only) ---
    async def handle_sprint(self, msg: SprintIngestionMessage) -> None:
        tracker = self._tracker.begin(
            source_type="sprint", source_id=msg.sprint_id, space_id=msg.space_id,
            issue_key=msg.sprint_name, source_label=msg.sprint_name, event_id=msg.event_id,
        )
        try:
            await self._handle_sprint(msg, tracker)
        except Exception as exc:
            await tracker.failed(exc)
            raise

    async def _handle_sprint(self, msg: SprintIngestionMessage, tracker) -> None:
        if msg.event_type == "sprint_deleted":
            removed = await self._store.delete_sprint(msg.sprint_id)
            await tracker.deleted(removed)
            logger.info("sprint deleted", extra={"sprint_name": msg.sprint_name, "removed": removed})
            return

        await tracker.stage(Stage.PARSE)

        # Sprints have no parent issue, so issue_id/issue_key hold the sprint's own id/name. Only the
        # goal is embedded — "which sprint is about accessibility" is about the goal.
        text = html_to_text(msg.goal)
        await tracker.stage(Stage.CHUNK)
        chunks = await self._chunks(
            text,
            chunk_type="sprint",
            issue_id=msg.sprint_id,
            issue_key=msg.sprint_name,
            space_id=msg.space_id,
            source_id=msg.sprint_id,
            key_prefix=f"sprint:{msg.sprint_id}",
        )
        await self._replace("sprint", msg.sprint_id, chunks, tracker)
        await tracker.indexed(len(chunks))
        logger.info("sprint upserted", extra={"sprint_name": msg.sprint_name, "chunks": len(chunks)})

    # --- Comments ---
    async def handle_comment(self, msg: CommentIngestionMessage) -> None:
        tracker = self._tracker.begin(
            source_type="comment", source_id=msg.comment_id, space_id=msg.space_id,
            issue_key=msg.issue_key, source_label=f"comment on {msg.issue_key}",
            event_id=msg.event_id,
        )
        try:
            await self._handle_comment(msg, tracker)
        except Exception as exc:
            await tracker.failed(exc)
            raise

    async def _handle_comment(self, msg: CommentIngestionMessage, tracker) -> None:
        if msg.event_type == "comment_deleted":
            removed = await self._store.delete_source("comment", msg.comment_id)
            await tracker.deleted(removed)
            logger.info(
                "comment deleted",
                extra={"issue_key": msg.issue_key, "comment_id": msg.comment_id, "removed": removed},
            )
            return

        await tracker.stage(Stage.PARSE)
        text = html_to_text(msg.content)
        await tracker.stage(Stage.CHUNK)
        chunks = await self._chunks(
            text,
            chunk_type="comment",
            issue_id=msg.issue_id,
            issue_key=msg.issue_key,
            space_id=msg.space_id,
            source_id=msg.comment_id,
            key_prefix=f"comment:{msg.comment_id}",
            provenance=comment_provenance(msg),
        )
        await self._replace("comment", msg.comment_id, chunks, tracker)
        await tracker.indexed(len(chunks))
        logger.info(
            "comment upserted",
            extra={"issue_key": msg.issue_key, "comment_id": msg.comment_id, "chunks": len(chunks)},
        )

    # --- Attachments ---
    async def handle_attachment(self, msg: AttachmentIngestionMessage) -> None:
        tracker = self._tracker.begin(
            source_type="attachment", source_id=msg.attachment_id, space_id=msg.space_id,
            issue_key=msg.issue_key, source_label=msg.filename, event_id=msg.event_id,
        )
        try:
            await self._handle_attachment(msg, tracker)
        except Exception as exc:
            await tracker.failed(exc)
            raise

    async def _handle_attachment(self, msg: AttachmentIngestionMessage, tracker) -> None:
        if msg.event_type == "attachment_deleted":
            removed = await self._store.delete_source("attachment", msg.attachment_id)
            await self._record_vlm_outcome(msg, [])
            await tracker.deleted(removed)
            logger.info(
                "attachment deleted",
                extra={"issue_key": msg.issue_key, "attachment_id": msg.attachment_id, "removed": removed},
            )
            return

        if msg.byte_size and msg.byte_size > self._settings.attachment_max_bytes:
            await tracker.skipped(
                f"declared byte_size {msg.byte_size} exceeds "
                f"attachment_max_bytes {self._settings.attachment_max_bytes}"
            )
            logger.info(
                "attachment too large; skipping",
                extra={"attachment_id": msg.attachment_id, "byte_size": msg.byte_size},
            )
            return

        await tracker.stage(Stage.FETCH)
        data = await fetch_attachment(self._settings, msg.storage_uri, msg.bucket, msg.storage_key)
        if not data:
            # fetch_attachment returns None instead of raising; record it, or the document vanishes
            # without a trace ("in Jira but never in object storage").
            await tracker.failed(
                FileNotFoundError(
                    f"attachment binary not retrievable from {msg.storage_uri or msg.bucket}/"
                    f"{msg.storage_key}"
                )
            )
            return
        # Kafka's size field comes from the producer; check the real bytes too before parsing.
        if len(data) > self._settings.attachment_max_bytes:
            await tracker.skipped(
                f"fetched {len(data)} bytes, exceeds "
                f"attachment_max_bytes {self._settings.attachment_max_bytes}"
            )
            logger.info(
                "fetched attachment too large; skipping",
                extra={"attachment_id": msg.attachment_id, "byte_size": len(data)},
            )
            return

        await tracker.stage(Stage.PARSE)
        sections = await parse_attachment_sections(
            data, msg.filename or "attachment", self._settings, self._vlm_describer
        )
        parse_errors = pop_parse_errors()
        vlm_failures = pop_vlm_failures()
        if not sections and parse_errors:
            # The parser swallowed a real error (corrupt file, Docling crash). Record `failed`, or it
            # would look like a blank file. Not re-raised: a corrupt file must be skipped, not retried
            # forever.
            await tracker.failed(RuntimeError("; ".join(parse_errors)))
            logger.warning(
                "attachment parse produced no sections after an error; recorded as failed",
                extra={"issue_key": msg.issue_key, "attachment_id": msg.attachment_id,
                       "errors": parse_errors},
            )
            return

        await tracker.stage(Stage.CHUNK)
        chunks = []
        parents: List[ChunkParent] = []
        chunk_index_offset = 0
        # Number section chunks deterministically, or two one-chunk pages would both be
        # `attachment:{id}` and overwrite each other.
        force_numbered_ids = len(sections) > 1 or any(s.page_number is not None for s in sections)
        if len(sections) > 1 and all(s.page_number is not None for s in sections):
            # A PDF is one document that happens to be paginated: chunk it whole, or anything running
            # over a page break is cut there.
            paged, parent_sections = chunk_pages_with_parents(
                [(s.page_number, s.content) for s in sections],
                self._settings.chunk_size_tokens,
                self._settings.chunk_overlap_tokens,
                self._settings.parent_max_tokens,
            )
            chunks = await self._chunks(
                "\n\n".join(s.content for s in sections),
                chunk_type="attachment",
                issue_id=msg.issue_id,
                issue_key=msg.issue_key,
                space_id=msg.space_id,
                source_id=msg.attachment_id,
                key_prefix=f"attachment:{msg.attachment_id}",
                source_label=msg.filename,
                provenance=sections[0].provenance,
                force_numbered_ids=True,
                paged=paged,
            )
            parents = attachment_parents(msg, parent_sections)
            sections = []
        for section in sections:
            # A DOCX, HTML file, slide or sheet: chunked on its own, with parents like a PDF's.
            paged = None
            if self._settings.chunk_strategy == "structural":
                paged, section_parents = chunk_text_with_parents(
                    section.content,
                    self._settings.chunk_size_tokens,
                    self._settings.chunk_overlap_tokens,
                    self._settings.parent_max_tokens,
                    page=section.page_number,
                )
                # Parent ids are numbered across the whole attachment, not per section.
                paged = [
                    replace(p, parent_indexes=tuple(len(parents) + n for n in p.parent_indexes))
                    for p in paged
                ]
                parents.extend(
                    attachment_parents(msg, section_parents, start=len(parents), provenance=section.provenance)
                )
            section_chunks = await self._chunks(
                section.content,
                chunk_type="attachment",
                issue_id=msg.issue_id,
                issue_key=msg.issue_key,
                space_id=msg.space_id,
                source_id=msg.attachment_id,
                key_prefix=f"attachment:{msg.attachment_id}",
                source_label=msg.filename,
                page_number=section.page_number,
                provenance=section.provenance,
                chunk_index_offset=chunk_index_offset,
                force_numbered_ids=force_numbered_ids,
                paged=paged,
            )
            chunks.extend(section_chunks)
            chunk_index_offset += len(section_chunks)
        await self._replace("attachment", msg.attachment_id, chunks, tracker, parents)
        await self._record_vlm_outcome(msg, vlm_failures)
        # Zero chunks = EMPTY, not INDEXED: a file with no readable text is what an operator needs to see.
        await tracker.indexed(len(chunks))
        logger.info(
            "attachment upserted",
            extra={"issue_key": msg.issue_key, "attachment_id": msg.attachment_id, "chunks": len(chunks)},
        )

    async def _record_vlm_outcome(self, msg: AttachmentIngestionMessage, vlm_failures: list[str]) -> None:
        """Queue the attachment for a VLM retry if any VLM call failed, otherwise drop any old entry.

        Best effort, like status tracking: the chunks are already stored, and failing the event here
        would only re-run the same degraded parse.
        """
        try:
            if not vlm_failures:
                await self._vlm_retry_queue.clear_vlm_retry(msg.attachment_id)
                return
            VLM_DEGRADED_ATTACHMENTS_TOTAL.inc()
            logger.warning(
                "attachment indexed without its VLM pass; queued for retry",
                extra={"issue_key": msg.issue_key, "attachment_id": msg.attachment_id,
                       "failed_calls": len(vlm_failures)},
            )
            await self._vlm_retry_queue.queue_vlm_retry(
                attachment_id=msg.attachment_id, issue_id=msg.issue_id, space_id=msg.space_id,
                event_id=msg.event_id, message=msg.model_dump(mode="json", by_alias=True),
                failed_calls=len(vlm_failures), last_error=vlm_failures[-1],
                retry_after_seconds=self._settings.vlm_retry_interval_seconds,
            )
        except Exception:  # noqa: BLE001 - see docstring
            logger.warning("VLM retry queue write failed", exc_info=True)


def section_id(key_prefix: str, index: int) -> str:
    """Id of a parent section, e.g. `attachment:42:section#3`."""
    return f"{key_prefix}:section#{index}"


def location_label(page: int | None, provenance: dict | None) -> str:
    """Where in the file a chunk or parent comes from, e.g. " — PDF pages 3-4" or " — PPTX slide 2"."""
    location = provenance or {}
    if page is not None:
        end = location.get("page_end")
        label = f"PDF pages {page}-{end}" if end else f"PDF page {page}"
    elif location.get("slide_number") is not None:
        label = f"PPTX slide {location['slide_number']}"
    elif location.get("sheet_name"):
        cell_range = f" ({location['cell_range']})" if location.get("cell_range") else ""
        label = f"XLSX sheet {location['sheet_name']}{cell_range}"
    elif location.get("page_number") is not None:
        label = f"DOCX page {location['page_number']}"
    else:
        label = ""
    return f" — {label}" if label else ""


def attachment_parents(
    msg: AttachmentIngestionMessage, parent_sections: List[ParentSection], start: int = 0,
    provenance: dict | None = None,
) -> List[ChunkParent]:
    """Parent sections of an attachment, tagged like its chunks so the agent knows which file they are
    from. `start` is the index of the first one; `provenance` is the slide/sheet of a non-PDF section.
    """
    parents = []
    for index, parent in enumerate(parent_sections, start):
        first, last = parent.first_page, parent.last_page
        location = dict(provenance or {})
        if last is not None and last != first:
            location["page_end"] = last
        where = location_label(first, location)
        header = f"[Attachment: {msg.filename or 'attachment'} — issue {msg.issue_key}{where}]"
        parents.append(
            ChunkParent(
                id=section_id(f"attachment:{msg.attachment_id}", index),
                chunk_type="attachment",
                issue_id=msg.issue_id,
                issue_key=msg.issue_key,
                space_id=msg.space_id,
                source_id=msg.attachment_id,
                content=f"{header}\n{parent.text}",
                page_number=first,
                page_end=last,
            )
        )
    return parents


def paged_location(piece: PagedChunk, base: dict | None) -> tuple[int | None, dict]:
    """Page number and provenance for a chunk of a merged PDF. `page_number` is the first page its
    text starts on; `page_end` is added only when the chunk runs over a page break.
    """
    provenance = dict(base or {})
    provenance.pop("extraction", None)  # the first page's recovery note doesn't describe this chunk
    if piece.first_page is not None:
        provenance["page_number"] = piece.first_page
    if piece.last_page is not None and piece.last_page != piece.first_page:
        provenance["page_end"] = piece.last_page
    else:
        provenance.pop("page_end", None)
    return piece.first_page, provenance


def comment_provenance(msg: CommentIngestionMessage) -> dict:
    """Author, time and thread for get_issue_comments. Kept out of the embedded text on purpose
    (measured: quoting the parent pushed it out of top-5, hit@5 0.75 -> 0.67)."""
    fields = {
        "author": msg.author_name,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
        "parent_id": msg.parent_id,
        "reply_to_id": msg.reply_to_id,
    }
    return {k: v for k, v in fields.items() if v is not None}
