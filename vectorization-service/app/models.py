"""Wire models for inbound Kafka payloads, plus the internal chunk types.

jira-backend sends camelCase JSON (Jackson); an alias generator maps it to snake_case attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Literal, NotRequired, Optional, TypedDict

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class IssueIngestionMessage(_CamelModel):
    event_id: str
    event_type: Literal["issue_upserted", "issue_deleted"]
    emitted_at: datetime
    issue_id: int
    issue_key: str
    space_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    # Only for subtasks: the parent's key/title. Also embedded into the subtask's text (see
    # IngestPipeline.handle_issue).
    parent_key: Optional[str] = None
    parent_title: Optional[str] = None


class SprintIngestionMessage(_CamelModel):
    """One sprint's searchable goal text, from jira-backend's SprintContentChangedEvent.

    Unlike an issue, a sprint has no comment/attachment chunks — its ``goal`` is the only embeddable
    text, and it's the whole reason this message exists: without it, "which sprint's goal was about
    accessibility" has nothing real to search against (see SprintContentChangedEvent's docstring).
    """

    event_id: str
    event_type: Literal["sprint_upserted", "sprint_deleted"]
    emitted_at: datetime
    sprint_id: int
    sprint_name: str
    space_id: int
    goal: Optional[str] = None


class CommentIngestionMessage(_CamelModel):
    event_id: str
    event_type: Literal["comment_upserted", "comment_deleted"]
    emitted_at: datetime
    comment_id: int
    issue_id: int
    issue_key: str
    space_id: int
    content: Optional[str] = None
    # Who/when/which thread. Optional because older jira-backend builds don't send them; stored as
    # chunk provenance (never embedded), so they can't change retrieval ranking.
    author_name: Optional[str] = None
    created_at: Optional[datetime] = None
    parent_id: Optional[int] = None
    reply_to_id: Optional[int] = None


class AttachmentIngestionMessage(_CamelModel):
    event_id: str
    # jira-backend emits "attachment_uploaded" / "attachment_deleted".
    event_type: Literal["attachment_uploaded", "attachment_deleted"]
    emitted_at: datetime
    attachment_id: int
    issue_id: int
    issue_key: str
    space_id: int
    filename: Optional[str] = None
    mime_type: Optional[str] = None
    byte_size: Optional[int] = None
    storage_backend: Optional[str] = None
    bucket: Optional[str] = None
    storage_key: Optional[str] = None
    storage_uri: Optional[str] = None


ChunkType = Literal["issue", "comment", "attachment", "sprint"]


class AttachmentProvenance(TypedDict, total=False):
    """Static schema for format-native attachment locators stored in chunks.provenance JSONB."""

    source_type: str
    page_number: NotRequired[int]
    page_end: NotRequired[int]  # last page of a chunk that runs over a page break
    slide_number: NotRequired[int]
    sheet_name: NotRequired[str]
    cell_range: NotRequired[str]
    bbox: NotRequired[List[float]]
    coordinate_system: NotRequired[str]
    extraction: NotRequired[str]
    section_ids: NotRequired[List[str]]  # parent sections this chunk was cut from (ChunkParent ids)


@dataclass(slots=True)
class ChunkParent:
    """A whole section that several chunks were cut from (parent-child retrieval). Not embedded and
    not searched: a chunk lists its parents in `provenance["section_ids"]`, and the agent reads one
    by id when that chunk alone is not enough.
    """

    id: str
    chunk_type: ChunkType
    issue_id: int
    issue_key: str
    space_id: int
    source_id: int
    content: str
    page_number: Optional[int] = None
    page_end: Optional[int] = None


@dataclass(slots=True)
class SearchHit:
    """One ranked result from VectorStore.search_* and /search. `retrievers` says which signal(s)
    found it (["vector"], ["lexical"], or both).
    """

    id: str
    chunk_type: ChunkType
    issue_id: int
    issue_key: str
    space_id: int
    source_id: int
    content: str
    score: float
    retrievers: List[str]
    # Null for sources without a page concept, and for legacy/Docling-only PDF chunks whose page
    # boundary was not available. PDF VLM page chunks carry a real, 1-based source page number.
    page_number: Optional[int] = None
    # Format-specific source locator, e.g. slide_number, sheet_name/cell_range, or image bbox.
    provenance: AttachmentProvenance = field(default_factory=dict)


@dataclass(slots=True)
class IssueComment:
    """One comment chunk, fetched by issue key — complete and unranked, not a search hit (see
    VectorStore.get_issue_comments).
    """

    id: str
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    chunk_index: int
    # author / created_at / parent_id / reply_to_id when ingested with them; {} for older chunks.
    provenance: dict = field(default_factory=dict)


@dataclass(slots=True)
class IssueCommentsQueryResult:
    """Aggregate + list result of a get_issue_comments fetch: exact count plus every comment
    (up to `limit`) for the requested issues, in thread order (see VectorStore.get_issue_comments)."""

    total_count: int
    comments: List[IssueComment]


@dataclass(slots=True)
class IssueDetail:
    """An issue's own title+description chunk, fetched by key — not a search hit. An issue's own chunk
    could rank below its comments; see VectorStore.get_issue_details.
    """

    id: str
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    chunk_index: int


@dataclass(slots=True)
class IssueDetailsQueryResult:
    """Aggregate + list result of a get_issue_details fetch: exact count plus every issue-body
    chunk (up to `limit`) for the requested issues, ordered by issue then chunk_index."""

    total_count: int
    details: List[IssueDetail]


@dataclass(slots=True)
class IssueAttachment:
    """One attachment chunk, fetched by key — not a search hit. Specific facts (an exact SKU) are hard
    to search for; see VectorStore.get_issue_attachments.
    """

    id: str
    issue_id: int
    issue_key: str
    source_id: int
    content: str
    chunk_index: int
    page_number: Optional[int] = None
    provenance: AttachmentProvenance = field(default_factory=dict)


@dataclass(slots=True)
class IssueAttachmentsQueryResult:
    """Aggregate + list result of a get_issue_attachments fetch: exact count plus every attachment
    chunk (up to `limit`) for the requested issues, ordered by issue then chunk_index."""

    total_count: int
    attachments: List[IssueAttachment]


@dataclass(slots=True)
class IngestionStatusRow:
    """One document's position in the ingestion pipeline (app/ingest/status.py).

    Per source, not per chunk: "did this PDF get indexed, and if not where did it die?" A source that
    produced zero chunks has no chunk row to hold its status — the case this table exists for.
    """

    source_type: str
    source_id: int
    space_id: Optional[int]
    issue_key: Optional[str]
    source_label: Optional[str]
    state: str
    stage: Optional[str]
    attempts: int
    chunk_count: Optional[int]
    error_type: Optional[str]
    error_message: Optional[str]
    event_id: Optional[str]
    request_id: Optional[str]
    duration_ms: Optional[int]
    stage_ms: dict
    first_seen_at: Optional[datetime]
    updated_at: Optional[datetime]


@dataclass(slots=True)
class VlmRetryItem:
    """One queued VLM retry: the original Kafka event to re-run, and which event it
    was, so a retry for content that has since changed is dropped."""

    attachment_id: int
    event_id: str
    message: dict
    attempts: int


@dataclass(slots=True)
class IngestionStatusQueryResult:
    """Aggregate + list result of a status query — counts cover every matching document:
    ``total_count`` and ``counts_by_state`` cover every match, ``rows`` is the limited sample."""

    total_count: int
    counts_by_state: dict[str, int]
    rows: List[IngestionStatusRow]


@dataclass(slots=True)
class Chunk:
    """One unit that gets embedded and stored as one vector-store row.

    `id` is deterministic (`issue:{id}`, `comment:{id}`, `attachment:{id}#{index}`), so re-processing
    overwrites instead of duplicating. `source_id` links a hit back to its issue/comment/attachment
    for citations; `space_id` drives the per-space permission filter.

    Sprint chunks have no parent issue, so `issue_id`/`issue_key` hold the sprint's own id/name.
    """

    id: str
    chunk_type: ChunkType
    issue_id: int
    issue_key: str
    space_id: int
    source_id: int
    chunk_index: int
    content: str
    # Source-document page, deliberately distinct from `chunk_index`: a long PDF page can yield
    # several chunks with the same page_number, while the chunk index stays unique per attachment.
    page_number: Optional[int] = None
    # Typed, format-native source locator serialized as JSONB in the vector store. Keep page_number
    # as a first-class PDF filter/index; this object carries slide/sheet/range/bbox metadata.
    provenance: AttachmentProvenance = field(default_factory=dict)
