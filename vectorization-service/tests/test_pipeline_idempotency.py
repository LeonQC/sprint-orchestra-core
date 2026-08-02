"""Pipeline behaviour with a fake embedder + in-memory store.

The key property under test is idempotency: re-processing the same event (which Kafka's
at-least-once delivery makes routine) must not accumulate duplicate vectors, and deletes must
actually purge — this is what makes the "commit-after-handle, redeliver-on-failure" consumer safe.
"""
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.ingest.embedder import FakeEmbedder
from app.ingest.chunker import PagedChunk
from app.ingest.pipeline import IngestPipeline
from app.models import (
    CommentIngestionMessage,
    IssueIngestionMessage,
    SprintIngestionMessage,
)


class InMemoryStore:
    """Mimics VectorStore's keyed replace/delete semantics with a dict."""

    def __init__(self):
        self.rows = {}  # id -> chunk

    async def delete_sprint(self, sprint_id) -> int:
        before = len(self.rows)
        self.rows = {
            cid: c for cid, c in self.rows.items()
            if not (c.chunk_type == "sprint" and c.source_id == sprint_id)
        }
        return before - len(self.rows)

    async def replace_source(self, chunk_type, source_id, chunks, vectors, parents=()):
        # Delete-then-insert, same as the real store.
        self.rows = {
            cid: c for cid, c in self.rows.items()
            if not (c.chunk_type == chunk_type and c.source_id == source_id)
        }
        for c in chunks:
            self.rows[c.id] = c

    async def get_source_chunks(self, chunk_type, source_id):
        return sorted(
            (c for c in self.rows.values() if c.chunk_type == chunk_type and c.source_id == source_id),
            key=lambda c: c.chunk_index,
        )

    async def refresh_lexical_index(self, chunk_type, source_id, chunks):
        self.lexical_refreshes = getattr(self, "lexical_refreshes", 0) + 1

    async def delete_source(self, chunk_type, source_id) -> int:
        before = len(self.rows)
        self.rows = {
            cid: c for cid, c in self.rows.items()
            if not (c.chunk_type == chunk_type and c.source_id == source_id)
        }
        return before - len(self.rows)

    async def delete_issue(self, issue_id) -> int:
        before = len(self.rows)
        self.rows = {cid: c for cid, c in self.rows.items() if c.issue_id != issue_id}
        return before - len(self.rows)


def _settings() -> Settings:
    return Settings(embedding_provider="fake", embedding_dim=8)


def _issue_msg(event_type="issue_upserted") -> IssueIngestionMessage:
    return IssueIngestionMessage(
        eventId="e1",
        eventType=event_type,
        emittedAt=datetime.now(timezone.utc),
        issueId=42,
        issueKey="PROJ-42",
        spaceId=7,
        title="Payment service outage",
        description="<p>root cause was a connection pool leak</p>",
        issueType="bug",
        status="done",
        priority="high",
        createdAt=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updatedAt=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _comment_msg(event_type="comment_upserted") -> CommentIngestionMessage:
    return CommentIngestionMessage(
        eventId="e2",
        eventType=event_type,
        emittedAt=datetime.now(timezone.utc),
        commentId=981,
        issueId=42,
        issueKey="PROJ-42",
        spaceId=7,
        content="<p>raising HikariCP max-lifetime fixed it</p>",
    )


@pytest.fixture
def pipeline():
    settings = _settings()
    return IngestPipeline(settings, FakeEmbedder(settings.embedding_dim), InMemoryStore()), settings


async def test_issue_upsert_is_idempotent(pipeline):
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    first = dict(pipe._store.rows)
    assert len(first) == 1

    # Reprocessing the identical event (Kafka redelivery) must not duplicate.
    await pipe.handle_issue(_issue_msg())
    assert len(pipe._store.rows) == 1
    assert set(pipe._store.rows) == set(first)


class _CountingEmbedder(FakeEmbedder):
    def __init__(self, dim):
        super().__init__(dim)
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return await super().embed(texts)


async def test_legacy_metadata_only_reingest_skips_embedding():
    # A status drag re-sends the whole issue. Re-embedding unchanged text costs money and, by
    # rewriting chunks, moved the index watermark and emptied ai-service's answer cache.
    settings = _settings()
    embedder = _CountingEmbedder(settings.embedding_dim)
    pipe = IngestPipeline(settings, embedder, InMemoryStore())
    await pipe.handle_issue(_issue_msg())

    dragged = _issue_msg()
    dragged = dragged.model_copy(update={"status": "in_progress"})
    await pipe.handle_issue(dragged)

    assert embedder.calls == 1
    # Redelivery is how a failed OpenSearch write gets retried, so the lexical docs are still rewritten.
    assert pipe._store.lexical_refreshes == 1


async def test_changed_text_is_re_embedded():
    settings = _settings()
    embedder = _CountingEmbedder(settings.embedding_dim)
    pipe = IngestPipeline(settings, embedder, InMemoryStore())
    await pipe.handle_issue(_issue_msg())

    edited = _issue_msg()
    edited.description = "<p>root cause was a DNS timeout</p>"
    await pipe.handle_issue(edited)

    assert embedder.calls == 2
    assert "DNS timeout" in next(iter(pipe._store.rows.values())).content


async def test_attachment_chunks_are_tagged_with_their_source_filename(pipeline):
    # An attachment's parsed content (a CSV row, a PDF's body text) essentially never mentions its own
    # filename or issue key naturally — found live, unlike issue text (which handle_issue prepends the
    # key into before chunking). Contextual retrieval's LLM-written sentence is optional (off by
    # default) so it can't be the only thing carrying this — this deterministic tag must fire for
    # every attachment chunk, single- or multi-chunk, contextual retrieval on or off.
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "Short attachment body.",
        chunk_type="attachment", issue_id=1, issue_key="ATC-1", space_id=1, source_id=99,
        key_prefix="attachment:99", source_label="report.pdf",
    )
    assert len(chunks) == 1
    assert chunks[0].content.startswith("[Attachment: report.pdf — issue ATC-1]")
    assert "Short attachment body." in chunks[0].content


async def test_pdf_attachment_chunks_keep_structured_page_metadata_and_a_readable_prefix(pipeline):
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "Invoice total: $42.",
        chunk_type="attachment", issue_id=1, issue_key="ATC-1", space_id=1, source_id=99,
        key_prefix="attachment:99", source_label="invoice.pdf", page_number=2,
        force_numbered_ids=True,
    )

    assert len(chunks) == 1
    assert chunks[0].id == "attachment:99#0"
    assert chunks[0].page_number == 2
    assert chunks[0].content.startswith("[Attachment: invoice.pdf — issue ATC-1 — PDF page 2]")


async def test_merged_pdf_chunk_that_runs_over_a_page_break_names_both_pages(pipeline):
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "whole document",
        chunk_type="attachment", issue_id=1, issue_key="ATC-1", space_id=1, source_id=99,
        key_prefix="attachment:99", source_label="contract.pdf",
        provenance={"source_type": "pdf", "page_number": 1, "extraction": "docling_cell_recovery"},
        force_numbered_ids=True,
        paged=[PagedChunk("Payment terms ...", 3, 4), PagedChunk("Notices ...", 4, 4)],
    )

    assert [(c.page_number, c.provenance.get("page_end")) for c in chunks] == [(3, 4), (4, None)]
    assert chunks[0].content.startswith("[Attachment: contract.pdf — issue ATC-1 — PDF pages 3-4]")
    assert chunks[1].content.startswith("[Attachment: contract.pdf — issue ATC-1 — PDF page 4]")
    assert "extraction" not in chunks[0].provenance


async def test_attachment_chunks_keep_native_non_pdf_provenance(pipeline):
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "Revenue by region.",
        chunk_type="attachment", issue_id=1, issue_key="ATC-1", space_id=1, source_id=99,
        key_prefix="attachment:99", source_label="revenue.xlsx",
        provenance={"source_type": "xlsx", "sheet_name": "Q2 Revenue"},
        force_numbered_ids=True,
    )

    assert chunks[0].page_number is None
    assert chunks[0].provenance == {"source_type": "xlsx", "sheet_name": "Q2 Revenue"}
    assert chunks[0].content.startswith(
        "[Attachment: revenue.xlsx — issue ATC-1 — XLSX sheet Q2 Revenue]"
    )


async def test_comment_chunks_are_always_tagged_even_the_first_one(pipeline):
    # Found live: unlike issue text, a comment's raw text (html_to_text(msg.content)) has NO issue_key
    # prefix at all — confirmed against a real comment in the corpus, chunk #0 read as plain prose with
    # zero mention of which issue it's on. So comments are tagged unconditionally, not just past
    # chunk #0 (contrast with the issue case below).
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "The fix landed in the retry handler.",
        chunk_type="comment", issue_id=1, issue_key="ATC-1", space_id=1, source_id=5,
        key_prefix="comment:5",
    )
    assert len(chunks) == 1
    assert chunks[0].content == "[Comment on ATC-1]\nThe fix landed in the retry handler."


async def test_issue_chunk_zero_is_not_tagged_but_later_chunks_are():
    # handle_issue prepends f"{issue_key} {body}" into the text BEFORE chunking (see handle_issue), so
    # chunk #0 always naturally starts with its own key already — tagging it too would be pure
    # redundant noise. But chunk #1+ (past the split point) has lost that natural prefix — found live,
    # ATC-43's own body is long enough to produce exactly this case in the real corpus. Force a split
    # here with a small chunk_size so the test doesn't depend on a huge fixture string.
    settings = Settings(embedding_provider="fake", embedding_dim=8, chunk_size_tokens=10, chunk_overlap_tokens=2)
    pipe = IngestPipeline(settings, FakeEmbedder(settings.embedding_dim), InMemoryStore())
    long_text = "ATC-1 " + " ".join(f"word{i}" for i in range(40))

    chunks = await pipe._chunks(
        long_text,
        chunk_type="issue", issue_id=1, issue_key="ATC-1", space_id=1, source_id=1,
        key_prefix="issue:1",
    )

    assert len(chunks) > 1
    assert not chunks[0].content.startswith("[Issue ATC-1]")
    assert all(c.content.startswith("[Issue ATC-1]") for c in chunks[1:])


async def test_sprint_chunks_are_always_tagged(pipeline):
    # A sprint's goal text (html_to_text(msg.goal)) has no sprint_name prefix either — same gap as
    # comments, for the same reason (see handle_sprint: the goal is embedded on its own).
    pipe, _ = pipeline
    chunks = await pipe._chunks(
        "Make the storefront accessible and observable.",
        chunk_type="sprint", issue_id=70, issue_key="Sprint 7", space_id=1, source_id=70,
        key_prefix="sprint:70",
    )
    assert chunks[0].content == "[Sprint Sprint 7]\nMake the storefront accessible and observable."


async def test_html_is_stripped_before_storing(pipeline):
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    (chunk,) = pipe._store.rows.values()
    assert "<p>" not in chunk.content
    assert "connection pool leak" in chunk.content


async def test_subtask_embeds_parent_context(pipeline):
    # A subtask's own text is often too terse to be findable or correctly characterized on its own —
    # embedding the parent's key+title directly into the subtask's own chunk means that context
    # doesn't depend on the parent's chunk also ranking into the same top-K.
    pipe, _ = pipeline
    msg = _issue_msg()
    msg = msg.model_copy(update={
        "parent_issue_id": 43, "parent_key": "ATC-43",
        "parent_title": "Checkout can create two orders from a double click",
    })
    await pipe.handle_issue(msg)
    (chunk,) = pipe._store.rows.values()
    assert "Subtask of ATC-43" in chunk.content
    assert "Checkout can create two orders from a double click" in chunk.content
    assert "connection pool leak" in chunk.content  # the issue's own body is still there too


async def test_non_subtask_issue_has_no_parent_prefix(pipeline):
    # No parent on the message (the common case: not every issue is a subtask) must not add any
    # "Subtask of" text — regression guard for the branch added alongside the above.
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    (chunk,) = pipe._store.rows.values()
    assert "Subtask of" not in chunk.content


async def test_comment_delete_purges_only_that_comment(pipeline):
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    await pipe.handle_comment(_comment_msg())
    assert len(pipe._store.rows) == 2

    await pipe.handle_comment(_comment_msg("comment_deleted"))
    remaining = list(pipe._store.rows.values())
    assert len(remaining) == 1
    assert remaining[0].chunk_type == "issue"


async def test_issue_delete_purges_issue_and_its_comments(pipeline):
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    await pipe.handle_comment(_comment_msg())
    assert len(pipe._store.rows) == 2

    await pipe.handle_issue(_issue_msg("issue_deleted"))
    assert pipe._store.rows == {}


async def test_blank_issue_clears_stale_vectors(pipeline):
    pipe, _ = pipeline
    await pipe.handle_issue(_issue_msg())
    assert len(pipe._store.rows) == 1

    blank = _issue_msg()
    blank.title = None
    blank.description = ""
    await pipe.handle_issue(blank)
    assert pipe._store.rows == {}


def _sprint_msg(event_type="sprint_upserted") -> SprintIngestionMessage:
    return SprintIngestionMessage(
        eventId="e-s1",
        eventType=event_type,
        emittedAt=datetime.now(timezone.utc),
        sprintId=501,
        sprintName="Sprint 7",
        spaceId=7,
        goal="Prove the storefront is accessible, observable, and ready for a larger public beta.",
        status="active",
        completedPoints=12,
        finalScopePoints=20,
    )


async def test_sprint_upsert_embeds_goal_without_structured_storage(pipeline):
    pipe, _ = pipeline
    await pipe.handle_sprint(_sprint_msg())

    (chunk,) = pipe._store.rows.values()
    assert chunk.chunk_type == "sprint"
    # No parent issue for a sprint chunk — issue_id/issue_key are reused as the sprint's own
    # id/name (see Chunk's docstring in app/models.py), not left null or pointing at a real issue.
    assert chunk.issue_id == 501
    assert chunk.issue_key == "Sprint 7"
    assert "accessible" in chunk.content


async def test_sprint_upsert_is_idempotent(pipeline):
    pipe, _ = pipeline
    await pipe.handle_sprint(_sprint_msg())
    await pipe.handle_sprint(_sprint_msg())
    assert len(pipe._store.rows) == 1


async def test_sprint_delete_purges_goal_chunk(pipeline):
    pipe, _ = pipeline
    await pipe.handle_sprint(_sprint_msg())
    assert len(pipe._store.rows) == 1

    await pipe.handle_sprint(_sprint_msg("sprint_deleted"))
    assert pipe._store.rows == {}


async def test_blank_sprint_goal_produces_no_chunk(pipeline):
    pipe, _ = pipeline
    blank = _sprint_msg()
    blank.goal = None
    await pipe.handle_sprint(blank)
    assert pipe._store.rows == {}


async def test_reply_carries_author_time_and_thread_as_provenance_not_embedded_text(pipeline):
    pipe, _ = pipeline
    store = pipe._store
    msg = _comment_msg().model_copy(update={
        "author_name": "Liam Ortiz",
        "created_at": datetime(2026, 6, 18, 11, 30, tzinfo=timezone.utc),
        "parent_id": 900,
        "reply_to_id": 950,
    })
    await pipe.handle_comment(msg)

    (chunk,) = store.rows.values()
    assert chunk.provenance == {
        "author": "Liam Ortiz", "created_at": "2026-06-18T11:30:00+00:00",
        "parent_id": 900, "reply_to_id": 950,
    }
    # The embedded text is unchanged: quoting the parent measurably hurt retrieval.
    assert chunk.content == "[Comment on PROJ-42]\nraising HikariCP max-lifetime fixed it"


async def test_comment_from_an_older_producer_gets_empty_provenance(pipeline):
    pipe, _ = pipeline
    await pipe.handle_comment(_comment_msg())
    (chunk,) = pipe._store.rows.values()
    assert chunk.provenance == {}
