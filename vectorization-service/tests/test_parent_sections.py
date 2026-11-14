"""Parent-child retrieval: a PDF section cut into several chunks is also stored whole, and each chunk
names it in provenance.section_ids so the agent can read it (POST /sections).
"""
from datetime import datetime, timezone

import pytest

from app.config import Settings
from app.db.vector_store import VectorStore
from app.ingest import docling_parser
from app.ingest import pipeline as pipeline_module
from app.ingest.docling_parser import ParsedAttachmentSection
from app.ingest.embedder import FakeEmbedder
from app.ingest.pipeline import IngestPipeline
from app.models import AttachmentIngestionMessage


def _steps(n: int) -> str:
    return "\n\n".join(f"Step {i}: run rollback command {i} and check the dashboard." for i in range(n))


class _Store:
    def __init__(self):
        self.chunks = []
        self.parents = []

    async def replace_source(self, chunk_type, source_id, chunks, vectors, parents=()):
        self.chunks = list(chunks)
        self.parents = list(parents)

    async def get_source_chunks(self, chunk_type, source_id):
        return []

    async def delete_source(self, chunk_type, source_id):
        return 0


@pytest.fixture
def three_page_pdf(monkeypatch):
    async def fake_fetch(*_args):
        return b"%PDF"

    async def fake_parse(data, filename, settings, describer):
        pages = [
            "## Runbook\n\n## 1. Overview\n\nThe checkout service runbook.",
            f"## 2. Rollback\n\n{_steps(40)}",
            "## 3. Contacts\n\nPage the on-call engineer.",
        ]
        return [
            ParsedAttachmentSection(text, page_number=n, provenance={"source_type": "pdf", "page_number": n})
            for n, text in enumerate(pages, start=1)
        ]

    monkeypatch.setattr(pipeline_module, "fetch_attachment", fake_fetch)
    monkeypatch.setattr(docling_parser, "_parse_attachment_sections", fake_parse)


async def test_a_split_section_is_stored_whole_and_its_chunks_point_at_it(three_page_pdf):
    store = _Store()
    settings = Settings(embedding_provider="fake", embedding_dim=8, chunk_size_tokens=200,
                        chunk_overlap_tokens=30)
    msg = AttachmentIngestionMessage(
        eventId="e1", eventType="attachment_uploaded", emittedAt=datetime(2026, 11, 8, tzinfo=timezone.utc),
        attachmentId=99, issueId=42, issueKey="PROJ-42", spaceId=7, filename="runbook.pdf",
        storageUri="s3://bucket/runbook.pdf",
    )

    await IngestPipeline(settings, FakeEmbedder(8), store).handle_attachment(msg)

    assert [p.id for p in store.parents] == ["attachment:99:section#0"]
    parent = store.parents[0]
    assert parent.content.startswith("[Attachment: runbook.pdf — issue PROJ-42 — PDF page 2]")
    assert "Step 0:" in parent.content and "Step 39:" in parent.content
    assert (parent.space_id, parent.source_id, parent.page_number) == (7, 99, 2)

    rollback_chunks = [c for c in store.chunks if "Step " in c.content]
    assert len(rollback_chunks) >= 2
    for chunk in store.chunks:
        expected = ["attachment:99:section#0"] if chunk in rollback_chunks else None
        assert chunk.provenance.get("section_ids") == expected


class _FakePool:
    def __init__(self):
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return [{
            "id": "attachment:99:section#0", "chunk_type": "attachment", "issue_id": 42,
            "issue_key": "PROJ-42", "space_id": 7, "source_id": 99, "content": "whole section",
            "page_number": 2, "page_end": 3,
        }]


async def test_sections_are_read_by_id_inside_the_callers_spaces():
    pool = _FakePool()

    found = await VectorStore(pool).get_sections([7], ["attachment:99:section#0"])

    sql, args = pool.calls[0]
    assert "space_id = ANY($1::bigint[])" in sql and "id = ANY($2::text[])" in sql
    assert args == ([7], ["attachment:99:section#0"])
    assert found[0].content == "whole section" and found[0].page_end == 3


async def test_a_docx_section_gets_parents_too(monkeypatch):
    async def fake_fetch(*_args):
        return b"PK"

    async def fake_parse(data, filename, settings, describer):
        text = f"# Runbook\n\n## 1. Overview\n\nThe checkout service runbook.\n\n## 2. Rollback\n\n{_steps(40)}"
        return [ParsedAttachmentSection(text, provenance={"source_type": "docx"})]

    monkeypatch.setattr(pipeline_module, "fetch_attachment", fake_fetch)
    monkeypatch.setattr(docling_parser, "_parse_attachment_sections", fake_parse)
    store = _Store()
    settings = Settings(embedding_provider="fake", embedding_dim=8, chunk_size_tokens=200,
                        chunk_overlap_tokens=30)
    msg = AttachmentIngestionMessage(
        eventId="e2", eventType="attachment_uploaded", emittedAt=datetime(2026, 11, 12, tzinfo=timezone.utc),
        attachmentId=98, issueId=42, issueKey="PROJ-42", spaceId=7, filename="runbook.docx",
        storageUri="s3://bucket/runbook.docx",
    )

    await IngestPipeline(settings, FakeEmbedder(8), store).handle_attachment(msg)

    assert [p.id for p in store.parents] == ["attachment:98:section#0"]
    parent = store.parents[0]
    assert parent.content.startswith("[Attachment: runbook.docx — issue PROJ-42]")
    assert "Step 0:" in parent.content and "Step 39:" in parent.content
    assert parent.page_number is None
    rollback_chunks = [c for c in store.chunks if "Step " in c.content]
    assert len(rollback_chunks) >= 2
    for chunk in store.chunks:
        expected = ["attachment:98:section#0"] if chunk in rollback_chunks else None
        assert chunk.provenance.get("section_ids") == expected
        assert chunk.provenance["source_type"] == "docx"
