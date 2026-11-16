-- The vector store's tables, in the `rag` schema of the shared PostgreSQL database.
-- Run as the schema owner / deployment administrator, never as the runtime rag role.
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.chunks (
    id TEXT PRIMARY KEY,
    embedding public.vector(1024) NOT NULL,
    chunk_type TEXT NOT NULL,
    issue_id BIGINT NOT NULL,
    issue_key TEXT NOT NULL,
    space_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL,
    chunk_index INT NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    page_number INT,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    CONSTRAINT chunks_page_number_positive CHECK (page_number IS NULL OR page_number >= 1)
);
CREATE INDEX IF NOT EXISTS idx_chunks_issue_id ON rag.chunks(issue_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON rag.chunks(chunk_type, source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON rag.chunks USING hnsw(embedding public.vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_attachment_page ON rag.chunks(source_id, page_number, chunk_index)
    WHERE chunk_type = 'attachment' AND page_number IS NOT NULL;

-- Parent-child retrieval: a whole section that several chunks were cut from. Not embedded; read by
-- id (chunks.provenance.section_ids) when one chunk is not enough to answer.
CREATE TABLE IF NOT EXISTS rag.chunk_parents (
    id TEXT PRIMARY KEY,
    chunk_type TEXT NOT NULL,
    issue_id BIGINT NOT NULL,
    issue_key TEXT NOT NULL,
    space_id BIGINT NOT NULL,
    source_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    page_end INT
);
CREATE INDEX IF NOT EXISTS idx_chunk_parents_source ON rag.chunk_parents(chunk_type, source_id);
CREATE INDEX IF NOT EXISTS idx_chunk_parents_issue_id ON rag.chunk_parents(issue_id);

CREATE TABLE IF NOT EXISTS rag.vlm_result_cache (
    cache_key TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS rag.ingestion_status (
    source_type TEXT NOT NULL,
    source_id BIGINT NOT NULL,
    space_id BIGINT,
    issue_key TEXT,
    source_label TEXT,
    state TEXT NOT NULL,
    stage TEXT,
    attempts INT NOT NULL DEFAULT 0,
    chunk_count INT,
    error_type TEXT,
    error_message TEXT,
    event_id TEXT,
    request_id TEXT,
    duration_ms INT,
    stage_ms JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_changed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(source_type, source_id)
);
CREATE INDEX IF NOT EXISTS idx_ingestion_status_state ON rag.ingestion_status(state, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ingestion_status_issue_key ON rag.ingestion_status(issue_key);
CREATE INDEX IF NOT EXISTS idx_ingestion_status_space ON rag.ingestion_status(space_id, state);
CREATE TABLE IF NOT EXISTS rag.vlm_retry_queue (
    attachment_id BIGINT PRIMARY KEY,
    issue_id BIGINT NOT NULL,
    space_id BIGINT,
    event_id TEXT NOT NULL,
    message JSONB NOT NULL,
    failed_calls INT NOT NULL,
    last_error TEXT,
    attempts INT NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL,
    queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_vlm_retry_queue_due ON rag.vlm_retry_queue(next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_vlm_retry_queue_issue ON rag.vlm_retry_queue(issue_id);
