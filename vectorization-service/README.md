# vectorization-service

The retrieval index for Sprint Orchestra. It consumes issue, comment and attachment change events
from Kafka, turns their text into chunks, embeds them and upserts them into pgvector (`pocdb.rag`) and
OpenSearch. It also owns the raw retrieval API over that index: `POST /search` combines dense (vector)
and lexical (BM25) search with Reciprocal Rank Fusion and a cross-encoder reranker.

Query-time logic (permission resolution beyond the `space_ids` a caller passes, answer synthesis,
re-retrieval, citation formatting) belongs to [`ai-service`](../ai-service), which calls `/search` as
a tool.

```
jira-backend ──(AFTER_COMMIT)──▶ Kafka: jira.content.ingestion ──▶ vectorization-service
  create/update/delete issue                jira.attachment.ingestion       │
  create/update/delete comment                                              ├─ HTML→text / Docling / OCR / VLM
  upload/delete attachment                                                  ├─ chunk (structure-aware)
                                                                            ├─ embed (text-embedding-3-small)
                                                                            └─ upsert → pgvector + OpenSearch
                                                                                        ▲
                                                          POST /search (vector ∪ BM25, RRF, rerank) ──▶ ai-service
```

## Run locally

```bash
# 1. Infrastructure, from the repository root: PostgreSQL, Kafka, OpenSearch, MinIO, LiteLLM
docker compose up -d --build

# 2. jira-backend with event publishing on
export APP_KAFKA_CONTENT_INGESTION_ENABLED=true
export APP_KAFKA_ATTACHMENT_INGESTION_ENABLED=true
./gradlew :jira-backend:bootRun

# 3. This service
cd vectorization-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # VEC_EMBEDDING_PROVIDER=fake runs without any model
uvicorn app.main:app --port 8100
```

Create or edit an issue in the app, then `curl localhost:8100/stats`.

To index content that existed before event publishing was on (or after a seed script, which writes
SQL directly):

```bash
python -m scripts.backfill_jira_backend --space-ids <id> --include-attachments
```

`--truncate-first` deletes only what the run writes back (the chosen spaces, and attachments only
with `--include-attachments`). Use it after an embedding-model change, since replaying unchanged text
re-embeds nothing.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Liveness, database ping, consumer status, OpenSearch `index_drift` |
| GET | `/stats` | Processed / skipped / failed counters, vector count, `vlm_retry_pending` |
| GET | `/metrics` | Prometheus metrics |
| POST | `/search` | `{query, space_ids, limit, mode}` → ranked chunks with citation metadata |
| POST | `/search/reindex` | Rebuild the OpenSearch chunk index from PostgreSQL |
| POST | `/embed` | `{text}` → `{embedding, dim}`; used by ai-service's answer cache |
| POST | `/issues/query` | Exact, space-scoped issue counts, breakdowns, filters and samples |
| POST | `/issues/history` | Issue transitions with time and event filters, including reopened detection |
| POST | `/issues/comments`, `/issues/details`, `/issues/attachments` | Complete, citation-bearing source text for known issue keys |
| POST | `/sections` | Whole parent sections by id (from a hit's `provenance.section_ids`) |
| POST | `/sprints/query` | Exact, space-scoped sprint state and counts |
| GET | `/index/watermark` | Opaque freshness token per space (see below) |
| GET | `/ingestion/health` | Per-state document counts and how many need attention |
| POST | `/ingestion/status` | Per-document status: state, stage, error, timings, attempts |

All endpoints are internal; keep them on the private network.

## Ingestion

### Events

- **Every content change is an event**: `issue_upserted` / `issue_deleted`, `comment_upserted` /
  `comment_deleted`, `attachment_uploaded` / `attachment_deleted`.
- **Deletes are explicit.** The index is a separate copy of the text, so database cascades never
  reach it. `issue_deleted` removes the issue's issue, comment and attachment chunks.
- **Upserts use deterministic chunk ids** (`issue:{id}`, `comment:{id}`, `attachment:{id}#{n}`), so
  Kafka's at-least-once redelivery overwrites in place. Identical content is detected and skips
  embedding.

### Text extraction

- **HTML** (issue and comment rich text) is stripped to plain text; inline images are indexed
  separately as attachments.
- **Documents** (PDF, DOCX, PPTX, XLSX, MD) are parsed with **Docling** to Markdown, which keeps
  reading order and renders tables in place. OCR and table-structure models are off by default
  (`VEC_DOCLING_DO_OCR`, `VEC_DOCLING_DO_TABLE_STRUCTURE`).
- **CSV, `.txt` and `.log`** are decoded directly.
- **Images** are read with **EasyOCR** on the full image, sorted into reading order.
- **VLM fallback** (`VEC_VLM_OCR_ENABLED`): when OCR drops low-confidence detections or reads nothing,
  the image goes to a vision-language model (`gemini-vlm` route), which can answer "illegible" instead
  of guessing.
- **PDF page fallback** (`VEC_VLM_PDF_ENABLED`): low-text and visual pages are rendered to PNG (200 DPI,
  at most 12 pages by default) and sent through the same VLM path. A fully scanned PDF is replaced by
  the VLM's page-labelled Markdown; a mixed document gets only the missing pages appended.
- VLM results are content-addressed in `vlm_result_cache`, so a Kafka retry never repeats a paid call.

### Chunking

- An issue (title + description) is one chunk; each comment is one chunk. Attachment text is split
  only where a section exceeds the budget (~800 tokens, ~120-token overlap inside a section).
- **Structure-aware splitting** (`VEC_CHUNK_STRATEGY=structural`, the default) cuts at the strongest
  boundary available — heading, blank line, newline, sentence end, then token window — and packs whole
  blocks up to the budget. Each chunk carries its heading breadcrumb
  (`Runbook: AtlasCart Payments Gateway > Rollback`). `VEC_CHUNK_STRATEGY=token` uses a plain token
  window.
- **PDFs are chunked as one document.** A section that crosses a page break stays together. Running
  headers and footers are dropped, extractor wrappers are removed, and heading levels are inferred
  from numbering (`3.` / `3.1` / `3.1.2`, `三、`, `第三章`). A chunk that spans pages stores
  `page_number` (first page) and `provenance.page_end`.
- **Tables that continue on the next page keep their header** (`carry_table_headers`) when the next
  page opens with a table of the same columns whose first row is data.
- **Parent sections.** A PDF section split into two or more chunks is also stored whole in
  `rag.chunk_parents` (sliced above `VEC_PARENT_MAX_TOKENS`, default 3000). Each of its chunks lists
  the parent ids in `provenance.section_ids`, and ai-service's `get_section` tool reads them through
  `POST /sections`.
- **Contextual retrieval** (`VEC_CONTEXTUAL_RETRIEVAL_ENABLED`, off by default): for a source that
  produces more than one chunk, a small model writes one sentence situating each chunk in its
  document, prepended before embedding (`app/ingest/contextualizer.py`). Generation failures leave the
  chunk unchanged and log a warning.

### Chunk metadata

Every chunk stores `space_id` (for permission filtering), `chunk_type`, `source_id`, `issue_key` and a
`provenance` object: PDF `page_number`, PPTX `slide_number`, XLSX `sheet_name` and `cell_range`, image
`bbox`. A locator is omitted when the format can't provide a stable one. `/search` and
`/issues/attachments` return the same metadata for citations and deep links; page-derived chunks start
with a label such as `[Attachment: invoice.pdf — issue ATC-46 — PDF page 2]`.

## Retrieval

`POST /search` modes:

| mode | What it does |
|------|--------------|
| `vector` | Cosine similarity in pgvector |
| `bm25` | Okapi BM25 in OpenSearch (`orchestra-chunks`, `app/search/chunk_index.py`) |
| `hybrid` (default) | Both, fused with Reciprocal Rank Fusion (`app/db/rrf.py`) |

RRF combines ranks rather than raw scores, because cosine distance and BM25 scores are on unrelated
scales; a chunk both retrievers rank highly beats either retriever's single top pick
(`tests/test_rrf.py`).

**Reranking** (`VEC_RERANK_ENABLED`, on by default): hybrid retrieval fetches a candidate pool
(`VEC_RERANK_CANDIDATE_POOL`, default 20), the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder
scores each `(query, chunk)` pair, and the top `limit` are returned. `VEC_RERANK_SCORE_THRESHOLD`
(unset by default) drops candidates below a cross-encoder score, so a query with nothing relevant can
return fewer results. Scores are raw logits, not 0-1, so tune any threshold against real queries.

**Consistency.** PostgreSQL is the source of truth for chunk text; OpenSearch is written right after
the PostgreSQL commit. The two writes are not atomic: `GET /healthz` reports `index_drift` and
`POST /search/reindex` repairs it.

**Embedding down.** A `hybrid` or `vector` request is answered with BM25 (still reranked) and returns
`degraded: true`, `mode: "bm25"`. Counted in `search_degraded_total{requested_mode}`.

## Structured queries

Counting, filtering and history questions can't be answered by top-k similarity search, so
`/issues/query`, `/issues/history` and `/sprints/query` read jira-backend's own tables through its
internal retrieval API (`/internal/retrieval/...`). The validated filters and authorized `space_ids`
are forwarded; counts, breakdowns and the row sample are read in one read-only repeatable-read
transaction. Archived spaces are excluded. Upstream failures return 502.

```bash
curl -s localhost:8100/issues/query -H 'content-type: application/json' \
  -d '{"space_ids":[5000014],"issue_types":["bug"]}' | jq '{total_count, counts_by_status}'
```

Configure `VEC_JIRA_BACKEND_URL` (local default `http://localhost:8081`) and
`VEC_INTERNAL_SERVICE_TOKEN` (the same value as jira-backend's `INTERNAL_SERVICE_TOKEN`). This service
holds no business-database credentials.

## Embeddings

`text-embedding-3-small` through the LiteLLM gateway:

```dotenv
VEC_EMBEDDING_PROVIDER=openai                 # the OpenAI-compatible wire format
VEC_EMBEDDING_MODEL=openai-embedding-3-small  # gateway route alias
VEC_OPENAI_BASE_URL=http://localhost:4000
VEC_OPENAI_API_KEY=sk-vec-service-local       # this service's LiteLLM virtual key
VEC_EMBEDDING_TRUNCATE_DIM=1024
```

The model emits 1536 dimensions; it is Matryoshka-trained, so vectors are truncated to 1024 and
re-normalized (`app/ingest/embedder.py`) to fit the `vector(1024)` column and its HNSW index. Don't
enable truncation for a model that wasn't trained this way. `VEC_EMBEDDING_PROVIDER=fake` gives
deterministic hash vectors for tests.

The VLM fallback (`gemini-vlm`) and contextual retrieval (`claude-context`) use the same gateway.

## Storage

One PostgreSQL instance (`orchestra-postgres`, port 5432): business records in `pocdb.public`, this
service's tables in `pocdb.rag` (`chunks`, `chunk_parents`, `vlm_result_cache`, `ingestion_status`,
`vlm_retry_queue`). The runtime pool uses `search_path=rag,public` and a restricted `rag` role with
no business-table privileges.

```dotenv
VEC_PG_DSN=postgresql://rag:rag123@localhost:5432/pocdb
```

New volumes get the schema from `schema/rag_schema.sql` and the role from
`docker/postgres/010_init_rag.sql`. On RDS, run `python -m scripts.bootstrap_rag` with
deployment-only credentials before starting workers.

## Index freshness token (`GET /index/watermark`)

ai-service keys its answer cache on this token: the latest `content_changed_at` and the document count
in `ingestion_status` for the requested spaces. It changes when searchable text changes (an upload, an
edited description, a delete) and not on status or sprint changes. Clients must treat it as opaque.

```bash
curl "localhost:8100/index/watermark?space_ids=5000018"
```

It carries no content, so it is not permission-checked; callers authorize their own `space_ids`.

## Index status tracking

Absence from `chunks` is ambiguous: a document has no chunks when it was never delivered, when the
file fetch failed, when parsing failed, when embedding failed, or when it is genuinely empty. So
ingestion writes one row per source into `ingestion_status`, keyed `(source_type, source_id)`, with
stage transitions written as they happen.

| state | meaning |
|-------|---------|
| `in_progress` | A stage is running, or the process died in it (see `stage` and `stale_after_seconds`) |
| `indexed` | Completed, chunks written |
| `empty` | Completed with zero chunks, correctly (a blank description, an image with nothing legible) |
| `skipped` | Deliberately not processed (over `attachment_max_bytes`) |
| `failed` | A stage raised; `stage` says where, `error_type` / `error_message` say what |
| `deleted` | A delete event removed the source's chunks |

Each row also has `attempts` (consecutive failed passes, reset on success), `stage_ms` (per-stage
timings) and `request_id` (joins logs and traces). Status writes are failure-isolated: a lost status
row never fails ingestion.

To check it from the command line:

```bash
curl -s localhost:8100/ingestion/health | jq

curl -s localhost:8100/ingestion/status -H 'content-type: application/json' \
  -d '{"states":["failed"]}' | jq '.rows[] | {source_label, state, stage, error_type, attempts}'

curl -s localhost:8100/ingestion/status -H 'content-type: application/json' \
  -d '{"issue_keys":["ATC-46"]}' | jq '.rows[] | {source_type, source_label, state, chunk_count}'

curl -s localhost:8100/ingestion/status -H 'content-type: application/json' \
  -d '{"stale_after_seconds":600}' | jq '.rows[] | {source_label, stage, updated_at}'
```

The `stage` column says who to look at: `fetch` is object storage, `parse` is Docling/OCR/VLM, `embed`
is the gateway or model, `store` is PostgreSQL/OpenSearch.

## When a model is down

The service never substitutes a different model: another embedding model's vectors wouldn't match the
index. It degrades instead.

- **Embedding, query side:** `/search` falls back to BM25 (see Retrieval).
- **Embedding, ingestion side:** the Kafka consumer doesn't commit; it rewinds and retries every 2 s.
  Nothing is lost.
- **VLM:** the attachment is indexed with its OCR/Docling text right away and queued in
  `vlm_retry_queue`. `app/ingest/vlm_retry.py` retries every `VEC_VLM_RETRY_INTERVAL_SECONDS` (600),
  starting with one probe; the wait per attachment doubles after each failure (capped at 6 h) up to
  `VEC_VLM_RETRY_MAX_ATTEMPTS` (8). A successful retry replaces the OCR chunks. Metrics:
  `ingest_vlm_degraded_attachments_total`, `ingest_vlm_retries_total{outcome}`.
- **OCR raising on a file** is recorded as `failed` at `parse` and not retried; a crash mid-parse
  leaves the Kafka offset uncommitted, so the event is redelivered.

## Metrics and tracing

`GET /metrics` exposes `http_request_duration_seconds` and, for `/search`, `search_embed_seconds`,
`search_retrieval_seconds{mode}` and `search_rerank_seconds`. Every request gets an `x-request-id`; the
`search completed` log line carries the same id and a `stage_ms` breakdown. With
`VEC_TRACING_ENABLED=true`, `search.*` spans join ai-service's trace.

## Tests

```bash
pytest   # chunking, RRF, pipeline idempotency, extraction fallbacks, ingestion status, VLM retry —
         # hermetic (fake embedder, in-memory store), no infrastructure
```

## Known limitations

- **At-most-once publish in jira-backend.** A crash between the database commit and the Kafka
  publish loses the event; an outbox or CDC publisher would fix it. This consumer is at-least-once.
- **No dead-letter topic.** Malformed messages are logged and skipped; transient failures are
  redelivered.
- **Reranker scores are raw cross-encoder logits**, useful for ordering but not as a probability.
- **The VLM retry lock is per process.** Several replicas need shared state or a Kafka-based retry.
