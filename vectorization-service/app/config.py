"""Runtime configuration from environment variables.

Every value has a local-dev default so `uvicorn app.main:app` works against the Docker Compose stack
as-is. Override in production via env vars.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Dev credentials that must never reach a deployed environment (same guard as ai-service/app/config.py).
_LOCAL_ONLY_DEFAULTS = {
    "pg_dsn": "postgresql://rag:rag123@localhost:5432/pocdb",
    "s3_access_key": "minioadmin",
    "s3_secret_key": "minioadmin",
    "openai_api_key": "sk-vec-service-local",
    "vlm_openai_api_key": "sk-vec-service-local",
    "contextual_openai_api_key": "sk-vec-service-local",
    "internal_service_token": "dev-internal-service-token-min-32-chars-long!!",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEC_", env_file=".env", extra="ignore")

    # --- Service ---
    service_name: str = "vectorization-service"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Distributed tracing (app/tracing.py) ---
    # Off by default (needs a collector). When on, spans join the caller's trace via W3C
    # `traceparent`, so you can see e.g. "5.2s of this 8s request was the reranker".
    tracing_enabled: bool = False
    tracing_collector_endpoint: str = "http://localhost:4317"
    # Blank = use service_name. Set it to tell several instances apart.
    tracing_service_name: str = ""
    # Anything other than "local" makes the dev-credential check below fatal.
    environment: str = "local"  # local | staging | production

    # Same business database, restricted rag role and schema; never a second PostgreSQL instance.
    pg_dsn: str = "postgresql://rag:rag123@localhost:5432/pocdb"
    pg_pool_min_size: int = 1
    pg_pool_max_size: int = 8

    # Authoritative structured reads; keep business database credentials inside jira-backend.
    jira_backend_url: str = "http://localhost:8081"
    internal_service_token: str = "dev-internal-service-token-min-32-chars-long!!"

    # --- OpenSearch (the BM25 half of hybrid search; see app/search/chunk_index.py) ---
    opensearch_url: str = "http://localhost:9200"
    opensearch_chunk_index: str = "orchestra-chunks"
    # Fail fast so a stuck OpenSearch shows up as a retrieval error, not a hung /search.
    opensearch_request_timeout_seconds: float = 10.0

    # --- Kafka ---
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_content_topic: str = "jira.content.ingestion"
    kafka_attachment_topic: str = "jira.attachment.ingestion"
    kafka_group_id: str = "vectorization-service"
    # "earliest" so a backlog produced before the consumer first ran still gets ingested.
    kafka_auto_offset_reset: str = "earliest"
    # Needs object-storage creds to fetch binaries, so off until those are set.
    kafka_consume_attachments: bool = False

    # --- Embeddings ---
    # OpenAI wire format, sent to the LiteLLM gateway (:4000). "openai" names the API format, so it also
    # covers any OpenAI-compatible server.
    embedding_provider: str = "openai"  # openai (any OpenAI-compatible server) | fake
    embedding_model: str = "openai-embedding-3-small"  # LiteLLM route alias -> text-embedding-3-small
    # Width of the vector(N) column in schema/rag_schema.sql.
    embedding_dim: int = 1024
    # This service's LiteLLM virtual key (scripts/litellm_provision_keys.sh).
    openai_api_key: str = "sk-vec-service-local"
    # Gateway by default so every call is rate-capped and traced in one place.
    openai_base_url: str = "http://localhost:4000"
    # Small on purpose for a single-instance embedding server, which handles one request at a time: 64
    # chunks once held its queue ~37s, past ai-service's 30s timeout, and Ask AI answered as if the
    # index were empty. The hosted endpoint batches concurrently, so this can be raised there.
    embedding_batch_size: int = 8
    embedding_max_retries: int = 3
    # Matryoshka truncation: keep the first N dims and re-normalize. Needed because pgvector's HNSW
    # index caps at 2000 dims; text-embedding-3-small emits 1536. Only valid for MRL-trained models;
    # set None for a model whose output already fits embedding_dim or that isn't MRL-trained.
    embedding_truncate_dim: int | None = 1024

    # --- Chunking ---
    # 800/130 beat 300 and 500 on all 4 RAGAS metrics for this corpus (tables fragment at small sizes).
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    # "structural" splits at headings/paragraphs/sentences and keeps the heading path; "token" is the
    # old fixed-window splitter, kept as a rollback switch. See app/ingest/chunker.py.
    chunk_strategy: str = "structural"  # structural | token
    # Parent-child retrieval: a document section cut into several chunks is also stored whole, so the agent
    # can read it (get_section) when one chunk is not enough. A longer section is stored in slices of
    # this size. ~3000 tokens is about 2 pages.
    parent_max_tokens: int = 3000

    # --- Per-document ingestion status (app/ingest/status.py) ---
    # The only way to answer "where did this attachment die?". Cheap, and failures are isolated.
    ingestion_status_enabled: bool = True

    # --- Contextual retrieval (https://www.anthropic.com/news/contextual-retrieval) ---
    # Off by default: one LLM call per chunk, only worth it for long multi-chunk attachments.
    contextual_retrieval_enabled: bool = False
    contextual_llm_provider: str = "openai_compatible"  # openai_compatible (the gateway) | anthropic
    contextual_anthropic_api_key: str = ""
    # A small model: this runs once per chunk at ingest time.
    contextual_anthropic_model: str = "claude-haiku-4-5"
    contextual_openai_base_url: str = "http://localhost:4000"
    contextual_openai_api_key: str = "sk-vec-service-local"
    contextual_openai_model: str = "claude-context"  # LiteLLM route alias -> claude-haiku-4-5

    # --- Reranking (cross-encoder over the fused candidate pool; app/db/reranker.py) ---
    # On by default: load tests measured only ~10ms P50 / ~50ms P95 extra at 50 concurrent requests.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Candidates handed to the reranker. Must be >= the largest `limit` callers ask for, or the
    # reranker can't promote anything from further down.
    rerank_candidate_pool: int = 20
    # Optional minimum rerank score. None = only reorder. This model outputs unbounded logits, not
    # 0-1 scores, so tune on real data instead of borrowing a cosine-style 0.65.
    rerank_score_threshold: float | None = None

    # --- Object storage (attachment binaries; MinIO locally) ---
    # On AWS set the endpoint and both keys to empty: real S3 with the ECS task role.
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_region: str = "us-east-1"

    # --- Attachment parsing (Docling) ---
    docling_do_ocr: bool = False  # OCR is slow; enable when attachments contain scans/screenshots
    docling_do_table_structure: bool = False
    attachment_max_bytes: int = 25 * 1024 * 1024  # skip binaries larger than this
    # Drop EasyOCR detections below this confidence. On a degraded test image, real words scored
    # 0.7-1.0 and garbage 0.006-0.06; the agent had presented one garbage fragment as a real SKU.
    # 0.4 sits in that gap. Not tuned across many image types.
    ocr_min_confidence: float = 0.4

    # PDF path only; standalone images always use EasyOCR directly. Picked by measurement
    # (31 facts on a scanned doc): easyocr 23, rapidocr 20,
    # tesseract 18. Apple Vision scored 27 but is macOS-only and crashed on another doc, so it was
    # dropped.
    docling_ocr_engine: str = "easyocr"  # default | easyocr | tesseract | rapidocr
    # "accurate" made no difference on that fixture (23 vs 23) and costs ~25% more parse time.
    docling_table_mode: str = "fast"  # fast | accurate

    # --- VLM fallback (app/ingest/vlm_describer.py) ---
    # The OCR threshold can only turn a wrong value into a missing one; a vision model can actually
    # read it or say it's illegible. Off by default (one paid call per escalated image), and only
    # used when EasyOCR shows signs of struggling.
    vlm_ocr_enabled: bool = False
    vlm_provider: str = "openai_compatible"  # anthropic | openai_compatible
    vlm_anthropic_api_key: str = ""
    vlm_anthropic_model: str = "claude-opus-4-8"  # needs vision; Haiku also supports it if cost matters more than accuracy here
    vlm_openai_base_url: str = "http://localhost:4000"
    vlm_openai_api_key: str = "sk-vec-service-local"
    # LiteLLM route alias -> gemini-3.8-flash. Not gpt-5.6-luna: luna returned a ~195-char fragment of a
    # dense page on every run (see claude-vlm's comment in litellm-gateway). Gemini transcribed that
    # page in full 3/3 and got 40/40 CM4 prices right.
    # claude-vlm (Sonnet) did as well but costs more; switched when the Anthropic credit ran out.
    vlm_openai_model: str = "gemini-vlm"
    # Render PDF pages Docling likely can't read (conditions A and B below) and send them to the VLM.
    # On by default: with a VLM that transcribes whole pages, MP-DocVQA (146q) went 0.445 -> 0.753
    # (p < 1e-6, replicated; measured with the earlier self-hosted VLM, not yet re-run on claude-vlm).
    # An earlier revert had measured a weaker VLM, not this feature. Two "only escalate hard pages"
    # routers were tried and both scored worse — escalating more pages scored better (2.1% -> 0.596,
    # 62.5% -> 0.647, 100% -> 0.667), because a form flattened by Docling leaves no visible trace to
    # detect. Cost is one VLM call per page, capped by vlm_pdf_max_pages; set false to opt out.
    vlm_pdf_enabled: bool = True
    vlm_pdf_render_dpi: int = 200
    # Covers real Jira attachment sizes (AtlasCart PDFs are 3-4 pages). Pages past the cap keep their
    # Docling text, and render_pdf_candidates logs which pages were skipped.
    vlm_pdf_max_pages: int = 15
    # Condition A: fewer embedded chars than this means the page is a picture of text. Known miss: a
    # court filing with a stamped 83-char docket header on every scanned page — condition B catches
    # those, which is why B isn't redundant. 80 sits in the widest gap of that cohort (..., 63 | 83, ...).
    vlm_pdf_min_text_chars: int = 80
    # Condition B: also escalate pages with charts/diagrams/image tables, even if they have text.
    # Measured on born-digital PDFs (112q): retrieval_correct 0.833 -> 0.937, 15 fixed vs 1 broken,
    # p = 0.0005.
    vlm_pdf_include_visual_pages: bool = True
    # Narrow trigger: escalate only pages where a Docling table cell is just a parenthetical with its
    # preceding line dropped.
    # Independent of vlm_pdf_enabled. A false positive only re-reads a page that was already suspect.
    vlm_pdf_table_defect_detection_enabled: bool = True
    # Narrow trigger: escalate a page when too much of Docling's own OCR is low-confidence (per-cell
    # bar = ocr_min_confidence). One noisy cell is normal, so a page needs this fraction below the bar.
    vlm_pdf_ocr_confidence_detection_enabled: bool = True
    vlm_pdf_low_confidence_page_ratio: float = 0.15
    # Narrow trigger: a table column that's filled in most rows but blank in one — Docling dropping a
    # value (seen live: a director's name missing from an attendance table). Zero false positives in
    # a 245-chunk sweep. Off by default anyway: escalation replaces the whole page, so the result
    # depends on the VLM. MP-DocVQA (146q, McNemar): gpt-5.6-luna 3 fixed / 18 broken (p=0.0015); the
    # earlier self-hosted VLM 14/4 and 16/5 (p~0.03). Enable only with a VLM that transcribes whole
    # pages. gemini-vlm does, but this trigger has not been re-measured with it.
    vlm_pdf_sparse_column_detection_enabled: bool = False
    # Escalate every page that does NOT have a Docling-parsed table (docling_parser.has_recognized_table).
    # Blanket escalation helped MP-DocVQA (0.514 -> 0.753) but hurt the clean regression set
    # (0.889 -> 0.852) by re-transcribing tables Docling had already parsed correctly. This keeps
    # the gain and skips those pages. Off until tested on the 81q set (MP-DocVQA is where the
    # supporting evidence came from, so it can't confirm it).
    vlm_pdf_difficulty_routing_enabled: bool = False
    # Images: escalate on column layout, not low confidence — a crisp table OCRs perfectly but comes
    # out as an unusable flat word stream (see docling_parser._column_row_ratio).
    vlm_image_column_detection_enabled: bool = True
    # Fraction of rows that must look like table/form rows (3+ columns, or 4+ aligned 2-column rows,
    # so one right-aligned date doesn't trigger). Low enough that prose plus one table still escalates.
    vlm_image_column_row_ratio: float = 0.15
    # Only accept the VLM text if it covers enough of what OCR found (applied when OCR found >= 500
    # chars). Seen live: OCR 2,259 chars with the answer, VLM 219 — replacing would lose ~90%.
    vlm_min_ocr_coverage_ratio: float = 0.5
    vlm_max_completion_tokens: int = 6_000
    # gpt-5.6-luna's default reasoning used the whole 2000-token budget before writing anything.
    # "none" is wrong here (table extraction benefits from a little reasoning); "low" was verified to
    # leave room for the full table.
    vlm_reasoning_effort: str = "low"

    # --- VLM retry (app/ingest/vlm_retry.py) ---
    # Attachments indexed with the OCR text because the VLM call failed are queued and re-ingested
    # once the VLM answers again. Only runs when vlm_ocr_enabled.
    vlm_retry_enabled: bool = True
    # First retry this long after the failure, then doubling per attempt (capped at 6 hours).
    vlm_retry_interval_seconds: int = 600
    vlm_retry_max_attempts: int = 8
    vlm_retry_batch_size: int = 20

    @model_validator(mode="after")
    def _reject_local_defaults_outside_local(self) -> "Settings":
        """Fail startup if a deployed environment is still using a development credential."""
        if self.environment.strip().lower() == "local":
            return self
        offenders = [
            f"VEC_{name.upper()}"
            for name, dev_value in _LOCAL_ONLY_DEFAULTS.items()
            if getattr(self, name, None) == dev_value
        ]
        if offenders:
            raise ValueError(
                f"environment={self.environment!r} but these still hold their local development "
                f"defaults: {', '.join(offenders)}. Set them from the deployment's secret store, "
                f"or set VEC_ENVIRONMENT=local if this really is a laptop."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
