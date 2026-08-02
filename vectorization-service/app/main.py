"""FastAPI entrypoint for vectorization-service. The lifespan wires the Postgres pool, vector store,
embedder, ingest pipeline and the Kafka consumer (stopped cleanly on shutdown).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db.pool import create_pool
from app.db.reranker import build_reranker
from app.db.vector_store import VectorStore
from app.ingest.contextualizer import build_context_generator
from app.ingest.embedder import build_embedder
from app.ingest.pipeline import IngestPipeline
from app.ingest.status import build_ingestion_tracker
from app.ingest.vlm_cache import CachedVlmImageDescriber
from app.ingest.vlm_describer import build_vlm_describer, vlm_cache_namespace
from app.ingest.vlm_retry import VlmRetryWorker
from app.kafka.consumer import IngestionConsumer
from app.logging_config import configure_logging
from app.observability import ObservabilityMiddleware, metrics_response
from app.search.factory import build_chunk_search_index
from app.structured_client import StructuredRetrievalClient
from app.tracing import configure_tracing, instrument_app

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    # Sets the real TracerProvider. `instrument_app` runs at import time instead (see below):
    # FastAPIInstrumentor must wrap the middleware stack before it's built, which happens before this
    # body runs. ProxyTracer connects the two.
    configure_tracing(settings)
    logger.info("starting %s", settings.service_name)

    pool = await create_pool(settings)
    embedder = build_embedder(settings)
    context_generator = build_context_generator(settings)
    search_index = build_chunk_search_index(settings)
    # Create the index before the consumer starts writing. If OpenSearch auto-created it from the first
    # document, it would guess the mapping and skip our analyzer — a silent ranking regression.
    await search_index.ensure_index()
    store = VectorStore(pool, search_index)
    raw_vlm_describer = build_vlm_describer(settings)
    # The cache is intentionally wrapped at the service boundary, not inside parsing: a successful
    # VLM response is persisted before the later embedding/vector-store steps can fail, so Kafka
    # redelivery after a crash reuses it rather than issuing another paid model call.
    vlm_describer = (
        CachedVlmImageDescriber(raw_vlm_describer, store, vlm_cache_namespace(settings))
        if settings.vlm_ocr_enabled
        else raw_vlm_describer
    )
    tracker = build_ingestion_tracker(settings, store)
    vlm_retry_active = settings.vlm_ocr_enabled and settings.vlm_retry_enabled
    pipeline = IngestPipeline(
        settings, embedder, store, context_generator, vlm_describer, tracker,
        vlm_retry_queue=store if vlm_retry_active else None,
    )
    consumer = IngestionConsumer(settings, pipeline)
    vlm_retry_worker = VlmRetryWorker(settings, pipeline, store) if vlm_retry_active else None
    reranker = build_reranker(settings)

    app.state.settings = settings
    app.state.pool = pool
    app.state.embedder = embedder
    app.state.store = store
    app.state.structured = StructuredRetrievalClient(
        settings.jira_backend_url, settings.internal_service_token,
    )
    app.state.consumer = consumer
    app.state.reranker = reranker
    app.state.search_index = search_index
    app.state.vlm_retry_worker = vlm_retry_worker

    await consumer.start()
    if vlm_retry_worker:
        vlm_retry_worker.start()
    try:
        yield
    finally:
        if vlm_retry_worker:
            await vlm_retry_worker.stop()
        await consumer.stop()
        await search_index.aclose()
        await embedder.aclose()
        await context_generator.aclose()
        await vlm_describer.aclose()
        await app.state.structured.aclose()
        await pool.close()
        logger.info("stopped %s", settings.service_name)


app = FastAPI(title="Vectorization Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(ObservabilityMiddleware)
# Demo-only convenience, not an access-control boundary. This service has no end-user authentication
# because it is an internal dependency of ai-service; keep :8100 private and never give it a public
# route.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(router)

# At import time, not in `lifespan`: the middleware stack is already built by then, so instrumenting
# there added no server span and ai-service's traceparent had nothing to attach to.
instrument_app(app, get_settings())


@app.get("/metrics")
async def metrics():
    return metrics_response()
