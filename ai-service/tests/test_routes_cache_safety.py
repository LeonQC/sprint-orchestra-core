"""Regression test for a bug found by a concurrency load test:
a semantic-cache backend failure (e.g. vectorization-service's /embed erroring or timing out under
load) must degrade POST /ask to "treat as a cache miss / skip the cache write," never crash the
whole request with an unhandled 500 — caching is an optimization, not a correctness dependency.
"""
from app.api.routes import (
    AskRequest,
    AskResponse,
    CitationOut,
    ModelUsageOut,
    _attachment_scoped_question,
    _safe_cache_get,
    _safe_cache_put,
    _safe_watermark,
    _write_outcome,
)


WATERMARK = "2026-09-03T10:00:00+00:00:100"


class _BrokenCache:
    async def get(self, question, space_ids, watermark, exact_only=False, execution_mode="single"):
        raise ConnectionError("embedding service unreachable")

    async def put(self, question, space_ids, response, watermark, execution_mode="single"):
        raise ConnectionError("embedding service unreachable")


class _WorkingCache:
    async def get(self, question, space_ids, watermark, exact_only=False, execution_mode="single"):
        return {"answer": "cached"}

    async def put(self, question, space_ids, response, watermark, execution_mode="single"):
        return None


class _BrokenWatermarkClient:
    async def watermark(self, space_ids):
        raise ConnectionError("vectorization-service unreachable")


class _WorkingWatermarkClient:
    async def watermark(self, space_ids):
        return WATERMARK


async def test_safe_cache_get_degrades_to_miss_on_backend_failure():
    result = await _safe_cache_get(_BrokenCache(), "question", [1], WATERMARK)
    assert result is None


async def test_safe_cache_put_swallows_backend_failure():
    await _safe_cache_put(_BrokenCache(), "question", [1], {"answer": "x"}, WATERMARK)  # no raise


async def test_safe_cache_get_passes_through_a_real_hit():
    result = await _safe_cache_get(_WorkingCache(), "question", [1], WATERMARK)
    assert result == {"answer": "cached"}


async def test_unavailable_watermark_bypasses_the_cache_rather_than_reusing_unchecked():
    """The one place this module's "degrade to a miss" rule must NOT be read as "degrade to a hit".

    A missing freshness token means the staleness check is unavailable, and the callers turn None
    into "compute the answer". Failing the other way would reuse answers without checking them,
    precisely when the system is already unhealthy.
    """
    assert await _safe_watermark(_BrokenWatermarkClient(), [1]) is None
    assert await _safe_watermark(_WorkingWatermarkClient(), [1]) == WATERMARK


def test_write_outcome_labels_distinguish_the_reasons_an_answer_was_not_cached():
    """A 0% hit rate has several causes; these labels are what tells them apart on the dashboard."""
    assert _write_outcome(True, "search_knowledge_base", False) == "stored"
    assert _write_outcome(True, "query_issues", False) == "skipped_route"
    assert _write_outcome(True, "search_knowledge_base", False, used_live_data=True) == "skipped_route"
    assert _write_outcome(True, "search_knowledge_base", True) == "skipped_abstained"
    # A follow-up, an attachment-scoped question, or an unavailable watermark: the read side already
    # declined, so the route is not the reason and must not be reported as one.
    assert _write_outcome(False, "search_knowledge_base", False) == "skipped_request"


def test_attachment_scope_serializes_in_camel_case_and_routes_without_answer_leakage():
    request = AskRequest(
        question="What is the title of this page?",
        space_ids=[7],
        attachment_issue_keys=["atc-46", "ATC-46"],
    )

    payload = request.model_dump(by_alias=True)
    assert payload["attachmentIssueKeys"] == ["atc-46", "ATC-46"]
    scoped = _attachment_scoped_question(request.question, request.attachment_issue_keys)
    assert scoped == "Using only the attachments on ATC-46, What is the title of this page?"


def test_citation_serializes_the_optional_pdf_page_in_camel_case():
    citation = CitationOut(
        issue_key="ATC-46", chunk_type="attachment", source_id=5000940,
        content="Invoice total: $42", page_number=2,
    )
    assert citation.model_dump(by_alias=True)["pageNumber"] == 2


def test_ask_response_serializes_per_model_usage_in_camel_case():
    response = AskResponse(
        answer="ATC-432",
        abstained=False,
        retrieval_rounds=1,
        queries_used=[],
        citations=[],
        input_tokens=2272,
        output_tokens=242,
        estimated_cost_usd=0.00048,
        route_source="model",
        route_tool="query_issues",
        model_usage=[ModelUsageOut(
            role="router", provider="Self-hosted", model="self-hosted-router", deployment="local",
            input_tokens=680, output_tokens=92, estimated_cost_usd=0,
            pricing_status="free_local",
        )],
    )

    payload = response.model_dump(by_alias=True)
    assert payload["routeSource"] == "model"
    assert payload["routeTool"] == "query_issues"
    assert payload["modelUsage"][0] == {
        "role": "router",
        "provider": "Self-hosted",
        "model": "self-hosted-router",
        "deployment": "local",
        "inputTokens": 680,
        "outputTokens": 92,
        "cachedInputTokens": 0,
        "estimatedCostUsd": 0.0,
        "pricingStatus": "free_local",
    }
