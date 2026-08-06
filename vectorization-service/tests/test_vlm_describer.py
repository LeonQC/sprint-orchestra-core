"""Tests for app/ingest/vlm_describer.py — no real network, mocked httpx/anthropic transports, same
approach as tests/test_contextual_retrieval.py uses for the analogous contextualizer.py.
"""
import json

import httpx
import pytest

from app.config import Settings
from app.ingest.vlm_describer import (
    NoopVlmImageDescriber,
    OpenAICompatibleVlmImageDescriber,
    build_vlm_describer,
    vlm_cache_namespace,
)


async def test_noop_describer_never_calls_out_and_returns_empty():
    describer = NoopVlmImageDescriber()
    assert await describer.describe(b"fake image bytes", "image/png") == ""
    await describer.aclose()  # must not raise


async def test_build_vlm_describer_returns_noop_when_disabled():
    settings = Settings(vlm_ocr_enabled=False)
    describer = build_vlm_describer(settings)
    assert isinstance(describer, NoopVlmImageDescriber)


async def test_build_vlm_describer_rejects_an_unknown_provider():
    settings = Settings(vlm_ocr_enabled=True, vlm_provider="not-a-real-provider")
    with pytest.raises(ValueError, match="unknown vlm_provider"):
        build_vlm_describer(settings)


class _FakeTransport(httpx.AsyncBaseTransport):
    def __init__(self, response_body: dict, status_code: int = 200):
        self._body = response_body
        self._status = status_code
        self.sent_requests = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.sent_requests.append(request)
        return httpx.Response(self._status, json=self._body)


async def test_openai_compatible_describer_sends_the_image_as_a_data_uri_and_returns_the_content():
    fake = _FakeTransport({"choices": [{"message": {"content": "SKU: W-7734"}}]})
    describer = OpenAICompatibleVlmImageDescriber("http://localhost:1234/v1", "local", "some-vlm")
    describer._client = httpx.AsyncClient(
        base_url="http://localhost:1234/v1", transport=fake
    )

    result = await describer.describe(b"\x89PNG raw bytes", "image/png")

    assert result == "SKU: W-7734"
    sent = json.loads(fake.sent_requests[0].content)
    assert sent["model"] == "some-vlm"
    image_part = next(p for p in sent["messages"][0]["content"] if p["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    await describer.aclose()


async def test_openai_compatible_describer_warns_when_the_transcription_was_truncated(caplog):
    """A budget-exhausted transcription is the one VLM failure that looks like a success.

    `finish_reason="length"` means the model stopped mid-page: the description is non-empty and
    well-formed, so every caller accepts it, and the rest of the document is silently absent from the
    index forever. Without this signal the only symptom is a later, confident "the document does not
    mention X" about a document that does.
    """
    fake = _FakeTransport(
        {"choices": [{"message": {"content": "## Invoice\n| Item | Qty |\n|---|---|\n| Widget"},
                      "finish_reason": "length"}]}
    )
    describer = OpenAICompatibleVlmImageDescriber(
        "http://localhost:1234/v1", "local", "some-vlm", max_completion_tokens=2_000
    )
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    with caplog.at_level("WARNING"):
        result = await describer.describe(b"\x89PNG", "image/png")

    # The partial text is still returned — a truncated page beats no page; the point is that it is
    # no longer SILENT.
    assert result.startswith("## Invoice")
    assert "TRUNCATED" in caplog.text
    await describer.aclose()


async def test_openai_compatible_describer_is_quiet_on_a_complete_transcription(caplog):
    fake = _FakeTransport({"choices": [{"message": {"content": "SKU: W-7734"},
                                        "finish_reason": "stop"}]})
    describer = OpenAICompatibleVlmImageDescriber("http://localhost:1234/v1", "local", "some-vlm")
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    with caplog.at_level("WARNING"):
        assert await describer.describe(b"\x89PNG", "image/png") == "SKU: W-7734"

    assert "TRUNCATED" not in caplog.text
    await describer.aclose()


async def test_openai_compatible_describer_sends_max_completion_tokens_not_max_tokens():
    # Regression test for a real bug hit live: this project's own configured production model (gpt-5.6-luna)
    # rejects the older `max_tokens` param with a 400. This module builds its own httpx client (it
    # isn't an instructor call, so it doesn't get instructor_client.py's shared rewrite transport),
    # so the request has to use the right name from the start.
    fake = _FakeTransport({"choices": [{"message": {"content": "x"}}]})
    describer = OpenAICompatibleVlmImageDescriber(
        "http://localhost:1234/v1", "local", "some-vlm", max_completion_tokens=2_000
    )
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    await describer.describe(b"bytes", "image/png")

    sent = json.loads(fake.sent_requests[0].content)
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == 2_000
    await describer.aclose()


async def test_openai_compatible_describer_sends_reasoning_effort():
    # Regression for a real live failure: gpt-5.6-luna, a reasoning-capable hosted model, spent its
    # ENTIRE max_completion_tokens budget on hidden reasoning for a dense table image and returned
    # empty content with finish_reason="length" — confirmed against the real API (reasoning_tokens
    # 2000/2000). `_maybe_escalate_to_vlm` treats empty content as "fall back to the OCR text", so
    # this failed with no error at all: a paid call that silently kept the worse result. Sending
    # `reasoning_effort` (verified live: "low" leaves ~1700 tokens for real content) is the fix.
    fake = _FakeTransport({"choices": [{"message": {"content": "x"}}]})
    describer = OpenAICompatibleVlmImageDescriber(
        "http://localhost:1234/v1", "local", "some-vlm", reasoning_effort="low"
    )
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    await describer.describe(b"bytes", "image/png")

    sent = json.loads(fake.sent_requests[0].content)
    assert sent["reasoning_effort"] == "low"
    await describer.aclose()


async def test_openai_compatible_describer_never_sends_temperature():
    # Regression lock for a real live break: this project's configured hosted model (gpt-5.6-luna)
    # hard-rejects any explicit `temperature` value with a 400 ("Only the default (1) value is
    # supported") — a known behavior of reasoning-tuned OpenAI-family models. A determinism-motivated
    # attempt to pin temperature=0 here (mirroring AnthropicVlmImageDescriber and contextualizer.py)
    # made every real call fail and silently fall back to worse OCR/Docling-only text (both
    # `_maybe_escalate_to_vlm` and `_pdf_sections_with_vlm` swallow exceptions and use the fallback
    # text) — 12/12 single-page image escalations broke in one real ingestion run before this was
    # caught. This test locks in "don't send it" so that regression can't silently come back.
    fake = _FakeTransport({"choices": [{"message": {"content": "x"}}]})
    describer = OpenAICompatibleVlmImageDescriber("http://localhost:1234/v1", "local", "some-vlm")
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    await describer.describe(b"bytes", "image/png")

    sent = json.loads(fake.sent_requests[0].content)
    assert "temperature" not in sent
    await describer.aclose()


def test_vlm_cache_namespace_changes_with_reasoning_effort():
    # A cached result from before this fix must not be reused after it — otherwise the empty/degraded
    # extraction the bug produced could sit in the durable cache indefinitely, silently outliving the
    # code fix that was supposed to correct it.
    settings = Settings(vlm_provider="openai_compatible", vlm_openai_model="m", vlm_reasoning_effort="low")
    other = Settings(vlm_provider="openai_compatible", vlm_openai_model="m", vlm_reasoning_effort="none")

    assert vlm_cache_namespace(settings) != vlm_cache_namespace(other)
    assert "low" in vlm_cache_namespace(settings)


async def test_build_vlm_describer_wires_reasoning_effort_from_settings():
    settings = Settings(
        vlm_ocr_enabled=True, vlm_provider="openai_compatible", vlm_reasoning_effort="low",
    )
    describer = build_vlm_describer(settings)

    assert isinstance(describer, OpenAICompatibleVlmImageDescriber)
    assert describer._reasoning_effort == "low"


async def test_openai_compatible_describer_raises_on_a_non_200_so_the_caller_can_fall_back():
    # docling_parser.py's _maybe_escalate_to_vlm relies on this raising rather than silently
    # returning "" — an HTTP error must be distinguishable from "the model legitimately found no text."
    fake = _FakeTransport({"error": "server error"}, status_code=500)
    describer = OpenAICompatibleVlmImageDescriber("http://localhost:1234/v1", "local", "some-vlm")
    describer._client = httpx.AsyncClient(base_url="http://localhost:1234/v1", transport=fake)

    with pytest.raises(httpx.HTTPStatusError):
        await describer.describe(b"bytes", "image/png")
    await describer.aclose()
