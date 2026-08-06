"""Vision-model (VLM) fallback for image attachments EasyOCR struggled with.

On degraded photos EasyOCR doesn't fail loudly — it outputs confident-looking wrong tokens ("Mx" for
"SKU: Pallet") that the agent then presents as real values. A confidence threshold only turns wrong
into missing. A VLM sees the whole image and can say "illegible" instead of guessing, which is why
production RAG tools (LlamaParse, Unstructured) moved this way.

Off by default (`VEC_VLM_OCR_ENABLED=false`, a paid call per image), and only used when OCR showed
signs of struggling (`_maybe_escalate_to_vlm`); most images here are clean screenshots.
"""
from __future__ import annotations

import base64
import logging
from typing import Protocol

import anthropic
import httpx

logger = logging.getLogger(__name__)

# Tells the model to mark illegible regions instead of guessing — the whole reason to use a VLM.
# v3 also summarizes photos (not just charts/diagrams): 17 of 42 wrong born-digital answers were about
# picture content, and with v2 a photo produced no text at all.
# Part of the VLM cache key, so bump the version with any prompt change, or cached v2 results are
# served and the comparison is fake.
_VLM_PROMPT_VERSION = "document-extraction-v3"

_VLM_PROMPT = """Extract high-fidelity, searchable reference content from this attachment image. The \
result will be embedded in a RAG system, so accuracy, source fidelity, and preserving structure matter \
more than prose quality.

Rules:
- Transcribe text you can read with real confidence exactly as written, including numbers, codes, \
and units.
- If part of the image is blurry, rotated, low-contrast, or otherwise illegible, write \
"[illegible: <what field or region this appears to be>]" for that part instead of guessing a plausible \
-looking value. A wrong value is worse than an honest gap — never invent one.
- If the image contains a table or a list of label/value pairs, preserve that structure as one \
"label: value" line per row.
- Render tables as Markdown when their rows and columns are readable. Do not invent missing cells.
- For ANY non-text visual — chart, diagram, workflow, photograph, illustration, map, icon — add a \
short `Visual summary:` describing only directly observable entities, relationships, labels, values, \
and trends. For a photograph or illustration, state the countable facts plainly: how many people are \
visible, what they are doing or wearing if plainly visible, the setting, and any legible signage or \
text within the image. Do not infer causes, intent, identity, emotion, or values that are not visible.
- Keep headings and sections in reading order. Do not add introductions or conclusions.
- If literally no text is visible anywhere in the image, respond with exactly: (no text found)"""


def _warn_if_truncated(stop_reason: str | None, budget_marker: str, text: str, budget: int) -> None:
    """Log when the model ran out of output tokens mid-transcription.

    The one VLM failure that looks fine: the text is non-empty and well-formed, just cut off, so the
    page is indexed half-done and later "the document doesn't mention X" answers are wrong. A log, not
    an error or retry — a partial page beats no page; it means `vlm_max_completion_tokens` is too small
    for your documents.
    """
    if stop_reason != budget_marker:
        return
    logger.warning(
        "VLM transcription hit the output token budget and is TRUNCATED — the tail of this page was "
        "not extracted and will be missing from the index; raise vlm_max_completion_tokens",
        extra={"max_completion_tokens": budget, "returned_chars": len(text), "stop_reason": stop_reason},
    )


class VlmImageDescriber(Protocol):
    async def describe(self, image_data: bytes, mime_type: str) -> str: ...

    async def aclose(self) -> None: ...


class NoopVlmImageDescriber:
    """Used when VLM escalation is disabled — never calls out, always returns ''."""

    async def describe(self, image_data: bytes, mime_type: str) -> str:
        return ""

    async def aclose(self) -> None:
        return None


class AnthropicVlmImageDescriber:
    def __init__(self, api_key: str, model: str, max_completion_tokens: int = 500):
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self._model = model
        self._max_completion_tokens = max_completion_tokens

    async def describe(self, image_data: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_data).decode()
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_completion_tokens,
            # temperature=0: two runs on the same page gave different digits ("Rs.76,90,22,489" vs
            # "Rs.76,80,22,489") and different made-up URLs — a correctness bug.
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        _warn_if_truncated(response.stop_reason, "max_tokens", text, self._max_completion_tokens)
        return text

    async def aclose(self) -> None:
        await self._client.close()


class OpenAICompatibleVlmImageDescriber:
    """The production path: a hosted vision model (e.g. gpt-5.6-luna) via OpenAI chat/completions with
    an `image_url` part.
    """

    def __init__(
        self, base_url: str, api_key: str, model: str, max_completion_tokens: int = 500,
        reasoning_effort: str = "low",
    ):
        self._model = model
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        )

    async def describe(self, image_data: bytes, mime_type: str) -> str:
        b64 = base64.b64encode(image_data).decode()
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                # `max_completion_tokens`, not `max_tokens` — gpt-5.6-luna rejects the old name (Case
                # Studies 28, 30). No shared transport here, so the request is just built right.
                "max_completion_tokens": self._max_completion_tokens,
                # Otherwise the model can spend the whole budget on hidden reasoning and return empty
                # content (seen: 2000/2000 reasoning tokens), which silently fell back to OCR text after
                # a paid call. "low", not "none" — see Settings.vlm_reasoning_effort.
                "reasoning_effort": self._reasoning_effort,
                # No temperature here (unlike the Anthropic describer): gpt-5.6-luna rejects any value
                # but the default ("Only the default (1) value is supported"), and since failures fall
                # back to OCR, setting it silently reverted every escalation (12/12 images) to worse
                # text. So this provider isn't byte-identical across runs — accepted.
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VLM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ],
                }],
            },
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        text = (choice["message"]["content"] or "").strip()
        _warn_if_truncated(choice.get("finish_reason"), "length", text, self._max_completion_tokens)
        return text

    async def aclose(self) -> None:
        await self._client.aclose()


def build_vlm_describer(settings) -> VlmImageDescriber:
    if not settings.vlm_ocr_enabled:
        return NoopVlmImageDescriber()
    if settings.vlm_provider == "anthropic":
        return AnthropicVlmImageDescriber(
            settings.vlm_anthropic_api_key,
            settings.vlm_anthropic_model,
            settings.vlm_max_completion_tokens,
        )
    if settings.vlm_provider == "openai_compatible":
        return OpenAICompatibleVlmImageDescriber(
            settings.vlm_openai_base_url,
            settings.vlm_openai_api_key,
            settings.vlm_openai_model,
            settings.vlm_max_completion_tokens,
            settings.vlm_reasoning_effort,
        )
    raise ValueError(f"unknown vlm_provider: {settings.vlm_provider}")


def vlm_cache_namespace(settings) -> str:
    """Version all inputs that can change extraction output before using a durable cache entry."""
    model = settings.vlm_anthropic_model if settings.vlm_provider == "anthropic" else settings.vlm_openai_model
    return (
        f"{_VLM_PROMPT_VERSION}:{settings.vlm_provider}:{model}:{settings.vlm_max_completion_tokens}"
        f":{settings.vlm_reasoning_effort}"
    )
