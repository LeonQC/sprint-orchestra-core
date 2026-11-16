"""Contextual retrieval (https://www.anthropic.com/news/contextual-retrieval).

A chunk on its own may lose what made it findable ("raising the pool size fixed it" — which pool?).
For multi-chunk sources, a small LLM writes one sentence placing each chunk in its document, and it's
prepended before embedding, so the vector, BM25 index and citation text all benefit. Single-chunk
sources are skipped. Off by default (one LLM call per chunk).
"""
from __future__ import annotations

import logging
from typing import Protocol

import anthropic
import httpx

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Give a short, succinct sentence of context (no more than 2 sentences) that situates this chunk \
within the overall document, for the purpose of improving search retrieval of the chunk. Answer \
only with the context sentence — no preamble, no quotes, nothing else."""


class ContextGenerator(Protocol):
    async def generate(self, document: str, chunk: str) -> str: ...

    async def aclose(self) -> None: ...


class NoopContextGenerator:
    """Used when contextual retrieval is disabled — never calls out, always returns ''."""

    async def generate(self, document: str, chunk: str) -> str:
        return ""

    async def aclose(self) -> None:
        return None


class AnthropicContextGenerator:
    def __init__(self, api_key: str, model: str):
        self._client = anthropic.AsyncAnthropic(api_key=api_key or None)
        self._model = model

    async def generate(self, document: str, chunk: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=100,
            # temperature=0: this sentence changes the vector and the BM25 text, so randomness here
            # moved rankings on 3 unrelated questions when unchanged pages were re-ingested.
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(document=document, chunk=chunk)}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    async def aclose(self) -> None:
        await self._client.close()


class OpenAICompatibleContextGenerator:
    """An OpenAI-compatible chat endpoint — by default the LiteLLM gateway's `claude-context` route."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=60.0
        )

    async def generate(self, document: str, chunk: str) -> str:
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "max_tokens": 100,
                # Reasoning models can spend the whole 100-token budget on hidden reasoning and return
                # empty content, silently skipping every call. "none" turns that off.
                "reasoning_effort": "none",
                # See AnthropicContextGenerator.generate's comment: pinned so re-ingesting unchanged
                # source content produces byte-identical situating sentences (and therefore embeddings)
                # across runs, instead of contributing sampling noise to retrieval-ranking comparisons.
                "temperature": 0,
                "messages": [{"role": "user", "content": _PROMPT_TEMPLATE.format(document=document, chunk=chunk)}],
            },
        )
        resp.raise_for_status()
        return (resp.json()["choices"][0]["message"]["content"] or "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()


def build_context_generator(settings) -> ContextGenerator:
    if not settings.contextual_retrieval_enabled:
        return NoopContextGenerator()
    if settings.contextual_llm_provider == "anthropic":
        return AnthropicContextGenerator(settings.contextual_anthropic_api_key, settings.contextual_anthropic_model)
    if settings.contextual_llm_provider == "openai_compatible":
        return OpenAICompatibleContextGenerator(
            settings.contextual_openai_base_url, settings.contextual_openai_api_key, settings.contextual_openai_model
        )
    raise ValueError(f"unknown contextual_llm_provider: {settings.contextual_llm_provider}")
