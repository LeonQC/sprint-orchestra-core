"""Gateway fallback: detection from the response, and pricing at the model that actually answered."""
from unittest.mock import AsyncMock

import pytest

from app.agent.crag_loop import CragAgent, RouteDecision
from app.llm.fallback import served_model
from app.llm.openai_compatible_client import OpenAICompatibleClient
from app.llm.pricing import estimate_cost_usd
from app.llm.types import Usage
from app.observability import LLM_FALLBACKS_TOTAL


def _fallback_count(requested: str, served: str) -> float:
    return LLM_FALLBACKS_TOTAL.labels(requested, served)._value.get()


def test_echoed_alias_is_not_a_fallback():
    assert served_model("openai-production", "openai-production") == "openai-production"


def test_dated_snapshot_of_the_same_model_is_not_a_fallback():
    # A provider called directly reports a dated snapshot of the requested model.
    assert served_model("gpt-5.6-luna", "gpt-5.6-luna-2026-09-01") == "gpt-5.6-luna"
    assert served_model("openai-production", "gpt-5.6-luna-2026-09-01") == "openai-production"


def test_missing_model_field_is_not_a_fallback():
    assert served_model("openai-production", None) == "openai-production"


def test_a_different_model_is_a_fallback_and_is_counted():
    before = _fallback_count("openai-production", "gemini-3.8-flash")
    assert served_model("openai-production", "gemini-3.8-flash") == "gemini-3.8-flash"
    assert _fallback_count("openai-production", "gemini-3.8-flash") == before + 1


def test_usage_addition_merges_fallback_tokens_per_model():
    total = Usage(10, 2) + Usage(100, 20, fallback={"gemini-3.8-flash": Usage(100, 20)}) \
        + Usage(50, 5, fallback={"gemini-3.8-flash": Usage(50, 5)})
    assert (total.input_tokens, total.output_tokens) == (160, 27)
    assert (total.fallback["gemini-3.8-flash"].input_tokens, total.fallback["gemini-3.8-flash"].output_tokens) == (150, 25)


async def test_client_marks_tokens_answered_by_a_fallback_model():
    class FakeResponse:
        def model_dump(self):
            return {
                "model": "gemini-3.8-flash",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
                "usage": {"input_tokens": 1000, "output_tokens": 50},
            }

    client = OpenAICompatibleClient(
        base_url="http://gateway.invalid", api_key="test", model="openai-production", max_output_tokens=100,
    )
    client._client.responses.create = AsyncMock(return_value=FakeResponse())

    turn = await client.next_turn("system", [{"role": "user", "content": "hi"}], [])

    assert turn.usage.input_tokens == 1000
    assert turn.usage.fallback["gemini-3.8-flash"].input_tokens == 1000
    await client.aclose()


def _agent() -> CragAgent:
    return CragAgent(
        llm=AsyncMock(), retrieval=AsyncMock(), max_iterations=4, top_k=5,
        model="openai-production", model_provider="litellm gateway", model_deployment="hosted",
    )


def test_fallback_tokens_are_priced_at_the_fallback_model():
    # OpenAI down, Gemini answered the whole /ask.
    usage = Usage(34214, 781, fallback={"gemini-3.8-flash": Usage(34214, 781)})

    cost = _agent()._cost(usage)

    assert cost == pytest.approx(estimate_cost_usd("gemini-3.8-flash", 34214, 781))
    assert cost > 3 * estimate_cost_usd("openai-production", 34214, 781)


def test_gemini_fallback_is_priced_under_both_names():
    # LiteLLM reports the real model ("gemini-3.8-flash"); called directly it is "gemini-production".
    assert estimate_cost_usd("gemini-3.8-flash", 1_000_000, 0) == pytest.approx(0.75)
    assert estimate_cost_usd("gemini-production", 0, 1_000_000) == pytest.approx(3.75)


def test_partial_fallback_prices_each_share_at_its_own_model():
    usage = Usage(1000, 100) + Usage(2000, 200, fallback={"gemini-3.8-flash": Usage(2000, 200)})

    cost = _agent()._cost(usage)

    assert cost == pytest.approx(
        estimate_cost_usd("openai-production", 1000, 100) + estimate_cost_usd("gemini-3.8-flash", 2000, 200)
    )


def test_model_usage_lists_the_fallback_model_separately():
    usage = Usage(1000, 100) + Usage(2000, 200, fallback={"gemini-3.8-flash": Usage(2000, 200)})
    route = RouteDecision("search_knowledge_base", "keyword", "high", "identified")

    entries = _agent()._model_usage(usage, Usage(), route)

    assert [(e.model, e.provider, e.input_tokens) for e in entries] == [
        ("openai-production", "litellm gateway", 1000),
        ("gemini-3.8-flash", "litellm gateway (fallback)", 2000),
    ]


def test_no_fallback_keeps_a_single_entry():
    route = RouteDecision("search_knowledge_base", "keyword", "high", "identified")

    entries = _agent()._model_usage(Usage(1000, 100), Usage(), route)

    assert [(e.model, e.input_tokens) for e in entries] == [("openai-production", 1000)]
