import pytest

from app.config import Settings
from app.llm.factory import build_router_llm_client, describe_llm_runtime
from app.llm.openai_compatible_client import OpenAICompatibleClient

from .test_crag_loop import FakeLLM


def test_router_provider_inherit_reuses_main_client_and_model():
    main = FakeLLM([])
    settings = Settings(
        _env_file=None,
        agent_model="hosted-answer-model",
        router_llm_provider="inherit",
    )

    client, model = build_router_llm_client(settings, main)

    assert client is main
    assert model == "hosted-answer-model"


async def test_router_provider_can_build_small_local_openai_compatible_client():
    main = FakeLLM([])
    settings = Settings(
        _env_file=None,
        router_llm_provider="openai_compatible",
        router_model="small-router",
        router_openai_base_url="http://localhost:1234/v1",
        router_openai_api_key="local",
        router_max_output_tokens=128,
        router_request_timeout_seconds=12,
        router_reasoning_effort="none",
    )

    client, model = build_router_llm_client(settings, main)

    assert isinstance(client, OpenAICompatibleClient)
    assert client is not main
    assert client._model == "small-router"
    assert client._max_tokens == 128
    assert client._reasoning_effort == "none"
    assert model == "small-router"
    await client.aclose()


def test_separate_router_provider_requires_an_explicit_model():
    settings = Settings(
        _env_file=None,
        router_llm_provider="openai_compatible",
        router_model="",
    )

    with pytest.raises(ValueError, match="AI_ROUTER_MODEL"):
        build_router_llm_client(settings, FakeLLM([]))


def test_runtime_description_distinguishes_lm_studio_from_hosted_openai():
    assert describe_llm_runtime("openai_compatible", "http://localhost:1234/v1") == (
        "Self-hosted", "local"
    )
    assert describe_llm_runtime("openai_compatible", "https://api.openai.com/v1") == (
        "OpenAI", "hosted"
    )
    assert describe_llm_runtime("anthropic") == ("Anthropic", "hosted")
