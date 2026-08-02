from __future__ import annotations

from urllib.parse import urlparse

from app.config import Settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.openai_compatible_client import OpenAICompatibleClient
from app.llm.pricing import has_known_price
from app.llm.types import LLMClient


def describe_llm_runtime(provider: str, base_url: str = "", model: str = "") -> tuple[str, str]:
    """Return a UI-safe provider label and local/hosted deployment classification.

    `deployment` ("local" | "hosted") gates cost accounting downstream (CragAgent._usage_cost skips
    pricing for "local"). A localhost URL used to be a safe proxy for "local and free" — it no longer
    is: the litellm gateway runs on localhost but fronts paid hosted models. So a model name the
    price table recognises (including gateway route aliases, see pricing._GATEWAY_ALIASES) is what
    settles it, checked before the host.
    """
    if provider == "anthropic":
        return "Anthropic", "hosted"
    if has_known_price(model):
        return "litellm gateway", "hosted"
    host = (urlparse(base_url).hostname or "").lower()
    if host in {"localhost", "127.0.0.1", "::1"}:
        return "Self-hosted", "local"
    if host == "api.openai.com" or host.endswith(".openai.com"):
        return "OpenAI", "hosted"
    return "OpenAI-compatible", "hosted"


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, settings.agent_model, settings.max_output_tokens)
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleClient(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.agent_model,
            settings.max_output_tokens,
            settings.llm_request_timeout_seconds,
            settings.reasoning_effort,
            response_cache=settings.llm_response_cache_enabled,
        )
    raise ValueError(f"unknown llm_provider: {settings.llm_provider}")


def build_router_llm_client(settings: Settings, main_llm: LLMClient) -> tuple[LLMClient, str]:
    """Build the narrow capability-classifier client independently from the answer model."""
    provider = settings.router_llm_provider
    if provider == "inherit":
        return main_llm, settings.agent_model
    if not settings.router_model:
        raise ValueError("AI_ROUTER_MODEL is required when AI_ROUTER_LLM_PROVIDER is not 'inherit'")
    if provider == "anthropic":
        return (
            AnthropicClient(
                settings.router_anthropic_api_key or settings.anthropic_api_key,
                settings.router_model,
                settings.router_max_output_tokens,
            ),
            settings.router_model,
        )
    if provider == "openai_compatible":
        return (
            OpenAICompatibleClient(
                settings.router_openai_base_url,
                settings.router_openai_api_key,
                settings.router_model,
                settings.router_max_output_tokens,
                settings.router_request_timeout_seconds,
                settings.router_reasoning_effort,
                response_cache=settings.llm_response_cache_enabled,
            ),
            settings.router_model,
        )
    raise ValueError(f"unknown router_llm_provider: {provider}")
