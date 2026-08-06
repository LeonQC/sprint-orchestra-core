"""Builds an instructor-patched client for structured-output features.

Separate from llm/factory.py's LLMClient: the agent loop needs raw stop reasons and tool calls, while
these features are one call in, one validated Pydantic object out — which instructor handles,
including re-asking with the validation error when parsing fails.

Mode per provider:
  * Anthropic: Mode.ANTHROPIC_TOOLS (standard).
  * OpenAI-compatible (the default, through the LiteLLM gateway): Mode.JSON_SCHEMA
    (`response_format: json_schema`), which enforces the schema server-side.
"""
from __future__ import annotations

import json
from typing import Tuple

import httpx
import instructor

from app.config import Settings
from app.llm.circuit_breaker import breaker_for
from app.observability import get_billing_user


def _rewrite_request_body(request: httpx.Request) -> httpx.Request:
    """Apply all outbound body rewrites in one pass (parse and re-encode once): `max_tokens` ->
    `max_completion_tokens`, and the end user in `user`. Wire-protocol concerns, so they live here,
    not at each call site.
    """
    if "chat/completions" not in str(request.url):
        return request
    try:
        body = json.loads(request.content)
    except (json.JSONDecodeError, ValueError):
        return request

    changed = _rename_max_tokens(body) | _stamp_billing_user(body)
    if not changed:
        return request

    content = json.dumps(body).encode()
    return httpx.Request(
        method=request.method,
        url=request.url,
        # Drop content-length: the rewritten body is a different byte length and a stale header would
        # truncate the request.
        headers={k: v for k, v in request.headers.items() if k.lower() != "content-length"},
        content=content,
        extensions=request.extensions,
    )


def _stamp_billing_user(body: dict) -> bool:
    """Put the authenticated end user in OpenAI's `user` field.

    Every structured-output feature goes through this client, so this covers all of them (and future
    ones), letting LiteLLM track cost and budgets per person. Never overwrites a `user` set on purpose,
    and omits it when there's no end user (eval runs).
    """
    if body.get("user"):
        return False
    user = get_billing_user()
    if not user:
        return False
    body["user"] = user
    return True


def _rename_max_tokens(body: dict) -> bool:
    """Rename `max_tokens` to `max_completion_tokens` on /chat/completions requests.

    Against hosted gpt-5.6-luna, every structured-output feature got a 400 ("Unsupported parameter:
    'max_tokens'") and quietly fell back to its degraded path — a total outage hidden behind
    `degraded=True`. Fixed here because it's a protocol detail: OpenAI deprecated `max_tokens`
    (reasoning models reject it). Only installed on the OpenAI-compatible path.
    """
    if "max_tokens" not in body:
        return False
    body["max_completion_tokens"] = body.pop("max_tokens")
    return True


def _request_model(request: httpx.Request) -> str | None:
    if "chat/completions" not in str(request.url):
        return None
    try:
        model = json.loads(request.content).get("model")
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None
    return model if isinstance(model, str) and model else None


class _RequestRewriteTransport(httpx.AsyncBaseTransport):
    """Applies `_rewrite_request_body` to every outgoing request; responses pass through untouched.

    Also feeds the same per-model circuit breaker as the agent's client. Here rather than at each call
    site because instructor sits between the call sites and the SDK; the openai SDK turns an open
    breaker into APIConnectionError, which `is_model_unavailable` recognizes.
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport):
        self._wrapped = wrapped

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request = _rewrite_request_body(request)
        model = _request_model(request)
        if model is None:
            return await self._wrapped.handle_async_request(request)
        breaker = breaker_for(model)
        breaker.check()
        try:
            response = await self._wrapped.handle_async_request(request)
        except httpx.TransportError:  # connection refused, timeout
            breaker.record_failure()
            raise
        except BaseException:
            breaker.release_probe()
            raise
        if response.status_code >= 500 or response.status_code == 429:
            breaker.record_failure()
        else:
            breaker.record_success()
        return response


def build_instructor_client(settings: Settings) -> Tuple[instructor.AsyncInstructor, str]:
    """Returns (client, model) — model is separate because the two providers name models differently."""
    if settings.llm_provider == "anthropic":
        import anthropic

        client = instructor.from_anthropic(
            anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None),
            mode=instructor.Mode.ANTHROPIC_TOOLS,
        )
        return client, settings.agent_model

    import openai

    http_client = httpx.AsyncClient(transport=_RequestRewriteTransport(httpx.AsyncHTTPTransport()))
    client = instructor.from_openai(
        # max_retries=0: the gateway owns retries and fallback (see openai_compatible_client.py).
        openai.AsyncOpenAI(
            base_url=settings.openai_base_url, api_key=settings.openai_api_key, http_client=http_client,
            max_retries=0,
        ),
        mode=instructor.Mode.JSON_SCHEMA,
    )
    return client, settings.agent_model
