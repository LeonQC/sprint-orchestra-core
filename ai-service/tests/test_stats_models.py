"""`GET /stats` must report which models produced the numbers it is counting.

Why this is worth a test rather than a comment: every accuracy figure this project publishes is a
property of a (system, model) pair, not of the system alone. The same PDF-escalation feature measured
0.342 with one VLM and 0.753 with another, and a feature was once reverted on the strength of the
first number before anyone noticed the model was the variable that had changed. The evaluation
harness records this block into every result file, so a result that silently lost it would be unreadable a month later — and silently is exactly how it would go.

Follows the direct-call pattern the rest of this codebase's route tests use rather than a FastAPI
TestClient (see `tests/test_sprint_recovery_routes.py`).
"""
import types

import pytest

from app.api import routes


def _request(agent_model: str, *, router_provider: str, router_model: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(
            settings=types.SimpleNamespace(
                agent_model=agent_model,
                llm_provider="openai_compatible",
                router_llm_provider=router_provider,
                router_model=router_model,
            ),
            stats=types.SimpleNamespace(as_dict=lambda: {"requests": 3, "total_cost_usd": 0.12}),
        )),
    )


async def test_reports_the_answering_model_alongside_the_cost_it_produced():
    body = await routes.stats(_request("gpt-5.6-luna", router_provider="inherit"))

    assert body["models"]["answer"] == "gpt-5.6-luna"
    assert body["requests"] == 3 and body["total_cost_usd"] == 0.12


async def test_inherit_resolves_the_router_to_the_agent_model_rather_than_reporting_inherit():
    """`inherit` is a configuration keyword, not a model. Passing it through would put the literal
    string "inherit" into a frozen result file, which tells a later reader nothing about what ran."""
    body = await routes.stats(_request("gpt-5.6-luna", router_provider="inherit", router_model="unused"))

    assert body["models"]["router"] == "gpt-5.6-luna"


async def test_a_separately_configured_router_is_reported_as_itself():
    body = await routes.stats(
        _request("gpt-5.6-luna", router_provider="openai_compatible", router_model="self-hosted-router")
    )

    assert body["models"]["router"] == "self-hosted-router"
    assert body["models"]["answer"] == "gpt-5.6-luna"


async def test_a_gateway_route_alias_is_resolved_to_the_model_it_actually_bills_as():
    """`AI_AGENT_MODEL=openai-production` is this project's real production config — and the gateway
    itself cannot be asked to resolve it: LiteLLM echoes back the alias the caller sent as the
    completion's `model` field, not the underlying model, by design (a gateway alias exists so the
    caller doesn't need to know or care). This endpoint's own docstring says the model block exists so
    an eval run can record what actually answered — an unresolved alias makes that unanswerable the
    moment someone repoints the alias to a different model, which a frozen eval result file would
    never reveal."""
    body = await routes.stats(_request("openai-production", router_provider="inherit"))

    assert body["models"]["answer"] == "gpt-5.6-luna"
    assert body["models"]["router"] == "gpt-5.6-luna"


async def test_an_unaliased_model_name_passes_through_unchanged():
    """The common case: most model strings are not gateway aliases at all."""
    body = await routes.stats(_request("claude-sonnet-5", router_provider="inherit"))

    assert body["models"]["answer"] == "claude-sonnet-5"


@pytest.mark.parametrize("leaky", ["api_key", "base_url", "openai_api_key", "anthropic_api_key"])
async def test_never_discloses_credentials_or_endpoints(leaky):
    """A metrics endpoint that grows a config dump is how a base URL or a key ends up in a log
    aggregator. Model names are the whole contract here."""
    request = _request("gpt-5.6-luna", router_provider="inherit")
    setattr(request.app.state.settings, leaky, "SHOULD-NOT-APPEAR")

    body = await routes.stats(request)

    assert "SHOULD-NOT-APPEAR" not in str(body)
    assert set(body["models"]) == {"answer", "answer_provider", "router", "router_provider"}
