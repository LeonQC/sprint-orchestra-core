"""End-to-end proof that an AI feature's spend reaches a trace attributed to the right person.

tests/test_usage_ledger.py covers the record's arithmetic in isolation. What that cannot catch is the
wiring failure this whole feature exists to fix: `X-User-Id` has always reached this service, and it
has always been dropped before it reached a span, so every dollar this service spent was anonymous.
These tests run real endpoints against a real (in-memory) OpenTelemetry exporter and assert on the
span that actually got exported.

Direct endpoint calls with a hand-built request, the pattern the rest of this codebase's route tests
use (tests/test_stats_models.py, tests/test_planning_route_flag.py) — a TestClient would add an ASGI
stack without adding coverage of anything under test here.
"""
import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agent.crag_loop import ModelUsage
from app.api import routes
from app.comment_digest.schemas import AttributedPoint, CommentDigest, Outcome
from app.llm.pricing import estimate_cost_usd
from app.llm.usage_ledger import FEATURE_ASK_AI, FEATURE_SUMMARIZE_COMMENTS
from app.stats import ServiceStats
from tests.test_comment_digest import FakeInstructorClient, _FakeCompletion, _FakeUsage


@pytest.fixture
def exporter(monkeypatch):
    """A real tracer whose spans land in memory.

    Two spans matter in a real request and both are covered here: the FastAPI server span (stood in
    for by `tracer`, which the tests open around the call) and the `CragAgent.ask` agent span `/ask`
    opens itself. The latter comes from the module-level tracer, which resolves through the *global*
    provider — patched rather than installed globally so this fixture cannot leak into other tests.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(routes, "_tracer", tracer)
    return SimpleNamespace(exporter=exporter, tracer=tracer)


def _digest():
    point = AttributedPoint(text="ship it", comment_ids=[1])
    return CommentDigest(
        context="c", discussion="d", outcome=Outcome(text="agreed", comment_ids=[1]),
        open_questions=[point], action_items=[],
    )


def _request(headers, client, *, stats=None, deployment="hosted"):
    return SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(
            instructor_client=client,
            instructor_model="gpt-5.6-luna",
            model_provider="litellm gateway",
            model_deployment=deployment,
            stats=stats or ServiceStats(),
        )),
    )


def _comments(n=3):
    return [
        {"id": i, "authorName": "michael", "createdAt": "2026-09-05T10:00:00Z", "content": f"comment {i}"}
        for i in range(1, n + 1)
    ]


async def _run_comment_digest(exporter, headers, *, usage=None, stats=None):
    client = FakeInstructorClient(_digest(), _FakeCompletion(usage or _FakeUsage(1_000_000, 1_000_000)))
    request = _request(headers, client, stats=stats)
    with exporter.tracer.start_as_current_span("POST /comment-digest"):
        await routes.comment_digest_endpoint(
            routes.CommentDigestRequest(comments=_comments()), request
        )
    return exporter.exporter.get_finished_spans()[0].attributes


async def test_the_authenticated_user_reaches_the_span(exporter):
    """The wiring bug this feature fixes: identity arrived on every request and never left the
    authorization check."""
    attributes = await _run_comment_digest(
        exporter, {"x-user-id": "42", "x-username": "michael"}
    )

    assert attributes["user.id"] == "42"
    assert json.loads(attributes["metadata"])["username"] == "michael"


async def test_spend_is_attributed_to_the_feature_that_made_it(exporter):
    """Seven features shared one process name in the trace, so "which product costs the money" was
    unanswerable — the first question a month-end invoice raises."""
    attributes = await _run_comment_digest(exporter, {"x-user-id": "42"})

    assert attributes["ai.feature"] == FEATURE_SUMMARIZE_COMMENTS
    assert attributes["ai.usage.cost_usd"] == pytest.approx(1.40)  # 1M in @ $0.20 + 1M out @ $1.20
    assert attributes["ai.usage.billed_models"] == "gpt-5.6-luna"


async def test_the_span_carries_the_itemized_breakdown_behind_the_total(exporter):
    attributes = await _run_comment_digest(exporter, {"x-user-id": "42"})

    record = json.loads(attributes["metadata"])
    components = {c["bucket"]: c for c in record["charges"][0]["components"]}
    assert components["input"]["rate_usd_per_million"] == 0.20
    assert components["output"]["tokens"] == 1_000_000
    assert record["detail"]["comment_count"] == 3


async def test_a_feature_that_spends_now_counts_toward_service_wide_cost(exporter):
    """Only /ask fed GET /stats before, so six paid features were missing from the service's own
    cost total — the number an operator checks first."""
    stats = ServiceStats()

    await _run_comment_digest(exporter, {"x-user-id": "42"}, stats=stats)

    assert stats.as_dict()["total_cost_usd"] == pytest.approx(1.40)
    assert stats.as_dict()["requests"] == 1


async def test_an_internal_caller_is_recorded_without_inventing_a_user(exporter):
    attributes = await _run_comment_digest(exporter, {})

    assert "user.id" not in attributes
    assert attributes["ai.feature"] == FEATURE_SUMMARIZE_COMMENTS  # still measured, just unattributed


async def test_a_degraded_call_still_records_what_it_burned(exporter):
    """A structured-output failure has already paid for its input tokens. A ledger that recorded only
    successes would understate exactly the traffic worth investigating."""
    client = FakeInstructorClient(RuntimeError("model unreachable"))
    request = _request({"x-user-id": "42"}, client)

    with exporter.tracer.start_as_current_span("POST /comment-digest"):
        response = await routes.comment_digest_endpoint(
            routes.CommentDigestRequest(comments=_comments()), request
        )

    attributes = exporter.exporter.get_finished_spans()[0].attributes
    assert response.degraded is True
    assert attributes["ai.feature"] == FEATURE_SUMMARIZE_COMMENTS
    assert json.loads(attributes["metadata"])["detail"]["degraded"] is True


async def test_a_local_model_is_recorded_as_measured_but_free(exporter):
    """Running against LM Studio must not silently look like production spend, and must not vanish
    from the trace either — token volume is still a capacity signal."""
    client = FakeInstructorClient(_digest(), _FakeCompletion(_FakeUsage(1_000_000, 1_000_000)))
    request = _request({"x-user-id": "42"}, client, deployment="local")

    with exporter.tracer.start_as_current_span("POST /comment-digest"):
        await routes.comment_digest_endpoint(
            routes.CommentDigestRequest(comments=_comments()), request
        )

    attributes = exporter.exporter.get_finished_spans()[0].attributes
    assert attributes["ai.usage.cost_usd"] == 0.0
    assert attributes["ai.usage.total_tokens"] == 2_000_000
    assert attributes["ai.usage.pricing_status"] == "free_local"


# --- POST /ask: the multi-model, multi-turn case ---------------------------------------------


class _StubAgent:
    """Returns a fixed answer whose spend is already split between the two models CragAgent uses."""

    def __init__(self, model_usage):
        self._model_usage = model_usage

    async def ask(self, question, space_ids, history=None, on_stage=None, on_token=None, space_labels=None):
        return SimpleNamespace(
            text="ATC-77 is blocked on the payment provider.",
            abstained=False, needs_clarification=False, clarification_options=[],
            retrieval_rounds=2, queries_used=["blocked issues"],
            citations=[], structured_evidence=[],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0),
            estimated_cost_usd=0.0,
            model_usage=self._model_usage,
            route_source="llm", route_tool="search_knowledge_base",
            stage_seconds={},
        )


class _NoCache:
    async def get(self, question, space_ids, watermark, exact_only=False, execution_mode="single"):
        return None

    async def put(self, question, space_ids, response, watermark, execution_mode="single"):
        return None


def _ask_request(headers, model_usage, stats=None):
    return SimpleNamespace(
        headers=headers,
        app=SimpleNamespace(state=SimpleNamespace(
            agent=_StubAgent(model_usage),
            cache=_NoCache(),
            index_watermark=SimpleNamespace(watermark=_always_none),
            space_membership=SimpleNamespace(validate=_always_ok),
            stats=stats or ServiceStats(),
        )),
    )


async def _always_none(space_ids):
    return None


async def _always_ok(user_id, username, space_ids):
    return None


def _model_usage(role, model, input_tokens, output_tokens):
    """The real dataclass CragAgent emits, priced the way CragAgent prices it — so the "both paths
    agree" test below compares two genuinely separate computations rather than a value with itself."""
    return ModelUsage(
        role=role, provider="litellm gateway", model=model, deployment="hosted",
        input_tokens=input_tokens, output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
        pricing_status="estimated",
    )


async def _run_ask(exporter, headers, history=(), model_usage=None, stats=None, question="why is ATC-77 blocked?"):
    usage = model_usage or [
        _model_usage("router", "gpt-5.6-luna", 500, 20),
        _model_usage("answer_agent", "openai-production", 1_000_000, 1_000_000),
    ]
    request = _ask_request(headers, usage, stats)
    with exporter.tracer.start_as_current_span("POST /ask"):
        response = await routes.ask(
            routes.AskRequest(question=question, spaceIds=[1], history=list(history)),
            request,
        )
    # The agent span the ledger is attached to, not the outer HTTP one.
    ledger_span = next(
        s for s in exporter.exporter.get_finished_spans() if "ai.feature" in s.attributes
    )
    return response, ledger_span.attributes


async def test_ask_records_each_model_separately_rather_than_as_one_blended_charge(exporter):
    """A cheap router and an expensive answer model on the same request. Collapsed into one number,
    "the router is 3% of spend" — the kind of finding that justifies a model change — is unavailable."""
    _, attributes = await _run_ask(exporter, {"x-user-id": "42"})

    record = json.loads(attributes["metadata"])
    by_role = {c["role"]: c for c in record["charges"]}
    assert by_role["router"]["billed_model"] == "gpt-5.6-luna"
    assert by_role["answer_agent"]["billed_model"] == "gpt-5.6-luna"  # resolved from the alias
    assert by_role["answer_agent"]["model"] == "openai-production"
    assert by_role["answer_agent"]["cost_usd"] == pytest.approx(1.40)


async def test_the_ledger_total_matches_the_cost_the_api_reports_to_the_caller(exporter):
    """Two independent paths compute this request's cost: CragAgent's, returned in the response, and
    the ledger's, re-derived from token counts. If they can disagree, one of them is lying to whoever
    is reconciling the invoice."""
    response, attributes = await _run_ask(exporter, {"x-user-id": "42"})

    charged = sum(item.estimated_cost_usd for item in response.model_usage)
    assert attributes["ai.usage.cost_usd"] == pytest.approx(charged)


async def test_follow_up_questions_land_in_the_same_session_as_the_first(exporter):
    """The per-conversation cost view: "this one chat cost $4.20" is the unit a user recognizes,
    and it only exists if turn 1, turn 2 and turn 5 all carry the same session id.

    Turn 1 is included deliberately, not just turn 2 vs turn 3: found live, tracing a real
    clarification round end to end, that turn 1 (whose `history` is always empty) fell back to
    `caller_from_headers`'s random-uuid branch, while turn 2 (whose `history[0]` is turn 1's own
    question) hashed that question into a *different*, deterministic session id — so this exact test,
    which only ever compared turn 2 against turn 3, could not have caught the bug its own name claims
    to guard against. See `_first_user_turn`'s docstring for the fix."""
    opening_question = "how many issues are blocked?"
    opening = [{"role": "user", "content": opening_question}]

    _, turn_one = await _run_ask(exporter, {"x-user-id": "42"}, question=opening_question)
    _, turn_two = await _run_ask(exporter, {"x-user-id": "42"}, history=opening)
    _, turn_three = await _run_ask(
        exporter, {"x-user-id": "42"},
        history=opening + [{"role": "assistant", "content": "three"}],
    )

    assert turn_one["session.id"] == turn_two["session.id"] == turn_three["session.id"]


async def test_response_carries_a_traceparent_naming_its_own_entrypoint_span(exporter):
    """`/ask` is stateless across HTTP calls — a clarification round is really two independent
    requests stitched together only by the frontend re-sending `history`, so each one gets its own
    trace by default. Found live: a clarification's follow-up question showed up as a disconnected
    trace instead of continuing the one where the agent asked the clarifying question in the first
    place. `AskResponse.trace_parent` is the fix's other half — see `_current_traceparent`'s
    docstring: the frontend sends this back as a real `traceparent` request header on its next call,
    and `FastAPIInstrumentor` (untested at this level, since these tests call the handler directly and
    open "POST /ask" by hand rather than going through the real ASGI stack — verified live instead,
    against a real curl call carrying a real `traceparent` header) parents that request's own
    entrypoint span under it, joining both into one trace.

    This test covers what *is* reachable at this level: the value handed back actually names the
    entrypoint span this request ran inside, not some other span active at the time — not, say, the
    `CragAgent.ask` span opened inside the handler, which would silently point a follow-up at the
    wrong node in the trace tree."""
    response, _ = await _run_ask(exporter, {"x-user-id": "42"})

    server_span = next(s for s in exporter.exporter.get_finished_spans() if s.name == "POST /ask")
    ctx = server_span.get_span_context()
    expected = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"
    assert response.trace_parent == expected


async def test_cache_hit_response_gets_its_own_fresh_traceparent(exporter):
    """A cache hit still runs inside its own real "POST /ask" span even though it skips CragAgent
    entirely — its `trace_parent` must reflect *that* span. The cached blob itself never carries a
    `trace_parent` at all (see `ask()`'s `_safe_cache_put` call, which excludes it) — this is what
    reaching into `AskResponse`'s *default* (`None`) would look like if the explicit override on the
    cache-hit path (`"trace_parent": trace_parent`) were ever dropped."""
    class _HitCache:
        async def get(self, question, space_ids, watermark, exact_only=False, execution_mode="single"):
            return {
                "answer": "ATC-77 is blocked.", "abstained": False, "retrievalRounds": 1,
                "queriesUsed": [], "citations": [], "structuredEvidence": [],
                "inputTokens": 0, "outputTokens": 0, "estimatedCostUsd": 0.0,
                "modelUsage": [], "cacheHit": True,
            }

        async def put(self, question, space_ids, response, watermark, execution_mode="single"):
            return None

    async def _watermark(space_ids):
        return "v1"

    request = SimpleNamespace(
        headers={"x-user-id": "42"},
        app=SimpleNamespace(state=SimpleNamespace(
            agent=_StubAgent([]), cache=_HitCache(),
            index_watermark=SimpleNamespace(watermark=_watermark),
            space_membership=SimpleNamespace(validate=_always_ok),
            stats=ServiceStats(),
        )),
    )

    with exporter.tracer.start_as_current_span("POST /ask") as server_span:
        response = await routes.ask(routes.AskRequest(question="why is ATC-77 blocked?", spaceIds=[1]), request)

    assert response.cache_hit is True
    ctx = server_span.get_span_context()
    expected = f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-{ctx.trace_flags:02x}"
    assert response.trace_parent == expected



# --- the tag reaches the actual LLM-call spans, not just the ledger span ----------------------


async def test_the_feature_tag_is_active_in_context_during_the_structured_output_call(exporter):
    """`app/llm/usage_ledger.py`'s `attach()` sets `tag.tags` on the one span it's given — the
    request's parent span. Querying real Langfuse data for one trace showed exactly the resulting
    split: the parent carried `tags=['ask_ai']`, every `Response`/GENERATION span nested under it
    (the rows Langfuse's own Observations table lists) carried `tags=[]`, because they were never
    passed to `attach()` — they're created automatically, deep inside the provider SDK, by
    OpenInference's instrumentor.

    `_tagged` fixes this by writing the feature into the OTel context (`using_tags`) around the
    call, which every span created while that context is active reads back — the same mechanism
    `CallerAttributeMiddleware` already uses for `user.id`/`session.id`. This test proves the
    context actually carries the value at the moment the provider call happens, by having the fake
    client read it back from inside its own `create_with_completion`."""
    from openinference.instrumentation import get_attributes_from_context
    from openinference.semconv.trace import SpanAttributes

    seen = {}
    client = FakeInstructorClient(_digest(), _FakeCompletion(_FakeUsage(100, 50)))
    original_call = client.completions.create_with_completion

    async def _spying_call(**kwargs):
        seen.update(dict(get_attributes_from_context()))
        return await original_call(**kwargs)

    client.completions.create_with_completion = _spying_call
    request = _request({"x-user-id": "42"}, client)

    with exporter.tracer.start_as_current_span("POST /comment-digest"):
        await routes.comment_digest_endpoint(routes.CommentDigestRequest(comments=_comments()), request)

    assert list(seen.get(SpanAttributes.TAG_TAGS, [])) == [FEATURE_SUMMARIZE_COMMENTS]


async def test_the_feature_tag_does_not_leak_into_a_later_unrelated_call(exporter):
    """`using_tags` must not outlive its `with` block — otherwise the *previous* request's feature
    would still be attached to spans a completely different endpoint creates afterward."""
    from openinference.instrumentation import get_attributes_from_context
    from openinference.semconv.trace import SpanAttributes

    tagged_client = FakeInstructorClient(_digest(), _FakeCompletion(_FakeUsage(100, 50)))
    with exporter.tracer.start_as_current_span("POST /comment-digest"):
        await routes.comment_digest_endpoint(
            routes.CommentDigestRequest(comments=_comments()), _request({"x-user-id": "42"}, tagged_client)
        )

    assert SpanAttributes.TAG_TAGS not in dict(get_attributes_from_context())


async def test_ask_tags_the_llm_call_context_not_just_the_ledger_span(exporter):
    """`/ask`'s agent span already gets `tag.tags` via `UsageLedger.attach()`. What that alone
    cannot reach is the `Response` GENERATION span OpenInference creates automatically inside
    `agent.ask()` — the exact gap found by inspecting real Langfuse data. `_StubAgent.ask` reads the
    context the same way a real provider-SDK call would, to prove the tag is actually there at that
    point, not just on the span the ledger touches afterward."""
    from openinference.instrumentation import get_attributes_from_context
    from openinference.semconv.trace import SpanAttributes

    seen = {}

    class _SpyingAgent(_StubAgent):
        async def ask(self, *args, **kwargs):
            seen.update(dict(get_attributes_from_context()))
            return await super().ask(*args, **kwargs)

    usage = [_model_usage("answer_agent", "openai-production", 1_000_000, 1_000_000)]
    request = SimpleNamespace(
        headers={"x-user-id": "42"},
        app=SimpleNamespace(state=SimpleNamespace(
            agent=_SpyingAgent(usage), cache=_NoCache(),
            index_watermark=SimpleNamespace(watermark=_always_none),
            space_membership=SimpleNamespace(validate=_always_ok),
            stats=ServiceStats(),
        )),
    )

    with exporter.tracer.start_as_current_span("POST /ask"):
        await routes.ask(routes.AskRequest(question="why is ATC-77 blocked?", spaceIds=[1]), request)

    assert list(seen.get(SpanAttributes.TAG_TAGS, [])) == [FEATURE_ASK_AI]


async def test_the_server_span_itself_is_tagged_not_just_the_agent_span(exporter):
    """`ledger.attach()` tags `CragAgent.ask` deliberately, not the outer `POST /ask` span — so the
    inbound HTTP entrypoint span used to carry no tag, no user, no session at all. Confirmed against
    real data in two stages: first `POST /ask/stream` had `tags=[]` while `POST /summarize-comments`
    (which never opens a second span, so the ledger's `trace.get_current_span()` happened to land on
    the server span by coincidence) carried `tags=['summarize_comments']`; then, after fixing that,
    `POST /ask` showed `tags=['ask_ai']` but `session_id`/`user_id` were *still* empty on that same
    span — `CallerAttributeMiddleware`'s context can't reach a span created before it runs, the
    identical structural gap one layer down. `_tag_entrypoint_span`, called first thing in the
    handler, sets all three directly on the current span instead of relying on context propagation,
    making this explicit for every endpoint instead of an accident of which one opens its own span."""
    from openinference.semconv.trace import SpanAttributes

    usage = [_model_usage("answer_agent", "openai-production", 100, 50)]
    request = _ask_request({"x-user-id": "42"}, usage)
    with exporter.tracer.start_as_current_span("POST /ask"):
        await routes.ask(routes.AskRequest(question="why is ATC-77 blocked?", spaceIds=[1]), request)

    server_span = next(s for s in exporter.exporter.get_finished_spans() if s.name == "POST /ask")
    assert list(server_span.attributes[SpanAttributes.TAG_TAGS]) == [FEATURE_ASK_AI]
    assert server_span.attributes[SpanAttributes.USER_ID] == "42"
    assert server_span.attributes[SpanAttributes.SESSION_ID]


async def test_server_span_and_ledger_agree_on_the_same_session_id(exporter):
    """`_tag_entrypoint_span` computes a `Caller` once and hands it to `_record_usage` — proving why
    that matters: with no `x-session-id` header and no history, `caller_from_headers` falls back to a
    random uuid (see its docstring), so calling it a second time independently would silently mint a
    *different* session id and the server span and the ledger's `CragAgent.ask` span would disagree
    about which conversation this was."""
    from openinference.semconv.trace import SpanAttributes

    usage = [_model_usage("answer_agent", "openai-production", 100, 50)]
    request = _ask_request({"x-user-id": "42"}, usage)
    with exporter.tracer.start_as_current_span("POST /ask"):
        await routes.ask(routes.AskRequest(question="why is ATC-77 blocked?", spaceIds=[1]), request)

    spans = {s.name: s for s in exporter.exporter.get_finished_spans()}
    assert (
        spans["POST /ask"].attributes[SpanAttributes.SESSION_ID]
        == spans["CragAgent.ask"].attributes[SpanAttributes.SESSION_ID]
    )
