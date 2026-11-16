"""Ask AI when parts of the stack are down: BM25-only retrieval, or no answer model at all."""
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from app import degradation
from app.agent.retrieval_tool import RetrievalClient, RetrievedChunk
from app.api import routes
from app.stats import ServiceStats


def _chunk(issue_key: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"issue:{issue_key}", chunk_type="issue", issue_id=1, issue_key=issue_key, source_id=1,
        content=content, score=0.9, retrievers=["bm25"],
    )


class _Agent:
    def __init__(self, exc: Exception | None = None, note: str | None = None):
        self._exc = exc
        self._note = note

    async def ask(self, question, space_ids, history=None, on_stage=None, on_token=None, space_labels=None):
        if self._note:
            degradation.note(self._note)
        if self._exc:
            raise self._exc
        return SimpleNamespace(
            text="ATC-77 is blocked on the payment provider.",
            abstained=False, needs_clarification=False, clarification_options=[],
            retrieval_rounds=1, queries_used=["blocked"], citations=[], structured_evidence=[],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0),
            estimated_cost_usd=0.001, model_usage=[],
            route_source="keyword", route_tool="search_knowledge_base", stage_seconds={},
        )


class _Retrieval:
    def __init__(self, hits=None, exc: Exception | None = None):
        self._hits = hits or []
        self._exc = exc

    async def search(self, query, space_ids, limit, mode="hybrid"):
        if self._exc:
            raise self._exc
        return self._hits


class _RecordingCache:
    def __init__(self):
        self.puts = []

    async def get(self, question, space_ids, watermark, exact_only=False, execution_mode="single"):
        return None

    async def put(self, question, space_ids, response, watermark, execution_mode="single"):
        self.puts.append(response)


async def _watermark(space_ids):
    return "v1"


async def _always_ok(user_id, username, space_ids):
    return None


def _request(agent, retrieval=None, cache=None):
    return SimpleNamespace(
        headers={"x-user-id": "42"},
        app=SimpleNamespace(state=SimpleNamespace(
            agent=agent, retrieval=retrieval or _Retrieval(), cache=cache or _RecordingCache(),
            index_watermark=SimpleNamespace(watermark=_watermark),
            space_membership=SimpleNamespace(validate=_always_ok),
            settings=SimpleNamespace(retrieval_top_k=5),
            stats=ServiceStats(),
        )),
    )


def _ask(question="why is ATC-77 blocked?"):
    return routes.AskRequest(question=question, spaceIds=[1])


async def test_answer_models_down_returns_retrieved_sources_instead_of_an_error():
    cache = _RecordingCache()
    hits = [_chunk("ATC-77", "Blocked on the payment provider sandbox."), _chunk("ATC-77", "second chunk"),
            _chunk("ATC-12", "Payment provider contract renewal.")]
    request = _request(_Agent(exc=RuntimeError("all providers failed")), _Retrieval(hits), cache)

    response = await routes.ask(_ask(), request)

    assert response.degraded is True
    assert response.degraded_reasons == [degradation.ANSWER_MODEL_UNAVAILABLE]
    assert response.answer.startswith(routes.MODEL_UNAVAILABLE_ANSWER)
    # One line per issue, so the same ticket isn't listed twice.
    assert response.answer.count("- ATC-77:") == 1 and "- ATC-12:" in response.answer
    assert [c.issue_key for c in response.citations] == ["ATC-77", "ATC-77", "ATC-12"]
    assert response.estimated_cost_usd == 0.0
    assert cache.puts == []


async def test_answer_models_down_and_nothing_found_says_so_plainly():
    request = _request(_Agent(exc=RuntimeError("down")), _Retrieval([]))

    response = await routes.ask(_ask(), request)

    assert response.answer == routes.MODEL_UNAVAILABLE_NO_HITS
    assert response.citations == []


async def test_answer_models_and_retrieval_both_down_is_still_a_502():
    request = _request(_Agent(exc=RuntimeError("down")), _Retrieval(exc=httpx.ConnectError("refused")))

    with pytest.raises(HTTPException) as excinfo:
        await routes.ask(_ask(), request)

    assert excinfo.value.status_code == 502


async def test_an_answer_built_on_bm25_only_retrieval_is_flagged_and_not_cached():
    cache = _RecordingCache()
    request = _request(_Agent(note=degradation.RETRIEVAL_BM25_ONLY), cache=cache)

    response = await routes.ask(_ask(), request)

    assert response.answer.startswith("ATC-77 is blocked")
    assert response.degraded_reasons == [degradation.RETRIEVAL_BM25_ONLY]
    assert cache.puts == []


async def test_a_healthy_answer_is_not_degraded_and_is_cached():
    cache = _RecordingCache()

    response = await routes.ask(_ask(), _request(_Agent(), cache=cache))

    assert (response.degraded, response.degraded_reasons) == (False, [])
    assert len(cache.puts) == 1


async def test_stream_answer_models_down_ends_with_a_degraded_result_not_an_error():
    hits = [_chunk("ATC-77", "Blocked on the payment provider sandbox.")]
    request = _request(_Agent(exc=RuntimeError("down")), _Retrieval(hits))

    streaming = await routes.ask_stream(_ask(), request)
    frames = [frame async for frame in streaming.body_iterator]

    last = frames[-1]
    assert last.startswith("event: result")
    payload = json.loads(last.split("data: ", 1)[1])
    assert payload["degraded"] is True and payload["citations"][0]["issueKey"] == "ATC-77"


async def test_retrieval_client_reports_a_bm25_fallback_from_vectorization_service():
    body = {"hits": [], "mode": "bm25", "reranked": False, "degraded": True}
    client = RetrievalClient("http://vec.invalid")
    client._client = httpx.AsyncClient(
        base_url="http://vec.invalid", transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=body)),
    )

    with degradation.collect() as reasons:
        result = await client.search_with_meta("payment outage", [1], 5)

    assert result.degraded is True
    assert reasons == {degradation.RETRIEVAL_BM25_ONLY}
    await client._client.aclose()
