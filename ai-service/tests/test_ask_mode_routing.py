"""POST /ask with adaptive single/multi routing: who answers, what gets charged, what gets cached."""
import json

import pytest
from types import SimpleNamespace

from app import degradation
from app.agent.crag_loop import AgentAnswer
from app.agent.mode_router import ModeDecision
from app.api import routes
from app.llm.types import Usage
from app.llm.usage_ledger import charge_for
from tests.test_ask_degraded import _Agent, _RecordingCache, _request

_ROUTER_CHARGE = charge_for("mode_router", "OpenAI", "gpt-5.6-luna", "hosted", Usage(40, 5))
_WORKER_CHARGE = charge_for("worker-1", "OpenAI", "gpt-5.6-luna", "hosted", Usage(900, 80))


class _Router:
    def __init__(self, mode: str, charges=(_ROUTER_CHARGE,)):
        self.decision = ModeDecision(mode, "model", "test", tuple(charges))

    async def choose(self, question, space_ids, **kwargs):
        return self.decision


class _Investigator:
    def __init__(self, errors=(), structured_evidence=()):
        self.errors = list(errors)
        self.structured_evidence = list(structured_evidence)
        self.calls = 0

    async def ask(self, question, space_ids, on_stage=None, space_labels=None):
        self.calls += 1
        self.question = question
        self.space_labels = space_labels
        answer = AgentAnswer(
            text="- ATC has 3 open bugs. (ATC-1)\n\n- PRM has 1 open bug. (PRM-2)",
            retrieval_rounds=2, queries_used=["atc bugs", "prm bugs"],
            route_source="langgraph_multi_agent", route_tool="investigate",
            structured_evidence=self.structured_evidence,
        )
        return SimpleNamespace(answer=answer, charges=[_WORKER_CHARGE], errors=self.errors)


async def _authorized_spaces(user_id, username):
    return [{"id": 1, "key": "ATC", "name": "AtlasCart"}, {"id": 2, "key": "PRM", "name": "Patient"},
            {"id": 3, "key": "SCRUM", "name": "Not asked about"}]


def _two_space_request(router, investigator=None, cache=None):
    request = _request(_Agent(), cache=cache)
    request.app.state.mode_router = router
    request.app.state.investigator = investigator
    request.app.state.space_membership.authorized_spaces = _authorized_spaces
    return request


def _ask():
    return routes.AskRequest(question="How are ATC and PRM doing?", spaceIds=[1, 2], mode="deep")


def _capture_charges(monkeypatch) -> list:
    recorded = []
    monkeypatch.setattr(
        routes, "_record_usage",
        lambda request, feature, charges, **kwargs: recorded.append([c.role for c in charges]),
    )
    return recorded


async def test_multi_answer_charges_the_router_and_every_worker(monkeypatch):
    recorded = _capture_charges(monkeypatch)
    cache = _RecordingCache()
    investigator = _Investigator()

    response = await routes.ask(_ask(), _two_space_request(_Router("multi"), investigator, cache))

    assert investigator.calls == 1
    assert investigator.space_labels == {1: "ATC", 2: "PRM"}
    assert response.execution_mode == "multi"
    assert response.route_tool == "investigate"
    assert recorded == [["mode_router", "worker-1"]]
    assert response.degraded is False
    assert len(cache.puts) == 1


async def test_a_multi_answer_built_on_issue_counts_is_not_cached(monkeypatch):
    """"ATC has 3 open bugs" from query_issues is stale after the next card drag, and a drag doesn't
    change the cache key (the watermark follows searchable text only)."""
    _capture_charges(monkeypatch)
    cache = _RecordingCache()
    investigator = _Investigator(structured_evidence=["Workspace ATC:\ntotal_count: 3"])

    response = await routes.ask(_ask(), _two_space_request(_Router("multi"), investigator, cache))

    assert response.route_tool == "investigate"
    assert cache.puts == []


async def test_a_failed_worker_is_one_fixed_degraded_reason_and_is_not_cached(monkeypatch):
    _capture_charges(monkeypatch)
    cache = _RecordingCache()
    investigator = _Investigator(errors=["worker-2: TimeoutError"])

    response = await routes.ask(_ask(), _two_space_request(_Router("multi"), investigator, cache))

    # Raw error strings stay out of the response and out of the Prometheus label.
    assert response.degraded_reasons == [degradation.MULTI_AGENT_PARTIAL]
    assert cache.puts == []


async def test_multi_without_an_investigator_falls_back_to_single_and_keeps_the_router_charge(
    monkeypatch,
):
    recorded = _capture_charges(monkeypatch)

    response = await routes.ask(_ask(), _two_space_request(_Router("multi"), investigator=None))

    assert response.execution_mode == "single"
    assert response.mode_route_source == "fallback"
    assert response.answer.startswith("ATC-77 is blocked")
    assert recorded == [["mode_router"]]


async def test_single_mode_never_calls_the_investigator(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    response = await routes.ask(_ask(), _two_space_request(_Router("single", ()), investigator))

    assert investigator.calls == 0
    assert response.execution_mode == "single"


def _ask_with_mode(mode: str, history=None):
    return routes.AskRequest(
        question="How are ATC and PRM doing?", spaceIds=[1, 2], mode=mode, history=history or [],
    )


_HISTORY = [
    {"role": "user", "content": "Compare the duplicate-request fixes in ATC and PRM."},
    {"role": "assistant", "content": "ATC added an idempotency key; PRM disabled the button."},
]


async def test_deep_lets_the_router_choose_multi(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    response = await routes.ask(
        _ask_with_mode("deep"), _two_space_request(_Router("multi"), investigator),
    )

    assert investigator.calls == 1
    assert response.execution_mode == "multi"
    assert response.mode_route_source == "model"


async def test_deep_still_answers_with_one_agent_when_the_router_says_single(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    response = await routes.ask(
        _ask_with_mode("deep"), _two_space_request(_Router("single", ()), investigator),
    )

    assert investigator.calls == 0
    assert response.execution_mode == "single"


async def test_fast_skips_both_the_router_and_the_investigator(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    response = await routes.ask(
        _ask_with_mode("fast"), _two_space_request(_NeverCalledRouter(), investigator),
    )

    assert investigator.calls == 0
    assert response.execution_mode == "single"
    assert response.mode_route_source == "user"


async def test_the_default_mode_is_fast(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    request = routes.AskRequest(question="How are ATC and PRM doing?", spaceIds=[1, 2])

    response = await routes.ask(request, _two_space_request(_NeverCalledRouter(), investigator))

    assert investigator.calls == 0
    assert response.mode_route_source == "user"


async def test_a_deep_follow_up_can_use_multi_and_carries_the_earlier_turns(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()

    response = await routes.ask(
        _ask_with_mode("deep", _HISTORY), _two_space_request(_Router("multi"), investigator),
    )

    assert response.execution_mode == "multi"
    assert "idempotency key" in investigator.question
    assert investigator.question.endswith("Current question: How are ATC and PRM doing?")


def test_a_question_without_history_goes_to_multi_unchanged():
    assert routes._question_for_multi("How are ATC and PRM doing?", []) == "How are ATC and PRM doing?"


def test_a_long_earlier_answer_reaches_multi_whole():
    answer = "ATC row " + "x" * 5000 + " END-OF-TABLE"
    turns = [{"role": "user", "content": "compare ATC and PRM"},
             {"role": "assistant", "content": answer}]

    text = routes._question_for_multi("and now?", turns)

    assert answer in text


class _Compactor:
    """Stands in for HistoryCompactor: summarizes everything before the latest question and answer."""

    def __init__(self):
        self.received_summary = None

    async def compact(self, history, summary=None, on_stage=None):
        from app.agent.history_summary import CompactHistory, HistorySummary, apply_summary
        self.received_summary = summary
        new = HistorySummary("Question 1: idempotency key for ATC checkout.", len(history) - 2)
        charge = charge_for("history_summary", "OpenAI", "gpt-5.6-luna", "hosted", Usage(3000, 200))
        return CompactHistory(apply_summary(history, new), new, (charge,))


_LONG_HISTORY = _HISTORY + [
    {"role": "user", "content": "And PRM?"},
    {"role": "assistant", "content": "PRM uses a check-in idempotency key (PRM-15)."},
]


class _HistoryAgent(_Agent):
    async def ask(self, question, space_ids, history=None, **kwargs):
        self.history = history
        return await super().ask(question, space_ids, history=history)


async def test_single_gets_the_summary_and_the_response_returns_it(monkeypatch):
    recorded = _capture_charges(monkeypatch)
    agent = _HistoryAgent()
    request = _request(agent)
    request.app.state.history_compactor = _Compactor()

    response = await routes.ask(
        routes.AskRequest(question="Which are done?", spaceIds=[1], history=_LONG_HISTORY), request,
    )

    assert response.history_summary.covers_turns == len(_LONG_HISTORY) - 2
    sent = agent.history
    assert sent[0]["content"].startswith("Summary of the earlier part of this chat:")
    assert sent[-1]["content"] == "PRM uses a check-in idempotency key (PRM-15)."
    assert "history_summary" in recorded[-1]


async def test_multi_gets_the_same_compacted_turns(monkeypatch):
    _capture_charges(monkeypatch)
    investigator = _Investigator()
    request = _two_space_request(_Router("multi"), investigator)
    request.app.state.history_compactor = _Compactor()

    await routes.ask(_ask_with_mode("deep", _LONG_HISTORY), request)

    assert "Summary of the earlier part of this chat:" in investigator.question
    assert "PRM uses a check-in idempotency key (PRM-15)." in investigator.question


async def test_the_summary_the_ui_sends_back_reaches_the_compactor(monkeypatch):
    _capture_charges(monkeypatch)
    compactor = _Compactor()
    request = _request(_HistoryAgent())
    request.app.state.history_compactor = compactor

    await routes.ask(
        routes.AskRequest(
            question="Which are done?", spaceIds=[1], history=_LONG_HISTORY,
            historySummary={"text": "Question 1: ATC checkout.", "coversTurns": 2},
        ),
        request,
    )

    assert compactor.received_summary.text == "Question 1: ATC checkout."
    assert compactor.received_summary.covers_turns == 2


class _NeverCalledRouter:
    async def choose(self, question, space_ids, **kwargs):
        raise AssertionError("the router must not run when the user picked the mode")


class _NeverCalled:
    async def choose(self, question, space_ids, **kwargs):
        raise AssertionError("the mode router must not run for a question about an unselected space")


def _atc_only_request():
    request = _request(_Agent())
    request.app.state.mode_router = _NeverCalled()
    request.app.state.space_membership.authorized_spaces = _authorized_spaces
    return request


async def test_an_issue_key_of_an_unselected_space_gets_a_notice_without_any_model_call(monkeypatch):
    recorded = _capture_charges(monkeypatch)
    req = routes.AskRequest(question="What does prm-20 do?", spaceIds=[1])

    response = await routes.ask(req, _atc_only_request())

    assert response.abstained is True
    assert response.answer.startswith("Your question mentions PRM, which isn't selected")
    assert response.estimated_cost_usd == 0.0
    assert recorded == []


async def test_a_bare_space_key_counts_but_only_in_its_own_case():
    request = _atc_only_request()

    named = await routes.ask(routes.AskRequest(question="PRM 有幾個 bug？", spaceIds=[1]), request)
    plain_request = _request(_Agent())
    plain_request.app.state.mode_router = _Router("single", ())
    plain_request.app.state.space_membership.authorized_spaces = _authorized_spaces
    # SCRUM is one of the user's spaces; "scrum" in lower case is just a word.
    plain_word = await routes.ask(
        routes.AskRequest(question="Which ATC issues follow our scrum process?", spaceIds=[1]), plain_request,
    )

    assert named.answer == "問題提到 PRM，但這次沒有勾選 PRM，所以搜尋不到它的資料。請勾選 PRM 後再問一次。"
    assert not plain_word.abstained


async def test_a_selected_space_or_an_unknown_prefix_is_not_a_scope_problem():
    request = _two_space_request(_Router("single", ()))

    response = await routes.ask(
        routes.AskRequest(question="Compare ATC-43 and PRM-14, and order INV-3300", spaceIds=[1, 2]), request,
    )

    assert response.answer.startswith("ATC-77 is blocked")


async def test_stream_sends_the_notice_as_the_only_result():
    req = routes.AskRequest(question="What does PRM-20 do?", spaceIds=[1])

    streaming = await routes.ask_stream(req, _atc_only_request())
    frames = [frame async for frame in streaming.body_iterator]

    assert len(frames) == 1 and frames[0].startswith("event: result")
    assert json.loads(frames[0].split("data: ", 1)[1])["abstained"] is True


async def test_the_answer_shows_the_mode_router_and_summary_costs(monkeypatch):
    _capture_charges(monkeypatch)
    request = _two_space_request(_Router("single"))
    request.app.state.history_compactor = _Compactor()

    response = await routes.ask(_ask_with_mode("deep", _LONG_HISTORY), request)

    roles = [u.role for u in response.model_usage]
    assert "mode_router" in roles and "history_summary" in roles
    extra = _ROUTER_CHARGE.cost_usd + sum(u.estimated_cost_usd for u in response.model_usage
                                          if u.role == "history_summary")
    assert response.estimated_cost_usd == pytest.approx(0.001 + extra)
