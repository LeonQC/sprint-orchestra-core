"""Unit tests for adaptive single vs multi Ask AI routing."""
from __future__ import annotations

import pytest

from app.agent.mode_router import ModeRouter, manual_mode
from app.llm.types import LLMTurn, ToolCall, Usage


class _FakeModeLLM:
    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0

    async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
        self.calls += 1
        call = ToolCall("c1", "choose_execution_mode", {"mode": self.mode, "reason": "test"})
        return LLMTurn("tool_use", "", [call], {"role": "assistant", "content": []}, Usage(40, 5))


@pytest.mark.asyncio
async def test_disabled_always_single():
    router = ModeRouter(enabled=False)
    decision = await router.choose("Compare ATC and DVQPLH bug counts", [1, 2])
    assert decision.mode == "single"
    assert decision.source == "disabled"


@pytest.mark.asyncio
async def test_two_spaces_without_compare_cue_stays_single():
    """Selecting ATC+PRM is not enough — a one-fact question must not fan out workers."""
    router = ModeRouter(llm=None, enabled=True)
    decision = await router.choose("What is ATC-43 about?", [10, 20])
    assert decision.mode == "single"
    assert decision.source == "fallback"


@pytest.mark.asyncio
async def test_two_or_three_spaces_are_decided_by_the_classifier():
    """No keyword rule: "Compare ATC-43 and ATC-59" compares two issues, not two projects, and a
    regex sent it to multi."""
    for spaces in ([10, 20], [10, 20, 30]):
        llm = _FakeModeLLM("multi")
        router = ModeRouter(llm, enabled=True)
        decision = await router.choose("Compare ATC and PRM open bug counts", spaces)
        assert decision.mode == "multi"
        assert decision.source == "model"
        assert llm.calls == 1


@pytest.mark.asyncio
async def test_compare_wording_alone_does_not_pick_multi():
    llm = _FakeModeLLM("single")
    router = ModeRouter(llm, enabled=True)
    decision = await router.choose("Compare ATC-43 and ATC-59: which was completed first?", [10, 20])
    assert decision.mode == "single"
    assert decision.source == "model"


@pytest.mark.asyncio
async def test_five_spaces_fall_back_to_single_crag():
    router = ModeRouter(enabled=True)
    decision = await router.choose(
        "Find all blocked issues across every workspace",
        [10, 20, 30, 40, 50],
    )
    assert decision.mode == "single"
    assert decision.source == "rule"
    assert "more than 3 workspaces" in decision.reason


@pytest.mark.asyncio
async def test_one_space_stays_single_even_with_compare_cue():
    """Same-space multi measured no better than single at ~2x cost (measured)."""
    llm = _FakeModeLLM("multi")
    router = ModeRouter(llm, enabled=True)
    decision = await router.choose("Compare ATC-42 and ATC-43", [10])
    assert decision.mode == "single"
    assert decision.source == "rule"
    assert llm.calls == 0, "one workspace never needs the classifier call"


@pytest.mark.asyncio
async def test_ordinary_which_bug_stays_single():
    """Bare 'which bug' used to false-positive into multi — keep it on one agent."""
    router = ModeRouter(llm=None, enabled=True)
    decision = await router.choose("Which bug is blocking the release of ATC-42?", [10])
    assert decision.mode == "single"


@pytest.mark.asyncio
async def test_comparison_table_wording_stays_single():
    router = ModeRouter(llm=None, enabled=True)
    decision = await router.choose(
        "What is the comparison table in the attachment for ATC-43?", [10],
    )
    assert decision.mode == "single"


@pytest.mark.asyncio
async def test_a_follow_up_is_still_decided_by_the_classifier_and_told_it_is_one():
    llm = _FakeModeLLM("multi")
    seen = []
    original = llm.next_turn

    async def spy(system, messages, tools, **kwargs):
        seen.append(messages[0]["content"])
        return await original(system, messages, tools, **kwargs)

    llm.next_turn = spy
    router = ModeRouter(llm=llm, enabled=True)

    decision = await router.choose("Compare ATC vs DVQPLH", [10, 20], has_history=True)

    assert decision.mode == "multi"
    assert "follow-up" in seen[0]


@pytest.mark.asyncio
async def test_simple_fact_defaults_single_without_llm():
    router = ModeRouter(llm=None, enabled=True)
    decision = await router.choose("What is ATC-43 about?", [10])
    assert decision.mode == "single"
    assert decision.source == "rule"


@pytest.mark.asyncio
async def test_chinese_bijiao_meaning_more_stays_single():
    """比較 alone usually means "more" ("which bug is more serious"), not "compare"."""
    router = ModeRouter(llm=None, enabled=True)
    decision = await router.choose("哪個 bug 比較嚴重？", [10, 20])
    assert decision.mode == "single"


@pytest.mark.asyncio
async def test_classifier_call_is_charged():
    """The classifier call costs tokens; the /ask ledger and quota must see it."""
    router = ModeRouter(
        _FakeModeLLM("multi"), enabled=True, model="gpt-5.6-luna", provider="OpenAI",
    )
    decision = await router.choose("How are ATC and PRM doing?", [10, 20])
    assert decision.mode == "multi"
    assert decision.source == "model"
    assert len(decision.charges) == 1
    assert decision.charges[0].role == "mode_router"
    assert decision.charges[0].input_tokens == 40


@pytest.mark.asyncio
async def test_classifier_failure_still_charges_and_stays_single():
    router = ModeRouter(_FakeModeLLM("banana"), enabled=True, model="gpt-5.6-luna")
    decision = await router.choose("How are ATC and PRM doing?", [10, 20])
    assert decision.mode == "single"
    assert decision.source == "fallback"
    assert len(decision.charges) == 1


def test_manual_deep_leaves_the_choice_to_the_router():
    assert manual_mode("deep") is None


def test_manual_fast_is_always_single_and_never_reaches_the_router():
    decision = manual_mode("fast")

    assert decision.mode == "single"
    assert decision.source == "user"
