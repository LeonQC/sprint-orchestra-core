"""Long Ask AI chats: earlier turns summarized once, the latest question and answer kept word for word."""
from app.agent.history_summary import (
    HistoryCompactor, HistorySummary, apply_summary, valid_summary,
)
from app.llm.types import LLMTurn, Usage


class _Llm:
    def __init__(self, text="Question 1: PRM high issues not done: PRM-3, PRM-26.", exc=None):
        self.text = text
        self.exc = exc
        self.calls = []

    async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
        self.calls.append(messages[0]["content"])
        if self.exc:
            raise self.exc
        return LLMTurn(
            stop_reason="end_turn", text=self.text, tool_calls=[],
            assistant_message={"role": "assistant", "content": self.text}, usage=Usage(3000, 150),
        )


def _chat(pairs: int, answer_chars: int = 2000) -> list[dict]:
    turns = []
    for i in range(pairs):
        turns.append({"role": "user", "content": f"question {i}"})
        turns.append({"role": "assistant", "content": f"answer {i} " + "x" * answer_chars})
    return turns


async def test_a_short_chat_is_sent_as_it_is_without_a_model_call():
    llm = _Llm()
    history = _chat(3)

    result = await HistoryCompactor(llm, over_tokens=8_000).compact(history)

    assert result.turns == history
    assert result.new_summary is None
    assert llm.calls == []


async def test_a_long_chat_is_summarized_and_the_latest_answer_kept_word_for_word():
    llm = _Llm()
    history = _chat(10, answer_chars=4000)  # about 9,000 tokens before the latest answer

    result = await HistoryCompactor(llm, over_tokens=8_000).compact(history)

    assert result.new_summary.covers_turns == 18
    assert len(llm.calls) == 1
    assert result.turns[0]["role"] == "user"
    assert result.turns[0]["content"].startswith("Summary of the earlier part of this chat:")
    assert result.turns[0]["content"].endswith("Then the user asked: question 9")
    assert result.turns[1] == history[-1]
    assert [c.role for c in result.charges] == ["history_summary"]


async def test_a_summary_sent_back_is_reused_without_a_model_call():
    llm = _Llm()
    history = _chat(11, answer_chars=4000)
    earlier = HistorySummary("Question 1..10 summarized.", 18)

    result = await HistoryCompactor(llm, over_tokens=8_000).compact(history, earlier)

    assert llm.calls == []
    assert result.new_summary is None
    assert "Question 1..10 summarized." in result.turns[0]["content"]
    # The turns after the summary are sent whole: one older pair plus the latest pair.
    assert result.turns[1:] == history[19:]


async def test_new_turns_past_the_limit_are_merged_into_the_earlier_summary():
    llm = _Llm()
    history = _chat(22, answer_chars=4000)
    earlier = HistorySummary("Question 1..10 summarized.", 18)

    result = await HistoryCompactor(llm, over_tokens=8_000).compact(history, earlier)

    assert "Earlier summary:\nQuestion 1..10 summarized." in llm.calls[0]
    assert "question 9" in llm.calls[0] and "question 8" not in llm.calls[0]
    assert result.new_summary.covers_turns == 42


async def test_a_failed_summary_sends_the_full_chat():
    llm = _Llm(exc=RuntimeError("gateway down"))
    history = _chat(10, answer_chars=4000)

    result = await HistoryCompactor(llm, over_tokens=8_000).compact(history)

    assert result.turns == history
    assert result.new_summary is None


def test_a_summary_that_would_cover_the_latest_answer_is_ignored():
    assert valid_summary(HistorySummary("s", 4), history_len=4) is None
    assert valid_summary(HistorySummary("s", 0), history_len=4) is None
    assert valid_summary(HistorySummary("s", 2), history_len=4) is not None


def test_a_summary_before_an_assistant_turn_gets_its_own_user_turn():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
               {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
               {"role": "assistant", "content": "a3"}]

    turns = apply_summary(history, HistorySummary("s", 3))

    assert turns[0] == {"role": "user", "content": "Summary of the earlier part of this chat:\ns"}
    assert turns[1:] == history[3:]
