"""Shorten a long Ask AI chat before the model sees it.

The UI holds the conversation and sends all of it with every question. Once the turns before the
latest question and answer pass SUMMARIZE_OVER_TOKENS, they are replaced by a short summary written
by the model. The summary goes back to the UI with the answer, and the UI sends it with the next
question, so the same turns are never summarized twice: a summary written once is reused for
every later question in that chat.

The latest question and answer always stay word for word. Follow-ups mostly point at them
("which of those are done?"), and a summary can drop the exact row the user means.

Single and multi use the same result (routes.py): single gets the turns as chat history, multi
gets them written into the question.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from app.llm.types import LLMClient
from app.llm.usage_ledger import ModelCharge, charge_for
from app.tracing import llm_step

logger = logging.getLogger(__name__)

SUMMARIZE_OVER_TOKENS = 8_000
# The summary must never be larger than what it replaces.
SUMMARY_MAX_CHARS = 4_000

_SUMMARY_PROMPT = """You shorten the earlier part of a chat between a user and a Jira assistant,
so the assistant can still follow later questions that refer back to it ("the list you gave me
first", "that count").

Write at most 300 words of plain text, no preamble. Go turn by turn, in order, numbering the user's
questions ("Question 1: ..."). For each, keep what the user asked and what the answer concluded.
Copy exactly: every issue key (ATC-12) with the status and priority the answer gave it, every
number and count, date, sprint name, project or workspace name, and person's name. Drop
explanations, caveats and repeated text.
If an earlier summary is given, start from it and continue the numbering."""


@dataclass(frozen=True, slots=True)
class HistorySummary:
    text: str
    # How many history turns, counted from the start of the chat, the summary stands in for.
    covers_turns: int


@dataclass(frozen=True, slots=True)
class CompactHistory:
    # What the model sees: the summary in place of the turns it covers, then the rest word for word.
    turns: list[dict]
    # A summary written during this request, for the UI to keep. None when nothing new was written.
    new_summary: Optional[HistorySummary] = None
    charges: tuple[ModelCharge, ...] = ()


def approx_tokens(text: str) -> int:
    # English averages about 4 characters per token; close enough for a threshold.
    return len(text) // 4


def valid_summary(summary: Optional[HistorySummary], history_len: int) -> Optional[HistorySummary]:
    """The summary the UI sent, or None if it can't belong to this history.

    It must leave the latest question and answer uncovered. A summary from another chat, or one
    edited by hand, would otherwise hide turns the model needs.
    """
    if summary is None or not summary.text.strip():
        return None
    if summary.covers_turns < 1 or summary.covers_turns > history_len - 2:
        return None
    return summary


def apply_summary(history: list[dict], summary: Optional[HistorySummary]) -> list[dict]:
    """The turns to send: the summary in place of the turns it covers."""
    if summary is None:
        return [dict(turn) for turn in history]
    rest = [dict(turn) for turn in history[summary.covers_turns:]]
    note = f"Summary of the earlier part of this chat:\n{summary.text.strip()}"
    # Folded into the next user turn, so user and assistant turns still alternate.
    if rest[0]["role"] == "user":
        rest[0]["content"] = f"{note}\n\nThen the user asked: {rest[0]['content']}"
        return rest
    return [{"role": "user", "content": note}] + rest


class HistoryCompactor:
    def __init__(
        self, llm: LLMClient | None, *, model: str = "", provider: str = "Unknown",
        deployment: str = "hosted", over_tokens: int = SUMMARIZE_OVER_TOKENS,
    ):
        self._llm = llm
        # Only used to price the summary call.
        self._model = model
        self._provider = provider
        self._deployment = deployment
        self._over_tokens = over_tokens

    async def compact(
        self,
        history: list[dict],
        summary: Optional[HistorySummary] = None,
        on_stage: Optional[Callable[[str], None]] = None,
    ) -> CompactHistory:
        summary = valid_summary(summary, len(history))
        keep_from = len(history) - 2  # the latest question and answer stay word for word
        start = summary.covers_turns if summary else 0
        older = history[start:keep_from]
        older_tokens = sum(approx_tokens(turn["content"]) for turn in older)
        if self._llm is None or keep_from < 1 or older_tokens <= self._over_tokens:
            return CompactHistory(apply_summary(history, summary))

        if on_stage is not None:
            on_stage("summarizing the earlier part of this chat")
        lines = []
        if summary is not None:
            lines.append(f"Earlier summary:\n{summary.text.strip()}\n")
        lines.append("Turns to add:")
        for turn in older:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['content']}")

        charges: tuple[ModelCharge, ...] = ()
        try:
            with llm_step("ask.history_summary"):
                turn = await self._llm.next_turn(
                    _SUMMARY_PROMPT, [{"role": "user", "content": "\n".join(lines)}], [],
                )
            charges = (
                charge_for("history_summary", self._provider, self._model, self._deployment, turn.usage),
            )
            text = turn.text.strip()[:SUMMARY_MAX_CHARS]
            if not text:
                raise ValueError("empty summary")
        except Exception as exc:  # noqa: BLE001 - a failed summary must not fail the question
            # Send the turns as they are; the next question tries again.
            logger.warning("history summary failed; sending the full chat: %s", exc)
            return CompactHistory(apply_summary(history, summary), None, charges)

        new_summary = HistorySummary(text, keep_from)
        return CompactHistory(apply_summary(history, new_summary), new_summary, charges)
