"""Adaptive single vs multi-agent RAG mode router.

This is a traffic light, not a research agent: it only chooses an execution path.
Default is conservative (prefer single). Multi needs BOTH of:
- two or more workspaces in scope (each worker gets one). On one workspace multi measured no
  better than single at ~2x the time and cost; its clear
  win was cross-project questions.
- a question that actually compares or splits across those workspaces. Selecting ATC+PRM in the
  UI and asking "What is ATC-42?" is still one search.

Code decides the facts (attachment scope, how many workspaces); with two or three workspaces the
classifier decides the meaning. The user's Fast pick skips all of this (manual_mode).

Wired into POST /ask when Settings.adaptive_rag_routing_enabled is true.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from app.llm.types import LLMClient
from app.llm.usage_ledger import ModelCharge, charge_for
from app.tracing import llm_step

logger = logging.getLogger(__name__)

ExecutionMode = Literal["single", "multi"]
ModeRouteSource = Literal["disabled", "rule", "model", "fallback", "user"]

_MODE_TOOL = {
    "name": "choose_execution_mode",
    "description": (
        "Choose whether one search agent is enough, or several independent search "
        "agents should investigate then combine answers."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {"type": "string", "enum": ["single", "multi"]},
            "reason": {"type": "string"},
        },
        "required": ["mode", "reason"],
    },
}

# The classifier only runs with 2+ workspaces, and multi then splits by workspace, not by topic, so a
# two-part question is not a reason for multi. "Several facets" used to be one; it sent 6 of the 50
# one-workspace eval questions to multi when two workspaces were selected
#.
_MODE_PROMPT = """You route Ask AI questions to one of two execution modes.

The request may authorize one or more workspaces. Authorized count alone does NOT mean multi:
if the question is about one issue, one fact, or one named project, choose single even when
several workspaces are authorized.

single — one Corrective RAG agent (default). Use for one fact, one issue, one sprint,
one document, simple lookups, or questions that only need one research thread. These are
single too, even with several workspaces authorized:
- a question with two parts about the same scope ("How many bugs are there in total, and how
  many are still open?")
- a list over everything in scope ("Which issues are currently blocked?")
- comparing two issues or two sprints ("Between ATC-43 and ATC-59, which was completed first?")

multi — one CRAG worker per workspace, in parallel → synthesize. Use ONLY when the answer
must be given per project/workspace or contrasts them, for example:
- compare/contrast two or more projects or workspaces
- "for each workspace separately" / cross-project risk comparisons

Do NOT choose multi just because multiple workspaces are authorized. Prefer single when unsure.
Call choose_execution_mode exactly once."""


@dataclass(frozen=True, slots=True)
class ModeDecision:
    mode: ExecutionMode
    source: ModeRouteSource
    reason: str
    # Cost of the classifier call, when one ran. The caller adds it to the request's charges so the
    # usage ledger and per-user quota see it.
    charges: tuple[ModelCharge, ...] = ()


def rule_mode(
    question: str,
    space_ids: list[int],
    *,
    has_attachment_scope: bool = False,
    max_multi_spaces: int = 3,
) -> ModeDecision | None:
    """The decision the rules alone can make, or None when the classifier has to decide.

    A plain function so evals can apply the same routing to recorded answers without a model call.
    """
    # An ask about one attached issue stays on the single CRAG contract (attachment keys are not
    # part of multi investigation). A follow-up does not: the route hands multi the earlier turns.
    if has_attachment_scope:
        return ModeDecision("single", "rule", "attachment-scoped ask uses one search agent")

    scopes = list(dict.fromkeys(space_ids))
    if len(scopes) > max_multi_spaces:
        return ModeDecision(
            "single", "rule",
            f"more than {max_multi_spaces} workspaces selected; using one search agent",
        )
    if len(scopes) < 2:
        return ModeDecision("single", "rule", "one workspace uses one search agent")

    # Two or three workspaces: whether the question compares them is meaning, not a fact, so the
    # classifier decides. A keyword regex used to decide here and sent "Compare ATC-43 and ATC-59"
    # to multi: it can't tell two projects from two issues.
    return None


def manual_mode(requested: str) -> ModeDecision | None:
    """The user's pick in the UI. Fast is one search agent; Deep returns None, so the router decides.

    Deep means "multi may be used if the question suits it", not "always multi": most questions
    still get one agent, which measured as accurate at about half the cost.
    """
    if requested == "fast":
        return ModeDecision("single", "user", "you chose Fast: one search agent")
    return None


class ModeRouter:
    """Choose single CRAG vs multi-agent investigation for one Ask request."""

    def __init__(
        self, llm: LLMClient | None = None, *, enabled: bool = False,
        model: str = "", provider: str = "Unknown", deployment: str = "hosted",
    ):
        self._llm = llm
        self._enabled = enabled
        # Only used to price the classifier call.
        self._model = model
        self._provider = provider
        self._deployment = deployment

    async def choose(
        self,
        question: str,
        space_ids: list[int],
        *,
        has_history: bool = False,
        has_attachment_scope: bool = False,
        max_multi_spaces: int = 3,
    ) -> ModeDecision:
        if not self._enabled:
            return ModeDecision("single", "disabled", "adaptive routing off")

        decision = rule_mode(
            question, space_ids,
            has_attachment_scope=has_attachment_scope, max_multi_spaces=max_multi_spaces,
        )
        if decision is not None:
            return decision
        scopes = list(dict.fromkeys(space_ids))

        # Ambiguous questions: optional small LLM classify; prefer single.
        if self._llm is None:
            return ModeDecision("single", "fallback", "defaulting to one search agent")

        charges: tuple[ModelCharge, ...] = ()
        try:
            with llm_step("ask.mode_router"):
                turn = await self._llm.next_turn(
                    _MODE_PROMPT,
                    [{"role": "user", "content": (
                        f"Authorized workspaces: {len(scopes)}\nQuestion: {question}"
                        + ("\nThis is a follow-up in a chat; the earlier turns are not shown."
                           if has_history else "")
                    )}],
                    [_MODE_TOOL],
                    tool_choice="choose_execution_mode",
                )
            charges = (
                charge_for("mode_router", self._provider, self._model, self._deployment, turn.usage),
            )
            calls = [c for c in turn.tool_calls if c.name == "choose_execution_mode"]
            if len(calls) != 1:
                raise ValueError("mode router returned no structured choice")
            mode = calls[0].input.get("mode")
            reason = str(calls[0].input.get("reason") or "")[:500]
            if mode not in ("single", "multi"):
                raise ValueError(f"invalid mode {mode!r}")
            return ModeDecision(mode, "model", reason or "model choice", charges)
        except Exception as exc:  # noqa: BLE001 - routing must not fail the ask
            logger.warning("mode router LLM failed; defaulting to single: %s", exc)
            return ModeDecision(
                "single", "fallback", "mode router failed; using one search agent", charges,
            )
