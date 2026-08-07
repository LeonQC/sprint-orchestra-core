"""Prompt rendering for the sprint-recovery workflow.

Wording lives in the templates (tuned and versioned like other prompts); data assembly stays in
graph.py next to the state it reads. Templates were generated from the old f-strings, so the text is
byte-identical to earlier eval runs; tests/test_sprint_recovery_graph.py pins the key wording.
"""
from __future__ import annotations

from typing import List, Optional

from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)


def render_diagnose_prompt(*, sprint_name: str, goal_text: str, signals_text: str, evidence_text: str) -> str:
    return _prompts.render(
        "diagnose_system.jinja",
        sprint_name=sprint_name, goal_text=goal_text,
        signals_text=signals_text, evidence_text=evidence_text,
    )


def render_plan_prompt(
    *, sprint_name: str, goal_text: str, snapshot_text: str, hypotheses_text: str,
    prior_note: str, known_issue_keys: List[str], human_note: Optional[str] = None,
) -> str:
    """`human_note` is the one conditional section (`{% if %}`), asked for only when a note is pending,
    so no tokens are spent and no reply invented otherwise.
    """
    return _prompts.render(
        "plan_system.jinja",
        sprint_name=sprint_name, goal_text=goal_text, snapshot_text=snapshot_text,
        hypotheses_text=hypotheses_text, prior_note=prior_note,
        known_issue_keys=", ".join(known_issue_keys), human_note=human_note,
    )
