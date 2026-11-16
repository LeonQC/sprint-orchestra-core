"""Prompt rendering for the CRAG agent loop.

These were the last f-string prompts; as .jinja files they get a version like every other prompt
(see prompt_registry.py). The templates were generated from the old f-strings, so the text sent to
the model is byte-identical to earlier eval runs; tests/test_crag_loop.py pins the key wording.
"""
from __future__ import annotations

from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)


def render_capability_router_prompt() -> str:
    """System prompt for the first-tool router — static, no interpolation."""
    return _prompts.render("crag_router_system.jinja")


def render_crag_system_prompt(now_utc: str, section_tool: bool = True) -> str:
    """Rendered per request because it includes the current time (for "10 minutes ago" questions). The
    time is rounded to 5 minutes so prompt caching can still hit (see `_current_time_for_prompt`).
    """
    return _prompts.render("crag_answer_system.jinja", now_utc=now_utc, section_tool=section_tool)
