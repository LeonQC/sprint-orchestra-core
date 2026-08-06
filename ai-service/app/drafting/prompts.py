"""Jinja2 prompt rendering for AI-assisted task creation.

This was the first prompt in the project to become a template file rather than an f-string, for two
reasons that have since made every other prompt follow it: it has a genuine conditional section —
whether to list `existing_labels` at all — which Jinja2's `{% if %}` expresses directly in the prompt
text instead of building the string up with Python branches, and the wording lives in one `.jinja`
file that can be diffed and versioned on its own, separate from the code that calls it. See
`app/llm/prompt_registry.py` for what "versioned" now means concretely.
"""
from __future__ import annotations

from typing import List, Optional

from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)


def render_task_draft_prompt(existing_labels: Optional[List[str]] = None) -> str:
    return _prompts.render("task_draft_system.jinja", existing_labels=existing_labels or [])
