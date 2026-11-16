"""Prompt rendering for precedent search. Templates are loaded and fingerprinted by
app/llm/prompt_registry.py."""
from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass
from typing import List, Optional

from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

# A precedent is judged on what it was about, not on every word of its body — and a handful of
# candidates at full length would crowd out the ones ranked below them.
_MAX_DESC_CHARS = 700


def plain_text(markup: Optional[str], limit: int) -> str:
    """Strip tags, decode entities, collapse whitespace, truncate. Used for the model's input AND the
    retrieval query — with tags left in, one issue found 0 candidates instead of 4.
    """
    text = _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", markup or ""))).strip()
    return text[:limit] + ("…" if len(text) > limit else "")


@dataclass(slots=True)
class Candidate:
    """One retrieved issue plus the status ranking needs. No comment thread, by design (see
    service.py).
    """

    issue_key: str
    title: str
    status: Optional[str]
    issue_type: Optional[str]
    description: str = ""

    @property
    def is_done(self) -> bool:
        return (self.status or "").lower() == "done"


def render_precedent_system_prompt() -> str:
    return _prompts.render("precedent_system.jinja")


def render_precedent_message(
    current_key: str, current_title: str, current_description: str, candidates: List[Candidate]
) -> str:
    lines = [
        "CURRENT ISSUE (the work about to start):",
        f"  key: {current_key}",
        f"  title: {current_title}",
        f"  description: {plain_text(current_description, _MAX_DESC_CHARS)}",
        "",
        f"CANDIDATE ISSUES already in this space ({len(candidates)}):",
    ]
    for candidate in candidates:
        lines.append("")
        lines.append(f"--- candidate {candidate.issue_key} ---")
        lines.append(f"  status: {candidate.status or '(unknown)'}"
                     f"{'  [DONE — finished work]' if candidate.is_done else '  [not finished]'}")
        lines.append(f"  type: {candidate.issue_type or '(unknown)'}")
        lines.append(f"  title: {candidate.title}")
        lines.append(f"  description: {plain_text(candidate.description, _MAX_DESC_CHARS)}")
    lines.append("")
    lines.append("--- end of candidates ---")
    return "\n".join(lines)
