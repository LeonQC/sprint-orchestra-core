"""Prompt rendering for description review. Templates are loaded and fingerprinted by
app/llm/prompt_registry.py."""
from __future__ import annotations

import html
import re

from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)

_BLOCK_END_RE = re.compile(r"</(?:p|div|li|h[1-6]|tr|blockquote|pre)>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACES_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def strip_description_html(markup: str) -> str:
    """Plain text of a description, keeping block boundaries as newlines.

    Line structure is kept because the model must quote spans back exactly; flattened to one line, it
    quoted across lines and the quote couldn't be matched. Entities are decoded so the model sees what
    readers see (an undecoded `&#x27;` was reported as a typo). Quotes containing an entity then fail
    to match and are dropped — the safe direction. Not a security boundary; nothing is re-rendered.
    """
    text = _BR_RE.sub("\n", markup)
    text = _BLOCK_END_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _SPACES_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKLINES_RE.sub("\n\n", text).strip()


def render_description_review_system_prompt() -> str:
    return _prompts.render("description_review_system.jinja")


def render_description_message(description_text: str, title: str | None = None) -> str:
    """Title is included as context only — it is NOT reviewed and must never be quoted in a finding,
    since the caller applies replacements to the description body alone."""
    lines = []
    if title:
        lines.append(f"(Issue title, for context only — do not review or quote this: {title})")
    lines.append("--- description begins ---")
    lines.append(description_text)
    lines.append("--- description ends ---")
    return "\n".join(lines)
