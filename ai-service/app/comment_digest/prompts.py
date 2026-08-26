"""Prompt rendering for comment-thread digests. Templates are loaded and fingerprinted by
app/llm/prompt_registry.py."""
from __future__ import annotations

import re
from typing import List

from app.comment_digest.schemas import CommentIn
from app.llm.prompt_registry import template_loader

_prompts = template_loader(__file__)

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def _strip_html(content: str) -> str:
    """Comment bodies are stored as HTML (see the frontend's commentHtml.ts). Only plain text is
    useful to the model here, and raw markup just burns tokens and adds noise — this is a rendering
    convenience, not a security boundary (nothing here is ever re-rendered as HTML)."""
    text = _TAG_RE.sub(" ", content)
    text = _WHITESPACE_RE.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def render_comment_digest_system_prompt() -> str:
    return _prompts.render("comment_digest_system.jinja")


def render_comment_thread_message(comments: List[CommentIn], truncated_count: int) -> str:
    lines = []
    if truncated_count:
        lines.append(
            f"(Showing the most recent {len(comments)} comments; {truncated_count} earlier "
            "comment(s) were omitted. If your summary would materially depend on earlier context "
            "you don't have, say so rather than guessing.)"
        )
    for comment in comments:
        lines.append(f"--- comment id={comment.id} author={comment.author_name} at={comment.created_at} ---")
        lines.append(_strip_html(comment.content) or "(empty comment)")
    lines.append("--- end of thread ---")
    return "\n".join(lines)
