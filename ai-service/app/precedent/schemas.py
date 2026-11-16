"""Structured output for "Link similar work items", built as a precedent finder.

"Issues with similar text" is already covered by /search (and the Create Issue modal uses it where
it prevents duplicates). The real question when picking up a ticket is "has someone here already
solved this?", which differs in two ways:
  1. A precedent should usually be DONE; an open lookalike is a duplicate risk, not a lesson.
     service.py ranks finished work first.
  2. The output says WHY it's worth opening, not just how similar it is.

Uses title + description only, never comment threads (see service.py); `takeaway` is filled only when
a description states an approach. Every `issue_key` is checked against the real candidates
(`service.py::_ground`). `verdict` makes "nothing useful here" a proper answer — a tool that always
shows three vague tickets teaches people to ignore it.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import Field

from app.llm.structured_output import StructuredOutput


class Precedent(StructuredOutput):
    issue_key: str = Field(
        ...,
        description="The issue key, copied exactly from the candidate list you were given. Never "
                    "invent a key and never cite one that isn't in that list — it is checked "
                    "against the real candidates and dropped if it doesn't match.",
    )
    relevance: str = Field(
        ..., min_length=1, max_length=280,
        description="Why this is worth opening BEFORE starting the current work, in one plain "
                    "spoken sentence. Say what it actually has in common — the same subsystem, the "
                    "same failure, the same decision — not that it is 'similar' or 'related', which "
                    "the reader can already see from the title.",
    )
    takeaway: Optional[str] = Field(
        None, max_length=280,
        description="The specific thing to reuse or watch out for, ONLY if the candidate's own "
                    "description actually states one — an approach, a constraint, a decision and "
                    "its reason. Null is the common answer: a description rarely spells out what "
                    "was learned, and you have no other source to draw on here. Do not guess at "
                    "what a team 'probably' found out, and do not infer a lesson from the title "
                    "alone.",
    )


class PrecedentReport(StructuredOutput):
    verdict: Literal["found", "none_relevant"] = Field(
        ...,
        description="'none_relevant' when nothing in the candidates would genuinely help someone "
                    "starting this work — a legitimate and common answer, especially for novel work. "
                    "Superficial topic overlap is not relevance: prefer 'none_relevant' over listing "
                    "a ticket whose only connection is shared vocabulary.",
    )
    precedents: List[Precedent] = Field(
        default_factory=list,
        description="At most 3, best first, and fewer is better than padded. Empty when the verdict "
                    "is 'none_relevant'.",
    )
