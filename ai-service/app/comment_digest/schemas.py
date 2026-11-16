"""Structured output schema for "Summarize comments".

Every claim carries `comment_ids` — the comments that support it — so an invented decision can't
pass as a summary, and the UI can jump to the source comment. `service.py::_ground` then drops any
point whose ids aren't in the real thread (the model can still invent ids).

`CommentIn` is request input, never parsed from model output, so it's a plain BaseModel; only
`CommentDigest` and its nested models extend StructuredOutput.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.llm.structured_output import StructuredOutput


class CommentIn(BaseModel):
    id: int
    author_name: str
    created_at: str
    content: str


class AttributedPoint(StructuredOutput):
    # 220 is a safety net against runaway sentences, not the brevity mechanism (the prompt does that).
    # A low hard cap cut sentences mid-thought and made output confusing, not concise.
    text: str = Field(..., min_length=1, max_length=220)
    comment_ids: List[int] = Field(
        default_factory=list,
        description="Id(s), from the numbered comment list you were given, of every comment that "
                    "actually supports this point. Never invent an id or reuse one that doesn't "
                    "genuinely back this specific claim — a point with no real id will be dropped "
                    "before the user sees it, so an empty list here is a wasted point, not a safe "
                    "default.",
    )


class ActionItemOut(AttributedPoint):
    """`text`/`comment_ids` inherited unchanged from `AttributedPoint` — an action item is an
    attributed point that also carries an owner."""

    owner: Optional[str] = Field(
        None, description="Who owns this action, only if a specific person is named or unambiguously "
                           "implied by the comments. Null when the thread doesn't say.",
    )


class Outcome(StructuredOutput):
    """Not `AttributedPoint`: that model's `text` is `min_length=1` (a list item is only ever
    included when it has real content), but an outcome is a required top-level field that must be
    able to legitimately say "nothing was resolved" — see the empty-string note below."""

    # No min_length: "" is also the degraded/skipped sentinel (service.py::_empty_digest) and the
    # legitimate "nothing was resolved".
    text: str = Field(
        ..., max_length=260,
        description="Plain, spoken-language sentence(s) — usually one — for where the thread landed "
                    "as of the last comment. Keep it short by default, but don't cut something the "
                    "reader needs just to save words; a confusing short sentence is worse than a "
                    "clear longer one. Empty string only when nothing was actually resolved — an "
                    "open thread is a legitimate outcome, don't manufacture a resolution to sound "
                    "complete.",
    )
    comment_ids: List[int] = Field(
        default_factory=list,
        description="Id(s) of the comment(s) that most directly establish this outcome — same rule "
                    "as AttributedPoint.comment_ids. Empty only when `text` is also empty.",
    )


class CommentDigest(StructuredOutput):
    """`context`/`discussion`/`outcome` replace what used to be a single `summary` field — found live
    (direct product feedback) that one terse "where it landed" sentence read as a restatement of a
    separate `decisions` list and nothing else, not a summary a PM could actually use to catch up on
    a thread they hadn't read. Splitting it into the three-part shape an actual status report has
    forces the model to address each part rather than collapsing everything into an outcome-only
    sentence; a `context` or `discussion` field that ends up empty is now visible on its own, not
    silently absent from a single paragraph.

    `decisions` as a standalone field was then dropped — also direct product feedback: showing "what
    was decided" twice (once as `outcome` prose, once as an itemized `decisions` entry citing the
    same comment) read as redundant, not as two different pieces of information. `outcome` now
    carries its own `comment_ids` (see `Outcome` above) so the one place that states the resolution
    is also the one place that cites it. `open_questions` and `action_items` stay separate lists:
    unlike a decision, an open question or an action item is genuinely NOT restated by `outcome` —
    the digest can (and often does) have several of either.

    The character limits below went through a round trip: first tightened hard (context/outcome to
    140, discussion to 260) because the digest's total length was creeping up on the comments' own
    length — then loosened back up (to 260/450) on the next round of feedback, because that tight a
    ceiling forced the model to cut things a reader actually needed, and made a genuinely tangled
    thread read as vague rather than concise. The limits here are a safety net against a runaway
    response, not the thing that keeps the digest short — that's the system prompt's job (see
    templates/comment_digest_system.jinja's "clarity first, brevity second" framing).
    """
    context: str = Field(
        ..., max_length=260,
        description="Plain, spoken-language sentence(s) — usually one — for why this thread exists: "
                    "what prompted it (a bug found, a question raised, a review requested, work "
                    "reported). Ground this in what the comments themselves say or clearly imply; if "
                    "the thread jumps straight into discussion with no stated trigger, say what the "
                    "earliest comment raises instead of inventing a reason. Keep it short by default, "
                    "but don't leave out the part a reader actually needs just to save words.",
    )
    discussion: str = Field(
        ..., max_length=450,
        description="Plain, spoken-language sentences — usually one or two — for what was actually "
                    "debated: differing views, objections, what changed someone's mind. This is the "
                    "part a chronological recap gets wrong by listing who-said-what instead of what "
                    "was actually at issue; name the disagreement and how it resolved. Give it the "
                    "room it needs to make sense — a genuinely tangled disagreement can run longer "
                    "than a simple one, and a vague compressed version that leaves the reader unsure "
                    "what was actually argued is worse than a clearer, slightly longer one. Empty "
                    "string only when the thread has no real back-and-forth (e.g. a single status "
                    "update or one comment answering another with no pushback) — do not invent a "
                    "debate to fill this in.",
    )
    outcome: Outcome
    open_questions: List[AttributedPoint] = Field(
        default_factory=list,
        description="Questions raised in the thread that are still unresolved as of the last comment.",
    )
    action_items: List[ActionItemOut] = Field(
        default_factory=list,
        description="Concrete next steps someone committed to or was asked to do.",
    )
