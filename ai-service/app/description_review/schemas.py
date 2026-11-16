"""Structured output schema for "Improve description".

Built to avoid the two things that make AI copy-editing untrustworthy — always finding something to
"improve", and quietly inventing content:

1. Saying "fine as is" is the main path. A tool that never says that can't be trusted when it does
   speak up.
2. The model never writes the new description. It reports findings quoting exact text; service.py
   checks each quote and applies the replacement itself. So nothing can be invented, and the HTML
   structure and attachment chips are never regenerated.

`kind` is a closed set: grammar, typos, unclear wording. "Add missing acceptance criteria" is not
allowed — that would be the model writing requirements nobody agreed to.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from app.llm.structured_output import StructuredOutput


class Finding(StructuredOutput):
    kind: Literal["typo", "grammar", "unclear", "ambiguous"] = Field(
        ...,
        description="What kind of problem this is. Only these four — you are copy-editing, not "
                    "rewriting requirements. Missing acceptance criteria, missing context, or "
                    "anything else that would need NEW information is out of scope: leave it alone.",
    )
    quote: str = Field(
        ..., min_length=1, max_length=300,
        description="The exact text from the description you want to change, copied character for "
                    "character so it can be found again. Quote only the span that actually needs "
                    "editing — a word or a phrase, not a whole paragraph. Never paraphrase the "
                    "quote, and never quote text that isn't in the description: it will be checked "
                    "against the real description and dropped if it doesn't match exactly.",
    )
    suggestion: str = Field(
        ..., min_length=1, max_length=300,
        description="What that exact span should say instead. Same meaning, said better — never new "
                    "facts, requirements, or details the description didn't already state.",
    )
    why: str = Field(
        ..., min_length=1, max_length=200,
        description="One short, plain sentence on what's wrong with the original wording. This is "
                    "what the reader uses to decide whether to accept the change, so 'reads as if "
                    "the retry is automatic when the next line says it's manual' is useful and "
                    "'improves clarity' is not.",
    )


class DescriptionReview(StructuredOutput):
    verdict: Literal["fine_as_is", "needs_changes"] = Field(
        ...,
        description="'fine_as_is' when the description is already clear enough to work from — this "
                    "is a good, common answer and the one to prefer when it's true. Choose "
                    "'needs_changes' only when you have at least one concrete finding; a verdict of "
                    "'needs_changes' with no findings is a contradiction, and 'fine_as_is' is the "
                    "right answer whenever the only things you could say are stylistic preferences.",
    )
    findings: List[Finding] = Field(
        default_factory=list,
        description="The specific changes you're proposing. Empty when the verdict is 'fine_as_is'. "
                    "Do not pad this with nitpicks to look thorough — every entry is something a "
                    "person has to read and approve, so a weak finding costs them more than it's "
                    "worth.",
    )
