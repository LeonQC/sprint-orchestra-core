"""Detect the section layout this board's issue descriptions use, by counting (no LLM call).

Why: grounding showed the planner eight real issues and asked it to "match their style", which
measured as a coin flip (13-11 in a blind test). Every finished issue on the board uses a
fixed template (Context / Acceptance criteria / Estimate reasoning), while the prompt told the planner
to write "1-3 plain sentences" — and the specific rule won. So the layout is now detected and stated
as an explicit output format instead of left for the model to imitate.

How: a section label appears in most descriptions and only once in each ("Context" once per issue,
"The" six times). The detected labels come back on the API response with their coverage, so a wrong
guess is visible.

The layout is applied even when patterns.py judges the history irrelevant: relevance decides what the
history teaches, not what the board's paperwork looks like.

Known weakness: a phrase that behaves exactly like a heading (e.g. every Context section starting
with the same word) can't be told apart by counting. `coverage` and `sample_size` on the response
make that visible.

Only section labels — not tone or vocabulary, which is the part that measured as a coin flip.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Fraction of issues a label must appear in. Not stricter: older issues may predate the template.
_MIN_COVERAGE = 0.6

# Average times a label may appear per issue. A heading appears once; "The" appears many times. This
# ratio does most of the work a stopword list would.
_MAX_MEAN_REPEATS = 1.3

# Longest label, in words ("Definition of done" is three).
_MAX_LABEL_WORDS = 3

# Share of issues the reported "typical" item-count range must cover.
_ITEM_COUNT_COVERAGE = 0.8

# Sentence openers that can slip past the repeat test on a tiny sample (e.g. "This").
_NOT_LABELS = frozenset({
    "the", "this", "that", "these", "those", "it", "they", "we", "a", "an", "and", "but", "if",
    "when", "then", "give", "add", "make", "use", "allow", "support", "show", "keep", "store",
})

# Sections an EPIC skips: epics carry no story points here, so an estimate section would explain a
# number that doesn't exist. (Acceptance criteria stay: epics on this board do have outcome-level
# criteria.) Matched on words, so "Sizing notes" and "Story point rationale" are caught too.
_EPIC_EXCLUDED_WORDS = frozenset({"estimate", "estimation", "estimating", "sizing", "points"})

# Letters and inner hyphens only. `[A-Za-z][A-Za-z-]*` matched "ATC-" out of "ATC-74", producing a
# phrase that no longer exists in the source text.
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

# A labelled list item like "AC1 —" (after HTML is stripped). Prefix capped at 5 letters so a
# capitalised word followed by a year doesn't match.
_ITEM_MARKER = re.compile(r"\b([A-Z]{1,5})(\d{1,2})\s*[—–-]")

# The chunker prepends "ATC-74 Subtask of ATC-51 (...)." to every body. It's in 100% of descriptions
# once each, so it would be detected as a heading — strip it first.
_CHUNK_PREFIX = re.compile(
    r"^\s*[A-Z][A-Z0-9]*-\d+\s*(?:Subtask of\s+[A-Z][A-Z0-9]*-\d+\s*\([^)]*\)\.?)?\s*"
)


@dataclass(slots=True)
class HouseFormat:
    """The section labels this board's descriptions are built from, in the order they appear."""

    #: Labels kept, ordered by where they typically appear in a description.
    sections: List[str] = field(default_factory=list)
    #: Usual and maximum number of labelled items per description. Needed because with the right
    #: sections the planner still wrote 3-4 criteria per issue on a board where none has four.
    item_count_low: Optional[int] = None
    item_count_high: Optional[int] = None
    item_count_max: Optional[int] = None
    #: List-item prefix, e.g. "AC" for `AC1 — …`. Other issues cite these ("ATC-74 AC1"), so a plan
    #: numbered 1., 2., 3. can't be referenced. Detected separately because the heading rule drops
    #: trailing all-caps tokens.
    item_marker: Optional[str] = None
    #: label -> fraction of sampled issues containing it, returned so the claim can be checked.
    coverage: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0

    @property
    def detected(self) -> bool:
        return bool(self.sections)

    @property
    def epic_sections(self) -> List[str]:
        """Sections for an epic: all except the estimate one.

        Computed here so the prompt gets two explicit lists instead of asking the model to subtract.
        """
        return [
            section for section in self.sections
            if not any(word in section.lower().split() for word in _EPIC_EXCLUDED_WORDS)
        ]


def _candidate_counts(text: str) -> Dict[str, int]:
    """Every 1..N-word phrase that could be a heading, with how often it occurs.

    Must start with a capitalised word ("Acceptance criteria" yes, "criteria for acceptance" no).
    """
    words = _WORD.findall(_CHUNK_PREFIX.sub("", text))
    counts: Dict[str, int] = {}
    for index, first in enumerate(words):
        if not first[:1].isupper() or first.lower() in _NOT_LABELS:
            continue
        for length in range(1, _MAX_LABEL_WORDS + 1):
            if index + length > len(words):
                break
            phrase = words[index:index + length]
            if any(word.lower() in _NOT_LABELS for word in phrase[1:]):
                break
            # A trailing acronym is a project prefix or list marker ("AC1" tokenizes as "AC"), not part
            # of a heading — otherwise "Acceptance criteria AC" hides the real label.
            if len(phrase[-1]) > 1 and phrase[-1].isupper():
                continue
            counts[" ".join(phrase)] = counts.get(" ".join(phrase), 0) + 1
    return counts


def _item_counts(descriptions: Sequence[str], marker: Optional[str]) -> tuple:
    """Labelled items per description: (typical low, typical high, max).

    Not an average — "1.65 criteria" describes no real issue; "usually 1-2, never more than 3" does.
    Only counts descriptions that have items, so issues without criteria don't drag it toward zero.
    """
    if not marker:
        return (None, None, None)
    pattern = re.compile(rf"\b{re.escape(marker)}\d{{1,2}}\s*[—–-]")
    counts = sorted(
        count for count in (len(pattern.findall(text)) for text in descriptions) if count > 0
    )
    if not counts:
        return (None, None, None)
    # Mode, not median, for the low end: with 49% one / 37% two / 14% three, the median is 2 and would
    # push the planner to always write two. Ties go to the smaller count, since the bug was writing
    # too many.
    frequency: Dict[int, int] = {}
    for count in counts:
        frequency[count] = frequency.get(count, 0) + 1
    low = min(value for value, times in frequency.items() if times == max(frequency.values()))
    high = low
    for value in counts:
        high = value
        if sum(1 for count in counts if count <= value) / len(counts) >= _ITEM_COUNT_COVERAGE:
            break
    return (low, max(high, low), counts[-1])


def detect_house_format(descriptions: Sequence[str]) -> HouseFormat:
    """Find the section labels most of `descriptions` are built from.

    Input is the flattened text retrieval returns (HTML already stripped), so a `<h3>Context</h3>`
    arrives as the bare word "Context". The two counting signals don't need the markup.
    """
    usable = [text for text in descriptions if text and text.strip()]
    if len(usable) < 2:
        # One description cannot establish a convention; every phrase in it has 100% coverage.
        return HouseFormat(sample_size=len(usable))

    document_frequency: Dict[str, int] = {}
    total_occurrences: Dict[str, int] = {}
    first_position: Dict[str, List[float]] = {}
    for text in usable:
        counts = _candidate_counts(text)
        for phrase, count in counts.items():
            document_frequency[phrase] = document_frequency.get(phrase, 0) + 1
            total_occurrences[phrase] = total_occurrences.get(phrase, 0) + count
            # Average relative position, so sections come out in reading order.
            index = text.find(phrase)
            # Recorded even if not found verbatim (punctuation between words), so ordering never
            # crashes on a missing position.
            first_position.setdefault(phrase, []).append(
                index / max(len(text), 1) if index >= 0 else 1.0
            )

    marker_frequency: Dict[str, int] = {}
    for text in usable:
        # Count once per document, not per item.
        for prefix in {match.group(1) for match in _ITEM_MARKER.finditer(text)}:
            marker_frequency[prefix] = marker_frequency.get(prefix, 0) + 1
    item_marker = next(
        (
            prefix for prefix, frequency in sorted(
                marker_frequency.items(), key=lambda pair: -pair[1]
            )
            if frequency / len(usable) >= _MIN_COVERAGE
        ),
        None,
    )
    item_low, item_high, item_max = _item_counts(usable, item_marker)

    kept: Dict[str, float] = {}
    for phrase, frequency in document_frequency.items():
        coverage = frequency / len(usable)
        if coverage < _MIN_COVERAGE:
            continue
        if total_occurrences[phrase] / frequency > _MAX_MEAN_REPEATS:
            continue
        kept[phrase] = coverage

    # When one label contains another, decide by coverage, not length:
    #   "Context" (100%) vs "Context Provider work" (67%) -> the shorter one; running into the prose
    #   can only lose documents.
    #   "Definition" (100%) vs "Definition of done" (100%) -> a tie goes to the longer one.
    redundant = set()
    for short in kept:
        for long in kept:
            if short == long or f" {short} " not in f" {long} ":
                continue
            if kept[long] < kept[short]:
                redundant.add(long)
            else:
                redundant.add(short)
    sections = sorted(
        (phrase for phrase in kept if phrase not in redundant),
        key=lambda phrase: sum(first_position[phrase]) / len(first_position[phrase]),
    )
    if sections:
        logger.info(
            "detected house description format: %s (item marker: %s)", sections, item_marker,
        )
    return HouseFormat(
        sections=sections,
        item_marker=item_marker,
        item_count_low=item_low,
        item_count_high=item_high,
        item_count_max=item_max,
        coverage={phrase: round(kept[phrase], 3) for phrase in sections},
        sample_size=len(usable),
    )


def section_compliance(description: str, house: HouseFormat) -> float:
    """Fraction of the house sections a generated description contains.

    Unlike "writing style", anyone can recompute this. Exposed per issue on the API response.
    """
    if not house.sections:
        return float("nan")
    text = (description or "").lower()
    return round(
        sum(1 for section in house.sections if section.lower() in text) / len(house.sections), 3
    )
