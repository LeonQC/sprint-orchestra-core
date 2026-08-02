"""Source-provenance rules for grounded answers.

Documents contain key-like ids (INV-3300, HLT-439) that help pick a document but aren't sources
themselves. They're mapped to a real carrier issue only when this turn's authorized retrieval gives
exactly one match; nothing is ever guessed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class CitationCandidate(Protocol):
    issue_key: str
    content: str


@dataclass(frozen=True, slots=True)
class ImplicitSourceResolution:
    source_key: str | None
    matched_identifiers: tuple[str, ...] = ()


def resolve_implicit_source(
    subject_keys: set[str],
    citations: Sequence[CitationCandidate],
) -> ImplicitSourceResolution:
    """Resolve an explicit carrier key or document id to one retrieved source.

    Either the carrier key is among this turn's citations, or a document id from the question appears
    in chunks of exactly one retrieved carrier. Several candidates -> no resolution. Matches on token
    boundaries, so HLT-41 can't match HLT-410.
    """
    normalized_subjects = {key.upper() for key in subject_keys}
    carrier_keys = {citation.issue_key.upper() for citation in citations if citation.issue_key}

    explicit_carriers = normalized_subjects & carrier_keys
    if len(explicit_carriers) == 1:
        return ImplicitSourceResolution(next(iter(explicit_carriers)))
    if len(explicit_carriers) > 1:
        return ImplicitSourceResolution(None)

    document_identifiers = normalized_subjects - carrier_keys
    carriers_by_identifier: dict[str, set[str]] = {}
    for identifier in document_identifiers:
        pattern = re.compile(rf"(?<![A-Z0-9]){re.escape(identifier)}(?![A-Z0-9])", re.IGNORECASE)
        matching_carriers = {
            citation.issue_key.upper()
            for citation in citations
            if citation.issue_key and pattern.search(citation.content)
        }
        if matching_carriers:
            carriers_by_identifier[identifier] = matching_carriers

    matched_carriers = {
        carrier
        for carriers in carriers_by_identifier.values()
        for carrier in carriers
    }
    if len(matched_carriers) != 1:
        return ImplicitSourceResolution(None)
    source_key = next(iter(matched_carriers))
    matched_identifiers = tuple(sorted(
        identifier
        for identifier, carriers in carriers_by_identifier.items()
        if source_key in carriers
    ))
    return ImplicitSourceResolution(source_key, matched_identifiers)


def append_source_citation(text: str, source_key: str) -> str:
    """Append a verified carrier key while preserving terminal punctuation."""
    value = text.rstrip()
    punctuation = value[-1] if value.endswith((".", "!", "?")) else ""
    if punctuation:
        value = value[:-1].rstrip()
    return f"{value} ({source_key}){punctuation}"


def replace_trailing_citation(
    text: str,
    source_key: str,
    trailing_match: re.Match[str],
) -> str:
    """Replace an unsupported document-ID citation with verified provenance."""
    punctuation_match = re.search(r"[.!?]\s*$", text)
    punctuation = punctuation_match.group(0).strip() if punctuation_match else ""
    prefix = text[:trailing_match.start()].rstrip()
    return f"{prefix} ({source_key}){punctuation}"


# Every attachment chunk starts with this tag (vectorization pipeline): "[Attachment: a.pdf — issue X-1 — ...".
_ATTACHMENT_NAME_RE = re.compile(r"^\[Attachment: (?P<name>.+?) — issue ")
# A shorter stem ("spec", "notes") is an ordinary word, not proof the answer named the file.
_MIN_FILE_STEM_CHARS = 8


def carrier_named_by_file(text: str, citations: Sequence[CitationCandidate]) -> str | None:
    """The one retrieved carrier whose attachment the answer names by file, e.g.
    "(PRA-SS2-21-Outsourcing.pdf, §9.9)" instead of "(DVQFRESH200-5)". None if no file is named or
    the named files belong to more than one carrier.
    """
    carriers = set()
    for citation in citations:
        match = _ATTACHMENT_NAME_RE.match(citation.content or "")
        if not match or not citation.issue_key:
            continue
        name = match.group("name").strip()
        stem = name.rsplit(".", 1)[0]
        names = [name] if len(stem) < _MIN_FILE_STEM_CHARS else [name, stem]
        for value in names:
            if re.search(rf"(?<![\w.-]){re.escape(value)}(?![\w-])", text, re.IGNORECASE):
                carriers.add(citation.issue_key.upper())
                break
    return next(iter(carriers)) if len(carriers) == 1 else None
