"""Pure source-provenance policy tests; no model, database, or network."""
from dataclasses import dataclass

from app.agent.provenance_policy import resolve_implicit_source


@dataclass
class Candidate:
    issue_key: str
    content: str


def test_document_identifier_resolves_to_one_retrieved_carrier():
    result = resolve_implicit_source(
        {"HLT-439"},
        [
            Candidate("DVQPLF2-7", "Health report HLT-439: availability 99.87%."),
            Candidate("DVQPLF2-8", "Health report HLT-468: availability 99.91%."),
        ],
    )

    assert result.source_key == "DVQPLF2-7"
    assert result.matched_identifiers == ("HLT-439",)


def test_explicit_retrieved_carrier_remains_the_strongest_source_identity():
    result = resolve_implicit_source(
        {"DVQPLF2-7", "HLT-439"},
        [Candidate("DVQPLF2-7", "Health report HLT-439: availability 99.87%.")],
    )

    assert result.source_key == "DVQPLF2-7"
    assert result.matched_identifiers == ()


def test_same_document_identifier_in_two_carriers_fails_closed():
    result = resolve_implicit_source(
        {"HLT-439"},
        [
            Candidate("DVQPLF2-7", "Health report HLT-439."),
            Candidate("DVQPLF2-41", "Archived copy of health report HLT-439."),
        ],
    )

    assert result.source_key is None


def test_identifier_matching_uses_token_boundaries():
    result = resolve_implicit_source(
        {"HLT-41"},
        [Candidate("DVQPLF2-7", "Health report HLT-410: availability 99.94%.")],
    )

    assert result.source_key is None


def test_unseen_identifier_never_selects_an_unrelated_single_candidate():
    result = resolve_implicit_source(
        {"ATC-123"},
        [Candidate("DVQPLF2-7", "Health report HLT-439: availability 99.87%.")],
    )

    assert result.source_key is None

