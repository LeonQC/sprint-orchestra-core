from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.agent.retrieval_tool import RetrievedChunk
from app.investigation.evidence import catalog, render_claims


def chunk(**kwargs):
    defaults = dict(id="search-1",chunk_type="attachment",issue_id=1,issue_key="ATC-1",source_id=11,
                    content="The actual failure and fix",score=.9,retrievers=["vector"],
                    page_number=1,provenance={"page_number":1,"bbox":[1,2,3,4]})
    return RetrievedChunk(**(defaults | kwargs))


def report(worker, scope, chunks=(), structured=(), **kwargs):
    return {"id":worker,"space_ids":scope,"source_keys":["ATC-1"],
            "result":SimpleNamespace(**(dict(citations=list(chunks),structured_evidence=list(structured),
                                     abstained=False,needs_clarification=False)|kwargs))}


def test_catalog_deduplicates_alias_ids_but_preserves_actual_source_positions_and_scopes():
    first=chunk()
    duplicate=replace(first,id="exact-1")
    other_page=replace(first,page_number=2,provenance={"page_number":2})
    pool=catalog([report("a",[7],[first]),report("b",[7],[duplicate,other_page]),
                  report("c",[8],[first])])
    assert len(pool)==3
    assert pool[0].workers==["a","b"]
    assert pool[1].chunk.page_number==2
    assert pool[2].space_ids==[8]
    assert "content" not in pool[0].payload()["source"]


def test_identical_zero_count_receipts_in_different_spaces_are_not_collapsed():
    zero="Structured issue query: total_count = 0"
    pool=catalog([report("a",[7],structured=[zero]),report("b",[8],structured=[zero]),
                  report("c",[7],structured=[zero])])
    assert len(pool)==2
    assert pool[0].workers==["a","c"]
    assert not pool[0].source_keys


def test_exact_claim_selection_does_not_grant_sibling_chunks_with_the_same_issue_key():
    pool=catalog([report("a",[7],[chunk(),chunk(id="irrelevant",content="Unrelated estimate")])])
    text, selected, links=render_claims({"claims":[{"text":"The failure was fixed.",
                                                    "evidence_ids":["E1","E1"]}],"limitations":[]},pool)
    assert selected==["E1"]
    assert text=="The failure was fixed. (ATC-1)"
    assert links[0]["evidence_ids"]==["E1"]


@pytest.mark.parametrize("ids", [[],["E999"],[1]])
def test_missing_or_fabricated_evidence_ids_fail_closed(ids):
    with pytest.raises(ValueError):
        render_claims({"claims":[{"text":"claim","evidence_ids":ids}],"limitations":[]},
                      catalog([report("a",[7],[chunk()])]))


def test_fake_citation_cannot_hide_before_the_code_rendered_real_citation():
    with pytest.raises(ValueError,match="unsupported source"):
        render_claims({"claims":[{"text":"A claim. (EVIL-999)","evidence_ids":["E1"]}],
                       "limitations":[]},catalog([report("a",[7],[chunk()])]))


def test_document_business_identifier_as_a_fact_is_not_used_as_its_carrier():
    text,_,_=render_claims({"claims":[{"text":"Invoice INV-123 total is 10.","evidence_ids":["E1"]}],
                           "limitations":[]},catalog([report("a",[7],[chunk()])]))
    assert text.endswith("(ATC-1)")


def test_only_actual_metadata_keys_in_selected_structured_tool_output_can_be_sources():
    pool=catalog([report("a",[7],structured=["[ATC-1] parent=ACT-999 description says EVIL-999"])])
    assert pool[0].source_keys=={"ATC-1"}


def test_successful_zero_receipt_survives_an_incorrect_worker_refusal_without_its_prose_or_chunks():
    pool=catalog([report("a",[7],[chunk()],structured=["Structured query total_count = 0"],abstained=True)])
    assert len(pool)==1
    assert pool[0].chunk is None
    assert pool[0].content=="Structured query total_count = 0"


def test_rendering_preserves_claim_source_order_and_coverage_text_exactly():
    first = chunk()
    second = chunk(id="other", issue_key="ATC-2", issue_id=2, source_id=12)
    pool = catalog([report("a", [7], [first, second])])
    payload = {
        "claims": [
            {"text": "  Combined fact.  ", "evidence_ids": ["E2", "E1", "E2"]},
            {"text": "Second fact.", "evidence_ids": ["E1"]},
        ],
        "limitations": ["  First restriction.  ", "Second restriction."],
    }
    text, selected, links = render_claims(payload, pool)
    assert text == (
        "- Combined fact. (ATC-2, ATC-1)\n\n"
        "- Second fact. (ATC-1)\n\n"
        "  First restriction.  \nSecond restriction."
    )
    assert selected == ["E2", "E1"]
    assert links[0] == {"text": "Combined fact.", "evidence_ids": ["E2", "E1"]}


def test_empty_claims_do_not_turn_limitations_into_an_answer():
    assert render_claims({"claims": [], "limitations": ["Missing context."]}, []) == ("", [], [])


def test_catalog_preserves_distinct_facts_and_does_not_modify_worker_reports():
    first = chunk()
    alias = replace(first, id="alias", provenance={"bbox": [1, 2, 3, 4], "page_number": 1})
    changed_fact = replace(first, content="A different fix at the same source position")
    reports = [report("a", [7], [first]), report("b", [7], [alias, changed_fact])]
    original = deepcopy(reports)

    pool = catalog(reports)

    assert len(pool) == 2
    assert pool[0].workers == ["a", "b"]
    assert pool[1].content == changed_fact.content
    assert reports == original
    assert pool[0].payload()["source"]["provenance"] == first.provenance


def test_claim_size_boundaries_accept_the_largest_existing_valid_answer():
    chunks = []
    for number in range(12):
        chunks.append(chunk(id=f"chunk-{number}", source_id=number))
    pool = catalog([report("a", [7], chunks)])
    ids = [entry.id for entry in pool]
    claims = []
    for _ in range(40):
        claims.append({"text": "x" * 4000, "evidence_ids": list(ids)})
    payload = {"claims": claims, "limitations": ["y" * 2000]}
    original = deepcopy(payload)

    text, selected, links = render_claims(payload, pool)

    assert selected == ids
    assert len(links) == 40
    assert text.count(" (ATC-1)") == 40
    assert text.endswith("\n\n" + "y" * 2000)
    assert payload == original


@pytest.mark.parametrize("claims, limitations, message", [
    ([{"text": "x" * 4001, "evidence_ids": ["E1"]}], [], "invalid claim text"),
    ([{"text": "Fact", "evidence_ids": ["E1"] * 13}], [], "unknown or missing evidence"),
    ([{"text": "Fact", "evidence_ids": ["E1"]}] * 41, [], "invalid synthesis shape"),
    ([], ["x" * 2001], "invalid coverage limitation"),
])
def test_oversized_synthesis_is_rejected(claims, limitations, message):
    pool = catalog([report("a", [7], [chunk()])])
    with pytest.raises(ValueError, match=message):
        render_claims({"claims": claims, "limitations": limitations}, pool)
