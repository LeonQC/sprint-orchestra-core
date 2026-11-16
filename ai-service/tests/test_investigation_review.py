import pytest
from app.investigation.review import validate_review


def test_review_tracks_unsupported_and_unnecessary_claims_independently():
    bad,missing=validate_review({"judgments":[
        {"index":0,"supported":False,"relevant":True,"reason":"unsupported"},
        {"index":1,"supported":True,"relevant":False,"reason":"unasked status"},
        {"index":2,"supported":True,"relevant":True,"reason":"direct answer"}],
        "missing_parts":["second project"]},3)
    assert bad=={0,1} and missing==["second project"]


@pytest.mark.parametrize("judgments",[
    [],
    [{"index":0,"supported":"false","relevant":True,"reason":"text"}],
    [{"index":True,"supported":True,"relevant":True,"reason":"bool is not an index"}],
    [{"index":99,"supported":True,"relevant":True,"reason":"unknown"}],
])
def test_invalid_or_incomplete_review_cannot_approve_claims(judgments):
    with pytest.raises(ValueError):
        validate_review({"judgments":judgments,"missing_parts":[]},1)


def test_duplicate_judgments_cannot_stand_in_for_a_missing_claim():
    judgment = {"index": 0, "supported": True, "relevant": True, "reason": "supported"}
    with pytest.raises(ValueError, match="duplicate claim judgments"):
        validate_review({"judgments": [judgment, judgment], "missing_parts": []}, 2)


def test_missing_parts_remain_in_the_reviewers_order():
    missing = ["Second project", "First project"]
    result = validate_review({"judgments": [], "missing_parts": missing}, 0)
    assert result == (set(), missing)


def test_complete_review_accepts_out_of_order_indices_and_size_boundaries():
    judgments = []
    for index in [2, 0, 1]:
        judgments.append({"index": index, "supported": True, "relevant": True, "reason": "r" * 2000})
    missing = []
    for _ in range(8):
        missing.append("m" * 1000)
    assert validate_review({"judgments": judgments, "missing_parts": missing}, 3) == (set(), missing)


@pytest.mark.parametrize("field, value", [
    ("supported", 1), ("supported", None), ("relevant", 0), ("relevant", "true"),
    ("index", -1), ("index", 0.0), ("reason", None), ("reason", "x" * 2001),
])
def test_invalid_judgment_types_and_boundaries_are_rejected(field, value):
    judgment = {"index": 0, "supported": True, "relevant": True, "reason": "supported"}
    judgment[field] = value
    with pytest.raises(ValueError, match="invalid claim judgment values"):
        validate_review({"judgments": [judgment], "missing_parts": []}, 1)


@pytest.mark.parametrize("missing", [["m"] * 9, ["m" * 1001], ["  "], [None]])
def test_invalid_missing_parts_are_rejected(missing):
    with pytest.raises(ValueError, match="invalid missing question parts"):
        validate_review({"judgments": [], "missing_parts": missing}, 0)
