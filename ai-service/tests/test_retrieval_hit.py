from eval import retrieval_hit
from eval.ragas_eval import RagasCase


def _case(*keys, category="semantic:one-source"):
    return RagasCase("q", "r", category, required_issue_keys=keys)


def test_chunks_collapse_to_issue_keys_in_rank_order():
    hits = [{"issue_key": "ATC-43"}, {"issue_key": "ATC-43"}, {"issue_key": "ATC-47"}, {"issue_key": "ATC-43"}]
    assert retrieval_hit.distinct_issue_keys(hits) == ["ATC-43", "ATC-47"]


def test_hit_recall_and_reciprocal_rank_by_issue_key():
    s = retrieval_hit.score(_case("ATC-10", "ATC-22"), ["ATC-5", "ATC-22", "ATC-9", "ATC-1", "ATC-2", "ATC-10"])
    assert s.hit == {"hit@1": False, "hit@3": True, "hit@5": True}
    assert s.recall == {"recall@5": 0.5, "recall@10": 1.0}
    assert s.reciprocal_rank == 0.5


def test_a_question_whose_issue_never_appears_scores_zero_and_is_listed_as_a_miss():
    scores = [retrieval_hit.score(_case("ATC-68"), ["ATC-1", "ATC-2"]), retrieval_hit.score(_case("ATC-43"), ["ATC-43"])]
    summary = retrieval_hit.summarize(scores, {})
    assert summary["means"]["hit@5"] == 0.5
    assert summary["means"]["mrr"] == 0.5
    assert [m["required"] for m in summary["missed_at_5"]] == [["ATC-68"]]


def test_means_are_broken_down_by_category():
    scores = [
        retrieval_hit.score(_case("ATC-1"), ["ATC-1"]),
        retrieval_hit.score(_case("ATC-2", category="semantic:linked-sources"), ["ATC-9"]),
    ]
    summary = retrieval_hit.summarize(scores, {})
    assert summary["by_category"]["semantic:one-source"]["hit@5"] == 1.0
    assert summary["by_category"]["semantic:linked-sources"]["hit@5"] == 0.0


def test_every_semantic_question_in_the_set_is_labelled_and_tool_questions_are_not():
    from eval.ragas_eval import CASES

    for case in CASES:
        semantic = case.category.startswith("semantic") and "abstention" not in case.category
        assert bool(case.required_issue_keys) == semantic, case.question
        for key in case.required_issue_keys:
            assert key in case.reference, (case.question, key)
