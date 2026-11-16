"""Tests for app/search/service.py's chunk-to-issue dedup — no network, pure function.

Note: `dedupe_by_issue` defaults to both `exclude_chunk_types={"attachment"}` and
`min_score=DEFAULT_MIN_SCORE` (0.7). Tests that are specifically about ranking/limit/dedup mechanics
(not about the attachment or score-floor behavior) pass `min_score=None` to isolate what they're
actually testing, rather than needing every fixture score to clear an unrelated threshold.
"""
from app.agent.retrieval_tool import RetrievedChunk
from app.search.service import (
    DEFAULT_MIN_SCORE,
    RERANKED_MIN_SCORE,
    dedupe_by_issue,
    relevance_confidence,
)


def _chunk(issue_id, score, chunk_id=None, chunk_type="issue"):
    return RetrievedChunk(
        id=chunk_id or f"chunk-{issue_id}-{score}",
        chunk_type=chunk_type,
        issue_id=issue_id,
        issue_key=f"SCRUM-{issue_id}",
        source_id=issue_id,
        content=f"content for issue {issue_id}",
        score=score,
        retrievers=["vector"],
    )


def test_dedupe_keeps_best_scoring_chunk_per_issue():
    chunks = [_chunk(1, 0.4), _chunk(1, 0.9), _chunk(1, 0.6)]

    result = dedupe_by_issue(chunks, limit=10, min_score=None)

    assert len(result) == 1
    assert result[0].score == 0.9


def test_dedupe_ranks_distinct_issues_by_score_descending():
    chunks = [_chunk(1, 0.3), _chunk(2, 0.9), _chunk(3, 0.6)]

    result = dedupe_by_issue(chunks, limit=10, min_score=None)

    assert [c.issue_id for c in result] == [2, 3, 1]


def test_dedupe_respects_limit_after_collapsing_duplicates():
    chunks = [_chunk(1, 0.9), _chunk(1, 0.8), _chunk(2, 0.7), _chunk(3, 0.6)]

    result = dedupe_by_issue(chunks, limit=2, min_score=None)

    assert [c.issue_id for c in result] == [1, 2]


def test_dedupe_handles_empty_input():
    assert dedupe_by_issue([], limit=10) == []


def test_dedupe_excludes_attachment_chunks_by_default():
    # Found live against real ingested data (see module docstring): OCR'd screenshot chunks are
    # frequently near-templated boilerplate that outranks genuinely relevant issue/comment content.
    chunks = [_chunk(1, 0.9, chunk_type="attachment"), _chunk(2, 0.8, chunk_type="issue")]

    result = dedupe_by_issue(chunks, limit=10, min_score=None)

    assert [c.issue_id for c in result] == [2]


def test_dedupe_still_ranks_an_issue_by_its_best_non_attachment_chunk():
    chunks = [
        _chunk(1, 0.99, chunk_id="a", chunk_type="attachment"),
        _chunk(1, 0.8, chunk_id="b", chunk_type="comment"),
    ]

    result = dedupe_by_issue(chunks, limit=10, min_score=None)

    assert len(result) == 1
    assert result[0].id == "b"


def test_dedupe_exclude_chunk_types_can_be_overridden_to_none():
    chunks = [_chunk(1, 0.9, chunk_type="attachment")]

    result = dedupe_by_issue(chunks, limit=10, exclude_chunk_types=None, min_score=None)

    assert len(result) == 1


def test_dedupe_drops_results_below_the_score_floor_instead_of_always_returning_top_n():
    # The exact ask this answers: don't hand back "the least-bad of a bad bunch" as if it were a
    # confident match — if nothing clears the floor, the caller should see nothing, not a fabricated
    # top-N.
    chunks = [_chunk(1, 0.5), _chunk(2, 0.4), _chunk(3, 0.3)]

    result = dedupe_by_issue(chunks, limit=10, min_score=0.7)

    assert result == []


def test_dedupe_keeps_only_results_at_or_above_the_score_floor():
    chunks = [_chunk(1, 0.9), _chunk(2, 0.71), _chunk(3, 0.69)]

    result = dedupe_by_issue(chunks, limit=10, min_score=0.7)

    assert [c.issue_id for c in result] == [1, 2]


def test_dedupe_default_min_score_is_applied_when_not_overridden():
    chunks = [_chunk(1, DEFAULT_MIN_SCORE - 0.01), _chunk(2, DEFAULT_MIN_SCORE)]

    result = dedupe_by_issue(chunks, limit=10)

    assert [c.issue_id for c in result] == [2]


def test_reranked_min_score_works_on_the_cross_encoder_scale_not_cosine_scale():
    # Real scores observed live against a reranked deployment (see module docstring): genuine matches
    # scored 5.39/1.71/0.47, a totally unrelated query scored every candidate around -11.4. Regression
    # test that RERANKED_MIN_SCORE (0.0) actually separates that real distribution, not just a
    # symbolic placeholder value.
    chunks = [
        _chunk(1, 5.39), _chunk(2, 1.71), _chunk(3, 0.47),  # genuine matches
        _chunk(4, -4.65), _chunk(5, -11.4), _chunk(6, -11.43),  # irrelevant
    ]

    result = dedupe_by_issue(chunks, limit=10, min_score=RERANKED_MIN_SCORE)

    assert {c.issue_id for c in result} == {1, 2, 3}


def test_confidence_keeps_a_cross_encoder_logit_inside_0_to_1():
    # The "640% match" bug: the UI rendered score * 100 as a percentage, and with reranking on the
    # score is an unbounded logit — 6.395 became "640%". Every reranked score, however large, must
    # come back as a real 0-1 confidence.
    for logit in (6.395, 5.39, 1.71, 0.47, 12.0, 40.0):
        assert 0.0 <= relevance_confidence(logit, reranked=True) <= 1.0


def test_confidence_puts_the_relevance_floor_at_the_midpoint():
    # RERANKED_MIN_SCORE is 0.0 — "the cross-encoder itself judged this relevant". Under sigmoid that
    # is exactly 0.5, so anything that survives filtering displays as >= 50% and the scale's midpoint
    # is the model's own decision boundary rather than an arbitrary cutoff.
    assert relevance_confidence(RERANKED_MIN_SCORE, reranked=True) == 0.5
    assert relevance_confidence(0.47, reranked=True) > 0.5
    assert relevance_confidence(-4.65, reranked=True) < 0.5


def test_confidence_is_monotonic_so_it_never_reorders_results():
    # The displayed number must not disagree with the ranking it is shown next to.
    logits = [-11.4, -4.65, 0.0, 0.47, 1.71, 5.39, 6.395]
    confidences = [relevance_confidence(x, reranked=True) for x in logits]

    assert confidences == sorted(confidences)


def test_confidence_survives_a_deeply_negative_logit():
    # Live no-match observations sat around -11.4 and nothing bounds the scale; a naive exp(-score)
    # sigmoid overflows on a large negative input instead of returning ~0.
    assert relevance_confidence(-11.4, reranked=True) < 0.01
    assert relevance_confidence(-800.0, reranked=True) == 0.0


def test_confidence_passes_cosine_through_when_not_reranked():
    # On the un-reranked path score already IS a 0-1 confidence, so it must not be squashed again —
    # sigmoid(0.82) would be 0.69, silently downgrading a strong cosine match.
    assert relevance_confidence(0.82, reranked=False) == 0.82
    assert relevance_confidence(1.4, reranked=False) == 1.0
    assert relevance_confidence(-0.2, reranked=False) == 0.0
