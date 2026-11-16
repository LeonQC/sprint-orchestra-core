"""Tests for precedent search (app/precedent/) — fake retrieval + fake instructor client, no network.

The properties worth defending: matching and judging both work from title+description only (comment
threads are never fetched — see app/precedent/service.py's module docstring for why that was cut
after a real, structural failure mode was found in testing), the current issue never appears as its
own precedent, finished work is ranked ahead of unfinished work before the model sees anything, a
cited issue that was never a candidate is dropped rather than shown, and every failure path degrades
instead of raising.
"""
from app.agent.retrieval_tool import IssueQueryResult, RetrievedChunk, SearchResult
from app.precedent.prompts import Candidate, plain_text, render_precedent_message
from app.precedent.schemas import Precedent, PrecedentReport
from app.precedent.service import _rank, find_precedents


class _FakeUsage:
    def __init__(self, prompt_tokens=400, completion_tokens=90):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self):
        self.usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self, result):
        self._result = result
        self.calls = []

    async def create_with_completion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result, _FakeCompletion()


class FakeInstructorClient:
    def __init__(self, result):
        self.completions = _FakeCompletions(result)
        self.chat = type("_Chat", (), {"completions": self.completions})()


class _Result:
    def __init__(self, total_count, **kw):
        self.total_count = total_count
        for k, v in kw.items():
            setattr(self, k, v)


class FakeRetrieval:
    """Stands in for RetrievalClient: semantic hits, structured rows, and bodies. No comment fetch
    — find_precedents no longer calls get_issue_comments at all."""

    def __init__(self, hits, rows=None, details=None, fail=False):
        self._hits = hits
        self._rows = rows or []
        self._details = details or []
        self._fail = fail

    async def search_with_meta(self, query, space_ids, limit, mode="hybrid"):
        self.last_query = query
        if self._fail:
            raise RuntimeError("vectorization-service unreachable")
        return SearchResult(hits=self._hits, reranked=False)

    async def query_issues(self, space_ids, filters):
        return IssueQueryResult(total_count=len(self._rows), counts_by_type={},
                                counts_by_status={}, issues=self._rows)

    async def get_issue_details(self, space_ids, issue_keys, limit=200):
        return _Result(len(self._details), details=self._details)


def _chunk(issue_key, score=0.9, issue_id=1, chunk_type="issue"):
    return RetrievedChunk(id=f"c{issue_id}", chunk_type=chunk_type, issue_id=issue_id,
                          issue_key=issue_key, source_id=issue_id, content="body", score=score,
                          retrievers=["vector"])


def _row(issue_key, status="done", issue_type="story", title="t"):
    return {"issue_key": issue_key, "status": status, "issue_type": issue_type, "title": title}


async def test_the_current_issue_is_never_its_own_precedent():
    # An issue is always its own strongest semantic match; without this it would be candidate #1.
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-72", 0.99, 1), _chunk("ATC-51", 0.8, 2)],
        rows=[_row("ATC-51")],
    )
    client = FakeInstructorClient(PrecedentReport(
        verdict="found",
        precedents=[Precedent(issue_key="ATC-51", relevance="same importer", takeaway=None)],
    ))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert "ATC-72" not in result.candidate_keys
    assert result.candidate_keys == ["ATC-51"]


async def test_a_precedent_citing_a_non_candidate_is_dropped():
    retrieval = FakeRetrieval(hits=[_chunk("ATC-51", 0.8, 2)], rows=[_row("ATC-51")])
    client = FakeInstructorClient(PrecedentReport(
        verdict="found",
        precedents=[Precedent(issue_key="ATC-999", relevance="invented", takeaway=None)],
    ))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.report.precedents == []
    assert result.dropped_precedents == 1
    # Nothing citable survived, so the verdict is normalized rather than left claiming a find.
    assert result.report.verdict == "none_relevant"


async def test_a_grounded_precedent_survives_with_a_description_only_takeaway():
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-51", 0.8, 2)],
        rows=[_row("ATC-51", title="Locale fallback for importer")],
        details=[{"issue_key": "ATC-51",
                  "content": "Tried index-per-locale; it was too slow, switched to a shared index."}],
    )
    client = FakeInstructorClient(PrecedentReport(
        verdict="found",
        precedents=[Precedent(issue_key="ATC-51", relevance="same importer path",
                              takeaway="index-per-locale was too slow")],
    ))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.report.verdict == "found"
    assert result.report.precedents[0].takeaway == "index-per-locale was too slow"
    assert result.candidates_considered == 1
    assert result.input_tokens == 400
    # The title is the candidate's real one from query_issues, not anything the model produced —
    # Precedent (the model's own schema) carries no title field at all.
    assert result.candidate_titles["ATC-51"] == "Locale fallback for importer"


async def test_no_candidates_above_the_floor_is_reported_not_treated_as_failure():
    retrieval = FakeRetrieval(hits=[])
    client = FakeInstructorClient(RuntimeError("should never be called"))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.no_candidates is True
    assert result.degraded is False
    assert result.report.verdict == "none_relevant"
    assert client.completions.calls == []


async def test_retrieval_failure_degrades_instead_of_raising():
    retrieval = FakeRetrieval(hits=[], fail=True)
    client = FakeInstructorClient(RuntimeError("should never be called"))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.degraded is True
    assert result.report.precedents == []
    assert "unreachable" in (result.error or "")


async def test_model_failure_degrades_instead_of_raising():
    retrieval = FakeRetrieval(hits=[_chunk("ATC-51", 0.8, 2)], rows=[_row("ATC-51")])
    client = FakeInstructorClient(RuntimeError("model unreachable"))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.degraded is True
    assert result.report.verdict == "none_relevant"
    assert result.candidates_considered == 1


def test_finished_work_outranks_unfinished_before_the_model_sees_it():
    # Ranking decides who survives the candidate budget, so an in-progress lookalike must not crowd
    # out a solved one.
    ranked = _rank([
        Candidate("A", "a", status="in_progress", issue_type="story"),
        Candidate("B", "b", status="done", issue_type="story"),
        Candidate("C", "c", status="done", issue_type="story"),
    ])

    assert [c.issue_key for c in ranked] == ["B", "C", "A"]


def test_the_prompt_marks_which_candidates_are_finished():
    message = render_precedent_message(
        "ATC-72", "Current", "Body",
        [Candidate("ATC-51", "Older", status="done", issue_type="story"),
         Candidate("ATC-60", "Open one", status="in_progress", issue_type="bug")],
    )

    assert "DONE — finished work" in message
    assert "[not finished]" in message


def test_the_prompt_strips_markup_from_bodies():
    message = render_precedent_message(
        "ATC-72", "Current", "<p>Body &amp; more</p>",
        [Candidate("ATC-51", "Older", status="done", issue_type="story",
                   description="<p>We <strong>reverted</strong> it</p>")],
    )

    assert "<p>" not in message
    assert "Body & more" in message
    assert "We reverted it" in message


async def test_matching_ignores_comment_chunks():
    """An issue must earn its way into the candidate set on its own title+description, not on a
    passing remark in its thread. Here ATC-60 is only reachable via a comment chunk — and because
    dedupe keeps the best chunk per issue, without this exclusion that single remark would present
    it as the strongest match in the space."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-60", 9.0, 3, chunk_type="comment"), _chunk("ATC-51", 1.0, 2)],
        rows=[_row("ATC-51"), _row("ATC-60")],
    )
    client = FakeInstructorClient(PrecedentReport(verdict="none_relevant", precedents=[]))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.candidate_keys == ["ATC-51"]


async def test_matching_still_ignores_attachment_chunks():
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-60", 9.0, 3, chunk_type="attachment"), _chunk("ATC-51", 1.0, 2)],
        rows=[_row("ATC-51")],
    )
    client = FakeInstructorClient(PrecedentReport(verdict="none_relevant", precedents=[]))

    result = await find_precedents(client, "fake-model", retrieval, [1], "ATC-72", "T", "D")

    assert result.candidate_keys == ["ATC-51"]


async def test_the_retrieval_query_is_plain_text_not_stored_html():
    """Found live: descriptions are stored as HTML and the frontend sends them exactly as stored.
    Embedding the raw markup measurably wrecks retrieval — the same real issue returned 4 candidates
    from its plain-text description and ZERO with the tags left in."""
    retrieval = FakeRetrieval(hits=[_chunk("ATC-51", 0.8, 2)], rows=[_row("ATC-51")])
    client = FakeInstructorClient(PrecedentReport(verdict="none_relevant", precedents=[]))

    await find_precedents(
        client, "fake-model", retrieval, [1], "ATC-72", "Index products",
        "<h3>Context</h3><p>Replace the <strong>basic</strong> match &amp; hide inactive.</p>",
    )

    assert "<" not in retrieval.last_query
    assert "&amp;" not in retrieval.last_query
    assert "Replace the basic match & hide inactive." in retrieval.last_query
    assert "Index products" in retrieval.last_query


def test_plain_text_truncates_with_an_ellipsis():
    assert plain_text("<p>" + "x" * 50 + "</p>", 10) == "x" * 10 + "…"
