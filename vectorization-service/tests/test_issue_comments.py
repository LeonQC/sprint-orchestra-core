"""get_issue_comments SQL construction — fake pool, no PostgreSQL required.

The deterministic fallback to /search: complete, unranked comment fetch for named issues (see
VectorStore.get_issue_comments docstring). These tests only check the query shape/params, the same
level as the other chunk lookup regressions.
"""
from app.db.vector_store import VectorStore, thread_order
from app.models import IssueComment


class FakePool:
    def __init__(self, rows=None):
        self.calls = []
        self._rows = rows if rows is not None else []

    async def fetchval(self, sql, *args):
        self.calls.append(("fetchval", sql, args))
        return len(self._rows)

    async def fetch(self, sql, *args):
        self.calls.append(("fetch", sql, args))
        return self._rows


async def test_filters_by_space_and_issue_key_with_no_ranking():
    pool = FakePool()
    store = VectorStore(pool)

    await store.get_issue_comments([5000014], ["ATC-30", "ATC-68"])

    for _, sql, args in pool.calls:
        assert "chunk_type = 'comment'" in sql
        assert "issue_key = ANY($2::text[])" in sql
        assert "space_id = ANY($1::bigint[])" in sql
        assert args[0] == [5000014]
        assert args[1] == ["ATC-30", "ATC-68"]
    # Chronological within an issue (created_at, falling back to comment id), not a relevance
    # ranking. chunk_index alone left comment order to chance: every single-chunk comment is 0.
    fetch_sql = next(sql for kind, sql, _ in pool.calls if kind == "fetch")
    assert "ORDER BY issue_key, provenance->>'created_at' NULLS LAST, source_id, chunk_index" in fetch_sql


async def test_result_maps_every_row_and_reports_the_exact_count():
    rows = [
        {"id": "comment:1", "issue_id": 30, "issue_key": "ATC-30", "source_id": 1,
         "content": "Reopening this because...", "chunk_index": 0},
        {"id": "comment:2", "issue_id": 30, "issue_key": "ATC-30", "source_id": 2,
         "content": "Verification result PASS", "chunk_index": 1},
    ]
    pool = FakePool(rows=rows)
    store = VectorStore(pool)

    result = await store.get_issue_comments([5000014], ["ATC-30"])

    assert result.total_count == 2
    assert [c.content for c in result.comments] == ["Reopening this because...", "Verification result PASS"]
    assert result.comments[0].issue_key == "ATC-30"


def _c(source_id, parent_id=None, issue_key="ATC-10"):
    prov = {"parent_id": parent_id} if parent_id else {}
    return IssueComment(id=f"comment:{source_id}", issue_id=10, issue_key=issue_key, source_id=source_id,
                        content=str(source_id), chunk_index=0, provenance=prov)


def test_thread_order_puts_each_reply_right_after_its_top_level_comment():
    # Chronological input: two threads interleaved in time.
    #   1 Alice: Safari white screen      2 Bob: raise estimate?
    #   3 -> 1   4 -> 2   5 -> 1
    chronological = [_c(1), _c(2), _c(3, 1), _c(4, 2), _c(5, 1)]
    assert [c.source_id for c in thread_order(chronological)] == [1, 3, 5, 2, 4]


def test_thread_order_keeps_a_reply_whose_parent_is_missing():
    # Parent deleted (or cut off by the limit): the reply must still be returned, in place.
    assert [c.source_id for c in thread_order([_c(2), _c(7, 1), _c(8)])] == [2, 7, 8]


def test_thread_order_leaves_legacy_comments_without_provenance_untouched():
    assert [c.source_id for c in thread_order([_c(3), _c(1), _c(2)])] == [3, 1, 2]


async def test_result_carries_provenance_and_is_thread_ordered():
    rows = [
        {"id": "comment:1", "issue_id": 10, "issue_key": "ATC-10", "source_id": 1, "content": "root",
         "chunk_index": 0, "provenance": {"author": "Ava Patel", "created_at": "2026-06-15T22:40:00+00:00"}},
        {"id": "comment:2", "issue_id": 10, "issue_key": "ATC-10", "source_id": 2, "content": "other root",
         "chunk_index": 0, "provenance": {"created_at": "2026-06-16T00:00:00+00:00"}},
        {"id": "comment:3", "issue_id": 10, "issue_key": "ATC-10", "source_id": 3, "content": "reply",
         "chunk_index": 0, "provenance": {"parent_id": 1, "reply_to_id": 1}},
        {"id": "comment:4", "issue_id": 10, "issue_key": "ATC-10", "source_id": 4, "content": "legacy",
         "chunk_index": 0, "provenance": None},
    ]
    result = await VectorStore(FakePool(rows=rows)).get_issue_comments([1], ["ATC-10"])

    assert [c.source_id for c in result.comments] == [1, 3, 2, 4]
    assert result.comments[0].provenance["author"] == "Ava Patel"
    assert result.comments[3].provenance == {}
