"""Backfill keeps only searchable content and scopes source reads."""
from scripts.backfill_jira_backend import (
    _backfilled_chunk_types,
    _delete_chunks,
    _fetch_issues,
    _fetch_sprints,
)


class FakePool:
    async def fetch(self, query, *args):
        self.query, self.args = query, args
        return []


async def test_sprint_backfill_fetches_goals_and_scopes_spaces():
    pool = FakePool()
    await _fetch_sprints(pool, [5000014])
    assert "FROM sprints" in pool.query
    assert "goal" in pool.query
    assert "completed_points" not in pool.query
    assert "space_id = ANY($1::bigint[])" in pool.query
    assert pool.args == ([5000014],)


async def test_issue_backfill_preserves_parent_context_without_metadata_copy():
    pool = FakePool()
    await _fetch_issues(pool, [5000014])
    assert "FROM issues" in pool.query
    assert "parent_key" in pool.query and "parent_title" in pool.query
    assert "i.status" not in pool.query and "i.priority" not in pool.query
    assert "space_id = ANY($1::bigint[])" in pool.query
    assert pool.args == ([5000014],)


class ExecutePool:
    def __init__(self):
        self.calls = []

    async def execute(self, query, *args):
        self.calls.append((query, args))


async def test_truncate_keeps_attachments_unless_they_are_backfilled_too():
    # A plain DELETE FROM chunks would drop every attachment in every space.
    assert _backfilled_chunk_types(sprints_only=False, include_attachments=False) == ["sprint", "issue", "comment"]
    assert "attachment" in _backfilled_chunk_types(sprints_only=False, include_attachments=True)
    assert _backfilled_chunk_types(sprints_only=True, include_attachments=True) == ["sprint"]


async def test_truncate_is_scoped_to_the_chosen_spaces_and_clears_parents_too():
    pool = ExecutePool()
    await _delete_chunks(pool, ["issue"], [5000074])
    assert [q.split()[2] for q, _ in pool.calls] == ["chunks", "chunk_parents"]
    for query, args in pool.calls:
        assert "space_id = ANY($2::bigint[])" in query
        assert args == (["issue"], [5000074])
