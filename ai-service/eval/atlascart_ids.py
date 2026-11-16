"""Resolve AtlasCart's space and sprint ids instead of pinning them in each eval.

`seed-data/atlascart/m3_seed.py`'s `INSERT INTO spaces` supplies no explicit id, so every
cleanup/inject cycle mints a new serial one -- and the sprint ids move with it. Any eval that hard-
codes those numbers silently selects nothing the first time the fixture is reinstalled, which is not
a loud failure: a query returns zero rows and the eval reports on an empty corpus. This resolves
the values from the one thing the seed keeps stable, the space key.

Read-only, and deliberately dependency-light (`psycopg` against jira-backend's own Postgres) so an
eval that already talks to that database gains nothing new to configure.
"""
from __future__ import annotations

import os

ATLASCART_SPACE_KEY = "ATC"
JIRA_BACKEND_DSN = os.getenv("JIRA_BACKEND_PG_DSN", "postgresql://poc:poc123@localhost:5432/pocdb")


def atlascart_space_id(dsn: str | None = None) -> int:
    """The current AtlasCart space id. Raises rather than returning a stale or default id: an eval
    pointed at a space that does not exist should stop, not quietly measure nothing."""
    import psycopg

    with psycopg.connect(dsn or JIRA_BACKEND_DSN) as conn:
        row = conn.execute(
            "SELECT id FROM spaces WHERE space_key = %s AND deleted_at IS NULL",
            (ATLASCART_SPACE_KEY,),
        ).fetchone()
    if not row:
        raise RuntimeError(
            f"no space with space_key={ATLASCART_SPACE_KEY!r}; install the fixture first "
            f"(seed-data/atlascart/m3_seed.py inject --yes)"
        )
    return int(row[0])

