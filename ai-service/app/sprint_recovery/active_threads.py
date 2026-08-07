"""Tracks the latest sprint-recovery thread per (space_id, sprint_id), so the UI can find an
in-progress workflow again.

Closing the modal, reloading, or a crash loses the thread_id from the browser, and /start always
creates a new thread — so the intact checkpoint couldn't be found. LangGraph's checkpoint tables are
keyed only by thread_id, so this small side table answers "what's the current thread for this
sprint?".
"""
from __future__ import annotations

from typing import Optional

import psycopg

_SCHEMA = "planning_workflows"
_TABLE = f"{_SCHEMA}.sprint_recovery_active_threads"


async def _ensure_schema(conn) -> None:
    """Create the schema too: `CREATE TABLE IF NOT EXISTS` fails if the schema is missing, which
    happens on a fresh CI database where the checkpointer hasn't run first.
    """
    await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_SCHEMA}")


async def ensure_active_threads_table(db_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        await _ensure_schema(conn)
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                space_id BIGINT NOT NULL,
                sprint_id BIGINT NOT NULL,
                thread_id TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (space_id, sprint_id)
            )
            """
        )


async def register_active_thread(db_url: str, space_id: int, sprint_id: int, thread_id: str) -> None:
    """Called when /start creates a thread. One row per sprint; a new check replaces the old pointer."""
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        await conn.execute(
            f"""
            INSERT INTO {_TABLE} (space_id, sprint_id, thread_id, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (space_id, sprint_id) DO UPDATE SET thread_id = EXCLUDED.thread_id, updated_at = now()
            """,
            (space_id, sprint_id, thread_id),
        )


async def find_active_thread_id(db_url: str, space_id: int, sprint_id: int) -> Optional[str]:
    async with await psycopg.AsyncConnection.connect(db_url, autocommit=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"SELECT thread_id FROM {_TABLE} WHERE space_id = %s AND sprint_id = %s",
                (space_id, sprint_id),
            )
            row = await cur.fetchone()
            return row[0] if row else None
