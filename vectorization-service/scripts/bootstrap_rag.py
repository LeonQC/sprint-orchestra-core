"""Install the current rag schema and its restricted runtime role using a deployment connection.

RAG_ADMIN_PG_DSN and RAG_DB_PASSWORD are required environment variables. No runtime service uses
these administrator credentials; neither credential is logged or written into the schema files.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg

SCHEMA_SQL = (Path(__file__).resolve().parents[1] / "schema" / "rag_schema.sql").read_text()


async def bootstrap(connection, password: str) -> None:
    if not password:
        raise ValueError("RAG_DB_PASSWORD must not be empty")
    async with connection.transaction():
        await connection.execute(SCHEMA_SQL)
        role = await connection.fetchrow("SELECT rolsuper, rolcreatedb, rolcreaterole FROM pg_roles WHERE rolname='rag'")
        if role and any(role.values()):
            raise ValueError("existing rag role has administrative privileges; refusing to reuse it")
        if role is None:
            await connection.execute("CREATE ROLE rag LOGIN")
        # DDL does not accept bind placeholders. Ask PostgreSQL to quote the value; never interpolate
        # environment text into SQL or log the rendered statement.
        password_sql = await connection.fetchval("SELECT format('ALTER ROLE rag LOGIN PASSWORD %L', $1::text)", password)
        await connection.execute(password_sql)
        await connection.execute("GRANT USAGE ON SCHEMA rag, public TO rag")
        await connection.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO rag")
        await connection.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag")
        database = await connection.fetchval("SELECT current_database()")
        database_sql = await connection.fetchval(
            "SELECT format('ALTER ROLE rag IN DATABASE %I SET search_path = rag, public', $1::text)", database,
        )
        await connection.execute(database_sql)
        connect_sql = await connection.fetchval("SELECT format('GRANT CONNECT ON DATABASE %I TO rag', $1::text)", database)
        await connection.execute(connect_sql)


async def main() -> None:
    connection = await asyncpg.connect(os.environ["RAG_ADMIN_PG_DSN"])
    try:
        await bootstrap(connection, os.environ["RAG_DB_PASSWORD"])
    finally:
        await connection.close()
    print("rag schema and restricted runtime role ready")


if __name__ == "__main__":
    asyncio.run(main())
