-- Runs once, when the local PostgreSQL volume is first created. On RDS, run scripts/bootstrap_rag.py instead.
-- Keep this as SQL: the entrypoint reads it without executing files from a host bind mount.
\ir /opt/rag-schema/rag_schema.sql
CREATE ROLE rag LOGIN PASSWORD 'rag123';
GRANT USAGE ON SCHEMA rag, public TO rag;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO rag;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag;
ALTER ROLE rag SET search_path = rag, public;
