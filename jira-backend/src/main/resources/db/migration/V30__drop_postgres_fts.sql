-- Elasticsearch now owns the global search (see com.jiraagentic.app.search), so everything V23-V29
-- built for Postgres full-text search has no reader left. This drops all of it.
--
-- What goes, and what it cost while it stayed:
--
--   * `search_vector` on issues and comments — STORED generated columns, so every row carried a
--     second tokenized copy of its own text, and every INSERT/UPDATE paid to recompute it and
--     maintain a GIN index.
--   * `label_names` / `attachment_names` on issues, plus the four functions and four triggers that
--     kept them in sync. This was V28's workaround for a real limitation: a generated column can
--     only see its own row, so denormalizing across issue_labels and issue_attachments needed
--     triggers. In a document store that join just happens at index time
--     (SearchDocumentFactory), and the whole apparatus disappears.
--   * The pg_trgm indexes from V26/V28 backing the fuzzy and substring tiers.
--   * `strip_html()` and `literal_snippet()` — both reimplemented in Java, where the search that
--     uses them now runs (SearchDocumentFactory.stripHtml, LiteralSnippet).
--
-- The `pg_trgm` EXTENSION itself is deliberately left installed. Dropping it is not free to undo on
-- a managed Postgres, it costs nothing to keep, and nothing else in this schema depends on it.
--
-- V23-V29 stay in the migration history rather than being edited away: they record how search
-- actually evolved, and a fresh database still replays them before this drop.

-- --- triggers first: they reference the functions and columns dropped below ---------------------
DROP TRIGGER IF EXISTS issue_labels_refresh_names ON issue_labels;
DROP TRIGGER IF EXISTS labels_rename_refresh_names ON labels;
DROP TRIGGER IF EXISTS issue_attachments_refresh_names ON issue_attachments;

DROP FUNCTION IF EXISTS trg_issue_labels_refresh();
DROP FUNCTION IF EXISTS trg_labels_rename_refresh();
DROP FUNCTION IF EXISTS trg_issue_attachments_refresh();
DROP FUNCTION IF EXISTS refresh_issue_label_names(BIGINT);
DROP FUNCTION IF EXISTS refresh_issue_attachment_names(BIGINT);

-- --- full-text indexes and the generated columns behind them -----------------------------------
DROP INDEX IF EXISTS idx_issue_search_vector;
DROP INDEX IF EXISTS idx_comment_search_vector;

ALTER TABLE issues DROP COLUMN IF EXISTS search_vector;
ALTER TABLE comments DROP COLUMN IF EXISTS search_vector;

ALTER TABLE issues DROP COLUMN IF EXISTS label_names;
ALTER TABLE issues DROP COLUMN IF EXISTS attachment_names;

-- --- trigram indexes for the fuzzy / substring tiers -------------------------------------------
DROP INDEX IF EXISTS idx_issue_key_lower_pattern;
DROP INDEX IF EXISTS idx_issue_key_trgm;
DROP INDEX IF EXISTS idx_issue_title_trgm;
DROP INDEX IF EXISTS idx_issue_description_trgm;
DROP INDEX IF EXISTS idx_comment_content_trgm;
DROP INDEX IF EXISTS idx_issue_label_names_trgm;
DROP INDEX IF EXISTS idx_issue_attachment_names_trgm;

-- --- SQL helpers, now reimplemented in Java -----------------------------------------------------
DROP FUNCTION IF EXISTS literal_snippet(text, text);
DROP FUNCTION IF EXISTS strip_html(text);
