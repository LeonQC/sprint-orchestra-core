-- Search coverage, round 4. V23-V25 built the issue/comment tsvector over title/description/content;
-- V26 added the trigram fuzzy tier. Three gaps remained against the search coverage guidance:
--
--   1. Labels and attachment filenames were not searchable at all — "find the issue tagged
--      `checkout`" or "the one with metrics-wire.csv attached" returned nothing.
--   2. title/description/comment content are stored as HTML. Postgres's parser already drops tag
--      tokens from the tsvector (the `tag` token type isn't mapped in the `english` config), so the
--      INDEX was fine — but `ts_headline()` runs on the raw text, so result SNIPPETS came back full
--      of "<p>", "<li>", "<strong>".
--   3. No reliable substring match. A mid-token query ("auth" -> "authentication") is something FTS
--      stemming structurally cannot do, and the V26 trigram tier is fuzzy + threshold-gated +
--      fallback-only. Handled in the repository/service layer (new SUBSTRING tier), but it needs
--      trigram indexes on the new columns to stay fast — added at the end of this file.
--
-- Approach for (1): labels/attachments live in other tables, and a GENERATED column can only see
-- its own row, so two plain text columns on `issues` are kept in sync by triggers, then folded into
-- the generated search_vector at lower weights (C = labels, D = filenames). A trigger UPDATE of
-- those columns makes Postgres recompute the STORED search_vector for that row automatically.

-- --------------------------------------------------------------------------------------------------
-- (2) HTML stripper — IMMUTABLE so it can be used inside a GENERATED column expression.
-- --------------------------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION strip_html(t text) RETURNS text AS $$
    SELECT regexp_replace(coalesce(t, ''), '<[^>]+>', ' ', 'g');
$$ LANGUAGE sql IMMUTABLE;

-- --------------------------------------------------------------------------------------------------
-- (1) Denormalized, trigger-maintained label / attachment text on `issues`.
-- --------------------------------------------------------------------------------------------------
ALTER TABLE issues ADD COLUMN label_names text;
ALTER TABLE issues ADD COLUMN attachment_names text;

CREATE OR REPLACE FUNCTION refresh_issue_label_names(p_issue_id BIGINT) RETURNS void AS $$
    UPDATE issues i
       SET label_names = (
           SELECT string_agg(l.name, ' ')
           FROM issue_labels il
           JOIN labels l ON l.id = il.label_id
           WHERE il.issue_id = i.id
       )
     WHERE i.id = p_issue_id;
$$ LANGUAGE sql;

CREATE OR REPLACE FUNCTION refresh_issue_attachment_names(p_issue_id BIGINT) RETURNS void AS $$
    UPDATE issues i
       SET attachment_names = (
           SELECT string_agg(a.original_filename, ' ')
           FROM issue_attachments a
           WHERE a.issue_id = i.id
       )
     WHERE i.id = p_issue_id;
$$ LANGUAGE sql;

-- issue_labels: a row is (issue_id, label_id); INSERT/DELETE change one issue's label set.
CREATE OR REPLACE FUNCTION trg_issue_labels_refresh() RETURNS trigger AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        PERFORM refresh_issue_label_names(OLD.issue_id);
        RETURN OLD;
    END IF;
    PERFORM refresh_issue_label_names(NEW.issue_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER issue_labels_refresh_names
    AFTER INSERT OR DELETE ON issue_labels
    FOR EACH ROW EXECUTE FUNCTION trg_issue_labels_refresh();

-- Renaming a label must re-materialize every issue that carries it.
CREATE OR REPLACE FUNCTION trg_labels_rename_refresh() RETURNS trigger AS $$
BEGIN
    UPDATE issues i
       SET label_names = (
           SELECT string_agg(l.name, ' ')
           FROM issue_labels il
           JOIN labels l ON l.id = il.label_id
           WHERE il.issue_id = i.id
       )
     WHERE i.id IN (SELECT issue_id FROM issue_labels WHERE label_id = NEW.id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER labels_rename_refresh_names
    AFTER UPDATE OF name ON labels
    FOR EACH ROW WHEN (OLD.name IS DISTINCT FROM NEW.name)
    EXECUTE FUNCTION trg_labels_rename_refresh();

-- issue_attachments: filename set / rename per issue.
CREATE OR REPLACE FUNCTION trg_issue_attachments_refresh() RETURNS trigger AS $$
BEGIN
    IF (TG_OP = 'DELETE') THEN
        PERFORM refresh_issue_attachment_names(OLD.issue_id);
        RETURN OLD;
    END IF;
    PERFORM refresh_issue_attachment_names(NEW.issue_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER issue_attachments_refresh_names
    AFTER INSERT OR DELETE OR UPDATE OF original_filename ON issue_attachments
    FOR EACH ROW EXECUTE FUNCTION trg_issue_attachments_refresh();

-- Backfill existing rows before the generated column is redefined to read them.
UPDATE issues i SET label_names = (
    SELECT string_agg(l.name, ' ')
    FROM issue_labels il JOIN labels l ON l.id = il.label_id
    WHERE il.issue_id = i.id
);
UPDATE issues i SET attachment_names = (
    SELECT string_agg(a.original_filename, ' ')
    FROM issue_attachments a WHERE a.issue_id = i.id
);

-- --------------------------------------------------------------------------------------------------
-- Redefine the generated tsvectors: strip HTML on title/description/content, and add labels (C) and
-- attachment filenames (D). Same drop-and-recreate the earlier FTS migrations used.
-- --------------------------------------------------------------------------------------------------
DROP INDEX IF EXISTS idx_issue_search_vector;
ALTER TABLE issues DROP COLUMN search_vector;
ALTER TABLE issues
    ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', regexp_replace(strip_html(title), '[-_/.~+]+', ' ', 'g')), 'A') ||
        setweight(to_tsvector('english', regexp_replace(strip_html(description), '[-_/.~+]+', ' ', 'g')), 'B') ||
        setweight(to_tsvector('english', regexp_replace(coalesce(label_names, ''), '[-_/.~+]+', ' ', 'g')), 'C') ||
        setweight(to_tsvector('english', regexp_replace(coalesce(attachment_names, ''), '[-_/.~+]+', ' ', 'g')), 'D')
    ) STORED;
CREATE INDEX idx_issue_search_vector ON issues USING GIN (search_vector);

DROP INDEX IF EXISTS idx_comment_search_vector;
ALTER TABLE comments DROP COLUMN search_vector;
ALTER TABLE comments
    ADD COLUMN search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('english', regexp_replace(strip_html(content), '[-_/.~+]+', ' ', 'g'))
    ) STORED;
CREATE INDEX idx_comment_search_vector ON comments USING GIN (search_vector);

-- --------------------------------------------------------------------------------------------------
-- (3) Trigram indexes backing the new SUBSTRING tier's `LIKE '%q%'` on issue_key and the new
-- label/attachment columns (title/description/comment content already covered by V26).
-- --------------------------------------------------------------------------------------------------
CREATE INDEX idx_issue_key_trgm ON issues USING GIN (lower(issue_key) gin_trgm_ops);
CREATE INDEX idx_issue_label_names_trgm ON issues USING GIN (lower(coalesce(label_names, '')) gin_trgm_ops);
CREATE INDEX idx_issue_attachment_names_trgm ON issues USING GIN (lower(coalesce(attachment_names, '')) gin_trgm_ops);
