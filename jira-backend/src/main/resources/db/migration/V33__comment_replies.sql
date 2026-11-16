-- Two-level comment threads: a reply always points at a top-level comment (never at another
-- reply), so threads never nest deeper than one level. NULL = top-level comment, which is what
-- every pre-existing row stays.
ALTER TABLE comments ADD COLUMN parent_id BIGINT;
ALTER TABLE comments ADD CONSTRAINT fk_comments_parent FOREIGN KEY (parent_id) REFERENCES comments (id);
CREATE INDEX idx_comment_parent ON comments (parent_id);
