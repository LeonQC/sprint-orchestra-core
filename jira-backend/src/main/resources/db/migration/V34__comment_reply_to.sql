-- parent_id says which thread a reply sits in (always the top-level comment); reply_to_id keeps
-- which comment it actually answered, so "replying to X" survives the two-level flattening.
-- SET NULL: deleting the answered reply just drops the "replying to" label, not the reply.
ALTER TABLE comments ADD COLUMN reply_to_id BIGINT;
ALTER TABLE comments ADD CONSTRAINT fk_comments_reply_to FOREIGN KEY (reply_to_id) REFERENCES comments (id) ON DELETE SET NULL;
CREATE INDEX idx_comment_reply_to ON comments (reply_to_id);
