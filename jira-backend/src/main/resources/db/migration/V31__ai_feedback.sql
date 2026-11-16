CREATE TABLE ai_feedback (
    id UUID PRIMARY KEY,
    answer_id UUID NOT NULL,
    reporter_id BIGINT NOT NULL REFERENCES users(id),
    rating VARCHAR(8) NOT NULL CHECK (rating IN ('up', 'down')),
    reason VARCHAR(32) NOT NULL DEFAULT '',
    comment TEXT NOT NULL DEFAULT '',
    snapshot JSONB NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'new'
        CHECK (status IN ('new', 'reviewing', 'resolved', 'dismissed')),
    root_cause VARCHAR(32) NOT NULL DEFAULT '',
    review_note TEXT NOT NULL DEFAULT '',
    expected_answer TEXT NOT NULL DEFAULT '',
    required_issue_keys JSONB NOT NULL DEFAULT '[]',
    reviewer_id BIGINT REFERENCES users(id),
    revision INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (answer_id, reporter_id)
);
CREATE INDEX ai_feedback_created_idx ON ai_feedback (created_at DESC, id);
CREATE INDEX ai_feedback_status_idx ON ai_feedback (status, created_at DESC);
