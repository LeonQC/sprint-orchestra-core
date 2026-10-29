-- Helpful votes are aggregate quality signals. They are not unresolved failure reports and must
-- not enter the root-cause review queue. Preserve the vote and answer snapshot for analytics.
UPDATE ai_feedback
SET status = 'resolved',
    root_cause = '',
    review_note = '',
    expected_answer = '',
    required_issue_keys = '[]'::jsonb,
    reviewer_id = NULL,
    revision = revision + 1,
    updated_at = now()
WHERE rating = 'up'
  AND status <> 'resolved';
