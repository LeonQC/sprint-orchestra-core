package com.jiraagentic.app.search;

import java.time.Instant;

/**
 * One comment as it is stored in OpenSearch.
 *
 * <p>{@code issueTitle} is denormalized in rather than joined at query time: a comment hit renders
 * with its parent issue's title, and fetching that per hit would be an N+1 against Postgres on every
 * keystroke of a search box. It is display-only — not indexed, not searchable — so a stale copy after
 * an issue rename costs a slightly out-of-date label on a result row, never a wrong or missing hit.
 * The issue's own re-index does not rewrite its comments; if that ever becomes visible, the fix is
 * to re-index the comments alongside the issue, not to join at read time.
 */
public record CommentSearchDocument(
        Long commentId,
        Long issueId,
        Long spaceId,
        String issueKey,
        String issueTitle,
        String content,
        String substringSource,
        Instant updatedAt
) {
    public String documentId() {
        return String.valueOf(commentId);
    }
}
