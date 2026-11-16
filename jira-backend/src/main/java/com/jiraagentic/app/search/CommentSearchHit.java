package com.jiraagentic.app.search;

import java.time.Instant;
import java.util.List;

/** One comment hit from OpenSearch. See {@link IssueSearchHit} for why {@code rank} is normalized. */
public record CommentSearchHit(
        Long commentId,
        Long issueId,
        String issueKey,
        Long spaceId,
        String issueTitle,
        String snippet,
        double rank,
        List<String> matchedFields,
        boolean exactMatch,
        String exactSnippet,
        Instant updatedAt
) {
}
