package com.jiraagentic.app.search;

import java.time.Instant;
import java.util.List;

/**
 * One OpenSearch issue hit, before {@link com.jiraagentic.app.service.SearchService} assigns its tier.
 *
 * <p>{@code rank} is normalized to (0, 1] within its tier. SearchService adds a per-tier base
 * (10000/1000/500/0) so a lower tier never outranks a higher one, which only holds if ranks stay
 * below 1 — and BM25 {@code _score} is unbounded.
 */
public record IssueSearchHit(
        Long issueId,
        String issueKey,
        Long spaceId,
        String title,
        String snippet,
        double rank,
        List<String> matchedFields,
        boolean exactMatch,
        String exactSnippet,
        Instant updatedAt
) {
}
