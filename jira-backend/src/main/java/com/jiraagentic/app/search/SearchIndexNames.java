package com.jiraagentic.app.search;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * The two indices behind the global search bar.
 *
 * <p>Two, because a comment is its own result and has no title, labels or filenames to weight.
 * Neither is vectorization-service's chunk index: RAG retrieves paragraphs, while a person wants one
 * row per thing (a 12-page attachment shouldn't get twelve chances to rank, and new issues must be
 * findable without waiting for embedding).
 *
 * <p>Configurable so integration tests can use throwaway indices.
 */
@Component
@Getter
public class SearchIndexNames {

    private final String issues;
    private final String comments;

    public SearchIndexNames(
            @Value("${app.opensearch.index.issues:orchestra-issues}") String issues,
            @Value("${app.opensearch.index.comments:orchestra-comments}") String comments) {
        this.issues = issues;
        this.comments = comments;
    }
}
