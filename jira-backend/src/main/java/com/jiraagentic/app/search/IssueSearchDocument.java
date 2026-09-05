package com.jiraagentic.app.search;

import java.time.Instant;
import java.util.List;

/**
 * One issue as stored in OpenSearch — only what search needs: title, description, labels, attachment
 * filenames, the space id, and the key and timestamp for display. Everything else stays in Postgres;
 * unqueried fields would just go stale.
 *
 * <p>Title and description are stored HTML-stripped, because highlighting and the literal-substring
 * logic read {@code _source}, where raw HTML used to show up in front of users.
 *
 * <p>{@code substringSource} is all searchable fields lowercased in one {@code wildcard} field, so the
 * SUBSTRING tier is a single leading-wildcard query ({@code wildcard} is built for {@code LIKE '%q%'}).
 */
public record IssueSearchDocument(
        Long issueId,
        Long spaceId,
        String issueKey,
        String title,
        String description,
        String labelNames,
        String attachmentNames,
        String substringSource,
        Instant updatedAt
) {
    /** Document id in the index. The issue's own id, so re-indexing overwrites rather than duplicates. */
    public String documentId() {
        return String.valueOf(issueId);
    }

    public static String buildSubstringSource(String issueKey, String title, String description,
                                              String labelNames, String attachmentNames) {
        return String.join(" ",
                        List.of(nullToEmpty(issueKey), nullToEmpty(title), nullToEmpty(description),
                                nullToEmpty(labelNames), nullToEmpty(attachmentNames)))
                .toLowerCase();
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }
}
