package com.jiraagentic.app.kafka;

import com.jiraagentic.app.event.IssueContentChangedEvent;
import com.jiraagentic.app.event.IssueDeletedEvent;
import java.time.Instant;
import java.util.UUID;

/** Searchable text and citation scope only; business metadata stays in jira-backend and is read through its structured API. */
public record IssueIngestionMessage(
        String eventId, String eventType, Instant emittedAt, long issueId, String issueKey,
        long spaceId, String title, String description, String parentKey, String parentTitle
) {
    public static IssueIngestionMessage upserted(IssueContentChangedEvent e, UUID eventId) {
        return new IssueIngestionMessage(eventId.toString(), "issue_upserted", e.occurredAt(),
                e.issueId(), e.issueKey(), e.spaceId(), e.title(), e.description(),
                e.parentKey(), e.parentTitle());
    }

    public static IssueIngestionMessage deleted(IssueDeletedEvent e, UUID eventId) {
        return new IssueIngestionMessage(eventId.toString(), "issue_deleted", e.occurredAt(),
                e.issueId(), e.issueKey(), e.spaceId(), null, null, null, null);
    }
}
