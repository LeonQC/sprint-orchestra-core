package com.jiraagentic.app.event;

import java.time.Instant;

/** Searchable issue text, bridged to Kafka after commit. Structured facts remain in the source DB. */
public record IssueContentChangedEvent(
        long issueId, String issueKey, long spaceId, String title, String description,
        String parentKey, String parentTitle, Instant occurredAt
) {}
