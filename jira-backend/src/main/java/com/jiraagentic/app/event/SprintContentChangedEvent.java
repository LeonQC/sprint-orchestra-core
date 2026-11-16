package com.jiraagentic.app.event;

import java.time.Instant;

/** Searchable sprint goal, bridged to Kafka after commit. Metrics stay in the source database. */
public record SprintContentChangedEvent(
        long sprintId, String sprintName, long spaceId, String goal, Instant occurredAt
) {}
