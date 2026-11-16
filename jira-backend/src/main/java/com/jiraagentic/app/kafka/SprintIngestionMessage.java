package com.jiraagentic.app.kafka;

import com.jiraagentic.app.event.SprintContentChangedEvent;
import com.jiraagentic.app.event.SprintDeletedEvent;
import java.time.Instant;
import java.util.UUID;

/** Searchable goal and citation scope; no duplicated sprint metrics. */
public record SprintIngestionMessage(
        String eventId, String eventType, Instant emittedAt, long sprintId,
        String sprintName, long spaceId, String goal
) {
    public static SprintIngestionMessage upserted(SprintContentChangedEvent e, UUID eventId) {
        return new SprintIngestionMessage(eventId.toString(), "sprint_upserted", e.occurredAt(),
                e.sprintId(), e.sprintName(), e.spaceId(), e.goal());
    }

    public static SprintIngestionMessage deleted(SprintDeletedEvent e, UUID eventId) {
        return new SprintIngestionMessage(eventId.toString(), "sprint_deleted", e.occurredAt(),
                e.sprintId(), e.sprintName(), e.spaceId(), null);
    }
}
