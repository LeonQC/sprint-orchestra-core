package com.jiraagentic.app.kafka;

import com.jiraagentic.app.event.CommentContentChangedEvent;
import com.jiraagentic.app.event.CommentDeletedEvent;
import com.jiraagentic.app.event.IssueContentChangedEvent;
import com.jiraagentic.app.event.IssueDeletedEvent;
import com.jiraagentic.app.event.SprintContentChangedEvent;
import com.jiraagentic.app.event.SprintDeletedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

import java.util.UUID;

/**
 * Bridges issue/comment events to the RAG content-ingestion Kafka topic. Publishes after commit and
 * fails soft (logs) if the broker is down. Every message is keyed by {@code issueKey}, so events for
 * one issue stay in order.
 *
 * <p><b>Known limitation:</b> after-commit + best-effort send is at-most-once — a crash between commit
 * and publish loses the event. An outbox or Debezium CDC would fix that; out of scope here.
 */
@Component
@ConditionalOnProperty(prefix = "app.kafka.content-ingestion", name = "enabled", havingValue = "true")
@RequiredArgsConstructor
@Slf4j
public class ContentIngestionKafkaPublisher {

    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Value("${app.kafka.content-ingestion.topic}")
    private String topic;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onIssueContentChanged(IssueContentChangedEvent event) {
        publish(event.issueKey(), IssueIngestionMessage.upserted(event, UUID.randomUUID()),
                "issue " + event.issueId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onIssueDeleted(IssueDeletedEvent event) {
        publish(event.issueKey(), IssueIngestionMessage.deleted(event, UUID.randomUUID()),
                "issue " + event.issueId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onSprintContentChanged(SprintContentChangedEvent event) {
        publish("sprint:" + event.sprintId(), SprintIngestionMessage.upserted(event, UUID.randomUUID()),
                "sprint " + event.sprintId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onSprintDeleted(SprintDeletedEvent event) {
        publish("sprint:" + event.sprintId(), SprintIngestionMessage.deleted(event, UUID.randomUUID()),
                "sprint " + event.sprintId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onCommentContentChanged(CommentContentChangedEvent event) {
        publish(event.issueKey(), CommentIngestionMessage.upserted(event, UUID.randomUUID()),
                "comment " + event.commentId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onCommentDeleted(CommentDeletedEvent event) {
        publish(event.issueKey(), CommentIngestionMessage.deleted(event, UUID.randomUUID()),
                "comment " + event.commentId());
    }

    private void publish(String partitionKey, Object message, String subject) {
        try {
            kafkaTemplate.send(topic, partitionKey, message)
                    .whenComplete((result, ex) -> {
                        if (ex != null) {
                            log.warn("Kafka content-ingestion publish failed for {}: {}", subject, ex.getMessage());
                        }
                    });
        } catch (Exception e) {
            log.warn("Kafka content-ingestion publish failed for {}: {}", subject, e.getMessage());
        }
    }
}
