package com.jiraagentic.app.search;

import com.jiraagentic.app.entity.Comment;
import com.jiraagentic.app.entity.Issue;
import com.jiraagentic.app.event.AttachmentDeletedEvent;
import com.jiraagentic.app.event.AttachmentUploadedEvent;
import com.jiraagentic.app.event.CommentContentChangedEvent;
import com.jiraagentic.app.event.CommentDeletedEvent;
import com.jiraagentic.app.event.IssueDeletedEvent;
import com.jiraagentic.app.event.IssueSearchIndexRequestedEvent;
import com.jiraagentic.app.repository.CommentRepository;
import com.jiraagentic.app.repository.IssueAttachmentRepository;
import com.jiraagentic.app.repository.IssueRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Keeps the OpenSearch indices in step with Postgres (the source of truth).
 *
 * <p>Runs after commit, like the Kafka publisher, so a rolled-back row never gets indexed. Also
 * at-most-once: a crash between commit and indexing leaves the index behind; repair with
 * {@code POST /internal/search/reindex}.
 *
 * <p>Each handler opens a new read-only transaction and reloads the issue: the original transaction has
 * committed (lazy labels/attachments would throw), and some changes (a new attachment) never touch the
 * issue row.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SearchIndexListener {

    private final IssueRepository issueRepository;
    private final CommentRepository commentRepository;
    private final IssueAttachmentRepository attachmentRepository;
    private final SearchDocumentFactory documentFactory;
    private final SearchIndexWriter writer;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Transactional(propagation = Propagation.REQUIRES_NEW, readOnly = true)
    public void onIssueIndexRequested(IssueSearchIndexRequestedEvent event) {
        reindexIssue(event.issueId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onIssueDeleted(IssueDeletedEvent event) {
        writer.deleteIssue(event.issueId());
    }

    /**
     * An attachment upload changes what its issue is findable by — migration V28 made filenames
     * searchable — but it writes only the attachments table, so no issue-level content event fires
     * for it. Under Postgres a trigger covered this; here the listener has to.
     */
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Transactional(propagation = Propagation.REQUIRES_NEW, readOnly = true)
    public void onAttachmentUploaded(AttachmentUploadedEvent event) {
        reindexIssue(event.issueId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Transactional(propagation = Propagation.REQUIRES_NEW, readOnly = true)
    public void onAttachmentDeleted(AttachmentDeletedEvent event) {
        reindexIssue(event.issueId());
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    @Transactional(propagation = Propagation.REQUIRES_NEW, readOnly = true)
    public void onCommentChanged(CommentContentChangedEvent event) {
        Comment comment = commentRepository.findById(event.commentId()).orElse(null);
        if (comment == null) {
            // Created and deleted before this listener ran. The delete event will have removed the
            // document (or there is none), so there is nothing to repair.
            return;
        }
        writer.indexComment(documentFactory.toDocument(comment));
    }

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onCommentDeleted(CommentDeletedEvent event) {
        writer.deleteComment(event.commentId());
    }

    private void reindexIssue(long issueId) {
        Issue issue = issueRepository.findById(issueId).orElse(null);
        if (issue == null) {
            log.debug("issue {} vanished before search indexing; delete event will clear it", issueId);
            return;
        }
        writer.indexIssue(documentFactory.toDocument(
                issue, attachmentRepository.findByIssueIdOrderByCreatedAtDesc(issueId)));
    }
}
