package com.jiraagentic.app.search;

import com.jiraagentic.app.entity.Comment;
import com.jiraagentic.app.entity.Issue;
import com.jiraagentic.app.repository.CommentRepository;
import com.jiraagentic.app.repository.IssueAttachmentRepository;
import com.jiraagentic.app.repository.IssueRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Rebuilds the search indices from Postgres — the repair path for the accepted gap (indexing after
 * commit can't be atomic with it), a cheap alternative to an outbox. Also fills indices the first
 * time and rolls out mapping changes (change mapping, delete index, call this).
 *
 * <p>Idempotent. Doesn't remove documents whose row is gone; delete the index first for that.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SearchReindexService {

    /**
     * Pages rather than {@code findAll()}: a full reindex is exactly the moment the table is large,
     * and loading every issue with its labels into one list is how a maintenance endpoint becomes
     * the thing that takes the service down.
     */
    private static final int PAGE_SIZE = 200;

    private final IssueRepository issueRepository;
    private final CommentRepository commentRepository;
    private final IssueAttachmentRepository attachmentRepository;
    private final SearchDocumentFactory documentFactory;
    private final SearchIndexWriter writer;
    private final SearchIndexBootstrap bootstrap;
    private final SearchIndexNames indices;

    /**
     * Transactional at this level, not on the two loops below: they are private and called from
     * inside this bean, so Spring's proxy would never see the call and the annotation would silently
     * do nothing — leaving {@code issue.getLabels()} to throw on a detached entity. One read-only
     * transaction for the whole rebuild also gives the reindex a consistent snapshot to work from.
     */
    @Transactional(readOnly = true)
    public ReindexResult reindexAll() throws IOException {
        bootstrap.ensureIndex(indices.getIssues(), "opensearch/issues-mapping.json");
        bootstrap.ensureIndex(indices.getComments(), "opensearch/comments-mapping.json");

        int issues = reindexIssues();
        int comments = reindexComments();

        writer.refresh(indices.getIssues());
        writer.refresh(indices.getComments());
        log.info("search reindex complete: {} issues, {} comments", issues, comments);
        return new ReindexResult(issues, comments);
    }

    private int reindexIssues() throws IOException {
        int total = 0;
        for (int page = 0; ; page++) {
            Page<Issue> batch = issueRepository.findAll(
                    PageRequest.of(page, PAGE_SIZE, Sort.by("id")));
            if (batch.isEmpty()) {
                return total;
            }
            List<IssueSearchDocument> documents = new ArrayList<>(batch.getNumberOfElements());
            for (Issue issue : batch) {
                documents.add(documentFactory.toDocument(
                        issue, attachmentRepository.findByIssueIdOrderByCreatedAtDesc(issue.getId())));
            }
            total += writer.bulkIndexIssues(documents);
            if (batch.isLast()) {
                return total;
            }
        }
    }

    private int reindexComments() throws IOException {
        int total = 0;
        for (int page = 0; ; page++) {
            Page<Comment> batch = commentRepository.findAll(
                    PageRequest.of(page, PAGE_SIZE, Sort.by("id")));
            if (batch.isEmpty()) {
                return total;
            }
            List<CommentSearchDocument> documents = new ArrayList<>(batch.getNumberOfElements());
            for (Comment comment : batch) {
                documents.add(documentFactory.toDocument(comment));
            }
            total += writer.bulkIndexComments(documents);
            if (batch.isLast()) {
                return total;
            }
        }
    }

    public record ReindexResult(int issues, int comments) {
    }
}
