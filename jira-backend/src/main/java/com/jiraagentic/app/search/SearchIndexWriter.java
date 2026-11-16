package com.jiraagentic.app.search;

import org.opensearch.client.opensearch.OpenSearchClient;
import org.opensearch.client.opensearch._types.FieldValue;
import org.opensearch.client.opensearch.core.BulkRequest;
import org.opensearch.client.opensearch.core.BulkResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * Write side of the search indices. Every method fails soft: indexing runs after commit, so throwing
 * would only turn "search is briefly stale" into "saving an issue returned 500". At-most-once, like
 * the Kafka publisher; the log line signals when {@code POST /internal/search/reindex} is needed.
 * Documents use the entity's id, so repeats overwrite instead of duplicating.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SearchIndexWriter {

    private final OpenSearchClient client;
    private final SearchIndexNames indices;

    public void indexIssue(IssueSearchDocument document) {
        try {
            client.index(r -> r
                    .index(indices.getIssues())
                    .id(document.documentId())
                    .document(document));
        } catch (IOException | RuntimeException e) {
            log.error("failed to index issue {} for search — run POST /internal/search/reindex to repair",
                    document.issueId(), e);
        }
    }

    public void indexComment(CommentSearchDocument document) {
        try {
            client.index(r -> r
                    .index(indices.getComments())
                    .id(document.documentId())
                    .document(document));
        } catch (IOException | RuntimeException e) {
            log.error("failed to index comment {} for search — run POST /internal/search/reindex to repair",
                    document.commentId(), e);
        }
    }

    public void deleteIssue(Long issueId) {
        try {
            client.delete(r -> r.index(indices.getIssues()).id(String.valueOf(issueId)));
            // An issue's comments are their own documents in their own index; nothing cascades in
            // OpenSearch, so a deleted issue would otherwise keep answering searches through its
            // comments — the row would render with the title of an issue that no longer exists.
            client.deleteByQuery(r -> r
                    .index(indices.getComments())
                    .query(q -> q.term(t -> t.field("issueId").value(FieldValue.of(issueId)))));
        } catch (IOException | RuntimeException e) {
            log.error("failed to remove issue {} from search — run POST /internal/search/reindex to repair",
                    issueId, e);
        }
    }

    public void deleteComment(Long commentId) {
        try {
            client.delete(r -> r.index(indices.getComments()).id(String.valueOf(commentId)));
        } catch (IOException | RuntimeException e) {
            log.error("failed to remove comment {} from search — run POST /internal/search/reindex to repair",
                    commentId, e);
        }
    }

    /** Bulk path, used only by the reindex job — a document at a time would be one round trip each. */
    public int bulkIndexIssues(List<IssueSearchDocument> documents) throws IOException {
        return bulk(documents, indices.getIssues(), IssueSearchDocument::documentId);
    }

    public int bulkIndexComments(List<CommentSearchDocument> documents) throws IOException {
        return bulk(documents, indices.getComments(), CommentSearchDocument::documentId);
    }

    private <T> int bulk(List<T> documents, String index, java.util.function.Function<T, String> idOf)
            throws IOException {
        if (documents.isEmpty()) {
            return 0;
        }
        BulkRequest.Builder request = new BulkRequest.Builder();
        for (T document : documents) {
            request.operations(op -> op.index(i -> i.index(index).id(idOf.apply(document)).document(document)));
        }
        BulkResponse response = client.bulk(request.build());
        if (response.errors()) {
            // Reindex is a maintenance operation someone is watching, so a partial failure has to be
            // loud — silently reporting "indexed N" while some of the N were rejected is how a
            // corrupt index gets declared healthy.
            throw new IOException("bulk indexing into " + index + " reported item-level failures");
        }
        return documents.size();
    }

    public void refresh(String index) throws IOException {
        client.indices().refresh(r -> r.index(index));
    }
}
