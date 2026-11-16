package com.jiraagentic.app.event;

import com.jiraagentic.app.entity.Comment;

import java.time.Instant;

/**
 * Published after a comment is created or edited, on the same transactional boundary. The
 * vectorization service upserts a single vector keyed by comment id (see the ingestion service's
 * {@code comment:{id}} chunk key), so repeated edits overwrite rather than accumulate stale copies.
 */
public record CommentContentChangedEvent(
        long commentId,
        long issueId,
        String issueKey,
        long spaceId,
        String content,
        Instant occurredAt,
        String authorName,
        Instant createdAt,
        Long parentId,
        Long replyToId
) {
    public static CommentContentChangedEvent of(Comment c) {
        return new CommentContentChangedEvent(
                c.getId(), c.getIssue().getId(), c.getIssue().getIssueKey(), c.getIssue().getSpace().getId(),
                c.getContent(), Instant.now(),
                c.getAuthor() != null ? c.getAuthor().getName() : null,
                c.getCreatedAt(), c.getParentId(), c.getReplyToId());
    }
}
