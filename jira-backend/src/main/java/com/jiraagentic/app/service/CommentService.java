package com.jiraagentic.app.service;

import com.jiraagentic.app.dto.CommentDto;
import com.jiraagentic.app.dto.CreateCommentRequest;
import com.jiraagentic.app.entity.Comment;
import com.jiraagentic.app.entity.Issue;
import com.jiraagentic.app.event.CommentContentChangedEvent;
import com.jiraagentic.app.event.CommentDeletedEvent;
import com.jiraagentic.app.repository.CommentRepository;
import com.jiraagentic.app.repository.IssueRepository;
import com.jiraagentic.app.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class CommentService {

    private final CommentRepository commentRepository;
    private final IssueRepository issueRepository;
    private final UserRepository userRepository;
    private final IssueHistoryService issueHistoryService;
    private final IssueAttachmentService issueAttachmentService;
    private final ActiveSpaceGuard activeSpaceGuard;
    private final ApplicationEventPublisher applicationEventPublisher;

    public List<CommentDto> findByIssue(Long issueId) {
        Issue issue = issueRepository.findById(issueId)
                .orElseThrow(() -> new RuntimeException("Issue not found: " + issueId));
        activeSpaceGuard.requireActive(issue.getSpace());
        return commentRepository.findByIssueIdOrderByCreatedAtAsc(issueId).stream()
                .map(CommentDto::from)
                .collect(Collectors.toList());
    }

    @Transactional
    public CommentDto create(Long issueId, CreateCommentRequest req) {
        Issue issue = issueRepository.findById(issueId)
                .orElseThrow(() -> new RuntimeException("Issue not found: " + issueId));
        activeSpaceGuard.requireActive(issue.getSpace());

        Comment comment = new Comment();
        comment.setIssue(issue);
        comment.setAuthor(userRepository.findById(req.getAuthorId())
                .orElseThrow(() -> new RuntimeException("User not found: " + req.getAuthorId())));
        comment.setContent(req.getContent());
        comment.setParentId(resolveThreadRoot(issueId, req.getParentId()));
        comment.setReplyToId(req.getParentId());
        Comment saved = commentRepository.save(comment);
        issueHistoryService.recordEvent(issue, req.getAuthorId(), "comment_created",
                saved.getParentId() == null ? "added a comment" : "replied to a comment");
        applicationEventPublisher.publishEvent(CommentContentChangedEvent.of(saved));
        return CommentDto.from(saved);
    }

    @Transactional
    public CommentDto update(Long commentId, String content) {
        Comment comment = commentRepository.findById(commentId)
                .orElseThrow(() -> new RuntimeException("Comment not found: " + commentId));
        activeSpaceGuard.requireActive(comment.getIssue().getSpace());
        String oldContent = comment.getContent();
        comment.setContent(content);
        Comment saved = commentRepository.save(comment);
        issueHistoryService.recordFieldChange(
                saved.getIssue(),
                saved.getAuthor() != null ? saved.getAuthor().getId() : null,
                "comment",
                oldContent,
                saved.getContent()
        );
        issueAttachmentService.reconcileEmbeddedAttachments(saved.getIssue().getId(),
                saved.getAuthor() != null ? saved.getAuthor().getId() : null);
        applicationEventPublisher.publishEvent(CommentContentChangedEvent.of(saved));
        return CommentDto.from(saved);
    }

    @Transactional
    public void delete(Long commentId) {
        Comment comment = commentRepository.findById(commentId).orElse(null);
        if (comment == null) {
            throw new RuntimeException("Comment not found: " + commentId);
        }
        activeSpaceGuard.requireActive(comment.getIssue().getSpace());
        issueHistoryService.recordEvent(
                comment.getIssue(),
                comment.getAuthor() != null ? comment.getAuthor().getId() : null,
                "comment_deleted",
                "deleted a comment"
        );
        Long issueId = comment.getIssue().getId();
        String issueKey = comment.getIssue().getIssueKey();
        Long spaceId = comment.getIssue().getSpace().getId();
        Long actorId = comment.getAuthor() != null ? comment.getAuthor().getId() : null;
        // A thread goes with its top-level comment (the UI warns first). Each reply still gets its
        // own delete event so search and the vector store drop it too.
        List<Long> replyIds = comment.getParentId() != null ? List.of()
                : commentRepository.findByParentIdOrderByCreatedAtAsc(commentId).stream().map(Comment::getId).toList();
        if (!replyIds.isEmpty()) {
            commentRepository.deleteAllByIdInBatch(replyIds);
        }
        commentRepository.deleteById(commentId);
        List<Long> deletedIds = new ArrayList<>(replyIds);
        deletedIds.add(commentId);
        issueAttachmentService.reconcileEmbeddedAttachments(issueId, actorId);
        Instant now = Instant.now();
        for (Long id : deletedIds) {
            applicationEventPublisher.publishEvent(new CommentDeletedEvent(id, issueId, issueKey, spaceId, now));
        }
    }

    /** Flattens reply-to-a-reply onto the top-level comment so threads stay two levels deep. */
    private Long resolveThreadRoot(Long issueId, Long parentId) {
        if (parentId == null) {
            return null;
        }
        Comment parent = commentRepository.findById(parentId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.BAD_REQUEST, "Parent comment not found: " + parentId));
        if (!parent.getIssue().getId().equals(issueId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Parent comment belongs to a different issue");
        }
        return parent.getParentId() != null ? parent.getParentId() : parent.getId();
    }
}
