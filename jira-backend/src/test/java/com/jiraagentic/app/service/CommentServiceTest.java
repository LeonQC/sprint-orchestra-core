package com.jiraagentic.app.service;

import com.jiraagentic.app.dto.CommentDto;
import com.jiraagentic.app.dto.CreateCommentRequest;
import com.jiraagentic.app.entity.Comment;
import com.jiraagentic.app.entity.Issue;
import com.jiraagentic.app.entity.Space;
import com.jiraagentic.app.entity.User;
import com.jiraagentic.app.event.CommentContentChangedEvent;
import com.jiraagentic.app.event.CommentDeletedEvent;
import com.jiraagentic.app.kafka.CommentIngestionMessage;
import com.jiraagentic.app.repository.CommentRepository;
import com.jiraagentic.app.repository.IssueRepository;
import com.jiraagentic.app.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class CommentServiceTest {

    @Mock CommentRepository commentRepository;
    @Mock IssueRepository issueRepository;
    @Mock UserRepository userRepository;
    @Mock IssueHistoryService issueHistoryService;
    @Mock IssueAttachmentService issueAttachmentService;
    @Mock ActiveSpaceGuard activeSpaceGuard;
    @Mock ApplicationEventPublisher applicationEventPublisher;
    @InjectMocks CommentService service;

    private Issue issue;
    private Issue otherIssue;

    @BeforeEach
    void setUp() {
        Space space = new Space();
        space.setId(1L);
        issue = issue(10L, "ATC-10", space);
        otherIssue = issue(20L, "ATC-20", space);
        User user = new User();
        user.setId(5L);
        user.setName("Liam Ortiz");
        when(issueRepository.findById(10L)).thenReturn(Optional.of(issue));
        when(userRepository.findById(5L)).thenReturn(Optional.of(user));
        when(commentRepository.save(any(Comment.class))).thenAnswer(inv -> {
            Comment c = inv.getArgument(0);
            c.setId(999L);
            c.setCreatedAt(java.time.Instant.parse("2026-06-18T11:30:00Z"));
            return c;
        });
    }

    @Test
    void topLevelCommentHasNoParent() {
        CommentDto dto = service.create(10L, request(null));
        assertThat(dto.getParentId()).isNull();
        assertThat(dto.getReplyToId()).isNull();
    }

    @Test
    void replyToTopLevelCommentPointsAtIt() {
        stubComment(100L, issue, null);
        CommentDto dto = service.create(10L, request(100L));
        assertThat(dto.getParentId()).isEqualTo(100L);
        assertThat(dto.getReplyToId()).isEqualTo(100L);
    }

    @Test
    void replyToAReplyIsFiledUnderTheTopLevelCommentButRemembersWhoItAnswered() {
        stubComment(101L, issue, 100L);
        CommentDto dto = service.create(10L, request(101L));
        assertThat(dto.getParentId()).isEqualTo(100L);
        assertThat(dto.getReplyToId()).isEqualTo(101L);
    }

    @Test
    void parentFromAnotherIssueIsRejected() {
        stubComment(200L, otherIssue, null);
        assertThatThrownBy(() -> service.create(10L, request(200L)))
                .isInstanceOf(ResponseStatusException.class);
        verify(commentRepository, never()).save(any());
    }

    @Test
    void deletingTopLevelCommentAlsoDeletesItsRepliesAndAnnouncesEach() {
        Comment root = stubComment(100L, issue, null);
        Comment r1 = stubComment(101L, issue, 100L);
        Comment r2 = stubComment(102L, issue, 100L);
        when(commentRepository.findByParentIdOrderByCreatedAtAsc(100L)).thenReturn(List.of(r1, r2));

        service.delete(root.getId());

        verify(commentRepository).deleteAllByIdInBatch(List.of(101L, 102L));
        verify(commentRepository).deleteById(100L);
        ArgumentCaptor<CommentDeletedEvent> events = ArgumentCaptor.forClass(CommentDeletedEvent.class);
        verify(applicationEventPublisher, times(3)).publishEvent(events.capture());
        assertThat(events.getAllValues()).extracting(CommentDeletedEvent::commentId)
                .containsExactlyInAnyOrder(100L, 101L, 102L);
    }

    @Test
    void deletingAReplyLeavesTheRestOfTheThread() {
        stubComment(101L, issue, 100L);

        service.delete(101L);

        verify(commentRepository, never()).findByParentIdOrderByCreatedAtAsc(any());
        verify(commentRepository, never()).deleteAllByIdInBatch(any());
        verify(commentRepository).deleteById(101L);
        verify(applicationEventPublisher, times(1)).publishEvent(any(CommentDeletedEvent.class));
    }

    private static Issue issue(long id, String key, Space space) {
        Issue i = new Issue();
        i.setId(id);
        i.setIssueKey(key);
        i.setSpace(space);
        return i;
    }

    private Comment stubComment(long id, Issue onIssue, Long parentId) {
        Comment c = new Comment();
        c.setId(id);
        c.setIssue(onIssue);
        c.setParentId(parentId);
        c.setContent("x");
        when(commentRepository.findById(id)).thenReturn(Optional.of(c));
        return c;
    }

    private static CreateCommentRequest request(Long parentId) {
        CreateCommentRequest req = new CreateCommentRequest();
        req.setAuthorId(5L);
        req.setContent("hello");
        req.setParentId(parentId);
        return req;
    }

    @Test
    void ragIngestionMessageCarriesAuthorTimeAndThread() {
        stubComment(101L, issue, 100L);
        service.create(10L, request(101L));

        ArgumentCaptor<CommentContentChangedEvent> event = ArgumentCaptor.forClass(CommentContentChangedEvent.class);
        verify(applicationEventPublisher).publishEvent(event.capture());
        CommentIngestionMessage msg = CommentIngestionMessage.upserted(event.getValue(), java.util.UUID.randomUUID());
        assertThat(msg.authorName()).isEqualTo("Liam Ortiz");
        assertThat(msg.createdAt()).isEqualTo(java.time.Instant.parse("2026-06-18T11:30:00Z"));
        assertThat(msg.parentId()).isEqualTo(100L);
        assertThat(msg.replyToId()).isEqualTo(101L);
    }
}
