package com.jiraagentic.app.repository;

import com.jiraagentic.app.entity.Comment;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface CommentRepository extends JpaRepository<Comment, Long> {

    List<Comment> findByIssueIdOrderByCreatedAtAsc(Long issueId);

    List<Comment> findByParentIdOrderByCreatedAtAsc(Long parentId);

    // One bulk statement, not the derived delete-by-entity: deleting a top-level comment row before
    // its replies one-by-one would trip fk_comments_parent.
    @Modifying
    @Query("DELETE FROM Comment c WHERE c.issue.id = :issueId")
    void deleteByIssue_Id(@Param("issueId") Long issueId);
}
