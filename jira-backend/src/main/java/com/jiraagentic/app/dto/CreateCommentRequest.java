package com.jiraagentic.app.dto;

import lombok.Data;

@Data
public class CreateCommentRequest {
    private Long authorId;
    private String content;
    /** Optional: the comment being answered. A reply to a reply is filed under that reply's top-level comment. */
    private Long parentId;
}
