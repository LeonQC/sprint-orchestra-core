package com.jiraagentic.app.search;

import com.jiraagentic.app.entity.Comment;
import com.jiraagentic.app.entity.Issue;
import com.jiraagentic.app.entity.IssueAttachment;
import com.jiraagentic.app.entity.Label;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Turns entities into the projections that go into OpenSearch.
 *
 * <p>This is where the denormalization that migration V28 did with database triggers now happens.
 * Labels and attachment filenames live in their own tables, and a Postgres generated column can only
 * see its own row — hence two extra columns on {@code issues} kept in sync by two triggers and two
 * functions. In a document store the join simply happens here, at index time, and all of that
 * machinery goes away.
 */
@Component
@RequiredArgsConstructor
public class SearchDocumentFactory {

    public IssueSearchDocument toDocument(Issue issue, List<IssueAttachment> attachments) {
        String title = stripHtml(issue.getTitle());
        String description = stripHtml(issue.getDescription());
        String labelNames = issue.getLabels() == null ? "" : issue.getLabels().stream()
                .map(Label::getName)
                .collect(Collectors.joining(" "));
        String attachmentNames = attachments.stream()
                .map(IssueAttachment::getOriginalFilename)
                .collect(Collectors.joining(" "));

        return new IssueSearchDocument(
                issue.getId(),
                issue.getSpace().getId(),
                issue.getIssueKey(),
                title,
                description,
                labelNames,
                attachmentNames,
                IssueSearchDocument.buildSubstringSource(
                        issue.getIssueKey(), title, description, labelNames, attachmentNames),
                issue.getUpdatedAt());
    }

    public CommentSearchDocument toDocument(Comment comment) {
        Issue issue = comment.getIssue();
        String content = stripHtml(comment.getContent());
        return new CommentSearchDocument(
                comment.getId(),
                issue.getId(),
                issue.getSpace().getId(),
                issue.getIssueKey(),
                stripHtml(issue.getTitle()),
                content,
                content.toLowerCase(),
                comment.getUpdatedAt());
    }

    /**
     * The Java equivalent of migration V28's {@code strip_html()}.
     *
     * <p>Applied when building the document, not left to the analyzer's {@code html_strip} char
     * filter, because {@code _source} is what highlight fragments and the literal-substring logic
     * read. The analyzer would keep tags out of the inverted index but not out of what a user sees —
     * which is precisely the bug V28 was written to fix, snippets rendering as "&lt;p&gt;&lt;li&gt;".
     *
     * <p>Replaces tags with a space rather than deleting them so "&lt;li&gt;a&lt;/li&gt;&lt;li&gt;b&lt;/li&gt;"
     * does not become the single token "ab".
     */
    static String stripHtml(String value) {
        if (value == null) {
            return "";
        }
        return value.replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
    }
}
