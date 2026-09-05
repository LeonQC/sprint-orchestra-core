package com.jiraagentic.app.dto;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.Instant;
import java.util.List;

@Data
@AllArgsConstructor
public class SearchResultDto {
    /** "ISSUE" or "COMMENT". */
    private String type;
    /**
     * Identity of the underlying row — the issue id for an ISSUE hit, the COMMENT id for a comment
     * hit. {@code type + resultId} is unique across a result set; {@code issueId} is NOT, because
     * several comments on one issue each produce their own row. Clients keying a list by
     * {@code issueId} silently collapse those, which in React means duplicate keys and stale rows
     * surviving a re-render.
     */
    private Long resultId;
    /** "EXACT_KEY", "FULL_TEXT", "SUBSTRING", or "FUZZY" — which search tier produced this hit. */
    private String matchType;
    private Long spaceId;
    private Long issueId;
    private String issueKey;
    private String title;
    private String snippet;
    private double rank;
    /**
     * Which field(s) the query actually matched on — some of "key", "title", "description",
     * "labels", "attachment", "comment". `matchType` says which TIER produced the hit; this says
     * WHY, which is what a user needs once labels and attachment filenames are searchable too
     * (a row whose title looks unrelated is explained by a "labels" match).
     */
    private List<String> matchedFields;
    /**
     * Whether the query text appears verbatim (case-insensitive) in some searched field. False for a
     * hit that only survived stemming ("orders" matching "order") or fuzzy typo-correction — which
     * is exactly what the UI's "Exact only" filter drops. A filter over the ranked set, not a
     * different query, so toggling it never reshuffles what is already on screen.
     */
    private boolean exactMatch;
    /**
     * Snippet built around the LITERAL occurrence of the query, with it wrapped in the same
     * "&lt;b&gt;" markers as {@code snippet}. Null exactly when {@link #exactMatch} is false.
     *
     * <p>Needed because {@code snippet} comes from OpenSearch's highlighter, which highlights the
     * matched STEMS: searching "payments" on a row that really does contain "payments" can still render a
     * snippet bolding "payment", which reads as a broken exact filter. Under "Exact only" the UI
     * shows this instead, so what is highlighted is always literally what was typed.
     */
    private String exactSnippet;
    private Instant updatedAt;
}
