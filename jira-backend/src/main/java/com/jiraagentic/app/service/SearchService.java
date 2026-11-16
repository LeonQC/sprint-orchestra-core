package com.jiraagentic.app.service;

import com.jiraagentic.app.dto.SearchResultDto;
import com.jiraagentic.app.repository.SpaceMemberRepository;
import com.jiraagentic.app.search.CommentSearchHit;
import com.jiraagentic.app.search.IssueSearchHit;
import com.jiraagentic.app.search.SearchQueries;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Tiered search — not a weighted blend of scores from unrelated scales. Tiers, in order:
 * <ol>
 *   <li>Exact/prefix issue-key match — always first.</li>
 *   <li>Full text (stemmed, over title, description, labels, attachment filenames) — the main signal.</li>
 *   <li>Literal substring — what stemming can't do ("auth" finds
 *       "authentication"). Always run for queries of 3+ chars; banding keeps it below full-text hits.</li>
 *   <li>Fuzzy (typo tolerance) — only when the tiers above don't fill a page.</li>
 * </ol>
 * Each tier's rank is offset into its own band, so a lower tier never outranks a higher one;
 * {@link SearchQueries} normalizes BM25 into (0, 1] per tier to keep that true.
 *
 * <p>The queries run in OpenSearch ({@link SearchQueries}). Tiering stays here because it's a
 * product rule, and OpenSearch's own clause combining is exactly the weighted blend this avoids.
 */
@Service
@RequiredArgsConstructor
public class SearchService {

    private static final int DEFAULT_LIMIT = 20;
    private static final int MAX_LIMIT = 50;

    /** Substring and fuzzy matching need at least 3 characters to mean anything. */
    private static final int MIN_SUBSTRING_QUERY_LENGTH = 3;

    private static final double TIER_EXACT_KEY = 10_000;
    private static final double TIER_FULL_TEXT = 1_000;
    private static final double TIER_SUBSTRING = 500;
    private static final double TIER_FUZZY = 0;

    private final SpaceMemberRepository spaceMemberRepository;
    private final SearchQueries searchQueries;

    public List<SearchResultDto> search(Long userId, String query, Integer limit) {
        if (query == null || query.isBlank()) {
            return List.of();
        }
        int cappedLimit = clampLimit(limit);
        List<Long> spaceIds = spaceMemberRepository.findActiveSpaceIdsByUserId(userId);
        if (spaceIds.isEmpty()) {
            return List.of();
        }

        String trimmedQuery = query.trim();
        List<SearchResultDto> results = new ArrayList<>();
        Set<Long> seenIssueIds = new HashSet<>();
        Set<Long> seenCommentIds = new HashSet<>();

        for (IssueSearchHit hit : searchQueries.searchIssuesByKeyPrefix(spaceIds, trimmedQuery, cappedLimit)) {
            if (seenIssueIds.add(hit.issueId())) {
                results.add(toIssueDto("EXACT_KEY", hit, TIER_EXACT_KEY));
            }
        }

        for (IssueSearchHit hit : searchQueries.searchIssues(spaceIds, trimmedQuery, cappedLimit)) {
            if (seenIssueIds.add(hit.issueId())) {
                results.add(toIssueDto("FULL_TEXT", hit, TIER_FULL_TEXT));
            }
        }
        for (CommentSearchHit hit : searchQueries.searchComments(spaceIds, trimmedQuery, cappedLimit)) {
            if (seenCommentIds.add(hit.commentId())) {
                results.add(toCommentDto("FULL_TEXT", hit, TIER_FULL_TEXT));
            }
        }

        // Tier 2b: literal substring — run unconditionally (not just on underfill), banded below FTS.
        if (trimmedQuery.length() >= MIN_SUBSTRING_QUERY_LENGTH) {
            for (IssueSearchHit hit : searchQueries.searchIssuesBySubstring(spaceIds, trimmedQuery, cappedLimit)) {
                if (seenIssueIds.add(hit.issueId())) {
                    results.add(toIssueDto("SUBSTRING", hit, TIER_SUBSTRING));
                }
            }
            for (CommentSearchHit hit : searchQueries.searchCommentsBySubstring(spaceIds, trimmedQuery, cappedLimit)) {
                if (seenCommentIds.add(hit.commentId())) {
                    results.add(toCommentDto("SUBSTRING", hit, TIER_SUBSTRING));
                }
            }
        }

        // Tier 3: fuzzy, issues only. A typo-tolerant scan of every comment in a space produces far
        // more noise than hits, and comment matches are already found by the tiers above.
        if (results.size() < cappedLimit && trimmedQuery.length() >= MIN_SUBSTRING_QUERY_LENGTH) {
            for (IssueSearchHit hit : searchQueries.searchIssuesFuzzy(spaceIds, trimmedQuery, cappedLimit)) {
                if (seenIssueIds.add(hit.issueId())) {
                    results.add(toIssueDto("FUZZY", hit, TIER_FUZZY));
                }
            }
        }

        results.sort(Comparator.comparingDouble(SearchResultDto::getRank).reversed());
        return results.size() > cappedLimit ? results.subList(0, cappedLimit) : results;
    }

    private SearchResultDto toIssueDto(String matchType, IssueSearchHit hit, double tierBase) {
        return new SearchResultDto("ISSUE", hit.issueId(), matchType, hit.spaceId(), hit.issueId(),
                hit.issueKey(), hit.title(), hit.snippet(), tierBase + hit.rank(),
                hit.matchedFields(), hit.exactMatch(), hit.exactSnippet(), hit.updatedAt());
    }

    /**
     * Note {@code resultId} is the COMMENT id while {@code issueId} is its parent issue — a client
     * needs the former to identify the row and the latter to navigate. Several comments on one issue
     * are several rows, so keying them by issue id collapses them.
     */
    private SearchResultDto toCommentDto(String matchType, CommentSearchHit hit, double tierBase) {
        return new SearchResultDto("COMMENT", hit.commentId(), matchType, hit.spaceId(), hit.issueId(),
                hit.issueKey(), hit.issueTitle(), hit.snippet(), tierBase + hit.rank(),
                hit.matchedFields(), hit.exactMatch(), hit.exactSnippet(), hit.updatedAt());
    }

    private int clampLimit(Integer limit) {
        int value = limit == null ? DEFAULT_LIMIT : limit;
        return Math.min(Math.max(value, 1), MAX_LIMIT);
    }
}
