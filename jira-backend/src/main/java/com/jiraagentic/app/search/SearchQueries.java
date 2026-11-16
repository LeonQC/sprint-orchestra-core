package com.jiraagentic.app.search;

import org.opensearch.client.opensearch.OpenSearchClient;
import org.opensearch.client.opensearch._types.SortOrder;
import org.opensearch.client.opensearch._types.query_dsl.Query;
import org.opensearch.client.opensearch.core.SearchRequest;
import org.opensearch.client.opensearch.core.SearchResponse;
import org.opensearch.client.opensearch.core.search.Hit;
import org.opensearch.client.opensearch.core.search.HighlightField;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The four tiered search-bar queries, in OpenSearch; {@link com.jiraagentic.app.service.SearchService}
 * does the tiering.
 * <ul>
 *   <li><b>Exact key</b> — a {@code prefix} query on a lowercase keyword.</li>
 *   <li><b>Full text</b> — {@code simple_query_string} + BM25 (bare text, "quoted phrases",
 *       -exclusions, AND by default). Field boosts 10/4/2/1 for title, description, labels and
 *       attachment filenames.</li>
 *   <li><b>Substring</b> — a {@code wildcard} query on {@code substringSource}.</li>
 *   <li><b>Fuzzy</b> — {@code fuzziness: AUTO} (edit distance) to catch typos. Uses AND like full
 *       text, so typo tolerance is the only difference.</li>
 * </ul>
 *
 * <p>Scores are normalized per tier: BM25 is unbounded, and SearchService's tier bases
 * (10000/1000/500/0) only keep tiers apart if each tier's rank stays below 1.
 */
@Component
@RequiredArgsConstructor
public class SearchQueries {

    /** Field boosts, most important first: title, description, labels, attachment filenames. */
    private static final List<String> ISSUE_FULLTEXT_FIELDS =
            List.of("title^10", "description^4", "labelNames^2", "attachmentNames^1");

    private static final int SNIPPET_FRAGMENT_CHARS = 200;
    private static final int PLAIN_SNIPPET_CHARS = 200;

    private final OpenSearchClient client;
    private final SearchIndexNames indices;

    // --- Tier 1: exact / prefix issue key -------------------------------------------------------

    public List<IssueSearchHit> searchIssuesByKeyPrefix(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<IssueSearchDocument> response = execute(SearchRequest.of(r -> r
                .index(indices.getIssues())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        // The keyword field carries a lowercase normalizer, so the term must be
                        // lowercased here too — a normalizer applies to indexed values and to
                        // `term`/`prefix` query terms, but only if the query value is what the
                        // analyzer would have produced.
                        .must(m -> m.prefix(p -> p.field("issueKey").value(query.toLowerCase())))))
                // Deterministic ordering, matching the retired SQL's "exact match first, then key
                // ascending". _score is uniform for a prefix query, so sorting by it would leave the
                // order up to Lucene's internal doc order.
                .sort(s -> s.field(f -> f.field("issueKey").order(SortOrder.Asc)))));

        List<IssueSearchHit> hits = new ArrayList<>();
        for (Hit<IssueSearchDocument> hit : response.hits().hits()) {
            IssueSearchDocument doc = hit.source();
            if (doc == null) {
                continue;
            }
            boolean exact = doc.issueKey().equalsIgnoreCase(query);
            hits.add(new IssueSearchHit(
                    doc.issueId(), doc.issueKey(), doc.spaceId(), doc.title(),
                    "",
                    exact ? 1.0 : 0.9,
                    List.of("key"),
                    true,
                    LiteralSnippet.of(doc.issueKey(), query),
                    doc.updatedAt()));
        }
        // An exact key must lead, whatever the alphabetical order says ("SP1-3" before "SP1-30").
        hits.sort((a, b) -> Double.compare(b.rank(), a.rank()));
        return hits;
    }

    // --- Tier 2: full text ----------------------------------------------------------------------

    public List<IssueSearchHit> searchIssues(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<IssueSearchDocument> response = execute(SearchRequest.of(r -> r
                .index(indices.getIssues())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        .must(m -> m.simpleQueryString(s -> s
                                .query(query)
                                .fields(ISSUE_FULLTEXT_FIELDS)
                                .defaultOperator(org.opensearch.client.opensearch._types.query_dsl.Operator.And)))))
                .highlight(h -> h
                        // Plain <b> markers: the frontend splits on them and renders text nodes.
                        .preTags("<b>").postTags("</b>")
                        .numberOfFragments(1)
                        .fragmentSize(SNIPPET_FRAGMENT_CHARS)
                        .fields("description", HighlightField.of(f -> f))
                        .fields("title", HighlightField.of(f -> f))
                        .fields("labelNames", HighlightField.of(f -> f))
                        .fields("attachmentNames", HighlightField.of(f -> f)))
                .sort(s -> s.score(sc -> sc.order(SortOrder.Desc)))
                .sort(s -> s.field(f -> f.field("updatedAt").order(SortOrder.Desc)))));

        // The highlight map's keys ARE the fields that matched.
        return normalizeIssueRanks(response, query, (doc, hit) -> {
            String snippet = firstHighlight(hit.highlight(), "description", "title", "labelNames", "attachmentNames");
            return snippet != null ? snippet : truncate(doc.description());
        });
    }

    public List<CommentSearchHit> searchComments(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<CommentSearchDocument> response = executeComments(SearchRequest.of(r -> r
                .index(indices.getComments())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        .must(m -> m.simpleQueryString(s -> s
                                .query(query)
                                .fields("content")
                                .defaultOperator(org.opensearch.client.opensearch._types.query_dsl.Operator.And)))))
                .highlight(h -> h
                        .preTags("<b>").postTags("</b>")
                        .numberOfFragments(1)
                        .fragmentSize(SNIPPET_FRAGMENT_CHARS)
                        .fields("content", HighlightField.of(f -> f)))
                .sort(s -> s.score(sc -> sc.order(SortOrder.Desc)))
                .sort(s -> s.field(f -> f.field("updatedAt").order(SortOrder.Desc)))));

        return normalizeCommentRanks(response, query, (doc, hit) -> {
            String snippet = firstHighlight(hit.highlight(), "content");
            return snippet != null ? snippet : truncate(doc.content());
        });
    }

    // --- Tier 2b: literal substring -------------------------------------------------------------

    public List<IssueSearchHit> searchIssuesBySubstring(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<IssueSearchDocument> response = execute(SearchRequest.of(r -> r
                .index(indices.getIssues())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        .must(substringQuery(query))))
                .sort(s -> s.field(f -> f.field("updatedAt").order(SortOrder.Desc)))));

        List<IssueSearchHit> hits = new ArrayList<>();
        for (Hit<IssueSearchDocument> hit : response.hits().hits()) {
            IssueSearchDocument doc = hit.source();
            if (doc == null) {
                continue;
            }
            hits.add(new IssueSearchHit(
                    doc.issueId(), doc.issueKey(), doc.spaceId(), doc.title(),
                    firstNonNull(LiteralSnippet.of(doc.description(), query), truncate(doc.description())),
                    substringRank(query, doc),
                    issueMatchedFieldsLiteral(doc, query),
                    // This tier's query IS literal containment, so re-testing per row would be dead
                    // work — the same shortcut the retired SQL took.
                    true,
                    issueExactSnippet(doc, query),
                    doc.updatedAt()));
        }
        hits.sort((a, b) -> Double.compare(b.rank(), a.rank()));
        return hits;
    }

    public List<CommentSearchHit> searchCommentsBySubstring(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<CommentSearchDocument> response = executeComments(SearchRequest.of(r -> r
                .index(indices.getComments())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        .must(substringQuery(query))))
                .sort(s -> s.field(f -> f.field("updatedAt").order(SortOrder.Desc)))));

        List<CommentSearchHit> hits = new ArrayList<>();
        for (Hit<CommentSearchDocument> hit : response.hits().hits()) {
            CommentSearchDocument doc = hit.source();
            if (doc == null) {
                continue;
            }
            hits.add(new CommentSearchHit(
                    doc.commentId(), doc.issueId(), doc.issueKey(), doc.spaceId(), doc.issueTitle(),
                    firstNonNull(LiteralSnippet.of(doc.content(), query), truncate(doc.content())),
                    (double) query.length() / Math.max(length(doc.content()), 1),
                    List.of("comment"),
                    true,
                    LiteralSnippet.of(doc.content(), query),
                    doc.updatedAt()));
        }
        hits.sort((a, b) -> Double.compare(b.rank(), a.rank()));
        return hits;
    }

    // --- Tier 3: fuzzy fallback -----------------------------------------------------------------

    public List<IssueSearchHit> searchIssuesFuzzy(Collection<Long> spaceIds, String query, int limit) {
        SearchResponse<IssueSearchDocument> response = execute(SearchRequest.of(r -> r
                .index(indices.getIssues())
                .size(limit)
                .query(q -> q.bool(b -> b
                        .filter(spaceFilter(spaceIds))
                        .must(m -> m.multiMatch(mm -> mm
                                .query(query)
                                .fields("title^2", "description")
                                .fuzziness("AUTO")
                                // AND, like full text, so typo tolerance is the only difference.
                                // With OR, "search recent orders" matched 55 of 80 issues (e.g.
                                // "Reject repeated..." since reject ~ recent); now it matches one.
                                .operator(org.opensearch.client.opensearch._types.query_dsl.Operator.And)))))
                .highlight(h -> h
                        .preTags("<b>").postTags("</b>")
                        .numberOfFragments(1)
                        .fragmentSize(SNIPPET_FRAGMENT_CHARS)
                        .fields("title", HighlightField.of(f -> f))
                        .fields("description", HighlightField.of(f -> f)))
                .sort(s -> s.score(sc -> sc.order(SortOrder.Desc)))
                .sort(s -> s.field(f -> f.field("updatedAt").order(SortOrder.Desc)))));

        return normalizeIssueRanks(response, query, (doc, hit) -> {
            String snippet = firstHighlight(hit.highlight(), "title", "description");
            return snippet != null ? snippet : truncate(doc.description());
        });
    }

    // --- shared helpers -------------------------------------------------------------------------

    private static Query spaceFilter(Collection<Long> spaceIds) {
        // Filter context, never query context: space membership is an authorization boundary, so it
        // must not contribute to the score, and OpenSearch caches it as a bitset.
        List<org.opensearch.client.opensearch._types.FieldValue> values = spaceIds.stream()
                .map(org.opensearch.client.opensearch._types.FieldValue::of)
                .toList();
        return Query.of(q -> q.terms(t -> t.field("spaceId").terms(v -> v.value(values))));
    }

    private static Query substringQuery(String query) {
        return Query.of(q -> q.wildcard(w -> w
                .field("substringSource")
                .value("*" + escapeWildcard(query.toLowerCase()) + "*")));
    }

    /**
     * {@code *} and {@code ?} are wildcard metacharacters. A user searching for a literal
     * "sprint?" would otherwise get single-character-wildcard semantics — a silently wrong result
     * set rather than an error.
     */
    private static String escapeWildcard(String value) {
        return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?");
    }

    /**
     * Rescales a tier's scores into (0, 1] so SearchService's tier bands stay disjoint. Ordering
     * within the tier is untouched — this is a monotonic transform of OpenSearch's own ranking.
     */
    private List<IssueSearchHit> normalizeIssueRanks(SearchResponse<IssueSearchDocument> response,
                                                     String query,
                                                     SnippetBuilder<IssueSearchDocument> snippetBuilder) {
        double top = topScore(response.hits().hits());
        List<IssueSearchHit> hits = new ArrayList<>();
        for (Hit<IssueSearchDocument> hit : response.hits().hits()) {
            IssueSearchDocument doc = hit.source();
            if (doc == null) {
                continue;
            }
            hits.add(new IssueSearchHit(
                    doc.issueId(), doc.issueKey(), doc.spaceId(), doc.title(),
                    snippetBuilder.build(doc, hit),
                    score(hit) / top,
                    matchedFieldsFrom(hit.highlight().keySet()),
                    issueExactMatch(doc, query),
                    issueExactSnippet(doc, query),
                    doc.updatedAt()));
        }
        return hits;
    }

    private List<CommentSearchHit> normalizeCommentRanks(SearchResponse<CommentSearchDocument> response,
                                                         String query,
                                                         SnippetBuilder<CommentSearchDocument> snippetBuilder) {
        double top = topScore(response.hits().hits());
        List<CommentSearchHit> hits = new ArrayList<>();
        for (Hit<CommentSearchDocument> hit : response.hits().hits()) {
            CommentSearchDocument doc = hit.source();
            if (doc == null) {
                continue;
            }
            hits.add(new CommentSearchHit(
                    doc.commentId(), doc.issueId(), doc.issueKey(), doc.spaceId(), doc.issueTitle(),
                    snippetBuilder.build(doc, hit),
                    score(hit) / top,
                    List.of("comment"),
                    LiteralSnippet.contains(doc.content(), query),
                    LiteralSnippet.of(doc.content(), query),
                    doc.updatedAt()));
        }
        return hits;
    }

    private static <T> double topScore(List<Hit<T>> hits) {
        double top = hits.isEmpty() ? 1.0 : score(hits.get(0));
        // A sort on a non-score field leaves _score null, and a tier where everything ties at zero
        // would otherwise divide by zero and rank every row NaN.
        return top > 0 ? top : 1.0;
    }

    private static <T> double score(Hit<T> hit) {
        return hit.score() == null ? 0.0 : hit.score();
    }

    /** Maps OpenSearch field names onto the names the API contract uses. */
    private static List<String> matchedFieldsFrom(Collection<String> highlightedFields) {
        Map<String, String> naming = new LinkedHashMap<>();
        naming.put("title", "title");
        naming.put("description", "description");
        naming.put("labelNames", "labels");
        naming.put("attachmentNames", "attachment");
        return naming.entrySet().stream()
                .filter(e -> highlightedFields.contains(e.getKey()))
                .map(Map.Entry::getValue)
                .toList();
    }

    private static String firstHighlight(Map<String, List<String>> highlight, String... preference) {
        for (String field : preference) {
            List<String> fragments = highlight.get(field);
            if (fragments != null && !fragments.isEmpty()) {
                return fragments.get(0);
            }
        }
        return null;
    }

    private static List<String> issueMatchedFieldsLiteral(IssueSearchDocument doc, String query) {
        List<String> matched = new ArrayList<>();
        if (LiteralSnippet.contains(doc.issueKey(), query)) {
            matched.add("key");
        }
        if (LiteralSnippet.contains(doc.title(), query)) {
            matched.add("title");
        }
        if (LiteralSnippet.contains(doc.description(), query)) {
            matched.add("description");
        }
        if (LiteralSnippet.contains(doc.labelNames(), query)) {
            matched.add("labels");
        }
        if (LiteralSnippet.contains(doc.attachmentNames(), query)) {
            matched.add("attachment");
        }
        return matched;
    }

    private static boolean issueExactMatch(IssueSearchDocument doc, String query) {
        return !issueMatchedFieldsLiteral(doc, query).isEmpty();
    }

    /** Field priority carried over from the retired SQL: description, title, labels, filenames, key. */
    private static String issueExactSnippet(IssueSearchDocument doc, String query) {
        for (String body : List.of(
                nullToEmpty(doc.description()), nullToEmpty(doc.title()),
                nullToEmpty(doc.labelNames()), nullToEmpty(doc.attachmentNames()),
                nullToEmpty(doc.issueKey()))) {
            String snippet = LiteralSnippet.of(body, query);
            if (snippet != null) {
                return snippet;
            }
        }
        return null;
    }

    /**
     * The old SQL tightness ratio: {@code length(q) / max(length(title), length(labels or key), 1)}, so
     * a query that is most of the field ranks higher. (A wildcard query scores every match the same.)
     * Labels are "" (not null) when empty, so the fallback to the key checks for blank.
     */
    private static double substringRank(String query, IssueSearchDocument doc) {
        String secondary = isBlank(doc.labelNames()) ? doc.issueKey() : doc.labelNames();
        int denominator = Math.max(Math.max(length(doc.title()), length(secondary)), 1);
        return (double) query.length() / denominator;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static int length(String value) {
        return value == null ? 0 : value.length();
    }

    private static String truncate(String body) {
        if (body == null) {
            return "";
        }
        return body.length() <= PLAIN_SNIPPET_CHARS ? body : body.substring(0, PLAIN_SNIPPET_CHARS);
    }

    private static String firstNonNull(String a, String b) {
        return a != null ? a : b;
    }

    private static String nullToEmpty(String value) {
        return value == null ? "" : value;
    }

    private SearchResponse<IssueSearchDocument> execute(SearchRequest request) {
        try {
            return client.search(request, IssueSearchDocument.class);
        } catch (IOException e) {
            throw new SearchUnavailableException(e);
        }
    }

    private SearchResponse<CommentSearchDocument> executeComments(SearchRequest request) {
        try {
            return client.search(request, CommentSearchDocument.class);
        } catch (IOException e) {
            throw new SearchUnavailableException(e);
        }
    }

    @FunctionalInterface
    private interface SnippetBuilder<T> {
        String build(T document, Hit<T> hit);
    }
}
