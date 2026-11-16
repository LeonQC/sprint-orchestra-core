package com.jiraagentic.app.search;

import org.opensearch.client.opensearch.OpenSearchClient;
import com.jiraagentic.app.dto.CreateCommentRequest;
import com.jiraagentic.app.dto.SearchResultDto;
import com.jiraagentic.app.dto.UpdateIssueRequest;
import com.jiraagentic.app.service.CommentService;
import com.jiraagentic.app.service.IssueService;
import com.jiraagentic.app.service.SearchService;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * End-to-end test of the global search, against a real Postgres and a real OpenSearch.
 *
 * <p>Nothing here is mocked, because almost nothing here is our code any more. Stemming, BM25
 * ranking, HTML stripping, separator normalization, fuzziness and highlighting are all OpenSearch
 * behaviors; a fake client would assert that we build the request we think we build, which was never
 * the part in doubt. The parts that ARE ours — the tier bands, deduplication, matchedFields,
 * exactMatch/exactSnippet — only mean anything on top of real engine output.
 *
 * <p>Infrastructure comes from the project's own {@code docker compose up}: {@code orchestra-postgres}
 * (a throwaway {@code searchit_test} database created here and dropped afterwards) and
 * {@code orchestra-opensearch} (throwaway {@code searchit-test-*} indices, likewise). Not
 * Testcontainers — its bundled docker-java cannot negotiate an API version with the Docker Engine 29
 * daemon this project targets, and the rest of the suite already assumes localhost infra is up.
 * Point {@code SEARCH_IT_JDBC_ADMIN} / {@code SEARCH_IT_ES_URI} elsewhere to run it against other
 * instances.
 *
 * <p>Seeding goes in through JDBC and is then published to OpenSearch with one
 * {@link SearchReindexService#reindexAll()} call, so every test starts from an index that provably
 * matches the database — and the reindex path, which is the repair mechanism for the one consistency
 * gap this design accepts, is exercised on every run rather than only in an incident.
 */
@SpringBootTest(properties = {
        // Kafka/Redis beans still load (orchestra-kafka / orchestra-redis are up alongside
        // orchestra-postgres) — only the DB and the OpenSearch indices are redirected to
        // throwaway ones. Just keep Kafka consumers quiet.
        // Default (MOCK) web environment: SecurityConfig needs an HttpSecurity bean, which only a
        // web context provides — same reason JiraAgenticAiApplicationTests keeps the web layer.
        "spring.kafka.listener.auto-startup=false",
        // Never the real index names: a test run must not wipe what a running dev stack is serving.
        "app.opensearch.index.issues=searchit-test-issues",
        "app.opensearch.index.comments=searchit-test-comments",
})
class SearchServiceIT {

    private static final String ADMIN_URL =
            System.getenv().getOrDefault("SEARCH_IT_JDBC_ADMIN", "jdbc:postgresql://localhost:5432/postgres");
    private static final String USER = System.getenv().getOrDefault("SEARCH_IT_DB_USER", "poc");
    private static final String PASSWORD = System.getenv().getOrDefault("SEARCH_IT_DB_PASSWORD", "poc123");
    private static final String TEST_DB = "searchit_test";
    private static final String ES_URI =
            System.getenv().getOrDefault("SEARCH_IT_ES_URI", "http://localhost:9200");

    @DynamicPropertySource
    static void infrastructure(DynamicPropertyRegistry registry) {
        registry.add("app.opensearch.uri", () -> ES_URI);
        registry.add("spring.datasource.url",
                () -> ADMIN_URL.replaceFirst("/[^/]+$", "/" + TEST_DB));
        registry.add("spring.datasource.username", () -> USER);
        registry.add("spring.datasource.password", () -> PASSWORD);
    }

    @BeforeAll
    static void createDatabase() throws Exception {
        runAdmin("DROP DATABASE IF EXISTS " + TEST_DB + " WITH (FORCE)");
        runAdmin("CREATE DATABASE " + TEST_DB);
    }

    /**
     * Teardown has to be static — the test lifecycle stays PER_METHOD so that Spring's context (and
     * with it Flyway) is created after {@link #createDatabase()}, not before it. So the indices are
     * dropped over plain HTTP rather than through the autowired client, which a static method cannot
     * reach. Leaving them behind would not break a later run (every seed drops them first) but it
     * litters a cluster a developer also uses for real data.
     */
    @AfterAll
    static void dropDatabaseAndIndices() throws Exception {
        runAdmin("DROP DATABASE IF EXISTS " + TEST_DB + " WITH (FORCE)");
        HttpClient.newHttpClient().send(
                HttpRequest.newBuilder(URI.create(
                                ES_URI + "/searchit-test-issues,searchit-test-comments?ignore_unavailable=true"))
                        .DELETE().build(),
                HttpResponse.BodyHandlers.discarding());
    }

    private static void runAdmin(String sql) throws Exception {
        try (Connection c = DriverManager.getConnection(ADMIN_URL, USER, PASSWORD);
             Statement s = c.createStatement()) {
            s.execute(sql);
        }
    }

    @Autowired
    SearchService searchService;
    @Autowired
    IssueService issueService;
    @Autowired
    CommentService commentService;
    @Autowired
    JdbcTemplate jdbc;
    @Autowired
    SearchReindexService reindexService;
    @Autowired
    OpenSearchClient opensearch;
    @Autowired
    SearchIndexNames indices;

    private static final long ALICE = 1L;      // member of both spaces
    private static final long BOB = 2L;        // member of ATLAS only
    private static final long ATLAS = 10L;
    private static final long OTHER = 11L;     // Alice is a member; Bob is NOT

    @BeforeEach
    void seed() {
        jdbc.update("DELETE FROM issue_attachments");
        jdbc.update("DELETE FROM issue_labels");
        jdbc.update("DELETE FROM comments");
        // Tests that mutate through IssueService (rather than seeding raw rows) record history, and
        // issue_history holds a foreign key to issues — so it has to be cleared before them.
        jdbc.update("DELETE FROM issue_history");
        jdbc.update("DELETE FROM sprint_issue_history");
        jdbc.update("DELETE FROM issues");
        jdbc.update("DELETE FROM labels");
        jdbc.update("DELETE FROM space_members");
        jdbc.update("DELETE FROM spaces");
        jdbc.update("DELETE FROM users");

        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO users(id, username, name, email, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                ALICE, "alice", "Alice", "alice@example.com", now, now);
        jdbc.update("INSERT INTO users(id, username, name, email, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                BOB, "bob", "Bob", "bob@example.com", now, now);
        space(ATLAS, "AtlasCart", "ATC");
        space(OTHER, "Other Space", "OTH");
        member(ATLAS, ALICE, "OWNER");
        member(ATLAS, BOB, "MEMBER");
        member(OTHER, ALICE, "OWNER");

        // id, key, title, description(HTML), issue_type, status
        issue(100, ATLAS, "ATC-77", "Cache the public product catalog safely",
                "<h3>Context</h3><p>Reduce repeated catalog reads for the public beta.</p>", "story", "in_progress");
        issue(101, ATLAS, "ATC-43", "Double-click checkout created duplicate orders",
                "<p>A slow checkout allowed two Place order clicks; two order records were created.</p>", "bug", "done");
        issue(102, ATLAS, "ATC-90", "Add webhook retry with exponential backoff",
                "<p>Delivery failures need bounded retries.</p>", "task", "planned");
        issue(103, ATLAS, "ATC-12", "Token authentication for the seller API",
                "<p>Seller endpoints must require a bearer token.</p>", "story", "done");
        issue(104, ATLAS, "ATC-401", "Storefront empty states", "<p>Show clear empty states.</p>", "story", "planned");
        // A hit that only the OTHER space should ever produce, for the isolation test.
        issue(200, OTHER, "OTH-1", "Secret checkout rework", "<p>internal</p>", "story", "planned");

        label(500, ATLAS, "catalog");
        label(501, ATLAS, "checkout");
        label(502, ATLAS, "infra");
        issueLabel(100, 500);
        issueLabel(101, 501);

        attachment(9000, 101, "metrics-wire-dlq-depth.csv");

        comment(7000, 100, ALICE, "<p>We should invalidate on <strong>seller update</strong> events.</p>");
        // Three comments on ONE issue, two of which only stem-match "payments". This is the shape
        // that broke the UI: several rows sharing an issue id, most of them non-exact.
        comment(7001, 101, ALICE, "<p>Operations approved the payment-state column.</p>");
        comment(7002, 101, BOB, "<p>Full shipping address and payment-provider details are missing.</p>");
        comment(7003, 101, ALICE, "<p>Load test ran with fixture payments against the sandbox.</p>");

        publishToOpenSearch();
    }

    /**
     * Rebuild the throwaway indices from what was just seeded through JDBC.
     *
     * <p>Deleting first rather than only re-indexing: the documents are keyed by entity id, so a
     * plain reindex would overwrite but never remove, and a row deleted by one test would keep
     * answering searches in the next. Dropping the index is also what forces the mapping to be
     * recreated from {@code opensearch/*.json}, so a broken analyzer definition fails here rather
     * than silently degrading relevance.
     */
    private void publishToOpenSearch() {
        try {
            opensearch.indices().delete(r -> r
                    .index(indices.getIssues(), indices.getComments())
                    .ignoreUnavailable(true));
            reindexService.reindexAll();
        } catch (Exception e) {
            throw new IllegalStateException(
                    "could not rebuild search indices — is orchestra-opensearch up at " + ES_URI + "?", e);
        }
    }

    @Test
    void stemmedWord_findsInflectedForms() {
        assertThat(keys(search("caching"))).contains("ATC-77");
        assertThat(keys(search("orders"))).contains("ATC-43");   // stored text says "order"
    }

    @Test
    void quotedPhrase_requiresAdjacency() {
        // "public product catalog" is a contiguous run in ATC-77's title -> a real phrase (FULL_TEXT) hit.
        assertThat(matchType(search("\"public product catalog\""), "ATC-77")).isEqualTo("FULL_TEXT");
        // Same three words, scrambled: they're all in the doc but not adjacent in this order, so the
        // phrase tier must NOT produce it (a looser tier — fuzzy — still might, and that's fine).
        assertThat(matchTypeOrNull(search("\"catalog product public\""), "ATC-77")).isNotEqualTo("FULL_TEXT");
    }

    @Test
    void substring_matchesMidToken_whichStemmingCannot() {
        assertThat(keys(search("heckout"))).contains("ATC-43");        // "checkout"
        assertThat(keys(search("atalog"))).contains("ATC-77");         // "catalog"
        assertThat(keys(search("authent"))).contains("ATC-12");        // "authentication"
        assertThat(matchType(search("heckout"), "ATC-43")).isEqualTo("SUBSTRING");
    }

    @Test
    void labelName_isSearchable() {
        assertThat(keys(search("catalog"))).contains("ATC-77");   // ATC-77 carries the `catalog` label
        assertThat(keys(search("infra"))).isEmpty();              // label exists but no issue carries it
    }

    @Test
    void attachmentFilename_isSearchable() {
        assertThat(keys(search("dlq"))).contains("ATC-43");       // metrics-wire-dlq-depth.csv, separator-normalized
    }

    @Test
    void issueKeyPrefix_isTierOne_andExactRanksFirst() {
        List<SearchResultDto> hits = search("ATC-4");
        assertThat(keys(hits)).contains("ATC-43", "ATC-401");
        assertThat(matchType(hits, "ATC-43")).isEqualTo("EXACT_KEY");
        assertThat(search("ATC-43").get(0).getIssueKey()).isEqualTo("ATC-43");
    }

    @Test
    void fuzzy_toleratesTypos_asAFallback() {
        assertThat(keys(search("webhok rety"))).contains("ATC-90");   // webhook retry
    }

    /**
     * Regression for the noise this tier used to produce. {@code multi_match} defaults to OR, so a
     * single fuzzy-matching term carried a whole row: "webhok checkout" returned ATC-90 ("webhok" is
     * one edit from webhook) <em>and</em> ATC-43 ("checkout"), neither of which is about both words.
     *
     * <p>Against the AtlasCart demo data the effect was much worse: "search recent orders" pulled 55
     * issues out of a space of about 80, "Reject repeated checkout requests" among them ("reject" is
     * two edits from "recent"). Tier banding kept every one of them below the real hits, which is
     * why this read as cosmetic — it was not. It filled the dropdown with rows nobody searched for,
     * and it made the UI's "Exact only" toggle announce 49 hidden near matches for a query with one
     * genuine answer.
     *
     * <p>The tier runs AND, matching the full-text tier above it, so typo tolerance is the only
     * thing separating the two.
     */
    @Test
    void fuzzy_requiresEveryTerm_soOneNearWordCannotCarryARow() {
        // Both terms land in ATC-90's title, typos and all — the tier still does its job.
        assertThat(keys(search("webhok rety"))).contains("ATC-90");
        // One term each from two different issues, and no issue about both: on OR this returned
        // ATC-90 and ATC-43.
        assertThat(keys(search("webhok checkout"))).isEmpty();
    }

    @Test
    void spaceIsolation_neverLeaksAnotherSpacesIssue() {
        assertThat(keys(searchAs(BOB, "checkout"))).contains("ATC-43").doesNotContain("OTH-1");
        assertThat(keys(searchAs(ALICE, "checkout"))).contains("ATC-43", "OTH-1"); // Alice IS in OTHER
    }

    @Test
    void snippet_isHtmlStripped() {
        // The comment is stored as "<p>We should invalidate on <strong>seller update</strong> ...</p>".
        // The highlighter adds its own <b>..</b> around matched terms (that's the highlight the UI renders),
        // but none of the SOURCE markup — <p>, <strong>, <h3> — must survive into the snippet.
        String snippet = search("seller update").stream()
                .filter(r -> "COMMENT".equals(r.getType())).findFirst().orElseThrow().getSnippet();
        assertThat(snippet).doesNotContain("<p>").doesNotContain("<strong>").doesNotContain("<h3>");
    }

    /**
     * Regression for the gap the OpenSearch migration opened and had to close explicitly.
     *
     * <p>Under Postgres FTS, labels were kept searchable by a trigger inside the same transaction, so
     * a label-only edit was correct no matter what any application event did. With search in
     * OpenSearch there is no trigger — and the existing {@code IssueContentChangedEvent} does NOT
     * fire on a label-only change, because its publish condition lists title, description, parent,
     * status, priority, sprint, assignee and story points, and not labels. Reusing it would have
     * meant adding a label left the issue unfindable by that label forever, with no error anywhere.
     * {@code IssueSearchIndexRequestedEvent} is published unconditionally for exactly this reason;
     * this test is what proves it.
     *
     * <p>It deliberately goes through {@code IssueService.update} rather than writing
     * {@code issue_labels} directly: writing the join table is how the old trigger-based test worked,
     * and it would now pass or fail for reasons that have nothing to do with the code path a user
     * actually takes.
     */
    @Test
    void labelOnlyEdit_reindexesTheIssue() {
        assertThat(keys(search("infra"))).isEmpty();

        UpdateIssueRequest addLabel = new UpdateIssueRequest();
        addLabel.setLabels(List.of("infra"));
        issueService.update(ATLAS, "ATC-401", addLabel, ALICE);
        awaitIndexed();

        assertThat(keys(search("infra"))).contains("ATC-401");

        UpdateIssueRequest clearLabels = new UpdateIssueRequest();
        clearLabels.setLabels(List.of());
        issueService.update(ATLAS, "ATC-401", clearLabels, ALICE);
        awaitIndexed();

        assertThat(keys(search("infra"))).isEmpty();
    }

    /**
     * Covers the listener path, not the reindex path. Everything else here is seeded through JDBC and
     * published in bulk, which never exercises {@code SearchIndexListener}: that runs AFTER_COMMIT in
     * a *new* read-only transaction, and has to reload the entity because the original one has
     * already closed. Getting that wrong throws a LazyInitializationException reaching
     * {@code comment.getIssue().getSpace()} — after the write has committed, so the comment is saved
     * and simply never becomes searchable, with only a log line to say so.
     */
    @Test
    void newComment_becomesSearchableWithoutAReindex() {
        assertThat(search("kafka rebalance")).isEmpty();

        CreateCommentRequest req = new CreateCommentRequest();
        req.setAuthorId(ALICE);
        req.setContent("<p>The consumer hit a <em>kafka rebalance</em> mid-batch.</p>");
        commentService.create(100L, req);
        awaitIndexed();

        SearchResultDto hit = search("kafka rebalance").stream()
                .filter(r -> "COMMENT".equals(r.getType())).findFirst().orElseThrow();
        assertThat(hit.getIssueKey()).isEqualTo("ATC-77");
        // The parent issue's title is denormalized onto the comment document at index time; if the
        // listener had failed to resolve the lazy Issue association this would be null.
        assertThat(hit.getTitle()).isEqualTo("Cache the public product catalog safely");
        assertThat(hit.getSnippet()).doesNotContain("<p>").doesNotContain("<em>");
    }

    // --- matchedFields: WHY a row matched --------------------------------------------------------

    @Test
    void matchedFields_pinpointTheFieldThatMatched() {
        // "webhook" is only in ATC-90's title.
        assertThat(fields(search("webhook"), "ATC-90")).containsExactly("title");
        // "bearer" is only in ATC-12's description.
        assertThat(fields(search("bearer"), "ATC-12")).containsExactly("description");
        // ATC-77 carries the `catalog` label AND says "catalog" in title+description — all reported.
        assertThat(fields(search("catalog"), "ATC-77")).contains("title", "description", "labels");
    }

    @Test
    void matchedFields_explainAnOtherwiseBafflingHit() {
        // "dlq" appears nowhere a user can see on the row — only inside the attachment filename.
        // Without this the hit looks like a bug; with it the row explains itself.
        assertThat(fields(search("dlq"), "ATC-43")).containsExactly("attachment");
    }

    @Test
    void matchedFields_areSetOnEveryTier() {
        assertThat(fields(search("ATC-43"), "ATC-43")).containsExactly("key");          // EXACT_KEY
        assertThat(fields(search("heckout"), "ATC-43")).contains("title");              // SUBSTRING
        assertThat(fields(search("webhok rety"), "ATC-90")).isNotEmpty();               // FUZZY
        SearchResultDto comment = search("seller update").stream()
                .filter(r -> "COMMENT".equals(r.getType())).findFirst().orElseThrow();
        assertThat(comment.getMatchedFields()).containsExactly("comment");
    }

    // --- exactMatch: literal vs stemmed/typo-corrected --------------------------------------------

    @Test
    void exactMatch_isTrueWhenTheLiteralTextIsPresent() {
        assertThat(exact(search("checkout"), "ATC-43")).isTrue();
        assertThat(exact(search("catalog"), "ATC-77")).isTrue();
    }

    @Test
    void exactMatch_isFalseForAStemmedOnlyHit() {
        // The stored text says "order"/"orders"; searching "ordering" stems to the same lexeme and
        // matches, but the literal string "ordering" appears nowhere — that's what the filter drops.
        assertThat(keys(search("ordering"))).contains("ATC-43");
        assertThat(exact(search("ordering"), "ATC-43")).isFalse();
    }

    @Test
    void exactMatch_isFalseForATypoCorrectedFuzzyHit() {
        assertThat(keys(search("webhok rety"))).contains("ATC-90");
        assertThat(exact(search("webhok rety"), "ATC-90")).isFalse();
    }

    @Test
    void exactMatch_isAlwaysTrueForKeyPrefixAndSubstringTiers() {
        assertThat(exact(search("ATC-43"), "ATC-43")).isTrue();     // EXACT_KEY
        assertThat(exact(search("heckout"), "ATC-43")).isTrue();    // SUBSTRING
    }

    // --- row identity: several comments on one issue are several rows -----------------------------

    /**
     * Regression for the reported bug: with "Exact only" ticked, non-exact comment rows stayed on
     * screen. The filter was correct; the rows were keyed by issue id, so the three comment hits on
     * one issue collided, and React never removed the duplicates. The DTO has to hand the client an
     * identity that is actually unique.
     */
    @Test
    void resultId_isUniquePerRow_evenWhenSeveralCommentsShareAnIssue() {
        List<SearchResultDto> comments = search("payment").stream()
                .filter(r -> "COMMENT".equals(r.getType())).toList();

        assertThat(comments).hasSizeGreaterThanOrEqualTo(3);
        assertThat(comments).extracting(SearchResultDto::getIssueId).containsOnly(101L);
        assertThat(comments).extracting(r -> r.getType() + "-" + r.getResultId())
                .doesNotHaveDuplicates();
        // ...and resultId is the COMMENT id, not the issue id, or it could not be unique.
        assertThat(comments).extracting(SearchResultDto::getResultId)
                .allSatisfy(id -> assertThat(id).isNotEqualTo(101L));
    }

    @Test
    void resultId_isTheIssueIdForAnIssueHit() {
        SearchResultDto hit = search("ATC-43").get(0);
        assertThat(hit.getType()).isEqualTo("ISSUE");
        assertThat(hit.getResultId()).isEqualTo(hit.getIssueId());
    }

    // --- exactSnippet: proving an exact hit to the eye --------------------------------------------

    /**
     * The highlighter marks the matched STEM, so a search for "payments" can bold "payment" — which
     * makes a genuine exact hit look like the filter is broken. exactSnippet is built around the
     * literal occurrence instead, and exists exactly when the hit is exact.
     */
    @Test
    void exactSnippet_highlightsTheLiteralText_notTheStem() {
        SearchResultDto literalHit = search("payments").stream()
                .filter(r -> "COMMENT".equals(r.getType()) && r.isExactMatch())
                .findFirst().orElseThrow();

        assertThat(literalHit.getExactSnippet()).contains("<b>payments</b>");
        // The default snippet is the stemmed one — that difference is the whole reason this exists.
        assertThat(literalHit.getSnippet()).contains("<b>payment");
    }

    @Test
    void exactSnippet_isNullExactlyWhenTheHitIsNotExact() {
        for (SearchResultDto hit : search("payments")) {
            if (hit.isExactMatch()) {
                assertThat(hit.getExactSnippet())
                        .as("exact hit %s must be able to show where the literal text is", hit.getIssueKey())
                        .isNotBlank();
            } else {
                assertThat(hit.getExactSnippet())
                        .as("non-exact hit %s has no literal occurrence to show", hit.getIssueKey())
                        .isNull();
            }
        }
    }

    @Test
    void exactSnippet_isSetOnEveryTier() {
        assertThat(search("ATC-43").get(0).getExactSnippet()).isNotBlank();          // EXACT_KEY
        assertThat(exactSnippet(search("heckout"), "ATC-43")).contains("<b>heckout</b>");  // SUBSTRING
        assertThat(exactSnippet(search("checkout"), "ATC-43")).contains("<b>checkout</b>"); // FULL_TEXT
    }

    // --- space scoping ---------------------------------------------------------------------------

    @Test
    void softDeletedSpace_dropsOutOfSearchEntirely() {
        assertThat(keys(searchAs(ALICE, "checkout"))).contains("OTH-1");
        jdbc.update("UPDATE spaces SET deleted_at = ? WHERE id = ?", Timestamp.from(Instant.now()), OTHER);
        // findActiveSpaceIdsByUserId filters `deletedAt IS NULL`, so the membership row alone is not enough.
        assertThat(keys(searchAs(ALICE, "checkout"))).contains("ATC-43").doesNotContain("OTH-1");
    }

    @Test
    void userWithNoMemberships_getsNothingRatherThanEverything() {
        long stranger = 99L;
        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO users(id, username, name, email, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                stranger, "stranger", "Stranger", "stranger@example.com", now, now);
        // An empty space list must short-circuit, never degrade to an unscoped query.
        assertThat(searchAs(stranger, "checkout")).isEmpty();
    }

    // --- space membership on the issue-list endpoint ----------------------------------------------
    // The quick-search dropdown has two halves: the FTS results above (scoped by
    // findActiveSpaceIdsByUserId) and a local quick-jump list fed by GET /api/spaces/{id}/issues.
    // That endpoint used to be guarded only by "is the space alive", never by membership.

    @Test
    @Transactional  // findBySpace maps lazy `labels`; a direct service call from a test has no open session
    void issueList_isReadableByAMemberOfTheSpace() {
        assertThat(issueService.findBySpace(ATLAS, BOB)).isNotEmpty();
    }

    @Test
    void issueList_isRefusedToANonMember() {
        // Bob is a member of ATLAS only; OTHER must be off limits.
        assertThatThrownBy(() -> issueService.findBySpace(OTHER, BOB))
                .isInstanceOf(ResponseStatusException.class)
                .hasMessageContaining("403");
    }

    @Test
    @Transactional  // see above — session needed to map the returned DTOs
    void issueList_stillServesTrustedInternalCallersWithNoUserIdentity() {
        // ai-service calls with the internal service token and carries no end-user id; a blanket
        // membership check would have broken every AI feature that reads a space's issues.
        assertThat(issueService.findBySpace(OTHER, null)).isNotEmpty();
    }

    // --- helpers ---------------------------------------------------------------------------------

    /**
     * OpenSearch is near-real-time: an indexed document is durable immediately but not searchable
     * until the next refresh, which by default is up to a second away. Production is happy with that;
     * a test that writes and immediately reads is not, so force the refresh rather than sleeping.
     */
    private void awaitIndexed() {
        try {
            opensearch.indices().refresh(r -> r.index(indices.getIssues(), indices.getComments()));
        } catch (Exception e) {
            throw new IllegalStateException("could not refresh search indices", e);
        }
    }

    private List<SearchResultDto> search(String q) {
        return searchService.search(ALICE, q, 20);
    }

    private List<SearchResultDto> searchAs(long userId, String q) {
        return searchService.search(userId, q, 20);
    }

    private static List<String> keys(List<SearchResultDto> hits) {
        return hits.stream().map(SearchResultDto::getIssueKey).toList();
    }

    private static String matchType(List<SearchResultDto> hits, String issueKey) {
        return hits.stream().filter(h -> issueKey.equals(h.getIssueKey())).findFirst().orElseThrow().getMatchType();
    }

    private static boolean exact(List<SearchResultDto> hits, String issueKey) {
        return hits.stream().filter(h -> issueKey.equals(h.getIssueKey()) && "ISSUE".equals(h.getType()))
                .findFirst().orElseThrow().isExactMatch();
    }

    private static String exactSnippet(List<SearchResultDto> hits, String issueKey) {
        return hits.stream().filter(h -> issueKey.equals(h.getIssueKey()) && "ISSUE".equals(h.getType()))
                .findFirst().orElseThrow().getExactSnippet();
    }

    private static List<String> fields(List<SearchResultDto> hits, String issueKey) {
        return hits.stream().filter(h -> issueKey.equals(h.getIssueKey()) && "ISSUE".equals(h.getType()))
                .findFirst().orElseThrow().getMatchedFields();
    }

    private static String matchTypeOrNull(List<SearchResultDto> hits, String issueKey) {
        return hits.stream().filter(h -> issueKey.equals(h.getIssueKey())).findFirst()
                .map(SearchResultDto::getMatchType).orElse(null);
    }

    private void space(long id, String name, String key) {
        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO spaces(id, name, space_key, issue_counter, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                id, name, key, 0, now, now);
    }

    private void member(long spaceId, long userId, String role) {
        jdbc.update("INSERT INTO space_members(space_id, user_id, role, joined_at) VALUES (?,?,?,?)",
                spaceId, userId, role, Timestamp.from(Instant.now()));
    }

    private void issue(long id, long spaceId, String key, String title, String descriptionHtml, String type, String status) {
        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO issues(id, issue_key, space_id, title, description, issue_type, status, created_at, updated_at) "
                        + "VALUES (?,?,?,?,?,?,?,?,?)",
                id, key, spaceId, title, descriptionHtml, type, status, now, now);
    }

    private void label(long id, long spaceId, String name) {
        jdbc.update("INSERT INTO labels(id, space_id, name, normalized_name, created_at) VALUES (?,?,?,?,?)",
                id, spaceId, name, name.toLowerCase(), Timestamp.from(Instant.now()));
    }

    private void issueLabel(long issueId, long labelId) {
        jdbc.update("INSERT INTO issue_labels(issue_id, label_id) VALUES (?,?)", issueId, labelId);
    }

    private void attachment(long id, long issueId, String filename) {
        jdbc.update("INSERT INTO issue_attachments(id, issue_id, original_filename, storage_filename, size_bytes, created_at) "
                        + "VALUES (?,?,?,?,?,?)",
                id, issueId, filename, "s-" + id, 1L, Timestamp.from(Instant.now()));
    }

    private void comment(long id, long issueId, long authorId, String contentHtml) {
        Timestamp now = Timestamp.from(Instant.now());
        jdbc.update("INSERT INTO comments(id, issue_id, author_id, content, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                id, issueId, authorId, contentHtml, now, now);
    }
}
