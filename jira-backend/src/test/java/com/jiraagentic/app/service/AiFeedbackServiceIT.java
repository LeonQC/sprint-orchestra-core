package com.jiraagentic.app.service;

import com.jiraagentic.app.entity.Space;
import com.jiraagentic.app.repository.SpaceMemberRepository;
import com.jiraagentic.app.repository.SpaceRepository;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;
import org.springframework.web.server.ResponseStatusException;

import java.sql.DriverManager;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import static com.jiraagentic.app.service.FeedbackReceiptVerifierTest.*;
import static org.assertj.core.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Uses an isolated temporary schema; never migrates or resets the application's tables. */
@EnabledIfEnvironmentVariable(named = "RUN_FEEDBACK_DB_TESTS", matches = "true")
class AiFeedbackServiceIT {
    private static final String URL = System.getenv().getOrDefault("FEEDBACK_IT_JDBC_URL", "jdbc:postgresql://localhost:5432/pocdb");
    private static final String USER = System.getenv().getOrDefault("FEEDBACK_IT_DB_USER", "poc");
    private static final String PASSWORD = System.getenv().getOrDefault("FEEDBACK_IT_DB_PASSWORD", "poc123");
    private static final String SCHEMA = "feedback_it_" + UUID.randomUUID().toString().replace("-", "");
    private static NamedParameterJdbcTemplate jdbc;
    private AiFeedbackService service;
    private SpaceRepository spaces;
    private SpaceMemberRepository members;

    @BeforeAll static void setupDatabase() throws Exception {
        try (var connection = DriverManager.getConnection(URL, USER, PASSWORD); var statement = connection.createStatement()) {
            statement.execute("CREATE SCHEMA " + SCHEMA);
        }
        var datasource = new DriverManagerDataSource(URL + (URL.contains("?") ? "&" : "?") + "currentSchema=" + SCHEMA, USER, PASSWORD);
        jdbc = new NamedParameterJdbcTemplate(datasource);
        jdbc.getJdbcTemplate().execute("CREATE TABLE users(id BIGINT PRIMARY KEY,name TEXT NOT NULL)");
        jdbc.getJdbcTemplate().execute("INSERT INTO users VALUES(1,'Reporter'),(2,'Owner'),(3,'Other owner')");
        new ResourceDatabasePopulator(new ClassPathResource("db/migration/V31__ai_feedback.sql")).execute(datasource);
        new ResourceDatabasePopulator(new ClassPathResource("db/migration/V32__helpful_feedback_no_review.sql")).execute(datasource);
    }

    @AfterAll static void cleanup() throws Exception {
        try (var connection = DriverManager.getConnection(URL, USER, PASSWORD); var statement = connection.createStatement()) {
            statement.execute("DROP SCHEMA IF EXISTS " + SCHEMA + " CASCADE");
        }
    }

    @BeforeEach void setup() {
        jdbc.getJdbcTemplate().execute("DELETE FROM ai_feedback");
        spaces = mock(SpaceRepository.class); members = mock(SpaceMemberRepository.class);
        when(members.findActiveSpaceIdsByUserId(1L)).thenReturn(List.of(10L, 20L));
        when(spaces.findByOwnerIdAndDeletedAtIsNull(2L)).thenReturn(List.of(space(10), space(20)));
        when(spaces.findByOwnerIdAndDeletedAtIsNull(3L)).thenReturn(List.of(space(10)));
        service = new AiFeedbackService(jdbc, MAPPER, new FeedbackReceiptVerifier(MAPPER, SECRET), spaces, members);
    }

    private static Space space(long id) { Space space = new Space(); space.setId(id); return space; }
    private UUID submit(long... scopes) throws Exception {
        return submit("down", scopes);
    }

    private UUID submit(String rating, long... scopes) throws Exception {
        var result = service.submit(1, new AiFeedbackService.Submission(
                sign(snapshot(1, scopes)), rating, "incomplete", "Missing ATC-2"));
        return UUID.fromString((String) result.get("id"));
    }

    @Test void helpfulRatingsAffectAggregateSignalsWithoutEnteringTheIssueInbox() throws Exception {
        UUID id = submit("up", 10);
        Map<String, Object> result = service.list(2, "", "", 0);
        assertThat(result.get("total")).isEqualTo(0L);
        @SuppressWarnings("unchecked")
        Map<String, Long> signals = (Map<String, Long>) result.get("signals");
        assertThat(signals).containsEntry("total", 1L).containsEntry("helpful", 1L)
                .containsEntry("reported", 0L);
        assertThatThrownBy(() -> service.detail(2, id)).isInstanceOf(ResponseStatusException.class)
                .hasMessageContaining("404");
    }

    @Test void retriesCreateOnlyOneRecordAndDoNotOverwriteReviewedVotes() throws Exception {
        String receipt = sign(snapshot(1, 10));
        var request = new AiFeedbackService.Submission(receipt, "down", "incomplete", "Missing ATC-2");
        assertThat(service.submit(1, request)).isEqualTo(service.submit(1, request));
        assertThat(service.list(2, "", "", 0).get("total")).isEqualTo(1L);
    }

    @Test void allSourceSpacesRequiredForListingDetailAndReview() throws Exception {
        UUID id = submit(10, 20);
        assertThat(service.list(3, "", "", 0).get("total")).isEqualTo(0L);
        assertThatThrownBy(() -> service.detail(3, id)).isInstanceOf(ResponseStatusException.class);
        assertThatThrownBy(() -> service.review(3, id, new AiFeedbackService.Review("reviewing", "generation", "", "ATC-2", List.of(), 0))).isInstanceOf(ResponseStatusException.class);
        assertThat(service.detail(2, id).path("question").asText()).contains("blocked");
    }

    @Test void revokedMembershipAndDeletedOwnershipCannotAccessFeedback() throws Exception {
        String receipt = sign(snapshot(1, 20));
        when(members.findActiveSpaceIdsByUserId(1L)).thenReturn(List.of(10L));
        assertThatThrownBy(() -> service.submit(1, new AiFeedbackService.Submission(receipt, "down", "", ""))).isInstanceOf(ResponseStatusException.class);
        UUID id = submit(10);
        when(spaces.findByOwnerIdAndDeletedAtIsNull(2L)).thenReturn(List.of());
        assertThatThrownBy(() -> service.detail(2, id)).isInstanceOf(ResponseStatusException.class);
    }

    @Test void reviewConflictsAndRequirementsAreEnforced() throws Exception {
        UUID id = submit(10);
        assertThatThrownBy(() -> service.review(2, id, new AiFeedbackService.Review("resolved", "generation", "", "ATC-2", List.of(), 0))).isInstanceOf(ResponseStatusException.class);
        var review = new AiFeedbackService.Review("reviewing", "generation", "Verified the missing issue", "ATC-1 and ATC-2", List.of("atc-2"), 0);
        assertThat(service.review(2, id, review).path("revision").asInt()).isEqualTo(1);
        assertThatThrownBy(() -> service.review(2, id, review)).isInstanceOf(ResponseStatusException.class)
                .hasMessageContaining("409");
        assertThat(service.detail(2, id).path("requiredIssueKeys").get(0).asText()).isEqualTo("ATC-2");
    }

    @Test void paginationSearchAndCountsOperateOnAuthorizedRows() throws Exception {
        for (int i = 0; i < 27; i++) submit(10);
        submit(10, 20);
        Map<String, Object> first = service.list(3, "new", "Reporter", 0);
        assertThat(first.get("total")).isEqualTo(27L);
        @SuppressWarnings("unchecked")
        Map<String, Long> signals = (Map<String, Long>) first.get("signals");
        assertThat(signals).containsEntry("reported", 27L).containsEntry("helpful", 0L);
        assertThat((List<?>) first.get("items")).hasSize(25);
        assertThat((List<?>) service.list(3, "new", "Reporter", 1).get("items")).hasSize(2);
        assertThat(service.list(3, "", "no match", 0).get("total")).isEqualTo(0L);
    }
}
