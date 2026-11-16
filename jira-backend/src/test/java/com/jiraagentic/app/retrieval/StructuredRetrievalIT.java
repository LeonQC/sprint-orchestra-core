package com.jiraagentic.app.retrieval;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jiraagentic.app.event.IssueContentChangedEvent;
import com.jiraagentic.app.kafka.IssueIngestionMessage;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import java.sql.DriverManager;
import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** Real source SQL, HTTP contracts and service authentication, in a disposable database only. */
@SpringBootTest(properties = {
        "spring.kafka.listener.auto-startup=false",
        "app.kafka.content-ingestion.enabled=false",
        "app.kafka.attachment-ingestion.enabled=false",
        "app.security.internal-service-token=retrieval-test-token",
        "app.security.jwt.private-key-path=${java.io.tmpdir}/retrieval-test-jwt-key.pem",
        "spring.jpa.show-sql=false"
})
@AutoConfigureMockMvc
class StructuredRetrievalIT {
    private static final String ADMIN = System.getenv().getOrDefault(
            "RETRIEVAL_IT_JDBC_ADMIN", "jdbc:postgresql://localhost:5432/postgres");
    private static final String USER = System.getenv().getOrDefault("RETRIEVAL_IT_DB_USER", "poc");
    private static final String PASSWORD = System.getenv().getOrDefault("RETRIEVAL_IT_DB_PASSWORD", "poc123");
    private static final String DATABASE = "retrievalit_" + UUID.randomUUID().toString().replace("-", "");
    private static boolean created;

    @DynamicPropertySource
    static void database(DynamicPropertyRegistry registry) throws Exception {
        try (var connection = DriverManager.getConnection(ADMIN, USER, PASSWORD);
             var statement = connection.createStatement()) {
            statement.execute("CREATE DATABASE " + DATABASE);
            created = true;
        }
        registry.add("spring.datasource.url", () -> ADMIN.replaceFirst("/[^/]+$", "/" + DATABASE));
        registry.add("spring.datasource.username", () -> USER);
        registry.add("spring.datasource.password", () -> PASSWORD);
    }

    @AfterAll
    static void cleanup() throws Exception {
        if (!created) return;
        try (var connection = DriverManager.getConnection(ADMIN, USER, PASSWORD);
             var statement = connection.createStatement()) {
            statement.execute("DROP DATABASE " + DATABASE + " WITH (FORCE)");
        }
    }

    @Autowired JdbcTemplate jdbc;
    @Autowired MockMvc mvc;
    @Autowired ObjectMapper mapper;
    @Autowired StructuredRetrievalService service;

    @BeforeEach
    void seed() {
        jdbc.execute("TRUNCATE spaces CASCADE");
        jdbc.update("""
                INSERT INTO users(id, username, name, email, created_at, updated_at)
                VALUES (10001, 'retrieval-owner', 'Alex', 'retrieval@example.test', now(), now())
                ON CONFLICT (id) DO UPDATE SET name = 'Alex'
                """);
        jdbc.update("""
                INSERT INTO spaces(id, name, space_key, issue_counter, created_at, updated_at, deleted_at)
                VALUES (1, 'Mine', 'MINE', 3, now(), now(), NULL),
                       (2, 'Other', 'OTHER', 1, now(), now(), NULL),
                       (3, 'Archived', 'ARCH', 1, now(), now(), now())
                """);
        jdbc.update("""
                INSERT INTO sprints(id, name, space_id, goal, start_date, end_date, status,
                                    completed_points, final_scope_points, created_at, updated_at)
                VALUES (10, 'Sprint 10', 1, 'Accessibility', '2026-01-01', '2026-01-15', 'active',
                        8, 13, now(), now()),
                       (11, 'Sprint 11', 1, NULL, NULL, NULL, 'future', NULL, NULL, now(), now()),
                       (20, 'Sprint 10', 2, 'Private', '2026-01-01', '2026-01-15', 'active',
                        99, 99, now(), now())
                """);
        jdbc.update("""
                INSERT INTO issues(id, issue_key, space_id, sprint_id, title, description, issue_type,
                                   status, priority, assignee_id, story_points, created_at, updated_at)
                VALUES (101, 'MINE-1', 1, 10, 'Checkout', 'Details', 'bug', 'done', 'high', 10001, 5,
                        '2026-01-01', '2026-01-02'),
                       (102, 'MINE-2', 1, 10, '', NULL, 'bug', 'done', 'high', NULL, NULL,
                        '2026-01-01', '2026-01-02'),
                       (103, 'MINE-3', 1, NULL, 'Parent', NULL, 'story', 'blocked', 'low', 10001, 8,
                        '2025-12-01', '2026-01-01'),
                       (201, 'OTHER-1', 2, 20, 'Private', NULL, 'bug', 'done', 'high', 10001, 99,
                        '2026-01-01', '2026-01-02'),
                       (301, 'ARCH-1', 3, NULL, 'Archived', NULL, 'bug', 'done', 'high', 10001, 99,
                        '2026-01-01', '2026-01-02')
                """);
        jdbc.update("UPDATE issues SET parent_id = 103 WHERE id = 101");
        jdbc.update("""
                INSERT INTO issue_history(id, issue_id, actor_id, event_type, field_name,
                                          from_value, to_value, description, created_at)
                VALUES (1, 101, 10001, 'field_change', 'status', 'done', 'in_progress', 'Reopened', '2026-01-02'),
                       (2, 101, NULL, 'field_change', 'status', 'done', 'done', 'Unchanged', '2026-01-03'),
                       (3, 102, NULL, 'field_change', 'status', 'done', NULL, 'Cleared', '2026-01-04'),
                       (4, 201, 10001, 'field_change', 'status', 'done', 'in_progress', 'Private', '2026-01-02')
                """);
    }

    @Test
    void exactCountsAndBreakdownsIncludeBlankIssuesBeyondTheRowLimit() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1],"issue_types":["bug"],"statuses":["done"],"priorities":["high"],
                         "sprint_ids":[10],"sprint_names":["Sprint 10"],"created_after":"2026-01-01T00:00:00Z",
                         "created_before":"2026-01-02T00:00:00Z","updated_after":"2026-01-02T00:00:00Z",
                         "updated_before":"2026-01-03T00:00:00Z","order":"asc","limit":1}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(2))
                .andExpect(jsonPath("$.counts_by_type.bug").value(2))
                .andExpect(jsonPath("$.counts_by_status.done").value(2))
                .andExpect(jsonPath("$.counts_by_assignee.Alex").value(1))
                .andExpect(jsonPath("$.counts_by_assignee['(unassigned)']").value(1))
                .andExpect(jsonPath("$.issues.length()").value(1))
                .andExpect(jsonPath("$.issues[0].issue_key").value("MINE-1"))
                .andExpect(jsonPath("$.issues[0].parent_key").value("MINE-3"))
                .andExpect(jsonPath("$.issues[0].story_points").value(5))
                .andExpect(jsonPath("$.issues[0].created_at").isString());
    }

    @Test
    void statusEditsAreVisibleImmediatelyWithoutKafkaOrEmbedding() throws Exception {
        jdbc.update("UPDATE issues SET status = 'in_progress' WHERE id = 101");
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1],"issue_keys":["MINE-1"],"statuses":["in_progress"]}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(1))
                .andExpect(jsonPath("$.issues[0].status").value("in_progress"));
    }

    @Test
    void parentKeysReturnOnlyThoseIssuesChildren() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1],\"parent_keys\":[\"MINE-3\"]}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(1))
                .andExpect(jsonPath("$.issues[0].issue_key").value("MINE-1"));
    }

    @Test
    void tiedTimestampsUseIssueIdToKeepPaginationStable() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1],\"limit\":2}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.issues[0].issue_key").value("MINE-2"))
                .andExpect(jsonPath("$.issues[1].issue_key").value("MINE-1"));
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1],\"order\":\"asc\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.issues[0].issue_key").value("MINE-3"))
                .andExpect(jsonPath("$.issues[1].issue_key").value("MINE-1"))
                .andExpect(jsonPath("$.issues[2].issue_key").value("MINE-2"));
    }

    @Test
    void blankAndNullAssigneeNamesShareOneFallbackCount() throws Exception {
        jdbc.update("UPDATE users SET name = '' WHERE id = 10001");
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1],\"issue_types\":[\"bug\"]}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.counts_by_assignee['(unassigned)']").value(2));
    }

    @Test
    void duplicateSpaceIdsDoNotDuplicateBusinessRows() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1,1]}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(3))
                .andExpect(jsonPath("$.issues.length()").value(3));
    }

    @Test
    void historyFiltersCountTransitionsAndKeepNullActorsAndDestinations() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/history").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1],"fields":["status"],"event_types":["field_change"],
                         "since":"2026-01-02T00:00:00Z","until":"2026-01-05T00:00:00Z",
                         "reopened_only":true,"limit":1}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(2))
                .andExpect(jsonPath("$.changes.length()").value(1))
                .andExpect(jsonPath("$.changes[0].issue_key").value("MINE-2"))
                .andExpect(jsonPath("$.changes[0].changed_at").isString());
        mvc.perform(post("/internal/retrieval/issues/history").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1],"issue_keys":["MINE-1"],"reopened_only":true}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(1))
                .andExpect(jsonPath("$.changes[0].actor_name").value("Alex"));
    }

    @Test
    void sprintMetricsAndNullDatesComeFromTheSource() throws Exception {
        mvc.perform(post("/internal/retrieval/sprints/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1],"statuses":["active"],"order_by":"end_date","order":"asc"}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(1))
                .andExpect(jsonPath("$.sprints[0].sprint_id").value(10))
                .andExpect(jsonPath("$.sprints[0].completed_points").value(8))
                .andExpect(jsonPath("$.sprints[0].start_date").value("2026-01-01"));
        mvc.perform(post("/internal/retrieval/sprints/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[1],\"limit\":1}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(2))
                .andExpect(jsonPath("$.counts_by_status.future").value(1))
                .andExpect(jsonPath("$.sprints.length()").value(1));
    }

    @Test
    void permissionsAndInputValidationCannotBeBypassed() throws Exception {
        for (String path : List.of("issues/query", "issues/history", "sprints/query")) {
            mvc.perform(post("/internal/retrieval/" + path).contentType("application/json")
                    .content("{\"space_ids\":[1]}")).andExpect(status().isUnauthorized());
            mvc.perform(post("/internal/retrieval/" + path).header("X-Internal-Token", "wrong")
                    .contentType("application/json").content("{\"space_ids\":[1]}"))
                    .andExpect(status().isUnauthorized());
        }
        for (String body : List.of("{}", "{\"space_ids\":[]}", "{\"space_ids\":[0]}",
                "{\"space_ids\":[null]}", "{\"space_ids\":[-1]}",
                "{\"space_ids\":[1],\"limit\":0}", "{\"space_ids\":[1],\"order\":\"ASC\"}",
                "{\"space_ids\":[1],\"order_by\":\"title; DROP TABLE issues\"}",
                "{\"space_ids\":[1],\"limit\":201}")) {
            mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                    .contentType("application/json").content(body)).andExpect(status().isBadRequest());
        }
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("""
                        {"space_ids":[1,3],"issue_keys":["OTHER-1","ARCH-1","' OR TRUE --"]}
                        """))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(0));
    }

    @Test
    void emptyAuthorizedScopeKeepsAllThreeResponseShapes() throws Exception {
        mvc.perform(post("/internal/retrieval/issues/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[999]}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(0))
                .andExpect(jsonPath("$.counts_by_type").isEmpty())
                .andExpect(jsonPath("$.counts_by_status").isEmpty())
                .andExpect(jsonPath("$.counts_by_assignee").isEmpty())
                .andExpect(jsonPath("$.issues").isEmpty());
        mvc.perform(post("/internal/retrieval/issues/history").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[999]}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(0))
                .andExpect(jsonPath("$.changes").isEmpty());
        mvc.perform(post("/internal/retrieval/sprints/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json").content("{\"space_ids\":[999]}"))
                .andExpect(status().isOk()).andExpect(jsonPath("$.total_count").value(0))
                .andExpect(jsonPath("$.counts_by_status").isEmpty())
                .andExpect(jsonPath("$.sprints").isEmpty());
    }

    @Test
    void nullableSprintDatesSurviveJdbcConversionAndNullsSortLast() throws Exception {
        mvc.perform(post("/internal/retrieval/sprints/query").header("X-Internal-Token", "retrieval-test-token")
                        .contentType("application/json")
                        .content("{\"space_ids\":[1],\"order_by\":\"start_date\",\"order\":\"asc\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sprints[0].sprint_id").value(10))
                .andExpect(jsonPath("$.sprints[0].start_date").value("2026-01-01"))
                .andExpect(jsonPath("$.sprints[1].sprint_id").value(11))
                .andExpect(jsonPath("$.sprints[1].start_date").value(org.hamcrest.Matchers.nullValue()))
                .andExpect(jsonPath("$.sprints[1].end_date").value(org.hamcrest.Matchers.nullValue()));
    }

    @Test
    void ingestionMessagesKeepSearchableParentContextWithoutBusinessMetadata() throws Exception {
        var event = new IssueContentChangedEvent(101, "MINE-1", 1, "Title", "Text",
                "MINE-3", "Parent title", Instant.now());
        var payload = mapper.readTree(mapper.writeValueAsString(IssueIngestionMessage.upserted(event, UUID.randomUUID())));
        assertThat(payload.get("parentKey").asText()).isEqualTo("MINE-3");
        assertThat(payload.has("status")).isFalse();
        assertThat(payload.has("sprintId")).isFalse();
    }
}
