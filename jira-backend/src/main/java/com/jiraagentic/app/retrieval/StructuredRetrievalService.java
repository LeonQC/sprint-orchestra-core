package com.jiraagentic.app.retrieval;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import javax.sql.DataSource;
import java.sql.Timestamp;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Authoritative, space-scoped read primitives for the AI tools. No replicated business tables. */
@Service
@Transactional(readOnly = true, isolation = Isolation.REPEATABLE_READ, timeout = 15)
public class StructuredRetrievalService {
    // Aliases preserve the existing AI wire contract. Joins are all many-to-one: an issue is one row,
    // even without text, a sprint, a parent or an assignee. Archived spaces are never returned.
    private static final String ISSUES = """
            SELECT i.id AS issue_id, i.issue_key, i.space_id, i.issue_type, i.status, i.priority,
                   i.sprint_id, s.name AS sprint_name, i.title, p.issue_key AS parent_key,
                   i.assignee_id, a.name AS assignee_name, i.story_points, i.created_at, i.updated_at
            FROM issues i
            JOIN spaces sp ON sp.id = i.space_id AND sp.deleted_at IS NULL
            LEFT JOIN sprints s ON s.id = i.sprint_id AND s.space_id = i.space_id
            LEFT JOIN issues p ON p.id = i.parent_id AND p.space_id = i.space_id
            LEFT JOIN users a ON a.id = i.assignee_id
            """;
    private static final String SPRINTS = """
            SELECT s.id AS sprint_id, s.name AS sprint_name, s.space_id, s.goal,
                   s.start_date, s.end_date, s.status, s.initial_committed_points,
                   s.initial_completed_points, s.final_scope_points, s.completed_points,
                   s.initial_issue_count, s.completed_issue_count, s.final_issue_count,
                   s.unestimated_issue_count
            FROM sprints s JOIN spaces sp ON sp.id = s.space_id AND sp.deleted_at IS NULL
            """;
    private static final String HISTORY = """
            SELECT h.id, i.issue_id, i.issue_key, i.space_id, h.event_type, h.field_name,
                   h.from_value, h.to_value, h.description, a.name AS actor_name,
                   h.created_at AS changed_at
            FROM issue_history h
            JOIN (SELECT id AS issue_id, issue_key, space_id FROM issues) i ON i.issue_id = h.issue_id
            JOIN spaces sp ON sp.id = i.space_id AND sp.deleted_at IS NULL
            LEFT JOIN users a ON a.id = h.actor_id
            """;

    private final NamedParameterJdbcTemplate jdbc;

    public StructuredRetrievalService(DataSource dataSource) {
        this.jdbc = new NamedParameterJdbcTemplate(dataSource);
        this.jdbc.getJdbcTemplate().setQueryTimeout(10);
    }

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record IssueQuery(List<Long> spaceIds, List<String> issueKeys, List<String> parentKeys,
                             List<String> issueTypes, List<String> statuses, List<String> priorities,
                             List<Long> sprintIds, List<String> sprintNames, OffsetDateTime createdAfter,
                             OffsetDateTime createdBefore, OffsetDateTime updatedAfter,
                             OffsetDateTime updatedBefore, String orderBy, String order, Integer limit) {}

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record HistoryQuery(List<Long> spaceIds, List<String> issueKeys, List<String> fields,
                               List<String> eventTypes, OffsetDateTime since, OffsetDateTime until,
                               boolean reopenedOnly, Integer limit) {}

    @JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
    public record SprintQuery(List<Long> spaceIds, List<String> statuses, String orderBy,
                              String order, Integer limit) {}

    public Map<String, Object> issues(IssueQuery request) {
        Filters filters = new Filters(request.spaceIds());
        filters.in("issue_key", request.issueKeys());
        // Children of these issues, so an epic -> child -> subtask tree is two lookups, not a scan of
        // every issue.
        filters.in("parent_key", request.parentKeys());
        filters.in("issue_type", request.issueTypes());
        filters.in("status", request.statuses());
        filters.in("priority", request.priorities());
        filters.in("sprint_id", request.sprintIds());
        filters.in("sprint_name", request.sprintNames());
        filters.date("created_at", ">=", request.createdAfter());
        filters.date("created_at", "<", request.createdBefore());
        filters.date("updated_at", ">=", request.updatedAfter());
        filters.date("updated_at", "<", request.updatedBefore());
        String selectPrefix = "WITH records AS (" + ISSUES + ") SELECT ";
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("total_count", count(selectPrefix, filters));
        result.put("counts_by_type", breakdown(selectPrefix, filters, "issue_type", "unknown"));
        result.put("counts_by_status", breakdown(selectPrefix, filters, "status", "unknown"));
        result.put("counts_by_assignee", breakdown(selectPrefix, filters, "assignee_name", "(unassigned)"));
        result.put("issues", rows(selectPrefix, filters, orderBy(request.orderBy(), "updated_at",
                Set.of("created_at", "updated_at")), request.order(), "issue_id", request.limit()));
        return result;
    }

    public Map<String, Object> history(HistoryQuery request) {
        Filters filters = new Filters(request.spaceIds());
        filters.in("issue_key", request.issueKeys());
        filters.in("field_name", request.fields());
        filters.in("event_type", request.eventTypes());
        filters.date("changed_at", ">=", request.since());
        filters.date("changed_at", "<", request.until());
        if (request.reopenedOnly()) {
            filters.clauses.add("field_name = 'status' AND from_value = 'done' AND to_value IS DISTINCT FROM 'done'");
        }
        String selectPrefix = "WITH records AS (" + HISTORY + ") SELECT ";
        long total = count(selectPrefix, filters);
        List<Map<String, Object>> changes = rows(selectPrefix, filters, "changed_at", "desc", "id", request.limit());
        // The public response never exposed database ids or the permission scope.
        for (Map<String, Object> change : changes) {
            change.keySet().removeAll(Set.of("id", "issue_id", "space_id"));
        }
        return Map.of("total_count", total, "changes", changes);
    }

    public Map<String, Object> sprints(SprintQuery request) {
        Filters filters = new Filters(request.spaceIds());
        filters.in("status", request.statuses());
        String selectPrefix = "WITH records AS (" + SPRINTS + ") SELECT ";
        return Map.of("total_count", count(selectPrefix, filters),
                "counts_by_status", breakdown(selectPrefix, filters, "status", "unknown"),
                "sprints", rows(selectPrefix, filters, orderBy(request.orderBy(), "start_date",
                        Set.of("start_date", "end_date")), request.order(), "sprint_id", request.limit()));
    }

    private long count(String selectPrefix, Filters filters) {
        Long count = jdbc.queryForObject(selectPrefix + "count(*) FROM records" + filters.where(), filters.params, Long.class);
        return count == null ? 0 : count;
    }

    private Map<String, Long> breakdown(String selectPrefix, Filters filters, String column, String missing) {
        Map<String, Long> result = new LinkedHashMap<>();
        for (Map<String, Object> row : jdbc.queryForList(selectPrefix + column + ", count(*) AS n FROM records"
                + filters.where() + " GROUP BY " + column, filters.params)) {
            Object groupValue = row.get(column);
            String groupName;
            if (groupValue == null || groupValue.toString().isEmpty()) {
                groupName = missing;
            } else {
                groupName = groupValue.toString();
            }
            result.merge(groupName, ((Number) row.get("n")).longValue(), Long::sum);
        }
        return result;
    }

    private List<Map<String, Object>> rows(String selectPrefix, Filters filters, String orderBy, String order,
                                           String idColumn, Integer requestedLimit) {
        int limit = requestedLimit == null ? 20 : requestedLimit;
        if (limit < 1 || limit > 200) {
            throw badRequest("limit must be between 1 and 200");
        }
        String direction = order == null ? "DESC" : switch (order) {
            case "asc" -> "ASC";
            case "desc" -> "DESC";
            default -> throw badRequest("order must be asc or desc");
        };
        filters.params.addValue("limit", limit);
        String query = selectPrefix + "* FROM records" + filters.where()
                + " ORDER BY " + orderBy + " " + direction + " NULLS LAST, " + idColumn + " "
                + direction + " LIMIT :limit";
        List<Map<String, Object>> rows = jdbc.queryForList(query, filters.params);
        // JDBC timestamps otherwise serialize as epoch milliseconds, violating the ISO-8601 contract.
        for (Map<String, Object> row : rows) {
            for (Map.Entry<String, Object> entry : row.entrySet()) {
                Object value = entry.getValue();
                if (value instanceof Timestamp timestamp) {
                    entry.setValue(timestamp.toInstant());
                } else if (value instanceof java.sql.Date date) {
                    entry.setValue(date.toLocalDate());
                }
            }
        }
        return rows;
    }

    private static String orderBy(String value, String fallback, Set<String> allowed) {
        if (value == null) {
            return fallback;
        }
        if (!allowed.contains(value)) {
            throw badRequest("unsupported order_by");
        }
        return value;
    }

    private static void validateSpaces(List<Long> spaces) {
        if (spaces == null || spaces.isEmpty()) {
            throw badRequest("space_ids must contain authorized positive space ids");
        }
        for (Long spaceId : spaces) {
            if (spaceId == null || spaceId <= 0) {
                throw badRequest("space_ids must contain authorized positive space ids");
            }
        }
    }

    private static ResponseStatusException badRequest(String message) {
        return new ResponseStatusException(HttpStatus.BAD_REQUEST, message);
    }

    private static final class Filters {
        final List<String> clauses = new ArrayList<>();
        final MapSqlParameterSource params = new MapSqlParameterSource();

        Filters(List<Long> spaces) {
            validateSpaces(spaces);
            params.addValue("spaces", spaces);
            clauses.add("space_id IN (:spaces)");
        }

        void in(String column, List<?> values) {
            if (values != null && !values.isEmpty()) {
                String key = "p" + params.getValues().size();
                params.addValue(key, values);
                clauses.add(column + " IN (:" + key + ")");
            }
        }

        void date(String column, String operator, OffsetDateTime value) {
            if (value != null) {
                String key = "p" + params.getValues().size();
                params.addValue(key, Timestamp.from(value.toInstant()));
                clauses.add(column + " " + operator + " :" + key);
            }
        }

        String where() {
            return " WHERE " + String.join(" AND ", clauses);
        }
    }
}
