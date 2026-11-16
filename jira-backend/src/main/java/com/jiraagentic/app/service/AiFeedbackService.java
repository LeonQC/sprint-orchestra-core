package com.jiraagentic.app.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.jiraagentic.app.entity.Space;
import com.jiraagentic.app.repository.SpaceMemberRepository;
import com.jiraagentic.app.repository.SpaceRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.namedparam.MapSqlParameterSource;
import org.springframework.jdbc.core.namedparam.NamedParameterJdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.*;

@Service
@RequiredArgsConstructor
public class AiFeedbackService {
    private final NamedParameterJdbcTemplate jdbc;
    private final ObjectMapper mapper;
    private final FeedbackReceiptVerifier receipts;
    private final SpaceRepository spaces;
    private final SpaceMemberRepository members;

    public record Submission(String receipt, String rating, String reason, String comment) {}
    public record Review(String status, String rootCause, String reviewNote, String expectedAnswer,
                         List<String> requiredIssueKeys, Integer revision) {}

    private static final Set<String> STATUSES = Set.of("new", "reviewing", "resolved", "dismissed");
    private static final Set<String> REASONS = Set.of("", "incorrect", "incomplete", "sources", "irrelevant", "style", "other");
    private static final Set<String> CAUSES = Set.of("", "retrieval", "routing", "freshness", "generation", "citation", "ux", "not_a_bug", "other");
    private static final String FROM = " FROM ai_feedback f JOIN users u ON u.id=f.reporter_id ";
    // ALL source spaces must be owned by the reviewer; an overlap would leak cross-space evidence.
    private static final String SCOPE = " (CAST(:spaces AS jsonb) @> (f.snapshot->'request'->'spaceIds')) ";

    @Transactional
    public Map<String, Object> submit(long userId, Submission body) {
        require(body != null, "Feedback is required");
        String rating = choice(body.rating(), Set.of("up", "down"), "rating");
        String reason = rating.equals("down") ? choice(body.reason(), REASONS, "reason") : "";
        String comment = rating.equals("down") ? bounded(body.comment(), 4000, "Comment") : "";
        JsonNode snapshot = receipts.verify(body.receipt(), userId);
        Set<Long> allowed = new HashSet<>(members.findActiveSpaceIdsByUserId(userId));
        spaces.findByOwnerIdAndDeletedAtIsNull(userId).forEach(s -> allowed.add(s.getId()));
        for (JsonNode scope : snapshot.path("request").path("spaceIds")) {
            if (!allowed.contains(scope.asLong())) throw new ResponseStatusException(HttpStatus.FORBIDDEN, "You no longer have access to this answer's spaces");
        }
        UUID id = UUID.randomUUID();
        var params = new MapSqlParameterSource().addValue("id", id)
                .addValue("answer", UUID.fromString(snapshot.path("answerId").asText()))
                .addValue("user", userId).addValue("rating", rating).addValue("reason", reason)
                .addValue("comment", comment).addValue("snapshot", snapshot.toString())
                .addValue("status", rating.equals("up") ? "resolved" : "new");
        jdbc.update("""
                INSERT INTO ai_feedback(id,answer_id,reporter_id,rating,reason,comment,snapshot,status)
                VALUES(:id,:answer,:user,:rating,:reason,:comment,CAST(:snapshot AS jsonb),:status)
                ON CONFLICT(answer_id,reporter_id) DO NOTHING
                """, params);
        return jdbc.queryForObject("SELECT id FROM ai_feedback WHERE answer_id=:answer AND reporter_id=:user",
                params, (rs, row) -> Map.of("id", rs.getString("id")));
    }

    public Map<String, Object> list(long userId, String status, String query, int page) {
        require(page >= 0 && page <= 10000, "Invalid page");
        var params = reviewerScope(userId);
        String where = " WHERE " + SCOPE + " AND f.rating='down'";
        if (status != null && !status.isBlank()) {
            params.addValue("status", choice(status, STATUSES, "status"));
            where += " AND f.status=:status";
        }
        String search = bounded(query, 200, "Search");
        if (!search.isBlank()) {
            params.addValue("query", search.toLowerCase(Locale.ROOT));
            where += " AND position(:query IN lower(concat_ws(' ', f.snapshot->'request'->>'question', f.comment, u.name))) > 0";
        }
        Long total = jdbc.queryForObject("SELECT count(*)" + FROM + where, params, Long.class);
        params.addValue("offset", page * 25);
        List<ObjectNode> items = jdbc.query("SELECT f.*,u.name AS reporter_name" + FROM + where
                + " ORDER BY f.created_at DESC,f.id LIMIT 25 OFFSET :offset", params, rowMapper(false));
        Map<String, Long> counts = new LinkedHashMap<>();
        for (String state : List.of("new", "reviewing", "resolved", "dismissed")) counts.put(state, 0L);
        jdbc.query("SELECT f.status,count(*) AS n" + FROM + " WHERE " + SCOPE
                        + " AND f.rating='down' GROUP BY f.status",
                params, (org.springframework.jdbc.core.RowCallbackHandler) rs -> counts.put(rs.getString("status"), rs.getLong("n")));
        Map<String, Long> signals = jdbc.queryForObject("""
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE f.rating='up') AS helpful,
                       count(*) FILTER (WHERE f.rating='down') AS reported
                """ + FROM + " WHERE " + SCOPE, params, (rs, row) -> Map.of(
                "total", rs.getLong("total"),
                "helpful", rs.getLong("helpful"),
                "reported", rs.getLong("reported")));
        return Map.of("items", items, "total", total == null ? 0 : total, "page", page,
                "pageSize", 25, "counts", counts, "signals", signals);
    }

    public ObjectNode detail(long userId, UUID id) {
        var params = reviewerScope(userId).addValue("id", id);
        var rows = jdbc.query("SELECT f.*,u.name AS reporter_name" + FROM
                + " WHERE " + SCOPE + " AND f.rating='down' AND f.id=:id", params, rowMapper(true));
        if (rows.isEmpty()) throw new ResponseStatusException(HttpStatus.NOT_FOUND, "Feedback not found");
        return rows.get(0);
    }

    @Transactional
    public ObjectNode review(long userId, UUID id, Review body) {
        require(body != null, "Review is required");
        ObjectNode feedback = detail(userId, id);
        String status = choice(body.status(), STATUSES, "status");
        String cause = choice(body.rootCause(), CAUSES, "root cause");
        String note = bounded(body.reviewNote(), 8000, "Review note");
        String expected = bounded(body.expectedAnswer(), 12000, "Expected answer");
        if (Set.of("resolved", "dismissed").contains(status)) require(!note.isBlank(), "Add a review note before closing feedback");
        if (status.equals("resolved")) require(!cause.isBlank(), "Choose a root cause before resolving feedback");
        require(body.revision() != null && body.revision() >= 0, "A review revision is required");
        List<String> keys = body.requiredIssueKeys() == null ? List.of() : body.requiredIssueKeys();
        require(keys.size() <= 100, "Use at most 100 required issue keys");
        List<String> normalized = new ArrayList<>();
        for (String key : keys) {
            String value = bounded(key, 50, "Issue key").toUpperCase(Locale.ROOT);
            require(value.matches("[A-Z][A-Z0-9_]*-[0-9]+"), "Invalid issue key: " + value);
            if (!normalized.contains(value)) normalized.add(value);
        }
        var params = reviewerScope(userId).addValue("id", id).addValue("status", status)
                .addValue("cause", cause).addValue("note", note).addValue("expected", expected)
                .addValue("keys", mapper.valueToTree(normalized).toString()).addValue("user", userId)
                .addValue("revision", body.revision());
        int changed = jdbc.update("""
                UPDATE ai_feedback f SET status=:status,root_cause=:cause,review_note=:note,
                expected_answer=:expected,required_issue_keys=CAST(:keys AS jsonb),reviewer_id=:user,
                revision=revision+1,updated_at=now() WHERE f.id=:id AND f.revision=:revision AND
                """ + SCOPE, params);
        if (changed != 1) throw new ResponseStatusException(HttpStatus.CONFLICT, "Another reviewer updated this feedback. Reload before saving.");
        return detail(userId, id);
    }

    private MapSqlParameterSource reviewerScope(long userId) {
        List<Long> owned = spaces.findByOwnerIdAndDeletedAtIsNull(userId).stream().map(Space::getId).toList();
        if (owned.isEmpty()) throw new ResponseStatusException(HttpStatus.FORBIDDEN, "AI Feedback is available to space owners");
        return new MapSqlParameterSource("spaces", mapper.valueToTree(owned).toString());
    }

    private RowMapper<ObjectNode> rowMapper(boolean full) {
        return (rs, index) -> {
            ObjectNode node = mapper.createObjectNode();
            for (String name : List.of("id", "reason", "comment", "status")) node.put(name, rs.getString(name));
            node.put("reporterName", rs.getString("reporter_name"));
            node.put("reporterId", rs.getLong("reporter_id"));
            node.put("createdAt", rs.getTimestamp("created_at").toInstant().toString());
            node.put("updatedAt", rs.getTimestamp("updated_at").toInstant().toString());
            node.put("revision", rs.getInt("revision"));
            JsonNode snapshot = json(rs, "snapshot");
            node.put("question", snapshot.path("request").path("question").asText());
            node.set("spaceIds", snapshot.path("request").path("spaceIds"));
            if (full) {
                node.set("snapshot", snapshot);
                node.put("rootCause", rs.getString("root_cause"));
                node.put("reviewNote", rs.getString("review_note"));
                node.put("expectedAnswer", rs.getString("expected_answer"));
                node.set("requiredIssueKeys", json(rs, "required_issue_keys"));
            }
            return node;
        };
    }

    private JsonNode json(ResultSet rs, String name) throws SQLException {
        try { return mapper.readTree(rs.getString(name)); }
        catch (Exception e) { throw new SQLException("Invalid stored feedback JSON", e); }
    }

    private static String choice(String value, Set<String> choices, String name) {
        String normalized = value == null ? "" : value;
        require(choices.contains(normalized), "Invalid " + name);
        return normalized;
    }

    private static String bounded(String value, int max, String name) {
        require(value == null || value.length() <= max, name + " is too long");
        return value == null ? "" : value.trim();
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new ResponseStatusException(HttpStatus.BAD_REQUEST, message);
    }
}
