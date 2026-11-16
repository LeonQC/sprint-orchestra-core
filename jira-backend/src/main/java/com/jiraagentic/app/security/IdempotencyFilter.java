package com.jiraagentic.app.security;

import com.jiraagentic.app.entity.IdempotencyKeyRecord;
import com.jiraagentic.app.repository.IdempotencyKeyRepository;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;
import org.springframework.web.util.ContentCachingResponseWrapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Optional;

/**
 * Replays the stored response for a retried mutating request instead of running it again.
 *
 * <p>The ai-service sprint-recovery commit loop sends a stable {@code Idempotency-Key}
 * ({@code sprint-recovery:thread_id:commit_attempt_id:action_index}) with each Jira write. If the
 * write succeeded but the response was lost, the retry gets the recorded 2xx response and the handler
 * doesn't run again (no duplicate comment).
 *
 * <p>Not a lock: two truly concurrent requests with the same key can both run. Fine for the real
 * failure mode, a single caller retrying after a timeout.
 *
 * <p>Applies only when the header is present; normal requests pass straight through.
 */
@Component
public class IdempotencyFilter extends OncePerRequestFilter {

    public static final String IDEMPOTENCY_KEY_HEADER = "Idempotency-Key";

    private final IdempotencyKeyRepository repository;

    public IdempotencyFilter(IdempotencyKeyRepository repository) {
        this.repository = repository;
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        return !request.getRequestURI().startsWith("/api/");
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        String key = request.getHeader(IDEMPOTENCY_KEY_HEADER);
        if (!StringUtils.hasText(key)) {
            filterChain.doFilter(request, response);
            return;
        }

        Optional<IdempotencyKeyRecord> existing = repository.findById(key);
        if (existing.isPresent()) {
            replay(response, existing.get());
            return;
        }

        ContentCachingResponseWrapper wrapped = new ContentCachingResponseWrapper(response);
        filterChain.doFilter(request, wrapped);

        if (wrapped.getStatus() >= 200 && wrapped.getStatus() < 300) {
            String body = new String(wrapped.getContentAsByteArray(), StandardCharsets.UTF_8);
            IdempotencyKeyRecord record = new IdempotencyKeyRecord(
                    key, wrapped.getStatus(), body, wrapped.getContentType(), Instant.now()
            );
            try {
                repository.save(record);
            } catch (DataIntegrityViolationException raced) {
                // A concurrent request with the same key won the insert first — the side effect it
                // guarded against duplicating already happened once either way; nothing left to do.
            }
        }
        wrapped.copyBodyToResponse();
    }

    private void replay(HttpServletResponse response, IdempotencyKeyRecord record) throws IOException {
        response.setStatus(record.getResponseStatus());
        if (StringUtils.hasText(record.getContentType())) {
            response.setContentType(record.getContentType());
        }
        if (record.getResponseBody() != null) {
            response.getWriter().write(record.getResponseBody());
        }
    }
}
