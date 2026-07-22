package com.jiraagentic.app.exception;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.server.ResponseStatusException;

@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(UserNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleUserNotFound(UserNotFoundException ex) {
        return ResponseEntity
                .status(HttpStatus.NOT_FOUND)
                .body(ErrorResponse.of(HttpStatus.NOT_FOUND.value(), ex.getMessage()));
    }

    /**
     * Search runs in a separate process now, so "the search cluster is unreachable" is a dependency
     * failure, not a bad request. 503 rather than the generic 500 below tells the caller retrying is
     * worth doing — and keeps a search outage from reading like an application bug in the logs.
     */
    @ExceptionHandler(com.jiraagentic.app.search.SearchUnavailableException.class)
    public ResponseEntity<ErrorResponse> handleSearchUnavailable(
            com.jiraagentic.app.search.SearchUnavailableException ex) {
        log.error("search backend unavailable", ex);
        return ResponseEntity
                .status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(ErrorResponse.of(HttpStatus.SERVICE_UNAVAILABLE.value(), "Search is temporarily unavailable"));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(IllegalArgumentException ex) {
        String msg = ex.getMessage();
        if (msg == null || msg.isBlank()) {
            msg = "Invalid request";
        }
        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(ErrorResponse.of(HttpStatus.BAD_REQUEST.value(), msg));
    }

    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ErrorResponse> handleDataIntegrity(DataIntegrityViolationException ex) {
        log.warn("Request conflicted with a database constraint", ex);
        return ResponseEntity
                .status(HttpStatus.CONFLICT)
                .body(ErrorResponse.of(HttpStatus.CONFLICT.value(), "The requested value already exists or is in use"));
    }

    /**
     * A thrower that already chose an HTTP status must keep it. {@link ResponseStatusException}
     * extends {@link RuntimeException}, so without this handler it fell through to
     * {@link #handleRuntime} and every deliberate 401/403/404 was reported to the client as a
     * generic 500 — verified live against the space-membership guard in {@code IssueService}, and
     * true of {@code GroupService}'s existing "Only group admins can manage this group" 403 too.
     */
    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<ErrorResponse> handleResponseStatus(ResponseStatusException ex) {
        HttpStatus status = HttpStatus.resolve(ex.getStatusCode().value());
        if (status == null) {
            status = HttpStatus.INTERNAL_SERVER_ERROR;
        }
        String msg = ex.getReason() == null || ex.getReason().isBlank() ? status.getReasonPhrase() : ex.getReason();
        return ResponseEntity.status(status).body(ErrorResponse.of(status.value(), msg));
    }

    @ExceptionHandler(RuntimeException.class)
    public ResponseEntity<ErrorResponse> handleRuntime(RuntimeException ex) {
        String msg = ex.getMessage();
        if (msg == null || msg.isBlank()) {
            msg = ex.getClass().getSimpleName();
        }
        if (msg.contains("not found")) {
            return ResponseEntity
                    .status(HttpStatus.NOT_FOUND)
                    .body(ErrorResponse.of(HttpStatus.NOT_FOUND.value(), msg));
        }
        if (msg.contains("already exists")) {
            return ResponseEntity
                    .status(HttpStatus.CONFLICT)
                    .body(ErrorResponse.of(HttpStatus.CONFLICT.value(), msg));
        }
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ErrorResponse.of(HttpStatus.INTERNAL_SERVER_ERROR.value(), msg));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneric(Exception ex) {
        // Spring sometimes wraps failures (e.g. transaction / servlet) so the
        // real message is on the cause chain — unwrap before returning a generic 500.
        Throwable t = ex;
        for (int i = 0; i < 12 && t.getCause() != null && t.getCause() != t; i++) {
            Throwable c = t.getCause();
            if (c instanceof RuntimeException) {
                return handleRuntime((RuntimeException) c);
            }
            t = c;
        }
        if (t instanceof RuntimeException) {
            return handleRuntime((RuntimeException) t);
        }
        log.error("Unhandled exception", ex);
        String msg = t.getMessage();
        if (msg == null || msg.isBlank()) {
            msg = "Unexpected error occurred";
        }
        return ResponseEntity
                .status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(ErrorResponse.of(HttpStatus.INTERNAL_SERVER_ERROR.value(), msg));
    }
}
