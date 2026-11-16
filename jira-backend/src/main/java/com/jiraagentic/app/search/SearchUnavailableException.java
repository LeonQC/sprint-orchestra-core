package com.jiraagentic.app.search;

/**
 * Thrown when OpenSearch cannot be reached or refuses a query.
 *
 * <p>A distinct type rather than a bare {@code RuntimeException} so the global exception handler can
 * answer 503 rather than 500. The difference is not cosmetic: with search living in a separate
 * process, "the search cluster is down" is a transient dependency failure the caller should retry,
 * while a 500 tells them the request itself was bad and retrying is pointless.
 */
public class SearchUnavailableException extends RuntimeException {

    public SearchUnavailableException(Throwable cause) {
        super("search backend unavailable", cause);
    }
}
