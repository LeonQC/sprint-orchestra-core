package com.jiraagentic.app.event;

/**
 * "Re-index this issue for search", published on every change that can affect search results.
 *
 * <p>Separate from {@link IssueContentChangedEvent}, which only fires for a few text fields because
 * re-embedding costs a model call. That event never fires for label or attachment changes — fine
 * under Postgres FTS (triggers kept those columns current), but with OpenSearch it would leave an
 * issue unfindable by a new label forever. Re-indexing is one cheap request, so this fires always.
 *
 * <p>Carries only the id; the listener reloads the issue with its labels and attachments.
 */
public record IssueSearchIndexRequestedEvent(long issueId) {
}
