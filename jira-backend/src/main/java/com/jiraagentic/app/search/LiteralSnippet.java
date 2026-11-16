package com.jiraagentic.app.search;

/**
 * A snippet centred on the first case-insensitive LITERAL occurrence of the query, wrapped in
 * {@code <b>…</b>}.
 *
 * <p>OpenSearch highlights stemmed matches, so searching "payments" can bold "payment", which looks
 * like a broken "Exact only" filter. This highlights what the user actually typed. Returns
 * {@code null} when the query doesn't literally occur — that's how {@code exactMatch} is derived.
 * Uses the same markers as the OpenSearch highlight, so the frontend renders both the same way
 * (text nodes, never innerHTML).
 */
public final class LiteralSnippet {

    /** Characters of lead-in, so the match lands in context rather than at the very left edge. */
    private static final int LEAD_IN = 40;
    private static final int TRAIL = 100;

    private LiteralSnippet() {
    }

    public static String of(String body, String query) {
        if (body == null || query == null || query.isEmpty()) {
            return null;
        }
        int pos = body.toLowerCase().indexOf(query.toLowerCase());
        if (pos < 0) {
            return null;
        }
        int start = Math.max(0, pos - LEAD_IN);
        int matchEnd = pos + query.length();
        int end = Math.min(body.length(), matchEnd + TRAIL);
        return (start > 0 ? "…" : "")
                + body.substring(start, pos)
                + "<b>" + body.substring(pos, matchEnd) + "</b>"
                + body.substring(matchEnd, end)
                + (body.length() > end ? "…" : "");
    }

    public static boolean contains(String body, String query) {
        return body != null && !query.isEmpty() && body.toLowerCase().contains(query.toLowerCase());
    }
}
