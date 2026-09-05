-- Search coverage, round 5 — making the "Exact only" filter verifiable by eye.
--
-- The filter itself is computed correctly (a `LIKE '%q%'` test per row, added alongside
-- `matchedFields`), but a user cannot CHECK it, because the snippet next to each row comes from
-- `ts_headline()`, which highlights the matched *lexemes* — the stems. Searching "payments" and
-- reading a row whose snippet bolds "payment" looks exactly like a broken filter, even when the
-- row really does contain "payments" somewhere else in the field.
--
-- So: a snippet built around the LITERAL occurrence, used whenever the caller wants to justify an
-- exact hit. It emits the same "<b>"/"</b>" markers ts_headline uses, so the frontend renderer
-- (which splits on those markers and builds text nodes — never innerHTML) needs no changes.
--
-- The body passed in is expected to be strip_html()'d already, so the markers this adds are the
-- only ones present.

CREATE OR REPLACE FUNCTION literal_snippet(body text, q text) RETURNS text AS $$
    WITH found AS (
        SELECT strpos(lower(body), lower(q)) AS pos
        WHERE body IS NOT NULL AND q IS NOT NULL AND q <> ''
    ),
    win AS (
        -- 40 chars of lead-in so the match lands in context rather than at the very left edge.
        SELECT pos, greatest(1, pos - 40) AS start FROM found WHERE pos > 0
    )
    SELECT CASE WHEN win.start > 1 THEN '…' ELSE '' END
        || substring(body from win.start for win.pos - win.start)
        || '<b>' || substring(body from win.pos for length(q)) || '</b>'
        || substring(body from win.pos + length(q) for 100)
        || CASE WHEN length(body) > win.pos + length(q) + 100 THEN '…' ELSE '' END
    FROM win;
$$ LANGUAGE sql IMMUTABLE;

COMMENT ON FUNCTION literal_snippet(text, text) IS
    'Snippet centered on the first case-insensitive literal occurrence of q in body, with the '
    'occurrence wrapped in <b>...</b>. NULL when q does not literally occur — which makes it a '
    'usable "is this an exact hit, and where?" probe, not just a formatter.';
