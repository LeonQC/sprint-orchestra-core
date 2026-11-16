package com.jiraagentic.app.controller;

import com.jiraagentic.app.search.SearchReindexService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;

/**
 * Operator actions on the search indices, deliberately NOT under {@code /api/}.
 *
 * <p>The edge only routes {@code /api/**} here, so these paths are unreachable from a browser;
 * only a caller holding the internal service token (an operator or deploy script) gets in. A full
 * reindex used to sit at {@code POST /api/search/reindex}, behind nothing more than "is logged in".
 */
@RestController
@RequestMapping("/internal/search")
@RequiredArgsConstructor
public class SearchAdminController {

    private final SearchReindexService reindexService;

    /**
     * Rebuild the search indices from Postgres (the source of truth). Needed for a new environment
     * (empty indices), for repairing drift (indexing after commit fails soft, so an OpenSearch outage
     * leaves unfindable rows — this is the cheap alternative to an outbox), and after a mapping change
     * (change mapping, delete index, call this).
     *
     * <p>Synchronous, so a 200 means the index is current.
     */
    @PostMapping("/reindex")
    public SearchReindexService.ReindexResult reindex() throws IOException {
        return reindexService.reindexAll();
    }
}
