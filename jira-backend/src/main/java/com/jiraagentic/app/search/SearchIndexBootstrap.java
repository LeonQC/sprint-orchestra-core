package com.jiraagentic.app.search;

import org.opensearch.client.Request;
import org.opensearch.client.RestClient;
import org.opensearch.client.opensearch.OpenSearchClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;

/**
 * Creates the search indices with explicit mappings on startup, before any document is written.
 * Otherwise OpenSearch auto-creates them with the standard analyzer (no HTML stripping, separator
 * handling or stemming) — no error, search just gets quietly worse.
 *
 * <p>Runs on {@code ApplicationReadyEvent} so a slow or missing cluster doesn't stop startup. On
 * failure search degrades and the log says why; repair with {@code POST /internal/search/reindex}.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class SearchIndexBootstrap {

    private final OpenSearchClient client;
    private final RestClient restClient;
    private final SearchIndexNames indices;

    @EventListener(ApplicationReadyEvent.class)
    public void createIndicesIfMissing() {
        try {
            ensureIndex(indices.getIssues(), "opensearch/issues-mapping.json");
            ensureIndex(indices.getComments(), "opensearch/comments-mapping.json");
        } catch (Exception e) {
            log.error("could not create search indices — search will be degraded until "
                    + "POST /internal/search/reindex is called", e);
        }
    }

    /** Idempotent: an index that already exists is left exactly as it is, mapping included. */
    public void ensureIndex(String index, String mappingResource) throws IOException {
        if (client.indices().exists(r -> r.index(index)).value()) {
            return;
        }
        String body;
        try (Reader settings = resource("opensearch/analysis.json");
             Reader mappings = resource(mappingResource)) {
            body = wrap(settings, mappings);
        }
        // Sent through the low-level client rather than the typed CreateIndexRequest builder. The
        // OpenSearch client cannot parse a raw JSON body into a builder, and the alternative —
        // deserializing these files into
        // typed IndexSettings/TypeMapping objects only to have the client serialize them back — adds
        // a translation layer that can only lose fidelity. The body IS the API's own format already.
        Request request = new Request("PUT", "/" + index);
        request.setJsonEntity(body);
        restClient.performRequest(request);
        log.info("created opensearch index {}", index);
    }

    /**
     * The create-index API takes settings and mappings as two keys of one body, but they are kept in
     * two files here so every index shares one analyzer definition rather than each restating it —
     * a query and its documents must be analyzed identically, and the surest way to guarantee that
     * is for there to be only one copy of the rule.
     */
    private static String wrap(Reader settings, Reader mappings) throws IOException {
        return "{\"settings\":" + read(settings) + ",\"mappings\":" + read(mappings) + "}";
    }

    private static Reader resource(String path) throws IOException {
        InputStream stream = new ClassPathResource(path).getInputStream();
        return new InputStreamReader(stream, StandardCharsets.UTF_8);
    }

    private static String read(Reader reader) throws IOException {
        StringBuilder out = new StringBuilder();
        char[] buffer = new char[4096];
        int n;
        while ((n = reader.read(buffer)) != -1) {
            out.append(buffer, 0, n);
        }
        return out.toString();
    }
}
