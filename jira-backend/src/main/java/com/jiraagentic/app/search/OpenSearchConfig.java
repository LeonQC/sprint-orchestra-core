package com.jiraagentic.app.search;

import org.opensearch.client.opensearch.OpenSearchClient;
import org.opensearch.client.json.jackson.JacksonJsonpMapper;
import org.opensearch.client.transport.rest_client.RestClientTransport;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpHost;
import org.opensearch.client.RestClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.time.Duration;

/**
 * The OpenSearch client this service uses for all text search.
 *
 * <p>Built by hand rather than through a Spring Data starter: the tiered
 * search in {@link com.jiraagentic.app.service.SearchService} issues four differently-shaped queries
 * per request and merges them itself, so a repository abstraction would be something to route around.
 *
 * <p>The connect/socket timeouts are deliberately short. Search is a foreground, user-typed
 * interaction: a wedged cluster should surface as a fast failure the caller can retry, not as a
 * request that occupies a servlet thread until the browser gives up.
 */
@Configuration
public class OpenSearchConfig {

    @Bean(destroyMethod = "close")
    public RestClient openSearchRestClient(
            @Value("${app.opensearch.uri:http://localhost:9200}") String uri,
            @Value("${app.opensearch.connect-timeout-ms:2000}") int connectTimeoutMs,
            @Value("${app.opensearch.socket-timeout-ms:5000}") int socketTimeoutMs) {
        return RestClient.builder(HttpHost.create(uri))
                .setRequestConfigCallback(config -> config
                        .setConnectTimeout((int) Duration.ofMillis(connectTimeoutMs).toMillis())
                        .setSocketTimeout((int) Duration.ofMillis(socketTimeoutMs).toMillis()))
                .build();
    }

    @Bean
    public OpenSearchClient openSearchClient(RestClient restClient, ObjectMapper objectMapper) {
        // Reuses Spring's ObjectMapper so the JavaTimeModule it already registers applies here too —
        // without it, serializing an Instant on a search document throws at index time rather than
        // at startup, i.e. on the first issue anyone edits.
        return new OpenSearchClient(new RestClientTransport(restClient, new JacksonJsonpMapper(objectMapper)));
    }
}
