package com.jiraagentic.app.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.Base64;
import java.util.HexFormat;
import java.util.UUID;

@Component
public class FeedbackReceiptVerifier {
    private final ObjectMapper mapper;
    private final String secret;

    public FeedbackReceiptVerifier(ObjectMapper mapper,
            @Value("${app.security.internal-service-token}") String secret) {
        this.mapper = mapper;
        this.secret = secret;
    }

    public JsonNode verify(String receipt, long userId) {
        try {
            if (receipt == null || receipt.length() > 1_000_000) throw new IllegalArgumentException();
            String[] parts = receipt.split("\\.", -1);
            if (parts.length != 2 || parts[1].length() != 64) throw new IllegalArgumentException();
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
            byte[] signature = mac.doFinal(("ask-ai-feedback-v1:" + parts[0]).getBytes(StandardCharsets.UTF_8));
            if (!MessageDigest.isEqual(signature, HexFormat.of().parseHex(parts[1]))) throw new IllegalArgumentException();
            JsonNode node = mapper.readTree(Base64.getUrlDecoder().decode(parts[0]));
            long now = Instant.now().getEpochSecond();
            if (node.path("schemaVersion").asInt() != 1 || node.path("userId").asLong() != userId
                    || node.path("expiresAt").asLong() <= now || node.path("issuedAt").asLong() > now + 60) {
                throw new IllegalArgumentException();
            }
            UUID.fromString(node.path("answerId").asText());
            JsonNode scopes = node.path("request").path("spaceIds");
            if (!scopes.isArray() || scopes.isEmpty() || scopes.size() > 500) throw new IllegalArgumentException();
            for (JsonNode scope : scopes) {
                if (!scope.isIntegralNumber() || scope.asLong() <= 0) throw new IllegalArgumentException();
            }
            if (!node.path("response").path("answer").isTextual()) throw new IllegalArgumentException();
            return node;
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                    "This answer cannot receive feedback. Ask again to create a fresh answer.");
        }
    }
}
