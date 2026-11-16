package com.jiraagentic.app.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.springframework.web.server.ResponseStatusException;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Base64;
import java.util.HexFormat;
import java.util.UUID;

import static org.assertj.core.api.Assertions.*;

class FeedbackReceiptVerifierTest {
    static final String SECRET = "feedback-test-secret-not-a-production-key";
    static final ObjectMapper MAPPER = new ObjectMapper();
    private final FeedbackReceiptVerifier verifier = new FeedbackReceiptVerifier(MAPPER, SECRET);

    static ObjectNode snapshot(long user, long... spaces) {
        long now = Instant.now().getEpochSecond();
        ObjectNode node = MAPPER.createObjectNode().put("schemaVersion", 1).put("answerId", UUID.randomUUID().toString())
                .put("userId", user).put("issuedAt", now).put("expiresAt", now + 604800);
        var request = node.putObject("request").put("question", "Which issues are blocked?");
        var ids = request.putArray("spaceIds");
        for (long space : spaces) ids.add(space);
        request.putArray("history");
        node.putObject("response").put("answer", "ATC-1 is blocked.").putArray("citations");
        return node;
    }

    static String sign(ObjectNode node) throws Exception {
        String payload = Base64.getUrlEncoder().withoutPadding().encodeToString(MAPPER.writeValueAsBytes(node));
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(SECRET.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
        return payload + "." + HexFormat.of().formatHex(mac.doFinal(("ask-ai-feedback-v1:" + payload).getBytes(StandardCharsets.UTF_8)));
    }

    @Test void acceptsUnicodeAndPreservesEvidence() throws Exception {
        var node = snapshot(1, 10);
        ((ObjectNode) node.get("response")).put("answer", "需要修正 ATC-1。");
        assertThat(verifier.verify(sign(node), 1).toString()).isEqualTo(node.toString());
    }

    @Test void rejectsChangedAnswerOrDifferentOwner() throws Exception {
        var node = snapshot(1, 10);
        String receipt = sign(node);
        assertThatThrownBy(() -> verifier.verify(receipt, 2)).isInstanceOf(ResponseStatusException.class);
        ((ObjectNode) node.get("response")).put("answer", "forged");
        String tampered = Base64.getUrlEncoder().withoutPadding().encodeToString(MAPPER.writeValueAsBytes(node)) + "." + receipt.split("\\.")[1];
        assertThatThrownBy(() -> verifier.verify(tampered, 1)).isInstanceOf(ResponseStatusException.class);
    }

    @Test void rejectsExpiredEmptyScopeAndOversizedReceipts() throws Exception {
        var expired = snapshot(1, 10).put("expiresAt", 1);
        String expiredReceipt = sign(expired);
        String emptyReceipt = sign(snapshot(1));
        assertThatThrownBy(() -> verifier.verify(expiredReceipt, 1)).isInstanceOf(ResponseStatusException.class);
        assertThatThrownBy(() -> verifier.verify(emptyReceipt, 1)).isInstanceOf(ResponseStatusException.class);
        assertThatThrownBy(() -> verifier.verify("x".repeat(1_000_001), 1)).isInstanceOf(ResponseStatusException.class);
    }
}
