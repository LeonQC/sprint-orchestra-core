package com.jiraagentic.app.security;

import com.nimbusds.jose.jwk.RSAKey;
import com.nimbusds.jose.jwk.gen.RSAKeyGenerator;
import com.nimbusds.jose.jwk.JWKSet;
import com.nimbusds.jose.jwk.source.ImmutableJWKSet;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.oauth2.jose.jws.SignatureAlgorithm;
import org.springframework.security.oauth2.jwt.JwsHeader;
import org.springframework.security.oauth2.jwt.JwtClaimsSet;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtEncoderParameters;
import org.springframework.security.oauth2.jwt.NimbusJwtEncoder;
import org.springframework.test.web.servlet.MockMvc;

import java.time.Instant;
import java.time.temporal.ChronoUnit;

import static org.hamcrest.Matchers.hasSize;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * Who gets into jira-backend: user JWTs, the internal service token, and nothing else. Needs the local Postgres,
 * like the other *IT tests.
 */
@SpringBootTest(properties = {
        "app.security.internal-service-token=test-internal-token",
        "app.security.jwt.private-key-path=${java.io.tmpdir}/jira-backend-test-jwt-key.pem",
        "management.health.db.enabled=false",
        "management.health.redis.enabled=false"
})
@AutoConfigureMockMvc
class ApiAuthenticationIT {

    private static final String SPACES = "/api/spaces?userId=-1";

    @Autowired
    private MockMvc mvc;

    @Autowired
    private JwtEncoder jwtEncoder;

    @Autowired
    private RSAKey jwtSigningKey;

    @Test
    void rejectsApiCallsWithoutCredentials() throws Exception {
        mvc.perform(get(SPACES)).andExpect(status().isUnauthorized());
    }

    @Test
    void acceptsValidToken() throws Exception {
        mvc.perform(get(SPACES).header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, 60)))
                .andExpect(status().isOk());
    }

    @Test
    void rejectsExpiredToken() throws Exception {
        mvc.perform(get(SPACES).header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, -60)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void rejectsTokenSignedByAnotherKey() throws Exception {
        RSAKey otherKey = new RSAKeyGenerator(2048).keyID(jwtSigningKey.getKeyID()).generate();
        JwtEncoder otherEncoder = new NimbusJwtEncoder(new ImmutableJWKSet<>(new JWKSet(otherKey)));
        mvc.perform(get(SPACES).header("Authorization", "Bearer " + token(otherEncoder, otherKey, 60)))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void ignoresForgedIdentityHeadersWithoutInternalToken() throws Exception {
        mvc.perform(get(SPACES).header("X-User-Id", "1").header("X-Username", "admin"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void internalTokenActsAsForwardedUser() throws Exception {
        mvc.perform(get(SPACES)
                        .header("X-Internal-Token", "test-internal-token")
                        .header("X-User-Id", "1")
                        .header("X-Username", "admin"))
                .andExpect(status().isOk());
    }

    @Test
    void rejectsWrongInternalToken() throws Exception {
        mvc.perform(get(SPACES)
                        .header("X-Internal-Token", "wrong")
                        .header("X-User-Id", "1")
                        .header("X-Username", "admin"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void internalTokenWithoutUserCannotCallUserApis() throws Exception {
        mvc.perform(get(SPACES).header("X-Internal-Token", "test-internal-token"))
                .andExpect(status().isForbidden());
    }

    @Test
    void internalEndpointsRejectUserTokens() throws Exception {
        mvc.perform(post("/internal/search/reindex")
                        .header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, 60)))
                .andExpect(status().isForbidden());
    }

    @Test
    void loginStillWorksWithAStaleTokenInTheBrowser() throws Exception {
        mvc.perform(get("/api/auth/config").header("Authorization", "Bearer not-a-jwt"))
                .andExpect(status().isOk());
    }

    @Test
    void publishesOnlyThePublicKey() throws Exception {
        mvc.perform(get("/.well-known/jwks.json"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.keys", hasSize(1)))
                .andExpect(jsonPath("$.keys[0].kid").value(jwtSigningKey.getKeyID()))
                .andExpect(jsonPath("$.keys[0].d").doesNotExist());
    }

    @Test
    void listingUsersRequiresSignIn() throws Exception {
        mvc.perform(get("/api/users")).andExpect(status().isUnauthorized());
        mvc.perform(get("/api/users").header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, 60)))
                .andExpect(status().isOk());
    }

    @Test
    void nobodyCanCreateAnAccountFromOutside() throws Exception {
        String body = "{\"username\":\"intruder\",\"name\":\"x\",\"email\":\"x@example.com\"}";
        mvc.perform(post("/api/users").contentType("application/json").content(body))
                .andExpect(status().isUnauthorized());
        mvc.perform(post("/api/users").contentType("application/json").content(body)
                        .header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, 60)))
                .andExpect(status().isForbidden());
    }

    @Test
    void signedInUsersCannotDeleteAccounts() throws Exception {
        mvc.perform(delete("/api/users/999999999")).andExpect(status().isUnauthorized());
        mvc.perform(delete("/api/users/999999999")
                        .header("Authorization", "Bearer " + token(jwtEncoder, jwtSigningKey, 60)))
                .andExpect(status().isForbidden());
    }

    @Test
    void demoAccountListIsPublicButHasNoEmails() throws Exception {
        mvc.perform(get("/api/auth/demo-accounts"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].username").exists())
                .andExpect(jsonPath("$[0].email").doesNotExist());
    }

    @Test
    void unlistedPathsStayClosed() throws Exception {
        mvc.perform(get("/actuator/env")).andExpect(status().isUnauthorized());
    }

    private static String token(JwtEncoder encoder, RSAKey key, long expiresInSeconds) {
        Instant now = Instant.now();
        JwtClaimsSet claims = JwtClaimsSet.builder()
                .issuer("jira-agentic-ai")
                .issuedAt(now.minus(2, ChronoUnit.MINUTES))
                .expiresAt(now.plusSeconds(expiresInSeconds))
                .subject("admin")
                .claim("uid", 1L)
                .build();
        JwsHeader header = JwsHeader.with(SignatureAlgorithm.RS256).keyId(key.getKeyID()).build();
        return encoder.encode(JwtEncoderParameters.from(header, claims)).getTokenValue();
    }
}
