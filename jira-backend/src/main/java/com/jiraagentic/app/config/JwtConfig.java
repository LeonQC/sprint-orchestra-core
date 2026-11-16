package com.jiraagentic.app.config;

import com.nimbusds.jose.JOSEException;
import com.nimbusds.jose.JWSAlgorithm;
import com.nimbusds.jose.jwk.JWKSet;
import com.nimbusds.jose.jwk.KeyUse;
import com.nimbusds.jose.jwk.RSAKey;
import com.nimbusds.jose.jwk.source.ImmutableJWKSet;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.oauth2.jwt.JwtDecoder;
import org.springframework.security.oauth2.jwt.JwtEncoder;
import org.springframework.security.oauth2.jwt.JwtValidators;
import org.springframework.security.oauth2.jwt.NimbusJwtDecoder;
import org.springframework.security.oauth2.jwt.NimbusJwtEncoder;
import org.springframework.util.StringUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.KeyPairGenerator;
import java.security.interfaces.RSAPrivateCrtKey;
import java.security.interfaces.RSAPublicKey;
import java.security.spec.PKCS8EncodedKeySpec;
import java.security.spec.RSAPublicKeySpec;
import java.util.Base64;

/**
 * RS256 signing: other services verify tokens with the public key from /.well-known/jwks.json, so no
 * shared secret leaves this service. Swapping in an external issuer (Cognito, Auth0) later only needs
 * {@code app.security.jwt.jwk-set-uri}.
 */
@Slf4j
@Configuration
public class JwtConfig {

    @Bean
    public RSAKey jwtSigningKey(
            @Value("${app.security.jwt.private-key:}") String privateKeyPem,
            @Value("${app.security.jwt.private-key-path}") String privateKeyPath,
            @Value("${app.security.jwt.allow-generated-key}") boolean allowGeneratedKey
    ) throws GeneralSecurityException, IOException, JOSEException {
        RSAPrivateCrtKey privateKey;
        if (StringUtils.hasText(privateKeyPem)) {
            privateKey = parsePkcs8(privateKeyPem);
        } else {
            Path path = Path.of(privateKeyPath);
            if (Files.exists(path)) {
                privateKey = parsePkcs8(Files.readString(path));
            } else if (allowGeneratedKey) {
                privateKey = generateAndSave(path);
            } else {
                // Each instance would mint its own key and reject the others' tokens.
                throw new IllegalStateException(
                        "No JWT signing key: set APP_JWT_PRIVATE_KEY, or allow a generated key for local dev");
            }
        }
        RSAPublicKey publicKey = (RSAPublicKey) KeyFactory.getInstance("RSA")
                .generatePublic(new RSAPublicKeySpec(privateKey.getModulus(), privateKey.getPublicExponent()));
        RSAKey key = new RSAKey.Builder(publicKey)
                .privateKey(privateKey)
                .keyUse(KeyUse.SIGNATURE)
                .algorithm(JWSAlgorithm.RS256)
                .build();
        // Thumbprint as kid: stable for a given key, so verifiers keep their cache across restarts.
        return new RSAKey.Builder(key).keyID(key.computeThumbprint().toString()).build();
    }

    @Bean
    public JwtEncoder jwtEncoder(RSAKey jwtSigningKey) {
        return new NimbusJwtEncoder(new ImmutableJWKSet<>(new JWKSet(jwtSigningKey)));
    }

    @Bean
    public JwtDecoder jwtDecoder(
            RSAKey jwtSigningKey,
            @Value("${app.security.jwt.issuer}") String issuer,
            @Value("${app.security.jwt.jwk-set-uri:}") String jwkSetUri
    ) throws JOSEException {
        NimbusJwtDecoder decoder = StringUtils.hasText(jwkSetUri)
                ? NimbusJwtDecoder.withJwkSetUri(jwkSetUri).build()
                : NimbusJwtDecoder.withPublicKey(jwtSigningKey.toRSAPublicKey()).build();
        decoder.setJwtValidator(JwtValidators.createDefaultWithIssuer(issuer));
        return decoder;
    }

    private static RSAPrivateCrtKey parsePkcs8(String pem) throws GeneralSecurityException {
        String base64 = pem
                .replace("-----BEGIN PRIVATE KEY-----", "")
                .replace("-----END PRIVATE KEY-----", "")
                .replaceAll("\\s", "");
        return (RSAPrivateCrtKey) KeyFactory.getInstance("RSA")
                .generatePrivate(new PKCS8EncodedKeySpec(Base64.getDecoder().decode(base64)));
    }

    private static RSAPrivateCrtKey generateAndSave(Path path) throws GeneralSecurityException, IOException {
        KeyPairGenerator generator = KeyPairGenerator.getInstance("RSA");
        generator.initialize(2048);
        RSAPrivateCrtKey privateKey = (RSAPrivateCrtKey) generator.generateKeyPair().getPrivate();
        String pem = "-----BEGIN PRIVATE KEY-----\n"
                + Base64.getMimeEncoder(64, "\n".getBytes(StandardCharsets.US_ASCII)).encodeToString(privateKey.getEncoded())
                + "\n-----END PRIVATE KEY-----\n";
        Files.createDirectories(path.toAbsolutePath().getParent());
        Files.writeString(path, pem);
        try {
            Files.setPosixFilePermissions(path, PosixFilePermissions.fromString("rw-------"));
        } catch (UnsupportedOperationException ignored) {
            // Non-POSIX filesystem; the key is a local dev key anyway.
        }
        log.warn("Generated a local dev JWT signing key at {}. Production must set APP_JWT_PRIVATE_KEY.", path);
        return privateKey;
    }
}
