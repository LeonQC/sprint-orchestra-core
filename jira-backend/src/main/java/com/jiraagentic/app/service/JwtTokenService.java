package com.jiraagentic.app.service;

import com.jiraagentic.app.entity.User;
import com.nimbusds.jose.jwk.RSAKey;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.oauth2.jose.jws.SignatureAlgorithm;
import org.springframework.security.oauth2.jwt.*;
import org.springframework.stereotype.Service;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class JwtTokenService {

    private final JwtEncoder jwtEncoder;
    private final RSAKey jwtSigningKey;

    @Value("${app.security.jwt.issuer}")
    private String issuer;

    @Value("${app.security.jwt.expires-minutes:120}")
    private long expiresMinutes;

    public String issueToken(User user) {
        Instant now = Instant.now();
        JwtClaimsSet claims = JwtClaimsSet.builder()
                .issuer(issuer)
                .issuedAt(now)
                .expiresAt(now.plus(expiresMinutes, ChronoUnit.MINUTES))
                .subject(user.getUsername())
                .claims(map -> map.putAll(Map.of(
                        "uid", user.getId(),
                        "name", user.getName()
                )))
                .build();

        JwsHeader header = JwsHeader.with(SignatureAlgorithm.RS256).keyId(jwtSigningKey.getKeyID()).build();
        return jwtEncoder.encode(JwtEncoderParameters.from(header, claims)).getTokenValue();
    }

    public long getExpiresMinutes() {
        return expiresMinutes;
    }
}

