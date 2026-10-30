package com.jiraagentic.app.security;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;

/**
 * Service-to-service auth: ai-service and ops scripts present a shared secret instead of a user JWT.
 * Needed because some callers have no user token to forward (e.g. sprint recovery resumed by a Kafka
 * event). With X-User-Id/X-Username the call acts as that user; without them it can only reach
 * /internal/**. Requests without the header fall through to JWT authentication.
 */
@Component
public class InternalServiceAuthFilter extends OncePerRequestFilter {

    public static final String INTERNAL_TOKEN_HEADER = "X-Internal-Token";
    public static final String USER_ID_HEADER = "X-User-Id";
    public static final String USERNAME_HEADER = "X-Username";
    public static final String ROLE_INTERNAL_SERVICE = "ROLE_INTERNAL_SERVICE";

    private final byte[] expectedToken;

    public InternalServiceAuthFilter(@Value("${app.security.internal-service-token}") String expectedToken) {
        this.expectedToken = expectedToken.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    protected void doFilterInternal(
            HttpServletRequest request,
            HttpServletResponse response,
            FilterChain filterChain
    ) throws ServletException, IOException {
        String presented = request.getHeader(INTERNAL_TOKEN_HEADER);
        if (!StringUtils.hasText(presented)) {
            filterChain.doFilter(request, response);
            return;
        }
        if (!MessageDigest.isEqual(expectedToken, presented.getBytes(StandardCharsets.UTF_8))) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid internal token");
            return;
        }

        String uidHeader = request.getHeader(USER_ID_HEADER);
        String username = request.getHeader(USERNAME_HEADER);
        if (StringUtils.hasText(uidHeader) != StringUtils.hasText(username)) {
            response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "X-User-Id and X-Username must be sent together");
            return;
        }
        if (StringUtils.hasText(uidHeader)) {
            long uid;
            try {
                uid = Long.parseLong(uidHeader.trim());
            } catch (NumberFormatException e) {
                response.sendError(HttpServletResponse.SC_UNAUTHORIZED, "Invalid user id header");
                return;
            }
            SecurityContextHolder.getContext().setAuthentication(
                    new UserAuthentication(uid, username, List.of(ROLE_INTERNAL_SERVICE)));
        } else {
            SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(
                    "internal-service", null, List.of(new SimpleGrantedAuthority(ROLE_INTERNAL_SERVICE))));
        }
        filterChain.doFilter(request, response);
    }
}
