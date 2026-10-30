package com.jiraagentic.app.security;

import org.springframework.security.core.Authentication;

public final class AuthSupport {

    private AuthSupport() {
    }

    public static Long extractUid(Authentication authentication) {
        if (authentication instanceof UserAuthentication g) {
            return g.getUserId();
        }
        throw new IllegalStateException("Expected an authenticated user");
    }

    public static Long extractUidOrNull(Authentication authentication) {
        if (authentication == null || !authentication.isAuthenticated()) {
            return null;
        }
        if (authentication instanceof UserAuthentication g) {
            return g.getUserId();
        }
        return null;
    }
}
