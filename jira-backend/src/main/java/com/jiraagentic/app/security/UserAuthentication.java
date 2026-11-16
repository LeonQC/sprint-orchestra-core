package com.jiraagentic.app.security;

import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;

import java.util.ArrayList;
import java.util.List;

/**
 * The signed-in user, from either a verified JWT or an internal service acting on the user's behalf.
 */
public class UserAuthentication extends AbstractAuthenticationToken {

    public static final String ROLE_USER = "ROLE_USER";

    private final Long userId;
    private final String username;

    public UserAuthentication(Long userId, String username) {
        this(userId, username, List.of());
    }

    public UserAuthentication(Long userId, String username, List<String> extraRoles) {
        super(roles(extraRoles));
        this.userId = userId;
        this.username = username;
        setAuthenticated(true);
    }

    private static List<GrantedAuthority> roles(List<String> extraRoles) {
        List<GrantedAuthority> authorities = new ArrayList<>();
        authorities.add(new SimpleGrantedAuthority(ROLE_USER));
        extraRoles.forEach(role -> authorities.add(new SimpleGrantedAuthority(role)));
        return authorities;
    }

    @Override
    public Object getCredentials() {
        return "";
    }

    @Override
    public Object getPrincipal() {
        return username;
    }

    public Long getUserId() {
        return userId;
    }
}
