package com.jiraagentic.app.security;

import org.springframework.core.convert.converter.Converter;
import org.springframework.security.authentication.AbstractAuthenticationToken;
import org.springframework.security.oauth2.jwt.Jwt;
import org.springframework.security.oauth2.server.resource.InvalidBearerTokenException;
import org.springframework.stereotype.Component;

/**
 * Verified JWT -> {@link UserAuthentication}, so controllers read the user the same way for JWT and
 * internal-service calls.
 */
@Component
public class JwtUserAuthenticationConverter implements Converter<Jwt, AbstractAuthenticationToken> {

    @Override
    public AbstractAuthenticationToken convert(Jwt jwt) {
        Object uid = jwt.getClaim("uid");
        long userId;
        if (uid instanceof Number n) {
            userId = n.longValue();
        } else if (uid instanceof String s) {
            try {
                userId = Long.parseLong(s);
            } catch (NumberFormatException e) {
                throw new InvalidBearerTokenException("Token uid claim is not a number");
            }
        } else {
            throw new InvalidBearerTokenException("Token has no uid claim");
        }
        return new UserAuthentication(userId, jwt.getSubject());
    }
}
