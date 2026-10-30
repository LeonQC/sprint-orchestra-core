package com.jiraagentic.app.security;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;

import static org.assertj.core.api.Assertions.assertThat;

class InternalServiceAuthFilterTest {

    private final InternalServiceAuthFilter filter = new InternalServiceAuthFilter("test-internal-token");

    @AfterEach
    void clearContext() {
        SecurityContextHolder.clearContext();
    }

    @Test
    void passesThroughUntouchedWithoutTheHeader() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/spaces");
        request.addHeader("X-User-Id", "1");
        request.addHeader("X-Username", "admin");
        MockFilterChain chain = run(request, new MockHttpServletResponse());

        assertThat(chain.getRequest()).isNotNull();
        assertThat(SecurityContextHolder.getContext().getAuthentication()).isNull();
    }

    @Test
    void forwardedUserGetsBothRoles() throws Exception {
        MockHttpServletRequest request = internal();
        request.addHeader("X-User-Id", "42");
        request.addHeader("X-Username", "alice");
        run(request, new MockHttpServletResponse());

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        assertThat(auth).isInstanceOf(UserAuthentication.class);
        assertThat(((UserAuthentication) auth).getUserId()).isEqualTo(42L);
        assertThat(auth.getAuthorities()).extracting(Object::toString)
                .containsExactlyInAnyOrder("ROLE_USER", "ROLE_INTERNAL_SERVICE");
    }

    @Test
    void withoutUserHeadersOnlyTheServiceRole() throws Exception {
        run(internal(), new MockHttpServletResponse());

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        assertThat(auth.getAuthorities()).extracting(Object::toString).containsExactly("ROLE_INTERNAL_SERVICE");
    }

    @Test
    void rejectsHalfAnIdentity() throws Exception {
        MockHttpServletRequest request = internal();
        request.addHeader("X-User-Id", "42");
        MockHttpServletResponse response = new MockHttpServletResponse();
        MockFilterChain chain = run(request, response);

        assertThat(response.getStatus()).isEqualTo(401);
        assertThat(chain.getRequest()).isNull();
    }

    @Test
    void rejectsNonNumericUserId() throws Exception {
        MockHttpServletRequest request = internal();
        request.addHeader("X-User-Id", "alice");
        request.addHeader("X-Username", "alice");
        MockHttpServletResponse response = new MockHttpServletResponse();
        run(request, response);

        assertThat(response.getStatus()).isEqualTo(401);
    }

    private static MockHttpServletRequest internal() {
        MockHttpServletRequest request = new MockHttpServletRequest("GET", "/api/spaces");
        request.addHeader("X-Internal-Token", "test-internal-token");
        return request;
    }

    private MockFilterChain run(MockHttpServletRequest request, MockHttpServletResponse response) throws Exception {
        MockFilterChain chain = new MockFilterChain();
        filter.doFilter(request, response, chain);
        return chain;
    }
}
