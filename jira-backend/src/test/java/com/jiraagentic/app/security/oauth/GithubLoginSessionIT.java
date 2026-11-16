package com.jiraagentic.app.security.oauth;

import jakarta.servlet.http.Cookie;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.session.Session;
import org.springframework.session.data.redis.RedisSessionRepository;
import org.springframework.test.web.servlet.MockMvc;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/**
 * With two jira-backend instances, GitHub's callback can reach a different one than the instance that
 * started the login. Needs the local Postgres and Redis, like the other *IT tests.
 */
@SpringBootTest(properties = {
        "app.oauth2.github.enabled=true",
        "app.oauth2.github.client-id=test-client-id",
        "app.oauth2.github.client-secret=test-client-secret",
        "management.health.db.enabled=false",
        "management.health.redis.enabled=false"
})
@AutoConfigureMockMvc
class GithubLoginSessionIT {

    @Autowired
    private MockMvc mvc;

    @Autowired
    private RedisSessionRepository sessions;

    @Test
    void loginStateIsStoredInRedisSoAnyInstanceCanFinishTheLogin() throws Exception {
        Cookie cookie = mvc.perform(get("/oauth2/authorization/github"))
                .andExpect(status().is3xxRedirection())
                .andReturn().getResponse().getCookie("SESSION");

        assertThat(cookie).isNotNull();
        String sessionId = new String(Base64.getDecoder().decode(cookie.getValue()), StandardCharsets.UTF_8);
        Session stored = sessions.findById(sessionId);
        assertThat(stored).isNotNull();
        assertThat(stored.getAttributeNames()).anyMatch(name -> name.endsWith(".AUTHORIZATION_REQUEST"));
    }

    @Test
    void anonymousApiCallsDoNotCreateSessions() throws Exception {
        Cookie cookie = mvc.perform(get("/api/spaces")).andReturn().getResponse().getCookie("SESSION");

        assertThat(cookie).isNull();
    }
}
