package com.jiraagentic.app.config;

import com.jiraagentic.app.security.IdempotencyFilter;
import com.jiraagentic.app.security.InternalServiceAuthFilter;
import com.jiraagentic.app.security.JwtUserAuthenticationConverter;
import com.jiraagentic.app.security.oauth.OAuth2LoginSuccessHandler;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.oauth2.client.registration.ClientRegistrationRepository;
import org.springframework.security.oauth2.client.web.DefaultOAuth2AuthorizationRequestResolver;
import org.springframework.security.oauth2.server.resource.web.BearerTokenResolver;
import org.springframework.security.oauth2.server.resource.web.DefaultBearerTokenResolver;
import org.springframework.security.oauth2.server.resource.web.authentication.BearerTokenAuthenticationFilter;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.savedrequest.NullRequestCache;

import java.util.List;

@Configuration
public class SecurityConfig {

    private static final String[] PUBLIC_PATHS = {
            "/swagger-ui/**",
            "/swagger-ui.html",
            "/v3/api-docs/**",
            "/api/auth/**",
            "/.well-known/jwks.json",
            "/oauth2/**",
            "/login/**",
            "/actuator/health/**",
            "/actuator/prometheus",
            "/error"
    };

    // Login must work even when the browser still holds an expired token.
    private static final List<String> NO_BEARER_PREFIXES = List.of(
            "/api/auth", "/.well-known/", "/oauth2/", "/login/", "/actuator/health", "/error");

    @Bean
    public SecurityFilterChain securityFilterChain(
            HttpSecurity http,
            ObjectProvider<ClientRegistrationRepository> clientRegistrationRepositoryProvider,
            ObjectProvider<OAuth2LoginSuccessHandler> oauth2LoginSuccessHandler,
            InternalServiceAuthFilter internalServiceAuthFilter,
            IdempotencyFilter idempotencyFilter,
            JwtUserAuthenticationConverter jwtUserAuthenticationConverter
    ) throws Exception {
        http
                .csrf(AbstractHttpConfigurer::disable)
                // Saving the request a 401 interrupted would put a session in Redis for every
                // anonymous call; nothing here replays it.
                .requestCache(cache -> cache.requestCache(new NullRequestCache()))
                .addFilterBefore(internalServiceAuthFilter, BearerTokenAuthenticationFilter.class)
                .addFilterAfter(idempotencyFilter, BearerTokenAuthenticationFilter.class)
                .oauth2ResourceServer(oauth2 -> oauth2
                        .bearerTokenResolver(bearerTokenResolver())
                        .jwt(jwt -> jwt.jwtAuthenticationConverter(jwtUserAuthenticationConverter)))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(PUBLIC_PATHS).permitAll()
                        .requestMatchers("/logout").permitAll()
                        .requestMatchers("/internal/**").hasAuthority(InternalServiceAuthFilter.ROLE_INTERNAL_SERVICE)
                        // Accounts come from GitHub sign-in; creating or deleting one directly is an ops action.
                        .requestMatchers(HttpMethod.POST, "/api/users").hasAuthority(InternalServiceAuthFilter.ROLE_INTERNAL_SERVICE)
                        .requestMatchers(HttpMethod.DELETE, "/api/users/**").hasAuthority(InternalServiceAuthFilter.ROLE_INTERNAL_SERVICE)
                        .requestMatchers("/api/**").hasAuthority("ROLE_USER")
                        .anyRequest().denyAll()
                )
                .logout(logout -> logout
                        .logoutUrl("/logout")
                        .logoutSuccessUrl("/swagger-ui/index.html")
                        .permitAll()
                );

        if (clientRegistrationRepositoryProvider.getIfAvailable() != null) {
            ClientRegistrationRepository registrations = clientRegistrationRepositoryProvider.getIfAvailable();
            DefaultOAuth2AuthorizationRequestResolver authorizationRequestResolver =
                    new DefaultOAuth2AuthorizationRequestResolver(registrations, "/oauth2/authorization");
            authorizationRequestResolver.setAuthorizationRequestCustomizer(builder ->
                    builder.additionalParameters(parameters -> parameters.put("prompt", "select_account")));

            http.oauth2Login(oauth2 -> {
                oauth2.authorizationEndpoint(endpoint ->
                        endpoint.authorizationRequestResolver(authorizationRequestResolver));
                oauth2LoginSuccessHandler.ifAvailable(oauth2::successHandler);
            });
        }

        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    private static BearerTokenResolver bearerTokenResolver() {
        DefaultBearerTokenResolver resolver = new DefaultBearerTokenResolver();
        return (HttpServletRequest request) -> {
            String path = request.getRequestURI();
            if (NO_BEARER_PREFIXES.stream().anyMatch(path::startsWith)) {
                return null;
            }
            return resolver.resolve(request);
        };
    }
}
