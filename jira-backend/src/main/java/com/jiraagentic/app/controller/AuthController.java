package com.jiraagentic.app.controller;

import com.jiraagentic.app.dto.AuthTokenResponse;
import com.jiraagentic.app.dto.UserDto;
import com.jiraagentic.app.entity.User;
import com.jiraagentic.app.service.JwtTokenService;
import com.jiraagentic.app.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final UserService userService;
    private final JwtTokenService jwtTokenService;

    @Value("${app.oauth2.github.enabled:false}")
    private boolean githubOAuthEnabled;

    @Value("${app.auth.demo-accounts-enabled:true}")
    private boolean demoAccountsEnabled;

    @GetMapping("/config")
    @Operation(summary = "Public auth options for the login UI")
    public Map<String, Boolean> authConfig() {
        return Map.of(
                "githubOAuthEnabled", githubOAuthEnabled
        );
    }

    @GetMapping("/demo-accounts")
    @Operation(summary = "Accounts the login page offers for one-click demo login")
    public List<Map<String, Object>> demoAccounts() {
        // Public, so only what the buttons show: no email, no GitHub-only accounts.
        if (!demoAccountsEnabled) {
            return List.of();
        }
        return userService.findAll().stream()
                .filter(u -> u.getPassword() != null)
                .map(u -> Map.<String, Object>of(
                        "id", u.getId(),
                        "username", u.getUsername(),
                        "name", u.getName() == null ? u.getUsername() : u.getName(),
                        "avatarColor", u.getAvatarColor() == null ? "" : u.getAvatarColor()))
                .toList();
    }

    @PostMapping("/token")
    @Operation(summary = "Username/password login and issue bearer JWT")
    public ResponseEntity<?> issueToken(@RequestBody Map<String, String> body) {
        String username = body.get("username");
        String password = body.get("password");
        if (username == null || password == null) {
            return ResponseEntity.badRequest().body(Map.of("message", "Username and password are required"));
        }
        try {
            User user = userService.authenticate(username, password);
            String token = jwtTokenService.issueToken(user);
            return ResponseEntity.ok(new AuthTokenResponse(
                    token,
                    "Bearer",
                    jwtTokenService.getExpiresMinutes(),
                    UserDto.from(user)
            ));
        } catch (RuntimeException e) {
            return ResponseEntity.status(401).body(Map.of("message", e.getMessage()));
        }
    }
}
