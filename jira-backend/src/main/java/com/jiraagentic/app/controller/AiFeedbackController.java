package com.jiraagentic.app.controller;

import com.fasterxml.jackson.databind.node.ObjectNode;
import com.jiraagentic.app.security.UserAuthentication;
import com.jiraagentic.app.service.AiFeedbackService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ResponseStatusException;

import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/ai-feedback")
@RequiredArgsConstructor
public class AiFeedbackController {
    private final AiFeedbackService service;

    @PostMapping
    public Map<String, Object> submit(Authentication auth, @RequestBody AiFeedbackService.Submission body) {
        return service.submit(user(auth), body);
    }

    @GetMapping
    public Map<String, Object> list(Authentication auth, @RequestParam(required=false) String status,
            @RequestParam(required=false) String query,
            @RequestParam(defaultValue="0") int page) {
        return service.list(user(auth), status, query, page);
    }

    @GetMapping("/{id}")
    public ObjectNode detail(Authentication auth, @PathVariable UUID id) { return service.detail(user(auth), id); }

    @PutMapping("/{id}")
    public ObjectNode review(Authentication auth, @PathVariable UUID id, @RequestBody AiFeedbackService.Review body) {
        return service.review(user(auth), id, body);
    }

    private long user(Authentication auth) {
        if (auth instanceof UserAuthentication forwarded && auth.isAuthenticated()) return forwarded.getUserId();
        throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Sign in to use AI Feedback");
    }
}
