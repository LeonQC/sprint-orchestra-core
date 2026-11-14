package com.jiraagentic.app.controller;

import com.jiraagentic.app.retrieval.StructuredRetrievalService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/** Private service-token endpoints. The AI boundary supplies the caller's authorized space_ids. */
@RestController
@RequestMapping("/internal/retrieval")
@RequiredArgsConstructor
public class StructuredRetrievalController {
    private final StructuredRetrievalService service;

    @PostMapping("/issues/query")
    public Map<String, Object> issues(@RequestBody StructuredRetrievalService.IssueQuery request) {
        return service.issues(request);
    }

    @PostMapping("/issues/history")
    public Map<String, Object> history(@RequestBody StructuredRetrievalService.HistoryQuery request) {
        return service.history(request);
    }

    @PostMapping("/sprints/query")
    public Map<String, Object> sprints(@RequestBody StructuredRetrievalService.SprintQuery request) {
        return service.sprints(request);
    }
}
