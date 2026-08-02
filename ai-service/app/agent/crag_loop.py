"""Agentic RAG with corrective retrieval (CRAG-style), via tool use.

Instead of retrieving once and answering, the agent checks whether what came back actually answers
the question and searches again with a different query if not. The CRAG paper uses a trained
grading classifier; here the agent model grades its own results inside its normal tool loop, so a
re-search is just another tool call.

Security: the model controls only the query text (and mode), never `space_ids`. Permission scope is
injected by this code from the caller's authenticated identity on every call, so a confused or
prompt-injected model can't search spaces the user can't access.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.agent.retrieval_tool import (
    IssueCommentsResult,
    IssueDetailsResult,
    IssueHistoryResult,
    IssueQueryResult,
    RetrievalClient,
    RetrievedChunk,
    SprintQueryResult,
)
from app.agent.prompts import render_capability_router_prompt, render_crag_system_prompt
from app.agent.provenance_policy import (
    append_source_citation,
    carrier_named_by_file,
    replace_trailing_citation,
    resolve_implicit_source,
)
from app.llm.pricing import estimate_cost_usd, has_known_price
from app.llm.types import LLMClient, Usage
from app.tracing import llm_step
from app.observability import (
    LLM_CALL_SECONDS,
    RETRIEVAL_CALL_SECONDS,
    RETRIEVAL_ROUNDS,
    ROUTING_DECISIONS_TOTAL,
    TOOL_FAILURES_TOTAL,
)

logger = logging.getLogger(__name__)

# Exact phrase the system prompt requires when nothing grounds an answer. The API and evals match
# on this string instead of guessing whether free text "sounds like" a refusal.
ABSTENTION_PHRASE = "There is no information about this in the knowledge base."

# Per-request stage timings for AskResponse.stage_timings. A ContextVar because one CragAgent is
# shared across concurrent requests; storing on `self` would mix their timings.
_stage_seconds: ContextVar[Dict[str, float] | None] = ContextVar("_stage_seconds", default=None)

# Per-request system prompt (it includes the current time), task-local for the same reason: a
# shared field could be swapped by another request between two LLM turns.
_request_system_prompt: ContextVar[str | None] = ContextVar(
    "_request_system_prompt", default=None
)

# Per-request space id -> key ("ATC"), for naming each workspace in per-workspace query results.
_request_space_labels: ContextVar[Dict[int, str] | None] = ContextVar(
    "_request_space_labels", default=None
)


def _add_stage_seconds(stage: str, elapsed: float) -> None:
    acc = _stage_seconds.get()
    if acc is not None:
        acc[stage] = acc.get(stage, 0.0) + elapsed


# Progress labels for `ask()`'s optional `on_stage` callback (POST /ask/stream).
_TOOL_STAGE_LABELS = {
    "search_knowledge_base": "searching the knowledge base",
    "query_issues": "looking up issue data",
    "query_sprints": "looking up sprint data",
    "get_issue_history": "checking issue change history",
    "get_issue_comments": "reading issue comments",
    "get_issue_details": "reading issue details",
    "get_issue_attachments": "reading issue attachments",
    "get_section": "reading the whole section",
    "ask_clarification": "asking a clarification question",
}

# Budget guard on one tool call's result count, so a single call can't flood the context.
_MAX_RESULT_COUNT = 20
# Valid modes (mirrors vectorization-service). Unknown modes used to pass through, get a 422, and look
# like "nothing in the knowledge base" — seen after `lexical` was retired.
_SEARCH_MODES = ("hybrid", "vector", "bm25")
_DEFAULT_SEARCH_MODE = "hybrid"


def _coerce_search_mode(raw: object) -> str:
    """Model-supplied mode, or the default when it is anything this service cannot serve."""
    return raw if isinstance(raw, str) and raw in _SEARCH_MODES else _DEFAULT_SEARCH_MODE


def _coerce_result_count(raw: object, default: int) -> int:
    """Model-supplied result count, clamped into [1, _MAX_RESULT_COUNT]."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return default
    return max(1, min(_MAX_RESULT_COUNT, raw))


def _strip_shared_label_suffix(labels: List[str]) -> List[str]:
    """Drop a trailing phrase that EVERY clarification option repeats.

    The model sometimes appends the same words to every option ("... — diagnostic log" x3), which
    disambiguates nothing. Done in code because it's random model variance, not a missing
    instruction. Compares whole words, so "Kestrel" vs "Kestrel Holdings" is untouched.
    """
    if len(labels) < 2:
        return labels
    split = [label.split() for label in labels]
    shortest = min(len(words) for words in split)
    shared = 0
    while shared < shortest and len({words[-(shared + 1)] for words in split}) == 1:
        shared += 1
    if not shared:
        return labels
    trimmed = [
        # Also drop the separator the removed phrase hung off ("... order —" -> "... order").
        " ".join(words[: len(words) - shared]).rstrip(" -–—:,;|/")
        for words in split
    ]
    # All or nothing: if stripping would blank a label, keep them all as written.
    return labels if any(not t for t in trimmed) else trimmed


_SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the Jira workspace's knowledge base (issues, comments, attachments) for content "
        "relevant to a question. Call this whenever you need information to answer — never answer "
        "from memory. If the results don't actually address the question, call this again with a "
        "differently-worded or more specific query before giving up — don't settle for a weak match."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language or keyword search query."},
            # The model picks its own evidence budget: one specific thing needs a few chunks; an
            # answer assembled from several documents needs more.
            "result_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_RESULT_COUNT,
                "description": (
                    "Optional. How many results to return (default 5). Keep the default for a "
                    "question about one specific thing. Ask for more (10-20) when the answer has to "
                    "be assembled from several different issues, comments or attachments — for "
                    "example when the question asks for every case matching some description, or "
                    "for a comparison across the workspace. Larger values cost proportionally more "
                    "tokens, so raise it deliberately, not by default."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["hybrid", "vector", "bm25"],
                "description": (
                    "'hybrid' (default) combines semantic and keyword matching — use it unless you "
                    "have a specific reason. Use 'bm25' for exact identifiers (e.g. an issue key "
                    "like 'PROJ-42') or rare exact technical terms. Use 'vector' for purely "
                    "conceptual questions with no useful keywords."
                ),
            },
        },
        "required": ["query"],
    },
}

_ISSUE_QUERY_TOOL = {
    "name": "query_issues",
    "description": (
        "Structured lookup over issue METADATA — use this for counting, listing, filtering, or "
        "recency questions, NOT search_knowledge_base. It returns an exact total count, a breakdown "
        "by type and by status, and an ordered sample of matching issues. Use it whenever the answer "
        "is about how many / which ones / most recent rather than about the content of an issue. "
        "Examples: 'how many bugs are there', 'list all the stories', 'how many issues in total', "
        "'which issues are blocked', 'what was updated most recently', 'what issue did I just change', "
        "'how many issues are in the active sprint' (call query_sprints first to find the active "
        "sprint's id, then filter here by sprint_ids). "
        "Each returned issue includes its parent_key (the epic/parent issue it belongs to, or null if "
        "it has none) — use this, not search_knowledge_base, for 'which epic is X in' or 'what is the "
        "parent of X' questions: call this with issue_keys=[the issue in question] and read parent_key "
        "directly off the result. "
        "Each returned issue also carries assignee, story points, and priority — use this tool, not "
        "search_knowledge_base, for 'who is assigned to X', 'who owns X', 'how many points is X', or "
        "'what priority is X'. Those are structured fields: an issue's prose usually does not state "
        "them at all, so not finding them in search results is not evidence they are unset. A field "
        "rendered as (unassigned) or (unestimated) genuinely has no value — that IS the answer. "
        "All filters are optional and combine with AND; omit them all to get every issue in scope. "
        "\n\n"
        "SCANNING. Some questions ask which issues share a property that is NOT one of the structured "
        "fields above — 'which of these came from the same underlying mistake', 'what do the recurring "
        "failures here have in common', 'which ones were caused by X'. Do not reach for "
        "search_knowledge_base first for those. Search ranks by similarity and returns the top few, so "
        "anything it ranks low is simply absent from the answer — and a question asking for ALL the "
        "matches is exactly the shape that loses by being truncated. Worse, the property that links "
        "them is usually stated nowhere in the corpus: it exists only in the question, so there is "
        "little for a similarity search to match on. "
        "Instead, narrow with the structured filters above until the candidate set is small enough to "
        "read — a few hundred at most — then list it with a high limit, read the titles, and pick the "
        "candidates yourself. Confirm them with get_issue_details or get_issue_comments, which is "
        "where the evidence for the answer actually lives. Titles are short, so reading a few hundred "
        "of them is usually cheaper than several rounds of searching, and unlike a ranked top-k it "
        "cannot silently omit a match. "
        "A SCAN IS ONLY VALID ON A LIST YOU RECEIVED IN FULL. The result says `showing first N of "
        "TOTAL` whenever it was truncated. If you see that, you did not scan the workspace — you read "
        "an arbitrary slice of it, and any answer of the form 'these are all the ones that match' "
        "built on that slice is wrong. Narrow the filters and list again, or state in your answer "
        "which subset you actually examined. Never present a truncated scan as a complete one. "
        "This trades recall for precision: a title is a summary, so scanning will surface candidates "
        "that turn out not to qualify, and it will miss ones whose title does not hint at the property "
        "(read the details before committing to a list). Prefer search_knowledge_base when the "
        "question is about the CONTENT of a specific thing rather than about which things belong "
        "together."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_types": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter to these issue types. Known values: bug, story, task, epic, subtask.",
            },
            "issue_keys": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Restrict current-state lookup to these exact issue keys. Use this after "
                    "search_knowledge_base identifies topic-relevant issues, so an unrelated global "
                    "issue is never mistaken for the answer to a topic-specific question."
                ),
            },
            "parent_keys": {
                "type": "array", "items": {"type": "string"},
                "description": (
                    "Return only the children of these issues (issues whose parent_key is one of "
                    "them), e.g. ['ATC-52'] for an epic's stories and tasks, or a story's key for its "
                    "subtasks. For a tree (epic -> children -> subtasks), query the epic's children, "
                    "then call again with those children's keys; don't guess child keys."
                ),
            },
            "statuses": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter to these statuses. Known values: done, in_progress, planned, blocked.",
            },
            "priorities": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter to these priorities. Known values: low, medium, high, highest. "
                               "Only current values are known — see get_issue_history for past "
                               "priority changes.",
            },
            "sprint_ids": {
                "type": "array", "items": {"type": "integer"},
                "description": "Filter to issues in these sprints, by id — get ids from query_sprints "
                               "(e.g. its 'active' sprint) rather than guessing.",
            },
            "sprint_names": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter to issues in these sprints, by name (e.g. 'AtlasCart Sprint 7').",
            },
            "created_after": {"type": "string", "description": "ISO 8601 datetime lower bound on creation time."},
            "created_before": {"type": "string", "description": "ISO 8601 datetime upper bound on creation time."},
            "updated_after": {"type": "string", "description": "ISO 8601 datetime lower bound on last-update time."},
            "updated_before": {"type": "string", "description": "ISO 8601 datetime upper bound on last-update time."},
            "order_by": {"type": "string", "enum": ["created_at", "updated_at"],
                         "description": "Which timestamp to order the returned sample by. Default updated_at."},
            "order": {"type": "string", "enum": ["asc", "desc"],
                      "description": "desc = newest first (use for 'most recent'); asc = oldest first. Default desc."},
            "limit": {"type": "integer", "description":
                      "Max issues to return in the sample (the count is always exact). Default 20. "
                      "Raise it (up to 200) when you are scanning the workspace to decide which "
                      "issues share some property — a truncated list silently drops candidates, and "
                      "for a scan the whole point is that nothing is dropped."},
        },
        "required": [],
    },
}

_SPRINT_QUERY_TOOL = {
    "name": "query_sprints",
    "description": (
        "Structured lookup over sprint METADATA — name, goal, dates, status, and velocity/points "
        "stats. Use it for questions about sprints as a whole, NOT the issues inside them. Examples: "
        "'how many sprints are there', 'which sprint is currently active', 'what's the goal of sprint "
        "N', 'list the completed sprints', 'how many points did we complete last sprint'. To find a "
        "sprint's GOAL by topic/meaning ('which sprint was about accessibility') prefer "
        "search_knowledge_base first — sprint goals are indexed there as chunk_type=sprint — then use "
        "this tool if you need the exact structured fields (dates, points) for a sprint you've already "
        "identified. To count/list the ISSUES inside a sprint, use query_issues with sprint_ids/"
        "sprint_names instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "statuses": {
                "type": "array", "items": {"type": "string"},
                "description": "Filter to these statuses. Known values: future, active, completed.",
            },
            "order_by": {"type": "string", "enum": ["start_date", "end_date"],
                         "description": "Which date to order the returned sample by. Default start_date."},
            "order": {"type": "string", "enum": ["asc", "desc"],
                      "description": "desc = latest first; asc = earliest first. Default desc."},
            "limit": {"type": "integer", "description": "Max sprints to return in the sample (the count is always exact). Default 20."},
        },
        "required": [],
    },
}

_ISSUE_HISTORY_TOOL = {
    "name": "get_issue_history",
    "description": (
        "Query the append-only CHANGE HISTORY of issues — who changed what, when, from what value to "
        "what value (including old/new title and description text). Use it for questions about "
        "transitions and edits, NOT current state. Examples: 'which issues were reopened' "
        "(reopened_only=true), 'what did I change in ATC-77 just now' (issue_keys=['ATC-77']), "
        "'what changed in the last hour' (since=<ISO timestamp>), 'who moved this to blocked'. "
        "For an issue's CURRENT status/type use query_issues instead; for the content/meaning of an "
        "issue use search_knowledge_base."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_keys": {
                "type": "array", "items": {"type": "string"},
                "description": "Restrict to these issues, e.g. ['ATC-77'].",
            },
            "fields": {
                "type": "array", "items": {"type": "string"},
                "description": "Restrict to changes of these fields. Known values: status, title, "
                               "description, priority, assignee, sprint, storyPoints, issueType.",
            },
            "event_types": {
                "type": "array", "items": {"type": "string"},
                "description": "Restrict to these change kinds. Known values: field_change, "
                               "issue_created, comment_created, attachment_uploaded, link_created.",
            },
            "since": {"type": "string", "description": "ISO 8601 datetime — only changes at/after this time."},
            "until": {"type": "string", "description": "ISO 8601 datetime — only changes before this time."},
            "reopened_only": {
                "type": "boolean",
                "description": "true = only status changes that LEFT 'done' (an issue that was finished "
                               "and got reopened).",
            },
            "limit": {"type": "integer", "description": "Max change records to return (count is always exact). Default 20."},
        },
        "required": [],
    },
}

_ISSUE_COMMENTS_TOOL = {
    "name": "get_issue_comments",
    "description": (
        "Fetch the COMPLETE, unranked comment text for specific issue(s) you have already identified "
        "by key — not a top-K semantic match. Use this as a FALLBACK after search_knowledge_base: if "
        "search's top results don't contain the specific detail you need (a reason, an explanation, a "
        "decision) but you already know which issue it's in, call this with that issue's key to see "
        "every comment on it before concluding the information isn't there. Example: after finding "
        "which issue was reopened via get_issue_history, if you still need WHY, call this with that "
        "issue's key rather than guessing another search_knowledge_base query. Only abstain if BOTH "
        "search_knowledge_base and this (when you had a specific issue key to check) come up empty."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_keys": {
                "type": "array", "items": {"type": "string"},
                "description": "Fetch every comment on these issues, e.g. ['ATC-30', 'ATC-68'].",
            },
        },
        "required": ["issue_keys"],
    },
}

_ISSUE_DETAILS_TOOL = {
    "name": "get_issue_details",
    "description": (
        "Fetch the COMPLETE, unranked title+description text for specific issue(s) you have already "
        "identified by key — the counterpart to get_issue_comments, but for the issue's OWN body, not "
        "its comments. Use this whenever a question asks what a specific, already-identified issue "
        "IS or SAYS (e.g. 'show me the details of ATC-43', 'what is ATC-43 about') and "
        "search_knowledge_base's top results don't already show that issue's own title/description. "
        "This matters even when search_knowledge_base found PLENTY of results for that key: an issue "
        "with many comments can have its own single title+description chunk rank BELOW several of "
        "those comments (ranking has no guarantee of surfacing the canonical record just because it "
        "exists), so a pile of comment hits about an issue is not proof its own description was "
        "actually retrieved — check for it explicitly. Only conclude an issue's own details aren't "
        "available after calling this with its key and getting nothing back."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_keys": {
                "type": "array", "items": {"type": "string"},
                "description": "Fetch the title+description of these issues, e.g. ['ATC-43'].",
            },
        },
        "required": ["issue_keys"],
    },
}

_ISSUE_ATTACHMENTS_TOOL = {
    "name": "get_issue_attachments",
    "description": (
        "Fetch the COMPLETE, unranked parsed text of every attachment (PDF, spreadsheet, screenshot, "
        "log, etc.) on specific issue(s) you have already identified by key — the counterpart to "
        "get_issue_comments/get_issue_details, but for attachment content. Use this whenever a "
        "question asks about the content of a document/file/attachment on an already-identified issue "
        "(e.g. 'what SKU does ATC-46's attachment use', 'what does the log attached to ATC-43 show') "
        "and search_knowledge_base's results don't already contain the specific detail you need. This "
        "matters especially for an exact code, ID, or number you don't already know the wording of: "
        "a semantic/hybrid search query can only be as good as the words you guess, so a search that "
        "comes back empty is NOT proof the fact isn't there — call this with the issue's key to read "
        "its attachment text directly and completely before concluding the information is missing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "issue_keys": {
                "type": "array", "items": {"type": "string"},
                "description": "Fetch every attachment's parsed text on these issues, e.g. ['ATC-46'].",
            },
        },
        "required": ["issue_keys"],
    },
}

# Parent-child retrieval: a search hit from a long document section lists `section_ids`; this reads the whole
# section. Never a first move (it needs a hit), so it's not in _EXECUTABLE_TOOL_NAMES.
_SECTION_TOOL = {
    "name": "get_section",
    "description": (
        "Read the WHOLE section that a search_knowledge_base result was cut from. Results from long "
        "documents show `section_ids=...` when their section was split over several chunks. Call "
        "this with those ids when a result is relevant but cut off or incomplete for the question — "
        "e.g. it has steps 4-7 of a procedure and the question asks for all steps, a table continues "
        "past the chunk, or a finding's details stop mid-way. Also call it when the question asks for "
        "everything one document or section says on a topic (all the requirements, conditions, steps, "
        "phases or items) and the relevant results show section_ids: read the section instead of "
        "answering from a few pieces of it or searching again. Do not call it when the result already "
        "answers the question. Use get_issue_attachments instead when you need every attachment "
        "on an issue rather than one section."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section_ids": {
                "type": "array", "items": {"type": "string"}, "maxItems": 3,
                "description": "Ids copied from a result's section_ids, e.g. ['attachment:42:section#3'].",
            },
        },
        "required": ["section_ids"],
    },
}

# A real tool for asking a clarification question, so the UI can show a picker. Written into the
# answer text, it looked like an answer, the UI couldn't offer choices, and RAGAS scored it as a bad
# answer. Not a first-move route (nothing is ambiguous before retrieval has run), so it's not in
# _EXECUTABLE_TOOL_NAMES.
_CLARIFICATION_TOOL = {
    "name": "ask_clarification",
    "description": (
        "Ask the user ONE concise question when the request is genuinely ambiguous and answering "
        "would mean silently guessing — see ground rule 2b. Call this INSTEAD of writing the "
        "question into your final answer text: a question sent as plain text renders identically to "
        "a real answer, and the user has no structured way to respond to it. This tool ENDS the "
        "turn — the user sees your question (and any options), answers, and that becomes their next "
        "message, so do not call any other tool in the same turn as this one. Only call this after "
        "search_knowledge_base or a structured tool has already run — never as a first move, and "
        "never for a vague-but-answerable question where one broader search would resolve it.\n\n"
        "Two different situations both count as ambiguous, and `options` tells them apart:\n"
        "1. Retrieval found several documents that are ALL genuinely in contention for the SAME "
        "specific fact the user asked about (several invoices when asked 'the total amount due', "
        "several diagnostic logs when asked 'what the diagnostic log says'). Pass those as `options` "
        "— a real, meaningfully narrowed set worth choosing between.\n"
        "2. The question itself carries no topic beyond a generic placeholder — 'the attachment', "
        "'the document', 'the log', 'the report', 'the file' — with no bug, customer, date, or "
        "subject attached to it, so retrieval returns whatever happens to contain that generic word "
        "across UNRELATED issues. Those hits are not real candidates for anything specific; listing "
        "them as if the search had narrowed something down is worse than not listing them — the user "
        "has to read five unrelated document titles to discover none of them are what they meant. "
        "Pass `options: []` and ask the open question a person would actually ask back: which issue "
        "or topic they mean. Never fill `options` with unrelated hits just to satisfy a minimum — an "
        "empty list plus a real question is the honest response here, not a forced menu."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "One short question naming what's ambiguous. Do not preview the requested "
                    "amount, category, assignee, or other answer value in it — see ground rule 2b."
                ),
            },
            "options": {
                "type": "array",
                "minItems": 0,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": (
                                "Human-readable option text — a title, customer, date, or other "
                                "source cue from the evidence the user will recognize. Carry ONLY "
                                "what tells this candidate apart from the others: a word that would "
                                "appear in every option ('— diagnostic log' on all three, when the "
                                "question already asks which diagnostic log) disambiguates nothing "
                                "and is stripped before display. Do not repeat the issue key here "
                                "either — it is rendered beside the label from issue_key."
                            ),
                        },
                        "issue_key": {
                            "type": "string",
                            "description": "This candidate's real issue key, if it names one.",
                        },
                    },
                    "required": ["label"],
                },
                "description": (
                    "Either [] (the question has no real candidate set — see the tool description's "
                    "situation 2) or 2-5 candidate sources, each independently plausible for the SAME "
                    "specific fact/topic asked (situation 1). Never exactly 1 — that is not a choice. "
                    "When present, labels must read as distinct choices side by side — what they "
                    "share belongs in `question`, not repeated in every option."
                ),
            },
        },
        "required": ["question", "options"],
    },
}

# Complete, unranked reads by issue key (see CragAgent._exact_read_with_fallback).
_EXACT_READ_TOOL_NAMES = frozenset(
    {_ISSUE_COMMENTS_TOOL["name"], _ISSUE_DETAILS_TOOL["name"], _ISSUE_ATTACHMENTS_TOOL["name"]}
)

_EXECUTABLE_TOOL_NAMES = frozenset(
    {
        _SEARCH_TOOL["name"],
        _ISSUE_QUERY_TOOL["name"],
        _SPRINT_QUERY_TOOL["name"],
        _ISSUE_HISTORY_TOOL["name"],
        _ISSUE_COMMENTS_TOOL["name"],
        _ISSUE_DETAILS_TOOL["name"],
        _ISSUE_ATTACHMENTS_TOOL["name"],
    }
)

# Routing-only: when the keyword layer can't decide, the model returns one validated capability
# choice. It can't answer or invent a capability here.
_CAPABILITY_ROUTER_TOOL = {
    "name": "select_data_source",
    "description": "Select the best first data source for the user's question; do not answer it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": sorted(_EXECUTABLE_TOOL_NAMES),
                "description": (
                    "Choose an exact/structured tool whenever it can answer the question. Choose "
                    "search_knowledge_base only when the answer depends on topic/meaning or no "
                    "more precise capability applies."
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence that the selected exact capability matches the request.",
            },
            "reason": {
                "type": "string",
                "description": "One short routing rationale; never include an answer.",
            },
            "source_scope": {
                "type": "string",
                "enum": ["identified", "corpus_wide", "ambiguous_single_source"],
                "description": (
                    "identified when the user gives a unique document/source cue; corpus_wide for "
                    "an intentional aggregate/list/comparison/topic question; "
                    "ambiguous_single_source when one fact is requested but the source is omitted."
                ),
            },
        },
        "required": ["tool", "confidence", "source_scope"],
    },
}


def _current_time_for_prompt() -> str:
    """Current UTC time, rounded down to 5 minutes.

    A prompt that changes every second never hits the prompt cache. 5 minutes matches Anthropic's
    default cache TTL, so calls in that window share one cache entry. Costs at most 5 minutes of
    drift on "just now"-style questions.
    """
    now = datetime.now(timezone.utc)
    rounded = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
    return rounded.strftime("%Y-%m-%dT%H:%M:%SZ")


_HISTORY_INTENT_RE = re.compile(
    r"\b(reopen(?:ed|ing)?|change history|changed? (?:in|on|to|by)|who (?:changed|moved)|"
    r"when did|just (?:edit|change)|previous(?:ly)? (?:status|state))\b"
)
_ISSUE_ENTITY_RE = re.compile(
    r"\b(issues?|bugs?|stories|tasks?|epics?|subtasks?|tickets?|work items?)\b"
)
_SPRINT_ENTITY_RE = re.compile(r"\bsprints?\b|\bsprint\s*\d+\b")
_STRUCTURED_ACTION_RE = re.compile(
    r"\b(how many|number of|count|list|which|active|current|latest|newest|oldest|"
    r"recent(?:ly)?|last updated|most recently|updated most recently|sort(?:ed)?|velocity|points?)\b"
)
# Words that introduce a topic ("issues related to checkout"), which is a content question
# query_issues can't answer. Ambiguous ones ("for", "with") are left out: "bugs for sprint 5".
_TOPICAL_CONNECTIVE_RE = re.compile(
    r"\b(about|regarding|concerning|involving|pertaining to|"
    r"relate[sd]? to|relating to|related to|to do with|dealing with|on the (?:topic|subject) of)\b"
)
_EXPLICIT_DATE_CUE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:\s*[-–]\s*\d{1,2})?(?:,?\s+\d{4})?\b|"
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
    re.IGNORECASE,
)


def _has_explicit_source_cue(question: str) -> bool:
    """Whether the wording already points at one source (a reference number, a date/period, a quoted
    title), so we shouldn't ask for clarification.
    """
    if _ISSUE_KEY_RE.search(question) or _EXPLICIT_DATE_CUE_RE.search(question):
        return True
    return bool(re.search(r'["“][^"”]{3,}["”]', question))


def _initial_tool_choice(question: str) -> str | None:
    """Cheap keyword routing for common, unambiguous question shapes.

    A match saves a routing LLM call; a miss goes to the capability router, not to semantic search by
    default. The model can still use other tools afterwards.
    """
    q = " ".join(question.lower().split())

    # A caller-selected attachment ("Using only the attachments on KEY") wins over question words
    # like "which" that would otherwise route to query_issues. Only the first tool choice changes.
    if _ISSUE_KEY_RE.search(q) and re.search(r"\battachments?\b", q):
        return _ISSUE_ATTACHMENTS_TOOL["name"]

    if _HISTORY_INTENT_RE.search(q) or any(
        phrase in q for phrase in ("變更紀錄", "誰改了", "何時修改", "剛剛修改")
    ):
        return _ISSUE_HISTORY_TOOL["name"]

    issue_terms = bool(_ISSUE_ENTITY_RE.search(q)) or any(
        term in q for term in ("議題", "問題單", "工單")
    )
    # Users commonly type "sprint7" without a space; treat it the same as "Sprint 7".
    sprint_terms = bool(_SPRINT_ENTITY_RE.search(q)) or "衝刺" in q
    structured_action = bool(_STRUCTURED_ACTION_RE.search(q)) or any(
        term in q for term in ("最新", "最近", "剛剛", "幾個", "列出", "目前", "哪一個")
    )

    # Resolve every sprint reference first, including "Sprint 7": query_issues uses exact sprint ids/
    # names, so guessing a short display name can silently return 0 for "AtlasCart Sprint 7".
    if sprint_terms and structured_action:
        return _SPRINT_QUERY_TOOL["name"]

    if issue_terms and structured_action:
        return _ISSUE_QUERY_TOOL["name"]

    # Mirror of the rule above: an entity qualified by a topic, with no structured action ("issues
    # related to checkout"), is a content question -> semantic search. The capability router was a
    # coin flip here (two near-identical wordings went to different tools).
    if (
        issue_terms
        and not structured_action
        and not sprint_terms
        and not _ISSUE_KEY_RE.search(q)
        and _TOPICAL_CONNECTIVE_RE.search(q)
    ):
        return _SEARCH_TOOL["name"]

    # "Which issues are blocked?" is structured; "what is blocking payments?" is topic-specific
    # and intentionally falls through to semantic search first.
    if re.search(r"\b(which|list|how many)\b.*\b(blocked|planned|done|in progress)\b", q):
        return _ISSUE_QUERY_TOOL["name"]

    # When the key and source kind are explicit, skip ranking entirely. These exact readers exist so
    # a known issue's own content cannot disappear merely because another chunk scored higher.
    if _ISSUE_KEY_RE.search(q):
        if re.search(r"\b(attachments?|files?|documents?|pdfs?|screenshots?|images?)\b", q):
            return _ISSUE_ATTACHMENTS_TOOL["name"]
        if re.search(r"\b(comments?|discussion|replies|notes?)\b", q):
            return _ISSUE_COMMENTS_TOOL["name"]
        if re.search(r"\b(details?|description|title|body|summary|summarize)\b", q):
            return _ISSUE_DETAILS_TOOL["name"]

    # A miss means "ask the capability router", not "use semantic search".
    return None


#: The only route whose answers may be cached. Public so routes.py imports it instead of repeating it.
SEARCH_TOOL_NAME = _SEARCH_TOOL["name"]


def is_answer_cacheable(
    route_tool: str, abstained: bool, needs_clarification: bool = False, used_live_data: bool = False,
) -> bool:
    """Whether a completed answer may be written to the answer cache.

    Decided after routing, using the route actually taken. (Deciding before the agent ran meant
    guessing, and the guess could never say "semantic", so nothing was ever cached.)

    Cache only when:
    - the route was semantic search, or a multi-agent investigation (`investigate`), and
    - no structured tool result (issue query, sprint query, issue history) went into the answer, in
      any round or any worker. Those are live Jira state: a card drag changes them, but the cache
      key's `data_version` only follows searchable text, so a cached "ATC has 5 in progress" would
      outlive the drag. Before this, an investigation that counted issues was cached.
    - the agent didn't abstain. "We have no X" is the first thing to become false after an upload,
      and abstaining is cheap anyway.
    - the agent didn't ask for clarification. A later user who already included the detail would get
      the question back.
    """
    return (
        route_tool in (SEARCH_TOOL_NAME, "investigate")
        and not used_live_data
        and not abstained
        and not needs_clarification
    )


def _initial_tool_input_overrides(question: str, tool_name: str | None) -> dict:
    """Arguments the code enforces for high-confidence routes; the model fills in the rest.

    Only facts directly in the user's text are enforced (e.g. "most recent" -> ordering); inferred
    filters stay with the model.
    """
    q = " ".join(question.lower().split())
    if tool_name == _ISSUE_QUERY_TOOL["name"]:
        asks_for_single_latest = bool(
            re.search(
                r"\b(most recent(?:ly)? updated|updated most recently|latest updated|"
                r"last updated)\b.*\b(issue|ticket|bug|story|task|epic|subtask)\b",
                q,
            )
        )
        if asks_for_single_latest:
            return {"order_by": "updated_at", "order": "desc", "limit": 1}

    if tool_name in {
        _ISSUE_COMMENTS_TOOL["name"],
        _ISSUE_DETAILS_TOOL["name"],
        _ISSUE_ATTACHMENTS_TOOL["name"],
    }:
        issue_keys = [key.upper() for key in _ISSUE_KEY_RE.findall(question)]
        if issue_keys:
            return {"issue_keys": list(dict.fromkeys(issue_keys))}
    return {}


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Validated initial route shared by the first agent turn and routing telemetry."""

    tool_name: str
    source: str  # keyword | model | fallback
    confidence: str
    source_scope: str = "identified"  # identified | corpus_wide | ambiguous_single_source


@dataclass(frozen=True, slots=True)
class ClaimMismatch:
    """One verification failure: a claim contradicted by a structured fact. `detail` is the exact
    correction sent back to the model, written by the verifier that found it."""

    issue_key: str
    detail: str


@dataclass(slots=True)
class ModelUsage:
    """One model role's request usage, suitable for API/UI attribution."""

    role: str
    provider: str
    model: str
    deployment: str  # local | hosted
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    pricing_status: str  # free_local | estimated | unavailable
    # Of `input_tokens`, how many came from the prompt cache. `input_tokens` is already the fresh
    # count; this lets the UI show "N cached" and match a Phoenix trace.
    cached_input_tokens: int = 0
    # Tokens spent writing a prompt-cache entry (Anthropic only), billed at 1.25x. Kept so the usage
    # ledger can rebuild the same cost as `estimated_cost_usd`.
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ClarificationOption:
    """One choice offered by ask_clarification. `issue_key` is "" when the options aren't issues (e.g.
    two attachments on one issue); a plain string so the API can pass it straight through.
    """

    label: str
    issue_key: str = ""


@dataclass(slots=True)
class AgentAnswer:
    text: str
    retrieval_rounds: int
    queries_used: List[str]
    citations: List[RetrievedChunk] = field(default_factory=list)
    # Set when the turn ended with ask_clarification instead of an answer. `text` still holds the
    # question, so existing callers keep working; the options let the UI show a picker. Options are
    # empty whenever this is False.
    needs_clarification: bool = False
    clarification_options: List[ClarificationOption] = field(default_factory=list)
    # Text results of structured tools used this turn (same text the model saw), so evals like RAGAS
    # faithfulness can find facts such as "ATC-77 is blocked" that `citations` never covers.
    structured_evidence: List[str] = field(default_factory=list)
    abstained: bool = False
    usage: Usage = field(default_factory=Usage)
    estimated_cost_usd: float = 0.0
    model_usage: List[ModelUsage] = field(default_factory=list)
    route_source: str = ""
    route_tool: str = ""
    # Seconds per stage for this request ("llm", "retrieval"). A missing key means the stage never ran.
    stage_seconds: Dict[str, float] = field(default_factory=dict)


class CragAgent:
    def __init__(
        self, llm: LLMClient, retrieval: RetrievalClient, max_iterations: int, top_k: int, model: str = "",
        faithfulness_check_enabled: bool = False,
        router_llm: LLMClient | None = None, router_model: str = "",
        model_provider: str = "Unknown", model_deployment: str = "hosted",
        router_provider: str = "Unknown", router_deployment: str = "hosted",
        parent_child_tool_enabled: bool = True,
    ):
        self._llm = llm
        self._router_llm = router_llm or llm
        self._retrieval = retrieval
        self._max_iterations = max_iterations
        self._top_k = top_k
        # Used only for $-cost estimation (app/llm/pricing.py) — the LLM client already knows its own
        # model for the actual API call; the agent needs the name separately just to price tokens.
        self._model = model
        self._router_model = router_model or model
        self._model_provider = model_provider
        self._model_deployment = model_deployment
        self._router_provider = router_provider
        self._router_deployment = router_deployment
        self._faithfulness_check_enabled = faithfulness_check_enabled
        # Off = no get_section tool and no section_ids on hits: the agent as it was before parent-child
        # retrieval, kept so the two can be compared (eval/parent_child_compare.py).
        self._parent_child_tool_enabled = parent_child_tool_enabled
        # Post-generation verifiers; add new ones here without touching the loop. _check_faithfulness
        # isn't listed because it needs the retrieved context, not a re-query.
        self._post_generation_verifiers = [
            self._check_subtask_claims,
            self._check_current_state_claims,
        ]

    async def ask(
        self, question: str, space_ids: Sequence[int], history: Sequence[dict] | None = None,
        clarification_asked: bool = False,
        on_stage: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_round: Optional[Callable[[int, List["RetrievedChunk"], List[dict]], None]] = None,
        space_labels: Optional[Dict[int, str]] = None,
    ) -> AgentAnswer:
        # space_labels: space id -> key ("ATC"), used to name each workspace when two or more are
        # queried. Optional; without it a workspace is named by its issue-key prefix.
        # Optional callbacks, all no-ops by default:
        # on_stage: progress labels at existing checkpoints (POST /ask/stream).
        stage = on_stage or (lambda _label: None)
        # on_token: streams the final answer only — not routing, tool turns or the verification
        # rewrite. The `result` frame still has the authoritative answer.
        token_cb = on_token or (lambda _text: None)
        # on_round: after each tool round, all citations so far — for measuring whether retrieval
        # improves round over round.
        round_cb = on_round or (lambda _n, _citations, _calls: None)
        # Snapshot "now" once per ask so every LLM turn in this conversation sees the same clock —
        # needed to resolve relative time ("just now", "past 3 months") into concrete tool filters.
        request_system_prompt = render_crag_system_prompt(
            _current_time_for_prompt(), self._parent_child_tool_enabled
        )
        # `history` holds earlier turns' final text only, not their tool calls. Carrying tool_use /
        # tool_result pairs across turns isn't needed and would drag all the retrieved text along.
        messages: List[dict] = [dict(m) for m in (history or [])]
        messages.append({"role": "user", "content": question})
        queries_used: List[str] = []
        citations: List[RetrievedChunk] = []
        structured_evidence: List[str] = []
        seen_citations: set = set()
        evidence_issue_keys: set[str] = set()
        # Keys of entities/chunks actually returned by tools (not key-like ids mentioned inside text).
        # A final `(ATC-123)` citation must name one of these; an order id like BETA-1042 in the body
        # is fine.
        source_issue_keys: set[str] = set()
        subject_issue_keys: set[str] = {k.upper() for k in _ISSUE_KEY_RE.findall(question)}
        # Keys the earlier turns named. A follow-up answer may name one again without this turn's
        # query returning it ("PRM-30 is planned, so it is left out"); see verified_answer.
        chat_issue_keys: set[str] = {
            k.upper() for m in (history or []) for k in _ISSUE_KEY_RE.findall(str(m.get("content", "")))
        }
        semantic_used = False
        structured_used = False
        retrieval_rounds = 0
        total_usage = Usage()
        stage_seconds: Dict[str, float] = {}
        stage_seconds_token = _stage_seconds.set(stage_seconds)
        system_prompt_token = _request_system_prompt.set(request_system_prompt)
        space_labels_token = _request_space_labels.set(dict(space_labels or {}))

        try:
            route, route_usage = await self._select_initial_route(
                question, messages, stage
            )
            total_usage = total_usage + route_usage
            if (
                route.source_scope == "ambiguous_single_source"
                and _has_explicit_source_cue(question)
            ):
                route = RouteDecision(
                    route.tool_name, route.source, route.confidence, "identified"
                )
            if clarification_asked and route.source_scope == "ambiguous_single_source":
                # Already asked once in this chat: the user's reply is the answer to work with.
                route = RouteDecision(route.tool_name, route.source, route.confidence, "identified")
            if route.source_scope == "ambiguous_single_source":
                # Pass the router's view of the intent (seen before any retrieval could anchor the
                # model on one top hit) into every answering turn.
                _request_system_prompt.set(
                    request_system_prompt
                    + "\n\nRequest-specific routing constraint: The source-scope classifier "
                    + "determined that this asks for one document-specific fact but does not "
                    + "identify which document. Retrieve candidate evidence, then call "
                    + "ask_clarification (ground rule 2b) — do not answer the fact by selecting a "
                    + "top-ranked source, and do not write the question into your final answer "
                    + "text. Still apply ground rule 2b's own test for whether the retrieved "
                    + "documents are real candidates for this fact or just an unrelated coincidence "
                    + "on a generic word: list them as options ONLY if they are genuinely competing "
                    + "answers to the SAME fact, carrying only source-identifying cues — do not "
                    + "preview the requested answer value in the question or any option's label — "
                    + "otherwise call it with an empty options list and ask what issue or topic is "
                    + "meant. One exception: if the question names its own subject (a feature, "
                    + "setting, system, vendor or incident) and NO retrieved document is about that "
                    + "subject at all, there is nothing to choose between — reply with the "
                    + "no-information sentence from ground rule 3 instead, because asking which "
                    + "document to use would wrongly imply one exists."
                )
            if clarification_asked:
                _request_system_prompt.set(
                    _request_system_prompt.get()
                    + "\n\nYou already asked this user a clarifying question earlier in this "
                    + "conversation, so you cannot ask an open question again. If several retrieved "
                    + "documents are genuinely the kind asked about, call ask_clarification with "
                    + "those as `options` (a picker is allowed; an empty options list is not). "
                    + "Otherwise work from their reply: answer if the evidence supports it, or reply "
                    + "with the no-information sentence from ground rule 3. Never pick one top-ranked "
                    + "document and state its value as the answer, and never use a number from an "
                    + "unrelated document (a shopping-cart total is not an amount due)."
                )
            initial_tool_choice = route.tool_name
            initial_tool_input_overrides = _initial_tool_input_overrides(
                question, initial_tool_choice
            )

            def remember_hits(hits: List[RetrievedChunk]) -> None:
                # Every returned chunk is a citation candidate; keys inside its text count as
                # evidence, but only the chunk's own key may be cited as a source.
                for h in hits:
                    evidence_issue_keys.add(h.issue_key.upper())
                    source_issue_keys.add(h.issue_key.upper())
                    evidence_issue_keys.update(k.upper() for k in _ISSUE_KEY_RE.findall(h.content))
                    if _citation_identity(h) not in seen_citations:
                        seen_citations.add(_citation_identity(h))
                        citations.append(h)

            async def verified_answer(draft: str) -> AgentAnswer:
                stage("verifying and finalizing the answer")
                # A key from the chat may be named, not cited, once the database confirms it exists
                # in these spaces. Without this, one such mention made the whole follow-up abstain
                #. History comes from the client, so
                # it is checked, not trusted.
                from_chat = unfound_issue_keys(draft, evidence_issue_keys) & chat_issue_keys
                confirmed_chat_keys = (
                    await self._existing_issue_keys(from_chat, space_ids) if from_chat else set()
                )
                text, selected_citations, verify_usage = await self._apply_verification(
                    draft, messages, citations, evidence_issue_keys, source_issue_keys,
                    semantic_used, structured_used, space_ids, subject_issue_keys,
                    structured_evidence, confirmed_chat_keys,
                )
                usage = total_usage + verify_usage
                return AgentAnswer(
                    text=text,
                    retrieval_rounds=retrieval_rounds,
                    queries_used=queries_used,
                    citations=selected_citations,
                    structured_evidence=structured_evidence,
                    abstained=ABSTENTION_PHRASE in text,
                    usage=usage,
                    estimated_cost_usd=self._cost(usage, route_usage),
                    model_usage=self._model_usage(usage, route_usage, route),
                    route_source=route.source,
                    route_tool=route.tool_name,
                    stage_seconds=dict(stage_seconds),
                )

            for iteration in range(self._max_iterations + 1):
                # Force the routed tool only on the first decision. A failed call isn't a retrieval
                # round, so keying on retrieval_rounds forced the same broken call every time.
                required_tool = initial_tool_choice if iteration == 0 else None
                stage(
                    "choosing the right data source" if iteration == 0
                    else "reviewing results and deciding whether more retrieval is needed"
                )
                tools = [_SEARCH_TOOL, _ISSUE_QUERY_TOOL, _SPRINT_QUERY_TOOL, _ISSUE_HISTORY_TOOL,
                         _ISSUE_COMMENTS_TOOL, _ISSUE_DETAILS_TOOL, _ISSUE_ATTACHMENTS_TOOL]
                if self._parent_child_tool_enabled:
                    tools.append(_SECTION_TOOL)
                tools.append(_CLARIFICATION_TOOL)
                with llm_step(f"ask.turn_{iteration + 1}", label_outcome=True):
                    turn = await self._timed_llm_call(
                        messages,
                        tools,
                        tool_choice=required_tool,
                        # A turn that ends up calling a tool emits no answer prose to stream (hosted
                        # tool calls are structured, not text); a turn that ends the loop emits the
                        # answer.
                        on_text_delta=token_cb,
                    )
                total_usage = total_usage + turn.usage
                messages.append(turn.assistant_message)

                if turn.stop_reason == "end_turn":
                    # No retrieval at all (every tool failed) means nothing can ground an answer.
                    if retrieval_rounds > 0 and turn.text.strip():
                        return await verified_answer(turn.text)
                    return await verified_answer(ABSTENTION_PHRASE)

                if iteration >= self._max_iterations:
                    # Hit the round cap: force a final answer so the loop stays bounded.
                    messages.append(self._forced_stop_message(turn))
                    with llm_step("ask.forced_final_answer"):
                        final = await self._timed_llm_call(messages, [], on_text_delta=token_cb)
                    total_usage = total_usage + final.usage
                    messages.append(final.assistant_message)
                    # tools=[] should stop tool calls, but not every OpenAI-compatible server
                    # honors it, so don't trust `final.text` blindly.
                    text = final.text if final.stop_reason == "end_turn" and final.text else ABSTENTION_PHRASE
                    # Verify here too: research-heavy answers always end on this branch.
                    return await verified_answer(text)

                tool_results = []
                for call in turn.tool_calls:
                    if required_tool and call.name != required_tool:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": (
                                    f"Wrong tool for this question. Call {required_tool} first; "
                                    f"{call.name} was not executed."
                                ),
                                "is_error": True,
                            }
                        )
                        continue
                    # A failing tool is a failed tool call, not a failed request: return a typed
                    # error so the next turn can try a fallback (it used to surface as "AI request
                    # failed").
                    try:
                        tool_input = dict(call.input)
                        if iteration == 0 and call.name == required_tool:
                            # Code-owned facts win, e.g. "most recently updated" always means
                            # updated_at DESC LIMIT 1.
                            tool_input.update(initial_tool_input_overrides)
                        # Every tool gets `space_ids` from the authenticated caller, never from
                        # the model (see module docstring).
                        stage(_TOOL_STAGE_LABELS.get(call.name, f"calling {call.name}"))
                        if call.name == _ISSUE_QUERY_TOOL["name"]:
                            result_text, result_keys = await self._run_issue_query(tool_input, space_ids)
                            evidence_issue_keys.update(result_keys)
                            source_issue_keys.update(result_keys)
                            structured_used = True
                            structured_evidence.append(result_text)
                            retrieval_rounds += 1
                        elif call.name == _SPRINT_QUERY_TOOL["name"]:
                            result_text = await self._run_sprint_query(tool_input, space_ids)
                            structured_used = True
                            structured_evidence.append(result_text)
                            retrieval_rounds += 1
                        elif call.name == _ISSUE_HISTORY_TOOL["name"]:
                            result_text, result_keys = await self._run_issue_history(tool_input, space_ids)
                            evidence_issue_keys.update(result_keys)
                            source_issue_keys.update(result_keys)
                            structured_used = True
                            structured_evidence.append(result_text)
                            retrieval_rounds += 1
                        elif call.name in _EXACT_READ_TOOL_NAMES:
                            issue_keys_arg = [
                                k for k in tool_input.get("issue_keys", []) if isinstance(k, str)
                            ]
                            # Keyword routes (an explicit source request) stay exact; see
                            # _exact_read_with_fallback for why other routes may fall back.
                            hits, result_text, fell_back = await self._exact_read_with_fallback(
                                call.name, issue_keys_arg, question, space_ids,
                                allow_fallback=route.source != "keyword",
                            )
                            retrieval_rounds += 1
                            if fell_back:
                                queries_used.append(question)
                                retrieval_rounds += 1
                                semantic_used = True
                            else:
                                # A complete fetch by key, not a ranked search, so it doesn't go
                                # through the strict citation gate meant for search. That gate made
                                # the same correct read abstain ~90% of the time (18/20) just
                                # because citation wording varies.
                                structured_used = True
                            remember_hits(hits)
                        elif call.name == _SEARCH_TOOL["name"]:
                            query = tool_input.get("query", "")
                            mode = _coerce_search_mode(tool_input.get("mode"))
                            result_count = _coerce_result_count(
                                tool_input.get("result_count"), self._top_k
                            )
                            queries_used.append(query)
                            hits = await self._timed_retrieval_call(
                                query, space_ids, mode, result_count
                            )
                            # Put chunks containing the user's exact identifier first; the
                            # reranker can prefer a near-identical sibling document.
                            hits = _prioritize_explicit_identifier_hits(hits, question)
                            semantic_used = True
                            retrieval_rounds += 1
                            remember_hits(hits)
                            result_text = _format_hits(hits, self._parent_child_tool_enabled)
                        elif call.name == _SECTION_TOOL["name"] and self._parent_child_tool_enabled:
                            section_ids = [
                                s for s in tool_input.get("section_ids", []) if isinstance(s, str)
                            ][:3]
                            hits = await self._run_sections(section_ids, space_ids)
                            retrieval_rounds += 1
                            remember_hits(hits)
                            result_text = _format_sections(hits, section_ids)
                        elif call.name == _CLARIFICATION_TOOL["name"]:
                            raw_options = tool_input.get("options")
                            valid = [
                                o
                                for o in (raw_options if isinstance(raw_options, list) else [])
                                if isinstance(o, dict) and str(o.get("label") or "").strip()
                            ]
                            options = [
                                ClarificationOption(
                                    label=label,
                                    issue_key=str(o.get("issue_key") or "").strip().upper(),
                                )
                                for o, label in zip(
                                    valid,
                                    _strip_shared_label_suffix(
                                        [str(o["label"]).strip() for o in valid]
                                    ),
                                )
                            ]
                            clarification_text = str(tool_input.get("question") or "").strip()
                            # 0 options is fine (an open question). Exactly 1 isn't a choice.
                            if clarification_asked and not options:
                                raise ValueError(
                                    "the user was already asked once: ask_clarification now needs "
                                    "at least 2 real candidate options, or answer instead"
                                )
                            if not clarification_text or len(options) == 1:
                                # Malformed call: return an error result like any bad tool input,
                                # so the model can fix it or just answer.
                                raise ValueError(
                                    "ask_clarification requires a non-empty question and either no "
                                    "options or at least 2"
                                )
                            # No verification: a clarification question makes no claims, and
                            # the citation gate could misread it.
                            return AgentAnswer(
                                text=clarification_text,
                                retrieval_rounds=retrieval_rounds,
                                queries_used=queries_used,
                                citations=citations,
                                structured_evidence=structured_evidence,
                                abstained=False,
                                needs_clarification=True,
                                clarification_options=options,
                                usage=total_usage,
                                estimated_cost_usd=self._cost(total_usage, route_usage),
                                model_usage=self._model_usage(total_usage, route_usage, route),
                                route_source=route.source,
                                route_tool=route.tool_name,
                                stage_seconds=dict(stage_seconds),
                            )
                        else:
                            raise ValueError(f"unknown tool {call.name!r}")
                    except Exception:  # noqa: BLE001 - isolate one failed dependency/tool boundary
                        # Tool names come from the model; bound the metric label to known
                        # names to cap Prometheus cardinality.
                        metric_tool = call.name if call.name in _TOOL_STAGE_LABELS else "unknown"
                        TOOL_FAILURES_TOTAL.labels(metric_tool).inc()
                        logger.exception("agent tool call failed", extra={"tool": call.name})
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": call.id,
                                "content": _tool_failure_result(call.name),
                                "is_error": True,
                            }
                        )
                        continue
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": call.id, "content": result_text}
                    )
                messages.append({"role": "user", "content": tool_results})
                round_cb(
                    retrieval_rounds, list(citations),
                    [{"name": b["name"], "input": b["input"]}
                     for b in turn.assistant_message["content"] if b.get("type") == "tool_use"],
                )

            # Defensive fallback — the loop above normally returns before reaching this point.
            return AgentAnswer(text=ABSTENTION_PHRASE, retrieval_rounds=retrieval_rounds,
                                queries_used=queries_used, citations=citations,
                                structured_evidence=structured_evidence, abstained=True,
                                usage=total_usage,
                                estimated_cost_usd=self._cost(total_usage, route_usage),
                                model_usage=self._model_usage(total_usage, route_usage, route),
                                route_source=route.source,
                                route_tool=route.tool_name,
                                stage_seconds=dict(stage_seconds))
        finally:
            # In `finally` so every exit (answer, abstention, forced stop) is counted — this shows how
            # often corrective retrieval actually fires.
            RETRIEVAL_ROUNDS.observe(retrieval_rounds)
            _request_system_prompt.reset(system_prompt_token)
            _request_space_labels.reset(space_labels_token)
            _stage_seconds.reset(stage_seconds_token)

    async def _select_initial_route(
        self,
        question: str,
        messages: List[dict],
        stage: Callable[[str], None],
    ) -> Tuple[RouteDecision, Usage]:
        """Three-layer route: deterministic fast path, model classifier, semantic fallback."""
        keyword_tool = _initial_tool_choice(question)
        if keyword_tool is not None:
            # Every current keyword route has an explicit structured entity/source cue or asks for
            # an intentional structured set. It never handles source-free document questions.
            source_scope = (
                "corpus_wide"
                if keyword_tool in {_ISSUE_QUERY_TOOL["name"], _SPRINT_QUERY_TOOL["name"]}
                else "identified"
            )
            decision = RouteDecision(keyword_tool, "keyword", "high", source_scope)
            ROUTING_DECISIONS_TOTAL.labels(decision.source, decision.tool_name).inc()
            logger.info(
                "Ask-AI route selected",
                extra={"route_source": decision.source, "route_tool": decision.tool_name},
            )
            return decision, Usage()

        stage("classifying the question's data source")
        usage = Usage()
        try:
            turn = await self._timed_router_call(
                messages,
                [_CAPABILITY_ROUTER_TOOL],
                tool_choice=_CAPABILITY_ROUTER_TOOL["name"],
            )
            usage = turn.usage
            route_calls = [
                call for call in turn.tool_calls
                if call.name == _CAPABILITY_ROUTER_TOOL["name"]
            ]
            if len(route_calls) != 1:
                raise ValueError("capability router did not return exactly one routing decision")
            route_input = route_calls[0].input
            selected = route_input.get("tool")
            confidence = route_input.get("confidence")
            source_scope = route_input.get("source_scope")
            if selected not in _EXECUTABLE_TOOL_NAMES:
                raise ValueError("capability router returned an unknown tool")
            if confidence not in {"high", "medium", "low"}:
                raise ValueError("capability router returned invalid confidence")
            if source_scope not in {"identified", "corpus_wide", "ambiguous_single_source"}:
                raise ValueError("capability router returned invalid source scope")

            # Low-confidence classifications are not allowed to force an arbitrary exact reader.
            # Semantic/hybrid retrieval is the final broad-recall fallback requested by the design.
            if confidence == "low":
                selected = _SEARCH_TOOL["name"]
                source = "fallback"
            else:
                source = "model"
            decision = RouteDecision(selected, source, confidence, source_scope)
        except Exception:  # noqa: BLE001 - routing failure has a safe bounded fallback
            logger.exception("Ask-AI capability routing failed; using semantic fallback")
            decision = RouteDecision(_SEARCH_TOOL["name"], "fallback", "low", "identified")

        ROUTING_DECISIONS_TOTAL.labels(decision.source, decision.tool_name).inc()
        logger.info(
            "Ask-AI route selected",
            extra={
                "route_source": decision.source,
                "route_tool": decision.tool_name,
                "route_confidence": decision.confidence,
                "route_source_scope": decision.source_scope,
            },
        )
        return decision, usage

    async def _timed_llm_call(
        self, messages: List[dict], tools: List[dict], tool_choice: str | None = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ):
        # Normal calls use the task-local prompt set by ask(), so concurrent requests can't overwrite
        # it. A call from outside ask() (the multi-agent final check) renders one with the real time;
        # a fixed placeholder date once made "last week" mean 1969.
        system = _request_system_prompt.get() or render_crag_system_prompt(
            _current_time_for_prompt(), self._parent_child_tool_enabled
        )
        return await self._timed_client_call(
            self._llm,
            system,
            messages,
            tools,
            tool_choice,
            on_text_delta,
        )

    async def _timed_router_call(
        self, messages: List[dict], tools: List[dict], tool_choice: str | None = None,
    ):
        with llm_step("ask.router"):
            return await self._timed_client_call(
                self._router_llm,
                render_capability_router_prompt(),
                messages,
                tools,
                tool_choice,
            )

    @staticmethod
    async def _timed_client_call(
        client: LLMClient,
        system: str,
        messages: List[dict],
        tools: List[dict],
        tool_choice: str | None,
        on_text_delta: Optional[Callable[[str], None]] = None,
    ):
        start = time.perf_counter()
        try:
            return await client.next_turn(
                system, messages, tools, tool_choice=tool_choice, on_text_delta=on_text_delta,
            )
        finally:
            elapsed = time.perf_counter() - start
            LLM_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("llm", elapsed)

    async def _timed_retrieval_call(
        self, query: str, space_ids: Sequence[int], mode: str, limit: int | None = None
    ):
        start = time.perf_counter()
        try:
            return await self._retrieval.search(query, space_ids, limit or self._top_k, mode)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)

    async def _existing_issue_keys(self, keys: set[str], space_ids: Sequence[int]) -> set[str]:
        """Which of `keys` exist in `space_ids`. Empty on failure, so the answer check stays strict."""
        try:
            result = await self._retrieval.query_issues(
                space_ids, {"issue_keys": sorted(keys), "limit": len(keys)}
            )
        except Exception:  # noqa: BLE001 - not confirming a key is the safe outcome
            logger.warning("could not confirm issue keys from the chat", exc_info=True)
            return set()
        return {str(issue.get("issue_key")).upper() for issue in result.issues if issue.get("issue_key")}

    async def _run_issue_query(
        self, model_input: dict, space_ids: Sequence[int]
    ) -> tuple[str, set[str]]:
        # Pass only the whitelisted, model-controllable filter fields downstream — never let the model
        # smuggle in space_ids or arbitrary keys. The retrieval client injects space_ids itself.
        allowed = ("issue_keys", "parent_keys", "issue_types", "statuses", "priorities", "sprint_ids", "sprint_names",
                   "created_after", "created_before", "updated_after", "updated_before", "order_by",
                   "order", "limit")
        filters = {k: model_input[k] for k in allowed if k in model_input and model_input[k] is not None}
        scopes = list(dict.fromkeys(space_ids))
        start = time.perf_counter()
        try:
            if len(scopes) < 2:
                results = [(None, await self._retrieval.query_issues(space_ids, filters))]
            else:
                # One query per workspace. A combined query returns one total, and the model had to
                # split the rows by key prefix itself: it read 47/31 as 42/36
                #.
                per_space = await asyncio.gather(
                    *(self._retrieval.query_issues([scope], filters) for scope in scopes)
                )
                results = list(zip(scopes, per_space))
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        keys: set[str] = set()
        for _, result in results:
            for issue in result.issues:
                if issue.get("issue_key"):
                    keys.add(str(issue.get("issue_key")).upper())
                # parent_key is in the tool result text, so a cited parent_key is grounded like any key.
                if issue.get("parent_key"):
                    keys.add(str(issue.get("parent_key")).upper())
        if len(results) == 1:
            return _format_issue_query(results[0][1], filters), keys
        sections = []
        for scope, result in results:
            label = _workspace_label(scope, result.issues)
            sections.append((label, result.total_count, _format_issue_query(result, filters)))
        return _format_per_workspace(sections), keys

    async def _run_sprint_query(self, model_input: dict, space_ids: Sequence[int]) -> str:
        allowed = ("statuses", "order_by", "order", "limit")
        filters = {k: model_input[k] for k in allowed if k in model_input and model_input[k] is not None}
        scopes = list(dict.fromkeys(space_ids))
        start = time.perf_counter()
        try:
            if len(scopes) < 2:
                results = [(None, await self._retrieval.query_sprints(space_ids, filters))]
            else:
                # One query per workspace, for the same reason as _run_issue_query.
                per_space = await asyncio.gather(
                    *(self._retrieval.query_sprints([scope], filters) for scope in scopes)
                )
                results = list(zip(scopes, per_space))
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        if len(results) == 1:
            return _format_sprint_query(results[0][1], filters)
        sections = []
        for scope, result in results:
            sections.append(
                (_workspace_label(scope, []), result.total_count, _format_sprint_query(result, filters))
            )
        return _format_per_workspace(sections)

    async def _run_issue_history(
        self, model_input: dict, space_ids: Sequence[int]
    ) -> tuple[str, set[str]]:
        # Same whitelist discipline as _run_issue_query: only these fields ever reach downstream,
        # space_ids is injected by the client from the authenticated caller.
        allowed = ("issue_keys", "fields", "event_types", "since", "until", "reopened_only", "limit")
        filters = {k: model_input[k] for k in allowed if k in model_input and model_input[k] is not None}
        start = time.perf_counter()
        try:
            result = await self._retrieval.query_issue_history(space_ids, filters)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        keys = {
            str(change.get("issue_key")).upper()
            for change in result.changes
            if change.get("issue_key")
        }
        return _format_issue_history(result, filters), keys

    async def _exact_read_with_fallback(
        self, tool_name: str, issue_keys: List[str], question: str, space_ids: Sequence[int],
        allow_fallback: bool,
    ) -> Tuple[List[RetrievedChunk], str, bool]:
        """Read comments/details/attachments by issue key. Returns (hits, tool result text, whether
        the semantic fallback ran).

        The model may mistake a document's own id (INC-412, an invoice number) for a Jira key. If the
        exact read comes back empty, retry the user's wording through semantic search.
        """
        if tool_name == _ISSUE_COMMENTS_TOOL["name"]:
            hits = await self._run_issue_comments(issue_keys, space_ids)
            result_text = _format_issue_comments(hits, issue_keys)
        elif tool_name == _ISSUE_DETAILS_TOOL["name"]:
            hits = await self._run_issue_details(issue_keys, space_ids)
            result_text = _format_issue_details(hits, issue_keys)
        else:
            hits = await self._run_issue_attachments(issue_keys, space_ids)
            result_text = _format_issue_attachments(hits, issue_keys)

        if hits or not allow_fallback:
            return hits, result_text, False

        hits = await self._timed_retrieval_call(question, space_ids, "hybrid")
        hits = _prioritize_explicit_identifier_hits(hits, question)
        result_text += (
            "\n\nAutomatic semantic fallback for a possible document-internal identifier:\n"
            + _format_hits(hits, self._parent_child_tool_enabled)
        )
        return hits, result_text, True

    async def _run_issue_comments(
        self, issue_keys: Sequence[str], space_ids: Sequence[int]
    ) -> List[RetrievedChunk]:
        """Complete, unranked comment fetch, returned as RetrievedChunks so it goes through the same
        citation/grounding path as search hits."""
        if not issue_keys:
            return []
        start = time.perf_counter()
        try:
            result = await self._retrieval.get_issue_comments(space_ids, issue_keys)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        return [
            RetrievedChunk(
                id=f"comment:{c.get('source_id')}",
                chunk_type="comment",
                issue_id=c.get("issue_id", 0),
                issue_key=str(c.get("issue_key", "")),
                source_id=c.get("source_id", 0),
                content=c.get("content", ""),
                score=0.0,
                retrievers=["issue_comments"],
                provenance=c.get("provenance") or {},
            )
            for c in result.comments
        ]

    async def _run_issue_details(
        self, issue_keys: Sequence[str], space_ids: Sequence[int]
    ) -> List[RetrievedChunk]:
        """Complete, unranked issue title+description fetch, returned as RetrievedChunks (same
        citation/grounding path as search hits)."""
        if not issue_keys:
            return []
        start = time.perf_counter()
        try:
            result = await self._retrieval.get_issue_details(space_ids, issue_keys)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        return [
            RetrievedChunk(
                id=f"issue_details:{d.get('source_id')}",
                chunk_type="issue",
                issue_id=d.get("issue_id", 0),
                issue_key=str(d.get("issue_key", "")),
                source_id=d.get("source_id", 0),
                content=d.get("content", ""),
                score=0.0,
                retrievers=["issue_details"],
            )
            for d in result.details
        ]

    async def _run_issue_attachments(
        self, issue_keys: Sequence[str], space_ids: Sequence[int]
    ) -> List[RetrievedChunk]:
        """Complete, unranked attachment-text fetch, returned as RetrievedChunks (same
        citation/grounding path as search hits)."""
        if not issue_keys:
            return []
        start = time.perf_counter()
        try:
            result = await self._retrieval.get_issue_attachments(space_ids, issue_keys)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        return [
            RetrievedChunk(
                # Include page_number: one multi-page attachment shares a source_id, and citations
                # are deduped by id, so pages 2+ were dropped (ATC-46's numbers on page 4 always
                # showed as page 1).
                id=f"issue_attachments:{a.get('source_id')}:{a.get('page_number')}",
                chunk_type="attachment",
                issue_id=a.get("issue_id", 0),
                issue_key=str(a.get("issue_key", "")),
                source_id=a.get("source_id", 0),
                content=a.get("content", ""),
                score=0.0,
                retrievers=["issue_attachments"],
                page_number=a.get("page_number"),
                provenance=a.get("provenance") or {},
            )
            for a in result.attachments
        ]

    async def _run_sections(
        self, section_ids: Sequence[str], space_ids: Sequence[int]
    ) -> List[RetrievedChunk]:
        """Whole parent sections, returned as RetrievedChunks (same citation path as search hits)."""
        if not section_ids:
            return []
        start = time.perf_counter()
        try:
            sections = await self._retrieval.get_sections(space_ids, section_ids)
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        hits = []
        for s in sections:
            provenance = {"source_type": "pdf"}
            if s.get("page_number") is not None:
                provenance["page_number"] = s["page_number"]
            if s.get("page_end") not in (None, s.get("page_number")):
                provenance["page_end"] = s["page_end"]
            hits.append(
                RetrievedChunk(
                    id=str(s.get("id")),
                    chunk_type="attachment",
                    issue_id=s.get("issue_id", 0),
                    issue_key=str(s.get("issue_key", "")),
                    source_id=s.get("source_id", 0),
                    content=s.get("content", ""),
                    score=0.0,
                    retrievers=["section"],
                    page_number=s.get("page_number"),
                    provenance=provenance,
                )
            )
        return hits

    async def _apply_verification(
        self,
        text: str,
        messages: List[dict],
        citations: List[RetrievedChunk],
        evidence_issue_keys: set[str],
        source_issue_keys: set[str],
        semantic_used: bool,
        structured_used: bool,
        space_ids: Sequence[int],
        subject_issue_keys: set[str],
        structured_evidence: List[str] = (),
        confirmed_chat_keys: set[str] = frozenset(),
    ) -> Tuple[str, List[RetrievedChunk], Usage]:
        """Ground the answer, then run every verifier once.

        Runs outside the tool loop's iteration budget, so verification always gets its turn — before,
        answers that used the whole budget skipped it. The correction is one direct LLM call with no
        tools: fixing a wrong claim is rewording, not more research.
        """
        text, selected_citations = _ground_answer(
            text, citations, evidence_issue_keys, source_issue_keys, semantic_used, structured_used,
            subject_issue_keys, confirmed_chat_keys,
        )
        usage = Usage()
        if ABSTENTION_PHRASE in text:
            return text, selected_citations, usage

        mismatches: List[ClaimMismatch] = []
        for verifier in self._post_generation_verifiers:
            try:
                found, verifier_usage = await verifier(text, space_ids, subject_issue_keys)
            except Exception:  # noqa: BLE001 - an optional verifier must not erase a grounded answer
                verifier_name = getattr(verifier, "__name__", "unknown_verifier")
                TOOL_FAILURES_TOTAL.labels(verifier_name).inc()
                logger.exception(
                    "post-generation verifier failed; preserving grounded answer",
                    extra={"verifier": verifier_name},
                )
                continue
            mismatches.extend(found)
            usage = usage + verifier_usage

        # Optional faithfulness check (off by default; see config). Separate from the loop above because
        # it needs the retrieved context passed in.
        if self._faithfulness_check_enabled:
            try:
                found, faith_usage = await self._check_faithfulness(
                    text, selected_citations, structured_evidence
                )
            except Exception:  # noqa: BLE001 - same availability contract as structured verifiers
                TOOL_FAILURES_TOTAL.labels("faithfulness_verifier").inc()
                logger.exception("faithfulness verifier failed; preserving grounded answer")
            else:
                mismatches.extend(found)
                usage = usage + faith_usage

        if not mismatches:
            return text, selected_citations, usage

        messages.append(_verification_correction_message(mismatches))
        with llm_step("ask.verify.rewrite"):
            corrected = await self._timed_llm_call(messages, [])
        usage = usage + corrected.usage
        corrected_text = (
            corrected.text if corrected.stop_reason == "end_turn" and corrected.text else ABSTENTION_PHRASE
        )
        corrected_text, selected_citations = _ground_answer(
            corrected_text, citations, evidence_issue_keys, source_issue_keys, semantic_used,
            structured_used, subject_issue_keys, confirmed_chat_keys,
        )
        return corrected_text, selected_citations, usage

    async def _check_faithfulness(
        self, text: str, citations: List[RetrievedChunk], structured_evidence: List[str]
    ) -> Tuple[List[ClaimMismatch], Usage]:
        """Live faithfulness check: are this answer's claims supported by what was retrieved?

        RAGAS only measures this offline on a fixed set; this can catch a hallucination before the
        user sees it. Corrects once, like the other verifiers.
        """
        context = "\n\n".join([c.content for c in citations] + list(structured_evidence))
        if not context.strip():
            # No evidence to check against (abstentions are filtered out earlier).
            return [], Usage()

        check_system = (
            "You are a strict fact-checking assistant. You only ever answer in the exact format "
            "requested."
        )
        check_user = (
            "Context (retrieved evidence):\n"
            f'"""\n{context}\n"""\n\n'
            "Answer to check:\n"
            f'"""\n{text}\n"""\n\n'
            "Does the answer contain any factual claim that is NOT supported by the context above? "
            "Ignore phrasing/style differences — only flag claims the context does not actually "
            "back up. If every claim is supported, respond with exactly: SUPPORTED\n"
            "Otherwise respond with exactly one line per unsupported claim, each starting with "
            '"UNSUPPORTED: " followed by the specific claim.'
        )
        start = time.perf_counter()
        try:
            with llm_step("ask.verify.faithfulness"):
                check = await self._llm.next_turn(
                    check_system, [{"role": "user", "content": check_user}], [],
                )
        finally:
            elapsed = time.perf_counter() - start
            LLM_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("llm", elapsed)

        lines = [ln.strip() for ln in check.text.splitlines() if ln.strip()]
        unsupported = [ln[len("UNSUPPORTED:"):].strip() for ln in lines if ln.upper().startswith("UNSUPPORTED:")]
        if not unsupported:
            return [], check.usage
        mismatches = [
            ClaimMismatch(
                issue_key="(faithfulness)",
                detail=f"Not supported by retrieved evidence: {claim}",
            )
            for claim in unsupported
        ]
        return mismatches, check.usage

    async def _check_subtask_claims(
        self, text: str, space_ids: Sequence[int], subject_keys: set[str]
    ) -> Tuple[List[ClaimMismatch], Usage]:
        """Check whether differently typed issues were falsely presented as equivalent subtasks."""
        all_keys = {k.upper() for k in _ISSUE_KEY_RE.findall(text)} - subject_keys
        if len(all_keys) < 2:
            return [], Usage()
        start = time.perf_counter()
        try:
            result = await self._retrieval.query_issues(
                space_ids, {"issue_keys": sorted(all_keys)}
            )
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        by_key = {
            str(issue.get("issue_key", "")).upper(): issue
            for issue in result.issues
        }
        subtask_keys = {
            key
            for key in all_keys
            if by_key.get(key, {}).get("issue_type") == "subtask"
        }
        real_parent_keys = {
            str(by_key[key].get("parent_key", "")).upper()
            for key in subtask_keys
            if by_key.get(key, {}).get("parent_key")
        }
        other_keys = all_keys - subtask_keys - real_parent_keys
        if not subtask_keys or not other_keys:
            return [], Usage()

        classification_system = (
            "You are a precise text-classification assistant. You only ever answer with exactly one "
            "word."
        )
        classification_user = (
            f"An answer text mentions these issues together: {', '.join(sorted(subtask_keys))} (real "
            f"subtasks) and {', '.join(sorted(other_keys))} (a DIFFERENT issue_type — NOT a subtask "
            "of the same parent).\n\n"
            "Does the text present the second group as if it were the same kind of thing as the real "
            "subtasks — e.g. an equivalent item in the same split/list/group — rather than clearly "
            "distinguishing it as a separate, different kind of issue?\n\n"
            f'Text:\n"""\n{text}\n"""\n\n'
            "Answer with exactly one word: EQUIVALENT or DISTINGUISHED."
        )
        start = time.perf_counter()
        try:
            with llm_step("ask.verify.subtask_claims"):
                classification = await self._llm.next_turn(
                    classification_system,
                    [{"role": "user", "content": classification_user}],
                    [],
                )
        finally:
            elapsed = time.perf_counter() - start
            LLM_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("llm", elapsed)
        if "DISTINGUISHED" in classification.text.upper():
            return [], classification.usage

        mismatches = [
            ClaimMismatch(
                issue_key=key,
                detail=(
                    f"{key}: actual issue_type is "
                    f"{by_key.get(key, {}).get('issue_type') or 'unknown/not found'!r}, "
                    f"different from the real subtasks ({', '.join(sorted(subtask_keys))}) it was "
                    "presented alongside — clarify it is a separate, related issue, not an "
                    "equivalent split-off item."
                ),
            )
            for key in sorted(other_keys)
        ]
        return mismatches, classification.usage

    async def _check_current_state_claims(
        self, text: str, space_ids: Sequence[int], subject_keys: set[str]
    ) -> Tuple[List[ClaimMismatch], Usage]:
        """Check whether the answer states a CURRENT status that contradicts query_issues (ground rule
        4: current vs historical state; an older comment once won over the closing comment).

        The regex only skips the check when no status word appears; it's not the detector.
        """
        if not _STATUS_HINT_RE.search(text):
            return [], Usage()
        mentioned_keys = {k.upper() for k in _ISSUE_KEY_RE.findall(text)}
        if not mentioned_keys:
            return [], Usage()

        start = time.perf_counter()
        try:
            result = await self._retrieval.query_issues(
                space_ids, {"issue_keys": sorted(mentioned_keys)}
            )
        finally:
            elapsed = time.perf_counter() - start
            RETRIEVAL_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("retrieval", elapsed)
        known = {
            str(issue.get("issue_key", "")).upper(): issue
            for issue in result.issues
            if issue.get("status")
        }
        if not known:
            return [], Usage()

        facts = "\n".join(f"- {key}: current status = {value['status']!r}" for key, value in sorted(known.items()))
        classification_system = (
            "You are a precise fact-checking assistant. You only ever answer with a JSON array."
        )
        classification_user = (
            "Actual current status of these issues, from structured data (authoritative — trust "
            f"this over the text below):\n{facts}\n\n"
            f'Answer text:\n"""\n{text}\n"""\n\n'
            "For each issue key above, does the answer text assert a CURRENT status/state for it "
            '(e.g. "is done", "still open", "currently blocked") that CONTRADICTS its actual current '
            "status? Ignore mentions of PAST/historical state that the text itself frames as history "
            '(e.g. "was originally blocked, then fixed last sprint") — only flag a contradiction '
            "about what the issue IS right now.\n\n"
            'Answer with a JSON array of the contradicting issue keys only, e.g. ["ATC-43"], or [] '
            "if none contradict."
        )
        start = time.perf_counter()
        try:
            with llm_step("ask.verify.current_state"):
                classification = await self._llm.next_turn(
                    classification_system,
                    [{"role": "user", "content": classification_user}],
                    [],
                )
        finally:
            elapsed = time.perf_counter() - start
            LLM_CALL_SECONDS.observe(elapsed)
            _add_stage_seconds("llm", elapsed)

        contradicted = _parse_json_key_list(classification.text) & known.keys()
        if not contradicted:
            return [], classification.usage

        mismatches = [
            ClaimMismatch(
                issue_key=key,
                detail=(
                    f"{key}: actual current status is {known[key]['status']!r} — the answer's claim "
                    "about its current status/state contradicts this; correct it (mentioning a past/"
                    "historical state is fine as long as it is clearly framed as history, not as the "
                    "current state)."
                ),
            )
            for key in sorted(contradicted)
        ]
        return mismatches, classification.usage

    def _cost(self, usage: Usage, router_usage: Usage | None = None) -> float:
        """Price main-agent and router tokens against the models that actually answered them."""
        routed = router_usage or Usage()
        main = self._subtract_usage(usage, routed)
        return sum(
            self._usage_cost(model, part, deployment)
            for configured, part_usage, deployment in (
                (self._model, main, self._model_deployment),
                (self._router_model, routed, self._router_deployment),
            )
            for model, part in self._split_by_served_model(configured, part_usage)
        )

    def _model_usage(
        self, usage: Usage, router_usage: Usage, route: RouteDecision
    ) -> List[ModelUsage]:
        main = self._subtract_usage(usage, router_usage)
        entries: List[ModelUsage] = []
        if route.source != "keyword" or router_usage.input_tokens or router_usage.output_tokens:
            entries += self._served_entries(
                "router", self._router_provider, self._router_model, self._router_deployment, router_usage,
            )
        entries += self._served_entries(
            "answer_agent", self._model_provider, self._model, self._model_deployment, main,
        )
        return entries

    def _served_entries(
        self, role: str, provider: str, model: str, deployment: str, usage: Usage
    ) -> List[ModelUsage]:
        """One entry for the configured model, plus one per gateway fallback model that answered."""
        parts = self._split_by_served_model(model, usage)
        configured_model, configured_usage = parts[0]
        entries = []
        if len(parts) == 1 or configured_usage.input_tokens or configured_usage.output_tokens:
            entries.append(self._model_usage_entry(role, provider, configured_model, deployment, configured_usage))
        for served, served_usage in parts[1:]:
            entries.append(self._model_usage_entry(role, f"{provider} (fallback)", served, deployment, served_usage))
        return entries

    @staticmethod
    def _split_by_served_model(model: str, usage: Usage) -> List[Tuple[str, Usage]]:
        """[(configured model, its tokens), (fallback model, its tokens), ...]."""
        rest = Usage(
            usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        )
        for served_usage in usage.fallback.values():
            rest = CragAgent._subtract_usage(rest, served_usage)
        return [(model, rest), *usage.fallback.items()]

    @staticmethod
    def _subtract_usage(usage: Usage, excluded: Usage) -> Usage:
        fallback = {}
        for model, served in usage.fallback.items():
            remaining = CragAgent._subtract_usage(served, excluded.fallback.get(model, Usage()))
            if remaining.input_tokens or remaining.output_tokens:
                fallback[model] = remaining
        return Usage(
            input_tokens=max(0, usage.input_tokens - excluded.input_tokens),
            output_tokens=max(0, usage.output_tokens - excluded.output_tokens),
            cache_creation_input_tokens=max(
                0, usage.cache_creation_input_tokens - excluded.cache_creation_input_tokens
            ),
            cache_read_input_tokens=max(
                0, usage.cache_read_input_tokens - excluded.cache_read_input_tokens
            ),
            fallback=fallback,
        )

    @staticmethod
    def _usage_cost(model: str, usage: Usage, deployment: str) -> float:
        if deployment == "local":
            return 0.0
        return estimate_cost_usd(
            model, usage.input_tokens, usage.output_tokens,
            usage.cache_creation_input_tokens, usage.cache_read_input_tokens,
        )

    def _model_usage_entry(
        self, role: str, provider: str, model: str, deployment: str, usage: Usage
    ) -> ModelUsage:
        if deployment == "local":
            pricing_status = "free_local"
        elif has_known_price(model):
            pricing_status = "estimated"
        else:
            pricing_status = "unavailable"
        return ModelUsage(
            role=role,
            provider=provider,
            model=model,
            deployment=deployment,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=self._usage_cost(model, usage, deployment),
            pricing_status=pricing_status,
            cached_input_tokens=usage.cache_read_input_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens,
        )

    @staticmethod
    def _forced_stop_message(turn) -> dict:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": "Retrieval budget exhausted. Answer now with what you have, or state "
                                f'"{ABSTENTION_PHRASE}" if it is not enough.',
                }
                for call in turn.tool_calls
            ],
        }


def _tool_failure_result(tool_name: str) -> str:
    """Safe hint for the model after a tool fails. Never includes the exception text (that goes to
    logs); tool-specific so the next turn picks a useful fallback.
    """
    alternatives = {
        "search_knowledge_base": (
            "For counts, lists, status, ownership, or recency, call query_issues. If an issue key is "
            "already known, use get_issue_details, get_issue_comments, or get_issue_attachments."
        ),
        "query_issues": (
            "Do not replace exact metadata with semantic-search guesses. State that structured issue "
            "data is temporarily unavailable unless another exact tool can answer the question."
        ),
        "query_sprints": (
            "Do not guess sprint metadata. State that structured sprint data is temporarily unavailable."
        ),
        "get_issue_history": "Do not infer history from current state or semantic similarity.",
        "get_issue_comments": "Try get_issue_details if the issue's own body could answer the question.",
        "get_issue_details": "Try get_issue_comments or get_issue_attachments when applicable.",
        "get_issue_attachments": (
            "The attachment may still be ingesting; do not claim its content is absent based on this failure."
        ),
        "get_section": "Answer from the search results you already have, or try get_issue_attachments.",
    }
    guidance = alternatives.get(tool_name, "Choose a different available evidence source if one applies.")
    return (
        f"Tool {tool_name} is temporarily unavailable and returned no evidence. {guidance} "
        "Never answer the user's factual question from memory."
    )


def _format_hits(hits: List[RetrievedChunk], show_sections: bool = True) -> str:
    if not hits:
        return "No results found."
    return "\n\n".join(
        f"[{h.issue_key}] (chunk_type={h.chunk_type}, retrievers={h.retrievers}"
        f"{f', page={h.page_number}' if h.page_number is not None else ''}"
        f"{_section_ids_label(h) if show_sections else ''})\n{h.content}"
        for h in hits
    )



def _section_ids_label(hit: RetrievedChunk) -> str:
    section_ids = (hit.provenance or {}).get("section_ids") or []
    return f", section_ids={','.join(section_ids)}" if section_ids else ""


def _format_sections(hits: List[RetrievedChunk], requested_ids: List[str]) -> str:
    if not hits:
        return (
            f"No section found for {requested_ids or 'the requested ids'}. Use only ids shown as "
            "section_ids on a search result; answer from the results you already have."
        )
    return "\n\n".join(f"[{h.issue_key}] (whole section {h.id})\n{h.content}" for h in hits)


_ISSUE_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]*-\d+\b", re.IGNORECASE)
_TRAILING_CITATION_RE = re.compile(
    r"\((?P<keys>[A-Z][A-Z0-9]*-\d+(?:\s*,\s*[A-Z][A-Z0-9]*-\d+)*)\)\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Gate for _check_current_state_claims: the known statuses plus common synonyms.
_STATUS_HINT_RE = re.compile(
    r"\b(done|resolved|closed|fixed|complete|completed|in[- ]?progress|ongoing|open|unresolved|"
    r"blocked|planned|active|reopened)\b",
    re.IGNORECASE,
)


def _parse_json_key_list(value: str) -> set[str]:
    match = re.search(r"\[.*\]", value, re.DOTALL)
    if not match:
        return set()
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return set()
    return {str(k).upper() for k in parsed if isinstance(k, str)}


def _verification_correction_message(mismatches: List["ClaimMismatch"]) -> dict:
    lines = [f"- {m.detail}" for m in mismatches]
    return {
        "role": "user",
        "content": (
            "Correction needed before this answer is final — structured data contradicts part of "
            "it:\n" + "\n".join(lines) +
            "\nRevise your answer to remove or correctly characterize each of these, then give ONLY "
            "the corrected answer text, exactly as it should be shown to the user. Do not mention "
            "this correction, do not say \"you're right\" or similar, and do not refer to a previous "
            "version of the answer — the user never saw it and has no idea a revision happened."
        ),
    }


_WORD_RE = re.compile(r"[a-z']{4,}")
# Common enough to appear in almost any sentence regardless of topic — excluded so the overlap score
# reflects shared SUBJECT MATTER, not shared grammar.
_STOPWORDS = frozenset({
    "this", "that", "with", "have", "been", "from", "were", "will", "would", "could", "should",
    "their", "there", "which", "when", "what", "into", "before", "after", "because", "then", "than",
    "only", "these", "those", "about", "while", "until", "each", "does", "being", "over",
})
def _citation_words(value: str) -> set[str]:
    return {word for word in _WORD_RE.findall(value.lower()) if word not in _STOPWORDS}


def _content_overlap_score(content: str, answer_text: str) -> int:
    """How many distinctive words a chunk shares with the answer — the main key for picking which
    chunk to cite for an issue. A cheap proxy for "does this chunk support what was said".
    """
    content_words = _citation_words(content)
    answer_words = _citation_words(answer_text)
    return len(content_words & answer_words)


def unfound_issue_keys(text: str, evidence_issue_keys: set[str]) -> set[str]:
    """Issue keys in `text` from a project the tools searched that no tool actually returned.

    Keys with another prefix (INV-3300, an order number) aren't issue references and are ignored.
    """
    evidence_prefixes = {key.rsplit("-", 1)[0] for key in evidence_issue_keys}
    referenced = {key.upper() for key in _ISSUE_KEY_RE.findall(text)}
    return {key for key in referenced if key.rsplit("-", 1)[0] in evidence_prefixes} - evidence_issue_keys


def _ground_answer(
    text: str,
    citations: List[RetrievedChunk],
    evidence_issue_keys: set[str],
    source_issue_keys: set[str],
    semantic_used: bool,
    structured_used: bool,
    subject_issue_keys: set[str] | None = None,
    confirmed_chat_keys: set[str] = frozenset(),
) -> tuple[str, List[RetrievedChunk]]:
    """Reject unsupported issue references and return only the sources the answer actually cites.

    Retrieved chunks are candidates, not citations. Every issue key in the answer must come from a
    real tool result, semantic-only answers must cite at least one, and sources are one chunk per
    cited issue.

    `confirmed_chat_keys`: keys from earlier turns that exist in the caller's spaces. The answer may
    name them, but only the second check lets them through; as a source they still need this turn's
    results.
    """
    if ABSTENTION_PHRASE in text:
        return ABSTENTION_PHRASE, []

    referenced_keys = {key.upper() for key in _ISSUE_KEY_RE.findall(text)}
    trailing_citation = _TRAILING_CITATION_RE.search(text)
    claimed_source_keys = (
        {key.upper() for key in _ISSUE_KEY_RE.findall(trailing_citation.group("keys"))}
        if trailing_citation else set()
    )
    citation_keys = {citation.issue_key.upper() for citation in citations}
    normalized_subject_keys = {key.upper() for key in (subject_issue_keys or set())}
    implicit_source = resolve_implicit_source(normalized_subject_keys, citations)

    # A citation claims provenance, so check it against keys actually returned this turn, whatever the
    # prefix. (The prefix-aware check below ignores other prefixes, which let a fake `(ATC-123)` pass.)
    unsupported_claimed_keys = claimed_source_keys - source_issue_keys
    # The final "(KEY)" can close a list item rather than cite the answer: "...regression coverage
    # (ATC-47)", where ATC-47 is a subtask named in ATC-43's retrieved comment. Named in tool output
    # and in the same project as a returned issue, it is grounded, not fabricated; this turned 1 in 9
    # correct ATC-43 answers into abstentions. A different prefix (a document id like HLT-439) still
    # goes through the carrier replacement below.
    source_prefixes = {key.rsplit("-", 1)[0] for key in source_issue_keys}
    unsupported_claimed_keys = {
        key for key in unsupported_claimed_keys
        if key not in evidence_issue_keys or key.rsplit("-", 1)[0] not in source_prefixes
    }
    if unsupported_claimed_keys:
        # The model may cite the document's own id, e.g. `(HLT-439)`, instead of the carrier issue
        # `DVQPLF2-7`. Replace it only if that id maps to exactly one retrieved carrier; otherwise fail
        # closed.
        matched_identifiers = set(implicit_source.matched_identifiers)
        if (
            trailing_citation
            and implicit_source.source_key
            and unsupported_claimed_keys <= normalized_subject_keys
            and unsupported_claimed_keys <= matched_identifiers
        ):
            text = replace_trailing_citation(text, implicit_source.source_key, trailing_citation)
            referenced_keys = {key.upper() for key in _ISSUE_KEY_RE.findall(text)}
            trailing_citation = _TRAILING_CITATION_RE.search(text)
            claimed_source_keys = {implicit_source.source_key}
        else:
            logger.warning(
                "rejecting answer with fabricated source citations",
                extra={"unsupported_source_keys": sorted(unsupported_claimed_keys)},
            )
            return ABSTENTION_PHRASE, []

    # If the user named exactly one issue/attachment and the answer used it but forgot the `(KEY)`
    # suffix, attach the citation instead of abstaining. Seen live: 5/10 correct attachment answers
    # omitted it and were turned into abstentions.
    # Safe: the key must be in the question AND returned this turn, and it's never picked among several.
    # Business ids (INV-3300, INC-429) aren't treated as citations unless they're a retrieved key.
    # Same for an answer that names the attachment's file instead of its issue ("(PRA-SS2-21-Outsourcing.pdf,
    # §9.9)"): the file belongs to one retrieved issue. Seen live: 6/6 correct answers to one question
    # abstained this way, with the file name or a paragraph number as the only citation.
    scoped_retrieved_keys = (subject_issue_keys or set()) & citation_keys
    referenced_source_keys = referenced_keys & citation_keys
    file_source_key = (
        carrier_named_by_file(text, citations)
        if not referenced_source_keys and not trailing_citation else None
    )
    if (
        (semantic_used or structured_used)
        and not referenced_source_keys
        and not trailing_citation
        and (len(scoped_retrieved_keys) == 1 or implicit_source.source_key or file_source_key)
    ):
        # Pure UX (attach a proven single source), so structured reads get it too — not a judgement
        # about trust. See test_single_explicit_attachment_source_is_cited_when_natural_answer_omits_suffix.
        if len(scoped_retrieved_keys) == 1:
            source_key = next(iter(scoped_retrieved_keys))
        elif implicit_source.source_key:
            source_key = implicit_source.source_key
        else:
            source_key = file_source_key
        text = append_source_citation(text, source_key)
        referenced_keys.add(source_key)
        referenced_source_keys.add(source_key)
    unfound_keys = unfound_issue_keys(text, evidence_issue_keys | confirmed_chat_keys)
    if unfound_keys:
        logger.warning(
            "rejecting answer with unsupported issue references",
            extra={"unsupported_issue_keys": sorted(unfound_keys)},
        )
        return ABSTENTION_PHRASE, []

    text_upper = text.upper()
    mentioned_citation_keys = {
        citation.issue_key.upper()
        for citation in citations
        if citation.issue_key.upper() in text_upper
    }

    # A semantic-only answer with no explicit reference to what it retrieved violates the prompt's
    # citation contract. Structured counts/sprint stats are self-grounding and do not need one.
    unverifiable = not citations or not (referenced_source_keys or mentioned_citation_keys)
    if unverifiable and semantic_used and not structured_used:
        return ABSTENTION_PHRASE, []
    # Exact reads (attachments/comments/details) are exempt only if their result was unambiguous —
    # the same test the auto-attach above uses. If that couldn't resolve a citation (e.g. the result
    # belongs to a different issue than asked), fail closed. Exempting all structured reads fixed the
    # 90% false abstentions but also accepted wrong-issue answers; this keeps both fixes.
    # query_issues/query_sprints/get_issue_history have no `citations`, so they stay exempt.
    if unverifiable and structured_used and citations and not (
        len(scoped_retrieved_keys) == 1 or implicit_source.source_key
    ):
        return ABSTENTION_PHRASE, []

    # Group candidates by issue key and pick the chunk that best matches the answer. Taking the first
    # chunk once cited comment #1 (an unrelated estimate note) for an answer drawn from comment #4.
    # Word overlap with the answer is the primary key, score only a tiebreak: scores aren't comparable
    # across tools (a search hit's score vs an exact fetch's flat 0.0), and score-first picked a
    # comment that merely mentioned the key over the chunk the answer actually quoted.
    candidates_by_key: Dict[str, List[RetrievedChunk]] = {}
    key_order: List[str] = []
    for citation in citations:
        key = citation.issue_key.upper()
        if key not in referenced_source_keys and key not in mentioned_citation_keys:
            continue
        if key not in candidates_by_key:
            candidates_by_key[key] = []
            key_order.append(key)
        candidates_by_key[key].append(citation)

    _MAX_CITATIONS_PER_KEY = 5
    selected: List[RetrievedChunk] = []
    for key in key_order:
        ranked = sorted(
            candidates_by_key[key],
            key=lambda citation: (_content_overlap_score(citation.content, text), citation.score),
            reverse=True,
        )
        overlapping = [
            citation
            for citation in ranked
            if _content_overlap_score(citation.content, text) > 0
        ]
        selected.extend(overlapping[:_MAX_CITATIONS_PER_KEY] if overlapping else ranked[:1])
    return text, selected


def _citation_identity(chunk: RetrievedChunk) -> tuple:
    """Search and exact reads name one chunk differently (`issue:5` vs `issue_details:5`), so
    deduplicating by id let the same text fill two of an issue's citation slots."""
    return (chunk.issue_key.upper(), " ".join(chunk.content.split()))


def _prioritize_explicit_identifier_hits(
    hits: Sequence[RetrievedChunk], question: str
) -> List[RetrievedChunk]:
    """Move chunks containing the user's literal identifier ahead of look-alikes (stable).

    Reranking can demote the exact invoice/incident when templated documents look alike. Scores are
    left unchanged; only the order changes.
    """
    identifiers = {key.upper() for key in _ISSUE_KEY_RE.findall(question)}
    if not identifiers:
        return list(hits)

    matched: List[RetrievedChunk] = []
    unmatched: List[RetrievedChunk] = []
    for hit in hits:
        content = hit.content.upper()
        (matched if any(key in content for key in identifiers) else unmatched).append(hit)
    return matched + unmatched


def _format_issue_query(result: IssueQueryResult, filters: dict) -> str:
    """Render an issue-query result for the model: total count first, then breakdowns, then a sample.

    Count first so the model reports the real number, not the sample size. Every IssueRowOut field
    is rendered — this text is the model's only view, and dropping assignee/story_points once made it
    answer "no assignee" for an issue assigned to Ava Patel. Empty values are written out as
    (unassigned) / (unestimated) so they read as answers.
    """
    lines = [f"Structured issue query (filters={filters or 'none - all issues in scope'}):",
             f"total_count = {result.total_count} (exact; this is the number to report for 'how many')"]
    if result.counts_by_type:
        lines.append("by type: " + ", ".join(f"{k}={v}" for k, v in sorted(result.counts_by_type.items())))
    if result.counts_by_status:
        lines.append("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(result.counts_by_status.items())))
    if result.counts_by_assignee:
        ranked = sorted(result.counts_by_assignee.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("by assignee (exact, most first): " + ", ".join(f"{k}={v}" for k, v in ranked))
    if result.issues:
        shown = len(result.issues)
        suffix = "" if shown >= result.total_count else f" (showing first {shown} of {result.total_count})"
        lines.append(f"matching issues{suffix}:")
        for it in result.issues:
            points = it.get("story_points")
            lines.append(
                f"  [{it.get('issue_key')}] type={it.get('issue_type')} status={it.get('status')} "
                f"priority={it.get('priority') or '(none)'} "
                f"assignee={it.get('assignee_name') or '(unassigned)'} "
                f"points={points if points is not None else '(unestimated)'} "
                f"sprint={it.get('sprint_name') or '(backlog)'} "
                f"parent_key={it.get('parent_key') or '(none)'} "
                f"updated={it.get('updated_at')} — {it.get('title')}"
            )
    return "\n".join(lines)


def _workspace_label(space_id: int, issues: Sequence[dict]) -> str:
    """The workspace key ("ATC") from the request's labels, else from an issue key, else the id."""
    label = (_request_space_labels.get() or {}).get(space_id)
    if label:
        return label
    for issue in issues:
        key = str(issue.get("issue_key") or "")
        if "-" in key:
            return key.rsplit("-", 1)[0].upper()
    return f"space {space_id}"


def _format_per_workspace(sections: List[tuple[str, int, str]]) -> str:
    """Join per-workspace query results; the combined total is added in code, not by the model."""
    combined = sum(count for _, count, _ in sections)
    parts = " + ".join(f"{label} {count}" for label, count, _ in sections)
    lines = [
        f"Queried each workspace separately. combined total_count = {combined} ({parts}). "
        "Report each workspace's own numbers from its section; use the combined total only when the "
        "question asks for one."
    ]
    for label, _, text in sections:
        lines.append(f"\n=== Workspace {label} ===\n{text}")
    return "\n".join(lines)


def _format_sprint_query(result: SprintQueryResult, filters: dict) -> str:
    """Render a sprint-query result, count first, with goal/dates/points for velocity questions."""
    lines = [f"Structured sprint query (filters={filters or 'none - all sprints in scope'}):",
             f"total_count = {result.total_count} (exact; this is the number to report for 'how many')"]
    if result.counts_by_status:
        lines.append("by status: " + ", ".join(f"{k}={v}" for k, v in sorted(result.counts_by_status.items())))
    if result.sprints:
        shown = len(result.sprints)
        suffix = "" if shown >= result.total_count else f" (showing first {shown} of {result.total_count})"
        lines.append(f"matching sprints{suffix}:")
        for sp in result.sprints:
            lines.append(
                f"  [{sp.get('sprint_id')}] {sp.get('sprint_name')} status={sp.get('status')} "
                f"dates={sp.get('start_date')}..{sp.get('end_date')} "
                f"points={sp.get('completed_points')}/{sp.get('final_scope_points')} "
                f"(final) or /{sp.get('initial_committed_points')} (committed) — goal: {sp.get('goal')}"
            )
    return "\n".join(lines)


def _format_issue_history(result: IssueHistoryResult, filters: dict) -> str:
    """Render change records: count first, then one line per change. from/to values are included
    (truncated) — they're the answer to "what did I change" and the evidence for reopens/blocks.
    """

    def _trunc(v, n=300):
        if v is None:
            return "(empty)"
        v = str(v)
        return v if len(v) <= n else v[:n] + "…[truncated]"

    lines = [f"Issue change history (filters={filters or 'none - all changes in scope'}):",
             f"total_count = {result.total_count} (exact)"]
    if not result.changes:
        lines.append("No matching changes.")
    shown = len(result.changes)
    if result.changes and shown < result.total_count:
        lines.append(f"showing the {shown} most recent of {result.total_count}:")
    for c in result.changes:
        who = c.get("actor_name") or "system"
        when = c.get("changed_at")
        if c.get("event_type") == "field_change":
            # Include the actor's note (e.g. why it was reopened) — it used to be dropped here. Bare
            # "Updated <Field>" notes carry nothing, so those stay terse.
            desc = c.get("description")
            is_generic = bool(desc) and desc.startswith("Updated ") and " — " not in desc
            suffix = f" — {_trunc(desc)}" if desc and not is_generic else ""
            lines.append(
                f"  [{c.get('issue_key')}] {when} {who} changed {c.get('field_name')}: "
                f"{_trunc(c.get('from_value'))} -> {_trunc(c.get('to_value'))}{suffix}"
            )
        else:
            lines.append(
                f"  [{c.get('issue_key')}] {when} {who}: {c.get('event_type')} — {c.get('description') or ''}"
            )
    return "\n".join(lines)


def _format_issue_comments(hits: List[RetrievedChunk], requested_keys: List[str]) -> str:
    if not hits:
        return (
            f"No comments found for {requested_keys or 'the requested issue(s)'} "
            "(either it has none, or the key doesn't exist in scope). If this key came from the "
            "user's document question, it may be a business identifier printed inside an "
            "attachment rather than a Jira issue key; search_knowledge_base for the original "
            "question before concluding the information is absent."
        )
    # vectorization-service already returns replies right after their top-level comment, in time order.
    authors = {h.source_id: h.provenance.get("author") for h in hits}
    blocks, seen = [], set()
    for h in hits:
        p = h.provenance
        body = _COMMENT_TAG_RE.sub("", h.content, count=1)
        if p.get("parent_id") is not None:
            body = "\n".join("    " + line for line in body.splitlines())
        if h.source_id in seen:  # later chunk of a long comment: no second header
            blocks[-1] += f"\n{body}"
            continue
        seen.add(h.source_id)
        who = ", ".join(x for x in (p.get("author"), (p.get("created_at") or "")[:10]) if x)
        who = f" by {who}" if who else ""
        if p.get("parent_id") is None:
            blocks.append(f"[{h.issue_key}] comment #{h.source_id}{who}\n{body}")
            continue
        target = authors.get(p.get("reply_to_id"))
        to = f" to #{p['reply_to_id']}" + (f" ({target})" if target else "") if p.get("reply_to_id") else ""
        blocks.append(f"    ↳ [{h.issue_key}] reply #{h.source_id}{to}{who}\n{body}")
    return "\n\n".join(blocks)


_COMMENT_TAG_RE = re.compile(r"^\[Comment on [^\]]+\]\n")


def _format_issue_details(hits: List[RetrievedChunk], requested_keys: List[str]) -> str:
    if not hits:
        return (
            f"No issue body found for {requested_keys or 'the requested issue(s)'} "
            "(either the key doesn't exist in scope, or that issue has no title/description text). "
            "If this key came from the user's document question, it may be a business identifier "
            "printed inside an attachment rather than a Jira issue key; search_knowledge_base for "
            "the original question before concluding the information is absent."
        )
    return "\n\n".join(f"[{h.issue_key}] (issue title+description)\n{h.content}" for h in hits)


def _format_issue_attachments(hits: List[RetrievedChunk], requested_keys: List[str]) -> str:
    if not hits:
        return (
            f"No attachment content found for {requested_keys or 'the requested issue(s)'} "
            "(either the key doesn't exist in scope, or that issue has no attachments/none could be "
            "parsed). If this key came from the user's document question, it may be a business "
            "identifier printed inside another attachment rather than a Jira issue key; "
            "search_knowledge_base for the original question before concluding the information is "
            "absent."
        )
    return "\n\n".join(f"[{h.issue_key}] (attachment)\n{h.content}" for h in hits)
