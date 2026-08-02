"""CRAG agent loop control-flow tests — scripted fake LLM + fake retrieval, no network.

These test the loop's *decisions* (does it re-retrieve, does it stop, does it respect the security
boundary on space_ids) independent of any real model's answer quality — that's what eval/ measures.
"""
import asyncio

import pytest

import app.agent.crag_loop as crag_loop
from app.agent.crag_loop import (
    ABSTENTION_PHRASE,
    CragAgent,
    _SEARCH_TOOL,
    _format_issue_history,
    _format_issue_query,
    _has_explicit_source_cue,
    _initial_tool_choice,
    _prioritize_explicit_identifier_hits,
    SEARCH_TOOL_NAME,
    is_answer_cacheable,
)
from app.agent.prompts import render_capability_router_prompt, render_crag_system_prompt
from app.agent.retrieval_tool import (
    IssueAttachmentsResult,
    IssueCommentsResult,
    IssueDetailsResult,
    IssueHistoryResult,
    IssueQueryResult,
    RetrievedChunk,
    SprintQueryResult,
)
from app.llm.types import LLMTurn, ToolCall, Usage


class FakeLLM:
    """Returns each scripted LLMTurn in order, one per call, ignoring the actual input."""

    def __init__(self, turns, router_decisions=None, router_usage=None):
        self._turns = list(turns)
        self._router_decisions = list(router_decisions or [])
        self._router_usage = router_usage or Usage()
        self.calls = 0
        self.messages_seen = []  # the `messages` list passed on each call, for assertions
        self.systems_seen = []
        self.tools_seen = []
        self.router_calls = []

    async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
        if tools and tools[0].get("name") == "select_data_source":
            self.router_calls.append({"messages": list(messages), "tool_choice": tool_choice})
            if self._router_decisions:
                decision = self._router_decisions.pop(0)
                selected, confidence = decision[:2]
                source_scope = decision[2] if len(decision) > 2 else "identified"
            else:
                # Existing control-flow tests script the first executable tool. Infer the router's
                # decision from that script without consuming it; focused router tests below pass an
                # explicit decision instead. Router calls are tracked separately so old assertions
                # about answer-generation call counts retain their meaning.
                next_turn = self._turns[0]
                selected = (
                    next_turn.tool_calls[0].name
                    if next_turn.tool_calls else "search_knowledge_base"
                )
                confidence = "high"
                source_scope = "identified"
            return LLMTurn(
                stop_reason="tool_use",
                text="",
                tool_calls=[ToolCall(
                    id="route_1",
                    name="select_data_source",
                    input={
                        "tool": selected,
                        "confidence": confidence,
                        "source_scope": source_scope,
                    },
                )],
                assistant_message={"role": "assistant", "content": []},
                usage=self._router_usage,
            )
        self.calls += 1
        self.systems_seen.append(system)
        self.tools_seen.append([t.get("name") for t in tools])
        # Snapshot now: `messages` is the loop's own mutable list, appended to after this call
        # returns (e.g. the assistant turn just produced) — capturing by reference would show
        # future mutations, not what the model actually saw at call time.
        self.messages_seen.append(list(messages))
        self.tool_choices_seen = getattr(self, "tool_choices_seen", [])
        self.tool_choices_seen.append(tool_choice)
        turn = self._turns.pop(0)
        # Mirror a real streaming client: emit the answer prose through the delta callback when this
        # turn ends the loop, so tests can assert POST /ask/stream forwards tokens.
        if on_text_delta is not None and turn.stop_reason == "end_turn" and turn.text:
            for word in turn.text.split(" "):
                on_text_delta(word + " ")
        return turn

    async def aclose(self):
        return None


class FakeRetrieval:
    def __init__(
        self, results_by_query=None, default=None, issue_result=None, history_result=None,
        sprint_result=None, comments_result=None, details_result=None, attachments_result=None,
        sections_result=None,
    ):
        self._results_by_query = results_by_query or {}
        self._default = default or []
        self._issue_result = issue_result or IssueQueryResult(
            total_count=0, counts_by_type={}, counts_by_status={}, issues=[]
        )
        self._history_result = history_result or IssueHistoryResult(total_count=0, changes=[])
        self._sprint_result = sprint_result or SprintQueryResult(
            total_count=0, counts_by_status={}, sprints=[]
        )
        self._comments_result = comments_result or IssueCommentsResult(total_count=0, comments=[])
        self._details_result = details_result or IssueDetailsResult(total_count=0, details=[])
        self._attachments_result = attachments_result or IssueAttachmentsResult(total_count=0, attachments=[])
        self.calls = []  # (query, space_ids, limit, mode)
        self.issue_calls = []  # (space_ids, filters)
        self.history_calls = []  # (space_ids, filters)
        self.sprint_calls = []  # (space_ids, filters)
        self.comments_calls = []  # (space_ids, issue_keys)
        self.details_calls = []  # (space_ids, issue_keys)
        self.attachments_calls = []  # (space_ids, issue_keys)
        self._sections_result = sections_result or []
        self.sections_calls = []  # (space_ids, section_ids)

    async def search(self, query, space_ids, limit, mode="hybrid"):
        self.calls.append((query, tuple(space_ids), limit, mode))
        return self._results_by_query.get(query, self._default)

    async def query_issues(self, space_ids, filters):
        self.issue_calls.append((tuple(space_ids), dict(filters)))
        return self._issue_result

    async def query_issue_history(self, space_ids, filters):
        self.history_calls.append((tuple(space_ids), dict(filters)))
        return self._history_result

    async def query_sprints(self, space_ids, filters):
        self.sprint_calls.append((tuple(space_ids), dict(filters)))
        return self._sprint_result

    async def get_issue_comments(self, space_ids, issue_keys, limit=200):
        self.comments_calls.append((tuple(space_ids), tuple(issue_keys)))
        return self._comments_result

    async def get_issue_details(self, space_ids, issue_keys, limit=200):
        self.details_calls.append((tuple(space_ids), tuple(issue_keys)))
        return self._details_result

    async def get_issue_attachments(self, space_ids, issue_keys, limit=200):
        self.attachments_calls.append((tuple(space_ids), tuple(issue_keys)))
        return self._attachments_result

    async def get_sections(self, space_ids, section_ids):
        self.sections_calls.append((tuple(space_ids), tuple(section_ids)))
        return self._sections_result

    async def aclose(self):
        return None


def _issue_query_turn(tool_input: dict, call_id: str = "iq_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="query_issues", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "query_issues", "input": tool_input}
        ]},
    )


def _sprint_query_turn(tool_input: dict, call_id: str = "sq_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="query_sprints", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "query_sprints", "input": tool_input}
        ]},
    )


def _issue_comments_turn(tool_input: dict, call_id: str = "ic_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="get_issue_comments", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "get_issue_comments", "input": tool_input}
        ]},
    )


def _issue_details_turn(tool_input: dict, call_id: str = "id_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="get_issue_details", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "get_issue_details", "input": tool_input}
        ]},
    )


def _issue_attachments_turn(tool_input: dict, call_id: str = "ia_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="get_issue_attachments", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "get_issue_attachments", "input": tool_input}
        ]},
    )


def _section_turn(section_ids: list, call_id: str = "sec_1") -> LLMTurn:
    tool_input = {"section_ids": section_ids}
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="get_section", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "get_section", "input": tool_input}
        ]},
    )


def _tool_use_turn(query: str, call_id: str = "call_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="search_knowledge_base", input={"query": query})],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "search_knowledge_base", "input": {"query": query}}
        ]},
    )


def _end_turn(text: str) -> LLMTurn:
    return LLMTurn(
        stop_reason="end_turn", text=text, tool_calls=[],
        assistant_message={"role": "assistant", "content": [{"type": "text", "text": text}]},
    )


class Hit:
    def __init__(self, id, issue_key="X-1", chunk_type="issue", source_id=1, content="c", score=1.0, retrievers=None, page_number=None):
        self.id, self.issue_key, self.chunk_type, self.source_id = id, issue_key, chunk_type, source_id
        self.content, self.score, self.retrievers = content, score, retrievers or ["vector"]
        self.page_number = page_number
        self.provenance = {}


def test_explicit_document_identifier_is_prioritized_over_semantic_lookalikes():
    lookalike = Hit("1", issue_key="DVQPLH-34", content="CORE-957 Assignee Lucas Meyer", score=8.0)
    exact = Hit("2", issue_key="DVQPLH-33", content="CORE-920 Assignee Amlna Okafor", score=-1.0)
    other = Hit("3", issue_key="DVQPLH-36", content="CORE-1031 Assignee Noah Williams", score=7.0)

    ordered = _prioritize_explicit_identifier_hits(
        [lookalike, exact, other], "Who is the assignee for CORE-920?"
    )

    assert [hit.id for hit in ordered] == ["2", "1", "3"]
    assert exact.score == -1.0  # reordering does not forge or rewrite retrieval scores


@pytest.mark.parametrize(
    "question",
    [
        "What total is due on invoice INV-3300?",
        "What was pending for September 8–14, 2026?",
        'In the “Renewal rate by region” chart, which region is highest?',
        "What changed on 2026-08-14?",
    ],
)
def test_normal_user_source_cues_prevent_false_ambiguity(question):
    assert _has_explicit_source_cue(question) is True


def test_source_free_single_fact_has_no_explicit_source_cue():
    assert _has_explicit_source_cue("Which category is highest?") is False
    assert _has_explicit_source_cue("What is the total amount due?") is False


def test_initial_router_matches_the_reported_regression_questions():
    assert _initial_tool_choice("how many sprints do we have so far") == "query_sprints"
    assert (
        _initial_tool_choice(
            "how many issues do we have so far? give me the result including sprint7 and without"
        )
        == "query_sprints"
    )
    assert _initial_tool_choice("which issues are currently blocked?") == "query_issues"
    # Exact production wording from the attachment/Ask-AI failure. The old `recent` word-boundary
    # regex did not match the adverb `recently`, so this ordinary ORDER BY question was forced onto
    # semantic search and exposed an unrelated vector-response schema failure.
    assert _initial_tool_choice("what is the most recently updated issue") == "query_issues"
    assert _initial_tool_choice("which issue was updated most recently?") == "query_issues"
    assert _initial_tool_choice("show the newest ticket") == "query_issues"
    assert _initial_tool_choice("最近更新的 issue 是哪一個？") == "query_issues"
    # Free-text topic wording with no entity noun is still not force-routed; the model chooses.
    assert _initial_tool_choice("what is blocking the payment system?") is None
    assert _initial_tool_choice("what does ATC-46's attachment say?") == "get_issue_attachments"
    assert _initial_tool_choice("show me the details of ATC-43") == "get_issue_details"


def test_topical_issue_lookup_routes_to_semantic_search_deterministically():
    """An entity noun + a topic connective + no structured action is a content-relevance question
    (which issue *text* is about this subject) that query_issues cannot answer. Before this rule the
    capability router was a coin flip on it — sending near-identical phrasings to different tools,
    only one of which is cacheable."""
    for q in (
        "is there a issue related to checkout ?",
        "are there issues related to checkout ?",
        "any issue related to checkout?",
        "any bug about refunds",
        "tickets regarding the payment provider",
        "do we have stories concerning onboarding",
    ):
        assert _initial_tool_choice(q) == "search_knowledge_base", q

    # A structured action word still wins — "how many"/"list" is a field query even with a topic.
    assert _initial_tool_choice("how many issues are related to checkout?") == "query_issues"
    assert _initial_tool_choice("list bugs about refunds") == "query_issues"
    # Sprint / issue-key wording is left to the more specific rules, not pulled onto semantic search.
    assert _initial_tool_choice("issues related to sprint 5") != "search_knowledge_base"
    assert _initial_tool_choice("comments related to the outage on ATC-9") != "search_knowledge_base"
    # No topic connective -> unchanged (still the router's call).
    assert _initial_tool_choice("is there a checkout issue?") is None


def test_selected_attachment_scope_beats_structured_ticket_vocabulary():
    question = (
        "Using only the attachments on DVQPLH-22, "
        "Which team is responsible for support ticket SUP-5341?"
    )
    assert _initial_tool_choice(question) == "get_issue_attachments"


def test_router_prompt_scopes_query_issues_away_from_free_text_facts():
    """Regression for the live ATC-432 misroute: "is there a flight info about LINGYI?" (a follow-up
    with no issue key and no "attachment" word, so the keyword layer above misses it entirely) was
    routed by the capability router to query_issues — a structured-metadata-only tool that can never
    contain a passenger name from an attachment — and the model then reported the name as absent
    without falling back to search_knowledge_base/get_issue_attachments. Pin the prompt language that
    fixes this so a future edit can't silently drop the scoping guidance.
    """
    assert "It has NO free-text" in render_capability_router_prompt()
    assert "a name, an ID, a quantity" in render_capability_router_prompt()
    assert "even if the question also names an issue" in render_capability_router_prompt()
    assert "ambiguous_single_source" in render_capability_router_prompt()


def test_router_does_not_assume_every_key_shaped_business_identifier_is_a_jira_issue():
    assert "not automatically a Jira issue key" in render_capability_router_prompt()
    assert "invoice numbers, incident references, support-ticket numbers" in render_capability_router_prompt()
    assert "printed inside an attachment" in render_capability_router_prompt()


def test_system_prompt_forbids_treating_structured_tool_silence_as_content_absence():
    """The main agent's ground rules must generalize rule 1's fallback chain beyond
    search_knowledge_base: a structured tool (query_issues/query_sprints/get_issue_history) not
    mentioning a value is not evidence that value is absent from the issue's free text, since those
    tools never carry prose/comment/attachment content in the first place."""
    system = render_crag_system_prompt("2026-08-11T00:00:00Z")
    assert "1b." in system
    assert "not evidence the value doesn't exist anywhere on the issue" in system


def test_system_prompt_requires_clarification_for_ambiguous_single_document_facts():
    system = render_crag_system_prompt("2026-08-11T00:00:00Z")
    assert "multiple independently plausible documents with different answers" in system
    assert "do NOT silently choose the top-ranked" in system
    assert "Do not ask for internal attachment IDs or issue keys" in system
    assert "aggregate, list, or comparison" in system
    assert "one plausible result from the first search does not prove" in system
    assert "one broader search_knowledge_base call" in system
    assert "do not preview the requested" in system


def test_empty_exact_readers_direct_business_identifiers_back_to_semantic_discovery():
    requested = ["INC-412"]
    for rendered in (
        crag_loop._format_issue_details([], requested),
        crag_loop._format_issue_comments([], requested),
        crag_loop._format_issue_attachments([], requested),
    ):
        assert "business identifier" in rendered
        assert "search_knowledge_base" in rendered


def _comment_hit(source_id, content, **provenance):
    return RetrievedChunk(
        id=f"comment:{source_id}", chunk_type="comment", issue_id=10, issue_key="ATC-10",
        source_id=source_id, content=f"[Comment on ATC-10]\n{content}", score=0.0,
        retrievers=["issue_comments"], provenance=provenance,
    )


def test_issue_comments_render_as_threads_with_author_date_and_reply_target():
    hits = [
        _comment_hit(5005116, "I am blocked on the duplicate rule.",
                     author="Ava Patel", created_at="2026-06-15T22:40:00+00:00"),
        _comment_hit(5005118, "Reject conflicting duplicates.", author="Liam Ortiz",
                     created_at="2026-06-18T11:30:00+00:00", parent_id=5005116, reply_to_id=5005116),
        _comment_hit(5005119, "Closer to eight points.", author="Emma Brooks",
                     created_at="2026-06-18T11:38:00+00:00", parent_id=5005116, reply_to_id=5005118),
    ]
    assert crag_loop._format_issue_comments(hits, ["ATC-10"]) == (
        "[ATC-10] comment #5005116 by Ava Patel, 2026-06-15\n"
        "I am blocked on the duplicate rule.\n\n"
        "    ↳ [ATC-10] reply #5005118 to #5005116 (Ava Patel) by Liam Ortiz, 2026-06-18\n"
        "    Reject conflicting duplicates.\n\n"
        "    ↳ [ATC-10] reply #5005119 to #5005118 (Liam Ortiz) by Emma Brooks, 2026-06-18\n"
        "    Closer to eight points."
    )


def test_issue_comments_without_provenance_still_render_one_block_each():
    hits = [_comment_hit(1, "first"), _comment_hit(2, "second")]
    assert crag_loop._format_issue_comments(hits, ["ATC-10"]) == (
        "[ATC-10] comment #1\nfirst\n\n[ATC-10] comment #2\nsecond"
    )


def test_a_long_comment_split_into_chunks_gets_one_header():
    hits = [_comment_hit(1, "part one", author="Ava Patel"), _comment_hit(1, "part two", author="Ava Patel")]
    assert crag_loop._format_issue_comments(hits, ["ATC-10"]) == (
        "[ATC-10] comment #1 by Ava Patel\npart one\npart two"
    )


def test_only_a_completed_semantic_search_answer_is_cacheable():
    """The write-side gate, decided from what the request actually did rather than guessed up front.

    The previous predicate took the question text and ran before routing, and so could only ever
    return False — `_initial_tool_choice` has no path that returns the semantic-search tool, since a
    keyword miss there means "ask the capability router". That made the answer cache unreachable.
    These assertions are the fix's contract: routes are judged after the fact, and the semantic route
    is the one that qualifies.
    """
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=False) is True
    # Multi-agent investigation answers are also cacheable (namespaced by execution_mode).
    assert is_answer_cacheable("investigate", abstained=False) is True

    # Structured/exact readers answer from live Jira state ("the most recently updated issue" is a
    # different issue an hour later), which no cache key can express. Never written.
    for tool in ("query_issues", "query_sprints", "get_issue_history", "get_issue_attachments"):
        assert is_answer_cacheable(tool, abstained=False) is False

    # An abstention is a claim about what the corpus lacks — the single most likely thing to stop
    # being true the moment the missing document is uploaded, and worth nothing to cache anyway.
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=True) is False
    assert is_answer_cacheable("investigate", abstained=True) is False

    # A request that never reached a route (an early failure) has no route to justify a write.
    assert is_answer_cacheable("", abstained=False) is False

    # Any structured result in the answer, from a later round or a multi worker, is live Jira state
    # too, and a card drag doesn't change the cache key.
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=False, used_live_data=True) is False
    assert is_answer_cacheable("investigate", abstained=False, used_live_data=True) is False


async def test_single_retrieval_then_answer():
    llm = FakeLLM([_tool_use_turn("payment errors"), _end_turn("root cause was X (X-1)")])
    retrieval = FakeRetrieval(default=[Hit("issue:1")])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("why did payment fail?", space_ids=[7])

    assert result.retrieval_rounds == 1
    assert result.queries_used == ["payment errors"]
    assert result.abstained is False
    assert len(result.citations) == 1


async def test_on_token_streams_the_final_answer_and_nothing_else():
    llm = FakeLLM([_tool_use_turn("payment errors"), _end_turn("root cause was X (X-1)")])
    retrieval = FakeRetrieval(default=[Hit("issue:1")])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    tokens: list[str] = []
    result = await agent.ask("why did payment fail?", space_ids=[7], on_token=tokens.append)

    # Every fragment concatenates back to exactly the answer — the tool-use turn streamed nothing.
    assert "".join(tokens).strip() == result.text
    assert len(tokens) > 1  # actually arrived in pieces, not one blob


async def test_on_token_is_optional():
    llm = FakeLLM([_tool_use_turn("q"), _end_turn("the answer (X-1)")])
    agent = CragAgent(llm, FakeRetrieval(default=[Hit("issue:1")]), max_iterations=4, top_k=5)
    result = await agent.ask("q?", space_ids=[7])
    assert result.text == "the answer (X-1)"


async def test_blank_end_turn_text_falls_back_to_abstention():
    # Regression test for a real bug found live (not hypothesized): qwen3.6-35b-a3b, via LM Studio,
    # returned stop_reason="end_turn" with empty text after retrieval found nothing relevant — not
    # the required ABSTENTION_PHRASE, just silence. An empty string is never a valid answer to hand
    # back to a caller.
    llm = FakeLLM([_tool_use_turn("obscure metric"), _end_turn("")])
    retrieval = FakeRetrieval(default=[])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what's our NPS score?", space_ids=[7])

    assert result.abstained is True
    assert result.text == ABSTENTION_PHRASE


async def test_corrective_retrieval_reformulates_and_retries():
    # First search returns nothing useful; model reformulates and searches again; then answers.
    llm = FakeLLM([
        _tool_use_turn("vague query", call_id="c1"),
        _tool_use_turn("HikariCP connection pool", call_id="c2"),
        _end_turn("fixed via pool tuning (X-1)"),
    ])
    retrieval = FakeRetrieval(results_by_query={
        "vague query": [],
        "HikariCP connection pool": [Hit("comment:9")],
    })
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("why did checkout fail?", space_ids=[7])

    assert result.retrieval_rounds == 2
    assert result.queries_used == ["vague query", "HikariCP connection pool"]
    assert len(result.citations) == 1


async def test_answer_may_cite_a_key_that_only_an_earlier_round_returned():
    # The key check pools every round of one question, not just the last one.
    llm = FakeLLM([
        _tool_use_turn("first query", call_id="c1"),
        _tool_use_turn("second query", call_id="c2"),
        _end_turn("The fix touched ATC-1 and ATC-2 (ATC-1, ATC-2)"),
    ])
    retrieval = FakeRetrieval(results_by_query={
        "first query": [Hit("issue:1", issue_key="ATC-1")],
        "second query": [Hit("issue:2", issue_key="ATC-2")],
    })
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what fixed checkout?", space_ids=[7])

    assert result.abstained is False
    assert {c.issue_key for c in result.citations} == {"ATC-1", "ATC-2"}


async def test_answer_naming_a_key_no_round_returned_abstains():
    llm = FakeLLM([
        _tool_use_turn("first query", call_id="c1"),
        _end_turn("The fix touched ATC-1 and ATC-99 (ATC-1)"),
    ])
    retrieval = FakeRetrieval(results_by_query={"first query": [Hit("issue:1", issue_key="ATC-1")]})
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what fixed checkout?", space_ids=[7])

    assert result.abstained is True


async def test_abstains_when_nothing_relevant_found():
    llm = FakeLLM([_tool_use_turn("obscure topic"), _end_turn(ABSTENTION_PHRASE)])
    retrieval = FakeRetrieval(default=[])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is our policy on quantum encryption?", space_ids=[7])

    assert result.abstained is True
    assert result.text == ABSTENTION_PHRASE


async def test_max_iterations_caps_retrieval_and_forces_final_answer():
    # Model tries to search on every turn up to (and past) the cap. With max_iterations=2, the loop
    # makes 3 tool-use decision points (iterations 0, 1, 2), then forces a final answer call — and
    # that final call is made with an EMPTY tools list, so a real model literally cannot emit
    # another tool_use there (nothing to call). FakeLLM can't express "no tools offered" since it's
    # a dumb scripted queue, so the 4th scripted turn stands in for "the model's only option now".
    turns = [_tool_use_turn(f"query {i}", call_id=f"c{i}") for i in range(3)]
    turns.append(_end_turn("giving up gracefully"))
    llm = FakeLLM(turns)
    retrieval = FakeRetrieval(default=[])
    agent = CragAgent(llm, retrieval, max_iterations=2, top_k=5)

    result = await agent.ask("anything", space_ids=[7])

    assert result.retrieval_rounds == 2  # capped, not 3
    assert result.text == ABSTENTION_PHRASE
    assert result.abstained is True
    assert llm.calls == 4  # 3 tool-use decisions + 1 forced final (tools=[])


async def test_forced_stop_falls_back_safely_if_model_ignores_empty_tools():
    # Defensive path: if a provider somehow still returns tool_use (or blank text) on the
    # tools=[] forced-final call, the loop must not silently return blank/garbage text.
    turns = [_tool_use_turn(f"query {i}", call_id=f"c{i}") for i in range(3)]
    turns.append(_tool_use_turn("still trying", call_id="c99"))  # ignores the empty tools hint
    llm = FakeLLM(turns)
    retrieval = FakeRetrieval(default=[])
    agent = CragAgent(llm, retrieval, max_iterations=2, top_k=5)

    result = await agent.ask("anything", space_ids=[7])

    assert result.text == ABSTENTION_PHRASE
    assert result.abstained is True


async def test_space_ids_are_never_taken_from_the_model():
    # The model's tool_use input only ever contains `query`/`mode` (see _SEARCH_TOOL's schema) — the
    # loop must pass the caller's space_ids on every call, regardless of what a (hypothetically
    # malicious or confused) model tries to put in the tool arguments.
    llm = FakeLLM([_tool_use_turn("q"), _end_turn("done")])
    retrieval = FakeRetrieval(default=[])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("question", space_ids=[42, 43])

    assert retrieval.calls[0][1] == (42, 43)


async def test_history_is_prepended_before_the_new_question():
    llm = FakeLLM([
        _tool_use_turn("bug issue keys"),
        _end_turn(ABSTENTION_PHRASE),
    ])
    agent = CragAgent(llm, FakeRetrieval(), max_iterations=4, top_k=5)
    history = [
        {"role": "user", "content": "how many bugs do we have?"},
        {"role": "assistant", "content": "There are 11 bugs, all done."},
    ]

    await agent.ask("what are their issue keys?", space_ids=[7], history=history)

    first_call_messages = llm.messages_seen[0]
    assert first_call_messages[0] == history[0]
    assert first_call_messages[1] == history[1]
    assert first_call_messages[2] == {"role": "user", "content": "what are their issue keys?"}


async def test_no_history_behaves_exactly_as_a_fresh_conversation():
    llm = FakeLLM([
        _tool_use_turn("a question"),
        _end_turn(ABSTENTION_PHRASE),
    ])
    agent = CragAgent(llm, FakeRetrieval(), max_iterations=4, top_k=5)

    await agent.ask("a question", space_ids=[7])  # history omitted entirely

    assert llm.messages_seen[0] == [{"role": "user", "content": "a question"}]


async def test_query_issues_tool_is_dispatched_for_count_questions():
    # A "how many bugs" question routes to the structured tool, not semantic search — and the exact
    # count flows through to the answer.
    result_data = IssueQueryResult(
        total_count=11, counts_by_type={"bug": 11}, counts_by_status={"done": 11}, issues=[]
    )
    llm = FakeLLM([
        _issue_query_turn({"issue_types": ["bug"]}),
        _end_turn("There are 11 bug issues."),
    ])
    retrieval = FakeRetrieval(issue_result=result_data)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("how many bugs do we have?", space_ids=[5000014])

    assert len(retrieval.issue_calls) == 1
    assert retrieval.calls == []  # semantic search was NOT used
    assert result.retrieval_rounds == 1
    assert "11" in result.text


async def test_most_recently_updated_issue_uses_structured_query_not_rag():
    latest = IssueQueryResult(
        total_count=1,
        counts_by_type={"task": 1},
        counts_by_status={"in_progress": 1},
        issues=[{
            "issue_key": "ATC-432",
            "title": "flight schedule : test1",
            "issue_type": "task",
            "status": "in_progress",
            "updated_at": "2026-08-11T12:05:46Z",
        }],
    )
    # The model intentionally omits every sort argument: routing code owns the unambiguous
    # updated_at DESC LIMIT 1 semantics rather than trusting model argument quality.
    llm = FakeLLM([
        _issue_query_turn({}),
        _end_turn("ATC-432 is the most recently updated issue."),
    ])
    retrieval = FakeRetrieval(issue_result=latest)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is the most recently updated issue", space_ids=[5000018])

    assert llm.tool_choices_seen[0] == "query_issues"
    assert llm.router_calls == []  # high-frequency fast path avoids an extra classifier call
    assert retrieval.issue_calls == [
        ((5000018,), {"order_by": "updated_at", "order": "desc", "limit": 1})
    ]
    assert retrieval.calls == []
    assert result.text == "ATC-432 is the most recently updated issue."


async def test_unseen_wording_is_routed_by_model_to_precise_tool_before_semantic_search():
    latest = IssueQueryResult(
        total_count=1,
        counts_by_type={"task": 1},
        counts_by_status={"in_progress": 1},
        issues=[{"issue_key": "ATC-432", "updated_at": "2026-08-11T12:05:46Z"}],
    )
    llm = FakeLLM(
        [_issue_query_turn({"order_by": "updated_at", "order": "desc", "limit": 1}),
         _end_turn("ATC-432 has the greatest modification timestamp.")],
        router_decisions=[("query_issues", "high")],
    )
    retrieval = FakeRetrieval(issue_result=latest)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(
        "Give me the work record whose modification timestamp is greatest.",
        space_ids=[5000018],
    )

    assert len(llm.router_calls) == 1
    assert llm.tool_choices_seen[0] == "query_issues"
    assert retrieval.issue_calls == [
        ((5000018,), {"order_by": "updated_at", "order": "desc", "limit": 1})
    ]
    assert retrieval.calls == []
    assert result.text.startswith("ATC-432")


async def test_empty_model_selected_exact_reader_falls_back_for_document_internal_key():
    """CORE-957 can be text inside an image carried by DVQPLH-34, not a Jira issue itself."""
    question = "Who is the assignee for CORE-957?"
    hit = Hit(
        "attachment:34",
        issue_key="DVQPLH-34",
        chunk_type="attachment",
        source_id=34,
        content="Issue CORE-957 — Assignee: Lucas Meyer",
    )
    llm = FakeLLM(
        [
            _issue_details_turn({"issue_keys": ["CORE-957"]}),
            _end_turn("CORE-957 is assigned to Lucas Meyer. (DVQPLH-34)"),
        ],
        router_decisions=[("get_issue_details", "high")],
    )
    retrieval = FakeRetrieval(default=[hit])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[5000034])

    assert retrieval.details_calls == [((5000034,), ("CORE-957",))]
    assert retrieval.calls == [(question, (5000034,), 5, "hybrid")]
    assert result.retrieval_rounds == 2
    assert result.text == "CORE-957 is assigned to Lucas Meyer. (DVQPLH-34)"
    assert [citation.issue_key for citation in result.citations] == ["DVQPLH-34"]


async def test_capability_router_can_use_a_separate_small_model_client():
    main_llm = FakeLLM([
        _issue_query_turn({"limit": 1}),
        _end_turn("ATC-432 is the result."),
    ])
    router_llm = FakeLLM([], router_decisions=[("query_issues", "high")])
    retrieval = FakeRetrieval()
    agent = CragAgent(
        main_llm,
        retrieval,
        max_iterations=4,
        top_k=5,
        router_llm=router_llm,
        router_model="local-router",
    )

    await agent.ask("Find the record with the greatest modification time.", space_ids=[7])

    assert len(router_llm.router_calls) == 1
    assert main_llm.router_calls == []
    assert main_llm.tool_choices_seen[0] == "query_issues"


def test_router_usage_is_not_priced_as_the_hosted_answer_model():
    agent = CragAgent(
        FakeLLM([]), FakeRetrieval(), max_iterations=4, top_k=5,
        model="gpt-5.6-luna", router_model="local-router",
    )
    total = Usage(input_tokens=1100, output_tokens=110)
    routed = Usage(input_tokens=100, output_tokens=10)

    # Only the main model's 1000 input + 100 output tokens are billed. The unknown local model is
    # deliberately $0 in the pricing table; its tokens remain in total usage for capacity metrics.
    assert agent._cost(total, routed) == pytest.approx(0.00032)


async def test_model_usage_attributes_local_router_and_hosted_answer_separately():
    query_turn = _issue_query_turn({"limit": 1})
    query_turn.usage = Usage(input_tokens=1000, output_tokens=100)
    answer_turn = _end_turn("ATC-432 is the result.")
    answer_turn.usage = Usage(input_tokens=500, output_tokens=50)
    main_llm = FakeLLM([query_turn, answer_turn])
    router_llm = FakeLLM(
        [],
        router_decisions=[("query_issues", "high")],
        router_usage=Usage(input_tokens=680, output_tokens=92),
    )
    agent = CragAgent(
        main_llm, FakeRetrieval(), max_iterations=4, top_k=5,
        model="gpt-5.6-luna", router_llm=router_llm, router_model="self-hosted-router",
        model_provider="OpenAI", model_deployment="hosted",
        router_provider="Self-hosted", router_deployment="local",
    )

    result = await agent.ask(
        "Return the Jira record with the maximum updated_at value.", space_ids=[7]
    )

    assert result.route_source == "model"
    assert result.route_tool == "query_issues"
    assert [(item.role, item.model) for item in result.model_usage] == [
        ("router", "self-hosted-router"), ("answer_agent", "gpt-5.6-luna")
    ]
    router_item, answer_item = result.model_usage
    assert (router_item.input_tokens, router_item.output_tokens) == (680, 92)
    assert router_item.deployment == "local"
    assert router_item.pricing_status == "free_local"
    assert router_item.estimated_cost_usd == 0
    assert (answer_item.input_tokens, answer_item.output_tokens) == (1500, 150)
    assert answer_item.deployment == "hosted"
    assert answer_item.pricing_status == "estimated"
    assert answer_item.estimated_cost_usd == pytest.approx(0.00048)
    assert result.estimated_cost_usd == pytest.approx(0.00048)


async def test_low_confidence_model_route_falls_back_to_semantic_search():
    llm = FakeLLM(
        [_tool_use_turn("release readiness concerns"), _end_turn("No concerns were found.")],
        router_decisions=[("query_issues", "low")],
    )
    retrieval = FakeRetrieval(default=[Hit("issue:1", issue_key="ATC-1")])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("Are we ready to ship this thing?", space_ids=[7])

    assert len(llm.router_calls) == 1
    assert llm.tool_choices_seen[0] == "search_knowledge_base"
    assert retrieval.calls[0][0] == "release readiness concerns"
    assert retrieval.issue_calls == []


async def test_router_ambiguity_classification_constrains_answer_turns_to_clarify():
    llm = FakeLLM(
        [
            _tool_use_turn("invoice total due alternatives"),
            _end_turn("Which invoice do you mean? (INV-1) (INV-2)"),
        ],
        router_decisions=[("search_knowledge_base", "high", "ambiguous_single_source")],
    )
    retrieval = FakeRetrieval(default=[
        Hit("1", issue_key="INV-1", content="Kestrel invoice total due $10"),
        Hit("2", issue_key="INV-2", content="Redwood invoice total due $20"),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("What is the total amount due?", space_ids=[7])

    assert llm.systems_seen
    assert all("source-scope classifier" in system for system in llm.systems_seen)
    assert all("do not answer the fact" in system for system in llm.systems_seen)
    assert all("ask_clarification" in system for system in llm.systems_seen)
    assert all(
        "do not write the question into your final answer text" in system
        for system in llm.systems_seen
    )
    assert all("do not preview the requested answer value" in system for system in llm.systems_seen)
    # A named subject with no matching document is "no information", not "which document?".
    assert all(
        "NO retrieved document is about that subject" in system for system in llm.systems_seen
    )


async def test_invalid_model_route_fails_closed_to_semantic_search():
    llm = FakeLLM(
        [_tool_use_turn("unclassified request"), _end_turn(ABSTENTION_PHRASE)],
        router_decisions=[("invented_database_tool", "high")],
    )
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Please investigate this unusual situation.", space_ids=[7])

    assert len(llm.router_calls) == 1
    assert llm.tool_choices_seen[0] == "search_knowledge_base"
    assert retrieval.calls[0][0] == "unclassified request"
    assert result.abstained is True


async def test_failed_tool_becomes_error_evidence_and_agent_can_choose_fallback():
    class SearchFails(FakeRetrieval):
        async def search(self, query, space_ids, limit, mode="hybrid"):
            self.calls.append((query, tuple(space_ids), limit, mode))
            raise RuntimeError("vector response schema mismatch")

    issue_result = IssueQueryResult(
        total_count=1,
        counts_by_type={"task": 1},
        counts_by_status={"planned": 1},
        issues=[{"issue_key": "ATC-432", "title": "flight schedule : test1"}],
    )
    llm = FakeLLM([
        _tool_use_turn("flight schedule"),
        _issue_query_turn({"issue_keys": ["ATC-432"]}, call_id="fallback"),
        _end_turn("ATC-432 is the matching issue."),
    ])
    retrieval = SearchFails(issue_result=issue_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("find the flight schedule work", space_ids=[5000018])

    assert result.text == "ATC-432 is the matching issue."
    assert result.retrieval_rounds == 1
    assert llm.tool_choices_seen[:2] == ["search_knowledge_base", None]
    failed_result = llm.messages_seen[1][-1]["content"][0]
    assert failed_result["is_error"] is True
    assert "temporarily unavailable" in failed_result["content"]
    assert "schema mismatch" not in failed_result["content"]  # internals stay server-side
    assert retrieval.issue_calls == [((5000018,), {"issue_keys": ["ATC-432"]})]


async def test_query_issues_space_ids_injected_not_taken_from_model():
    # Same security boundary as semantic search: the loop injects the caller's space_ids into the
    # structured query, never the model's tool input.
    llm = FakeLLM([_issue_query_turn({"issue_types": ["story"]}), _end_turn("done")])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("list the stories", space_ids=[42, 43])

    # One query per workspace, each one of the caller's.
    assert [call[0] for call in retrieval.issue_calls] == [(42,), (43,)]


def _history_turn(tool_input: dict, call_id: str = "h_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="get_issue_history", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "get_issue_history", "input": tool_input}
        ]},
    )


async def test_issue_history_tool_is_dispatched_for_reopen_questions():
    history = IssueHistoryResult(
        total_count=2,
        changes=[
            {"issue_key": "ATC-68", "event_type": "field_change", "field_name": "status",
             "from_value": "done", "to_value": "in_progress", "actor_name": "Daniel Park",
             "changed_at": "2026-07-02T16:00:00Z"},
            {"issue_key": "ATC-30", "event_type": "field_change", "field_name": "status",
             "from_value": "done", "to_value": "in_progress", "actor_name": "Noah Kim",
             "changed_at": "2026-05-14T17:20:00Z"},
        ],
    )
    llm = FakeLLM([
        _history_turn({"reopened_only": True}),
        _end_turn("2 issues were reopened: ATC-68 and ATC-30."),
    ])
    retrieval = FakeRetrieval(history_result=history)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("which issues were reopened?", space_ids=[5000014])

    assert retrieval.history_calls == [((5000014,), {"reopened_only": True})]
    # No search_knowledge_base call; each registered verifier performs its own structured
    # issue-type/status lookup over the same mentioned keys (subtask-claim check, then
    # current-state check — the answer text says "reopened", which is a status hint).
    assert retrieval.calls == []
    assert retrieval.issue_calls == [
        ((5000014,), {"issue_keys": ["ATC-30", "ATC-68"]}),
        ((5000014,), {"issue_keys": ["ATC-30", "ATC-68"]}),
    ]
    assert "ATC-68" in result.text


async def test_issue_history_space_ids_injected_and_input_whitelisted():
    llm = FakeLLM([
        _history_turn({"issue_keys": ["ATC-77"], "space_ids": [999], "sql": "drop table"}),
        _end_turn("done"),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("what did I change in ATC-77?", space_ids=[7])

    space_ids, filters = retrieval.history_calls[0]
    assert space_ids == (7,)
    assert filters == {"issue_keys": ["ATC-77"]}  # model-supplied space_ids/sql stripped


async def test_system_prompt_carries_current_time_for_relative_time_questions():
    # "what did I change 10 minutes ago" is unanswerable without a clock — the system prompt must
    # embed the ask-time UTC timestamp, not a stale module-load constant.
    captured = {}

    class CapturingLLM(FakeLLM):
        async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
            captured["system"] = system
            captured["tools"] = [t["name"] for t in tools]
            return await super().next_turn(system, messages, tools, tool_choice=tool_choice)

    llm = CapturingLLM([
        _tool_use_turn("anything"),
        _end_turn(ABSTENTION_PHRASE),
    ])
    agent = CragAgent(llm, FakeRetrieval(), max_iterations=4, top_k=5)
    await agent.ask("anything", space_ids=[7])

    import re
    from datetime import datetime, timezone
    m = re.search(r"current UTC time is (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)", captured["system"])
    assert m, "system prompt is missing the current-time stamp"
    stamped = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    # Fresh (within one 5-minute rounding window of now) — built at ask() time, not a stale constant.
    assert 0 <= (datetime.now(timezone.utc) - stamped).total_seconds() < 300
    # Rounded to a 5-minute mark, not second-precision — a prompt-caching requirement (see
    # _current_time_for_prompt's docstring): the stamp must repeat across calls within the same
    # window, or the "identical prefix" a cache hit needs never occurs.
    assert stamped.second == 0 and stamped.minute % 5 == 0
    assert captured["tools"] == [
        "search_knowledge_base", "query_issues", "query_sprints", "get_issue_history",
        "get_issue_comments", "get_issue_details", "get_issue_attachments", "get_section",
        "ask_clarification",
    ]


async def test_concurrent_asks_keep_their_own_system_prompt(monkeypatch):
    """A shared agent must not swap request A's clock for request B's between LLM turns."""
    timestamps = iter(["2030-01-01T00:00:00Z", "2040-02-02T00:00:00Z"])
    monkeypatch.setattr(crag_loop, "_current_time_for_prompt", lambda: next(timestamps))
    second_request_started = asyncio.Event()

    class InterleavingLLM:
        def __init__(self):
            self.calls_by_question = {}
            self.systems_by_question = {}

        async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
            question = messages[0]["content"]
            if tools and tools[0].get("name") == "select_data_source":
                return LLMTurn(
                    stop_reason="tool_use",
                    text="",
                    tool_calls=[ToolCall(
                        id=f"route-{question}",
                        name="select_data_source",
                        input={
                            "tool": "search_knowledge_base",
                            "confidence": "high",
                            "source_scope": "identified",
                        },
                    )],
                    assistant_message={"role": "assistant", "content": []},
                )
            call_number = self.calls_by_question.get(question, 0) + 1
            self.calls_by_question[question] = call_number
            self.systems_by_question.setdefault(question, []).append(system)
            if question == "request A" and call_number == 1:
                await second_request_started.wait()
            elif question == "request B" and call_number == 1:
                second_request_started.set()
            return _tool_use_turn(question, f"{question}-{call_number}") if call_number == 1 else _end_turn("done")

    llm = InterleavingLLM()
    agent = CragAgent(llm, FakeRetrieval(), max_iterations=4, top_k=5)

    first = asyncio.create_task(agent.ask("request A", space_ids=[7]))
    await asyncio.sleep(0)
    second = asyncio.create_task(agent.ask("request B", space_ids=[7]))
    await asyncio.gather(first, second)

    assert all("2030-01-01T00:00:00Z" in prompt for prompt in llm.systems_by_question["request A"])
    assert all("2040-02-02T00:00:00Z" in prompt for prompt in llm.systems_by_question["request B"])


async def test_query_issues_strips_non_whitelisted_fields_from_model_input():
    # A model that tries to smuggle space_ids (or any unknown key) into the structured tool input must
    # not have it forwarded downstream — only whitelisted filter fields pass through.
    llm = FakeLLM([
        _issue_query_turn({"issue_types": ["bug"], "space_ids": [999], "evil": "x"}),
        _end_turn("done"),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("how many bugs", space_ids=[7])

    space_ids, filters = retrieval.issue_calls[0]
    assert space_ids == (7,)  # caller's, not the model's 999
    assert filters == {"issue_types": ["bug"]}  # space_ids/evil stripped


async def test_query_issues_passes_sprint_filters_through():
    # sprint_ids/sprint_names must actually reach the retrieval client — a regression here would be
    # the tool schema advertising a filter the whitelist silently drops.
    llm = FakeLLM([
        _sprint_query_turn({}),
        _issue_query_turn({"sprint_ids": [501], "issue_types": ["bug"]}),
        _end_turn("done"),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("how many bugs in sprint 7", space_ids=[7])

    assert retrieval.sprint_calls == [((7,), {})]
    assert retrieval.issue_calls[0][1] == {"sprint_ids": [501], "issue_types": ["bug"]}


async def test_query_issues_passes_priorities_filter_through():
    # Regression test: priorities was advertised in the tool schema and documented in the system
    # prompt, but the whitelist in _run_issue_query omitted it, so it was silently dropped before
    # reaching the retrieval client — every priority-filtered question silently fell back to an
    # unfiltered query.
    llm = FakeLLM([
        _issue_query_turn({"priorities": ["highest"]}),
        _end_turn("done"),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("how many issues are highest priority", space_ids=[7])

    assert retrieval.issue_calls[0][1] == {"priorities": ["highest"]}


async def test_query_issues_passes_issue_keys_for_topic_current_state_intersection():
    llm = FakeLLM([
        _issue_query_turn({"issue_keys": ["ATC-34", "ATC-55"], "statuses": ["blocked"]}),
        _end_turn("ATC-34 is currently blocked."),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("which issues are blocked?", space_ids=[7])

    assert retrieval.issue_calls[0][1] == {
        "issue_keys": ["ATC-34", "ATC-55"],
        "statuses": ["blocked"],
    }


async def test_obvious_sprint_count_forces_structured_tool_first():
    llm = FakeLLM([_sprint_query_turn({}), _end_turn("There are exactly 7 sprints.")])
    retrieval = FakeRetrieval(
        sprint_result=SprintQueryResult(
            total_count=7, counts_by_status={"completed": 6, "active": 1}, sprints=[]
        )
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("how many sprints do we have so far?", space_ids=[7])

    assert llm.tool_choices_seen[0] == "query_sprints"
    assert retrieval.sprint_calls == [((7,), {})]
    assert "7" in result.text


async def test_unsupported_issue_key_in_answer_is_rejected_and_sources_are_hidden():
    llm = FakeLLM([
        _tool_use_turn("payment blocker"),
        _end_turn("The payment system is blocked by ATC-77."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit("issue:34", issue_key="ATC-34", content="Checkout beta is blocked."),
        Hit("issue:55", issue_key="ATC-55", content="Sign-in was unblocked."),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is blocking the payment system?", space_ids=[7])

    assert result.text == ABSTENTION_PHRASE
    assert result.abstained is True
    assert result.citations == []


async def test_sources_only_include_issue_keys_actually_cited_by_answer():
    llm = FakeLLM([
        _tool_use_turn("payment blocker"),
        _end_turn("Checkout is blocked by request-ID work (ATC-34)."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit("issue:34", issue_key="ATC-34", content="Checkout beta is blocked."),
        Hit("issue:55", issue_key="ATC-55", content="Sign-in was unblocked."),
        Hit("comment:34", issue_key="ATC-34", content="Duplicate chunk for the same issue."),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is blocking the payment system?", space_ids=[7])

    assert result.abstained is False
    assert [citation.issue_key for citation in result.citations] == ["ATC-34"]


async def test_order_id_in_answer_is_not_mistaken_for_an_unsupported_issue_key():
    # Regression: the corpus's own text contains non-issue-key identifiers of the same PREFIX-NNN
    # shape — e.g. order references like "BETA-1042" quoted inside a comment. When the model
    # faithfully quotes them from a retrieved chunk, the hallucination guard must NOT treat them as
    # unsupported issue references and nuke the whole (correct, grounded) answer. Found live: every
    # question about the double-order incident abstained for exactly this reason. Only tokens sharing
    # a project prefix with retrieved evidence (here ATC-) are held to the "must be in evidence" rule.
    llm = FakeLLM([
        _tool_use_turn("double order checkout"),
        _end_turn("Clicking Place order twice created two orders, BETA-1042 and BETA-1043, and "
                  "dropped stock by two (ATC-43)."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit("comment:43", issue_key="ATC-43",
            content="Two order rows show up: BETA-1042 and BETA-1043. Stock drops by two."),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("how was the double-order checkout bug reproduced?", space_ids=[7])

    assert result.abstained is False
    assert "BETA-1042" in result.text
    assert [citation.issue_key for citation in result.citations] == ["ATC-43"]


async def test_hallucinated_key_sharing_a_real_project_prefix_is_still_rejected():
    # The prefix-aware relaxation above must NOT weaken the guard for genuine hallucinations: a made-up
    # ATC-9999 shares the ATC- prefix that IS in evidence, so it's still an unsupported issue key.
    llm = FakeLLM([
        _tool_use_turn("payment blocker"),
        _end_turn("Checkout is blocked by ATC-9999."),
    ])
    retrieval = FakeRetrieval(default=[Hit("issue:34", issue_key="ATC-34", content="Checkout beta.")])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is blocking the payment system?", space_ids=[7])

    assert result.text == ABSTENTION_PHRASE
    assert result.abstained is True


async def test_hallucinated_trailing_citation_with_a_different_prefix_is_rejected():
    """Regression for the live DocVQA/Qwen-72B fabrication `Strain hardening (ATC-123)`.

    The only real structured result was DVQA-4. Prefix-aware prose validation deliberately ignores
    a different prefix so real order IDs such as BETA-1042 survive, but a key in the final citation
    slot is claiming to be a source and therefore must match an actual tool-result entity.
    """
    llm = FakeLLM([
        _issue_query_turn({}),
        _end_turn("Strain hardening (ATC-123)"),
    ])
    retrieval = FakeRetrieval(issue_result=IssueQueryResult(
        total_count=1,
        counts_by_type={"task": 1},
        counts_by_status={"todo": 1},
        issues=[{"issue_key": "DVQA-4", "title": "DocVQA fixture"}],
    ))
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Which technique stretches the lamina by 50-100%?", space_ids=[7])

    assert result.text == ABSTENTION_PHRASE
    assert result.abstained is True


async def test_issue_key_mentioned_only_inside_a_retrieved_chunks_own_text_is_not_hallucination():
    # Regression: a retrieved chunk's own prose can legitimately reference OTHER issue keys as
    # backstory (e.g. ATC-43's comment says "split into ATC-44, ATC-45..."). Those keys were never
    # independently retrieved this turn, so without scanning retrieved content for them, the model
    # faithfully quoting that backstory got rejected as "unsupported issue references." Found live:
    # A broad question about ATC-43 abstained because its answer named the follow-up issues its own
    # retrieved comment mentioned. (Explicit "details" questions now route to the exact body reader,
    # so this wording deliberately exercises semantic-search citation grounding.)
    llm = FakeLLM([
        _tool_use_turn("ATC-43"),
        _end_turn("ATC-43 was split into follow-ups ATC-44 (idempotency fix) and ATC-45 (button "
                  "mitigation)."),
    ])
    retrieval = FakeRetrieval(
        default=[
            Hit("comment:43", issue_key="ATC-43",
                content="Split the follow-up into ATC-44 for the idempotency fix and ATC-45 for the "
                        "button mitigation."),
        ]
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what happened around ATC-43?", space_ids=[7])

    assert result.abstained is False
    assert "ATC-44" in result.text and "ATC-45" in result.text


async def test_a_list_item_ending_in_a_key_named_by_retrieved_text_is_not_a_fabricated_citation():
    # The same as above, but the key sits in the final "(KEY)" slot, which is checked as the answer's
    # source claim. Found in the 2026-11-01 RAGAS run: "What is ATC-43 about?" abstained 1 in 9 asks
    # whenever the answer ended with a subtask bullet such as "... (ATC-47)".
    llm = FakeLLM([
        _tool_use_turn("ATC-43"),
        _end_turn("ATC-43 is a double-order checkout bug. (ATC-43)\n\nIts subtasks cover:\n"
                  "- An API idempotency fix (ATC-44)\n- Slow-checkout regression coverage (ATC-47)"),
    ])
    retrieval = FakeRetrieval(
        default=[
            Hit("comment:43", issue_key="ATC-43",
                content="Split the follow-up into ATC-44 for the API idempotency fix and ATC-47 for "
                        "the slow-response regression exit criteria."),
        ]
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("What is ATC-43 about?", space_ids=[7])

    assert result.abstained is False
    assert "(ATC-47)" in result.text


async def test_a_final_key_that_no_tool_returned_or_named_is_still_rejected():
    llm = FakeLLM([
        _tool_use_turn("ATC-43"),
        _end_turn("ATC-43 is a double-order checkout bug. (ATC-43)\n\n- A follow-up fix (ATC-99)"),
    ])
    retrieval = FakeRetrieval(
        default=[Hit("comment:43", issue_key="ATC-43", content="Split the follow-up into ATC-44.")]
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("What is ATC-43 about?", space_ids=[7])

    assert result.abstained is True


async def test_sprint_name_citation_via_semantic_search_alone_is_not_forced_to_abstain():
    # Regression: a sprint chunk's provenance label is its sprint NAME (e.g. "AtlasCart Sprint 7"),
    # not an issue key — it never matches the PREFIX-NNN issue-key shape, so a semantic-only answer
    # that legitimately cites a sprint goal could never satisfy "must reference something retrieved"
    # and always got rejected. Found live: a follow-up ("does it have a goal?") that answered purely
    # from a semantic sprint-goal hit (no structured tool call that turn) abstained despite being
    # correct and fully grounded.
    llm = FakeLLM([
        _tool_use_turn("AtlasCart Sprint 7 goal"),
        _end_turn('Yes — AtlasCart Sprint 7 has the goal: "Prove the storefront is accessible."'),
    ])
    retrieval = FakeRetrieval(default=[
        Hit("sprint:7", issue_key="AtlasCart Sprint 7", chunk_type="sprint",
            content="Prove the storefront is accessible."),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what was focused on accessibility?", space_ids=[7])

    assert result.abstained is False
    assert [c.issue_key for c in result.citations] == ["AtlasCart Sprint 7"]


async def test_get_issue_comments_is_dispatched_and_grounds_the_answer_with_citations():
    # Plan-B fallback: search_knowledge_base's top-K comes back empty/weak, so the model falls back
    # to get_issue_comments on the issue it already knows the key for, and the returned comment text
    # is what actually grounds the answer. The scripted model chooses semantic search first, then the
    # exact reader after seeing that ranked retrieval was insufficient.
    comments_result = IssueCommentsResult(
        total_count=1,
        comments=[{
            "issue_id": 30, "issue_key": "ATC-30", "source_id": 501,
            "content": "Reopening this because the beta incident shows one browser action can "
                       "submit twice.",
        }],
    )
    llm = FakeLLM([
        _tool_use_turn("ATC-30 extra work needed"),
        _issue_comments_turn({"issue_keys": ["ATC-30"]}),
        _end_turn("ATC-30 needed extra work because the beta incident showed one browser action "
                  "could submit the checkout twice (ATC-30)."),
    ])
    retrieval = FakeRetrieval(default=[], comments_result=comments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("why did ATC-30 need extra work?", space_ids=[7])

    assert retrieval.comments_calls == [((7,), ("ATC-30",))]
    assert result.abstained is False
    assert [c.issue_key for c in result.citations] == ["ATC-30"]


async def test_get_issue_details_is_dispatched_and_grounds_the_answer_with_citations():
    # Reproduces a real live bug: search_knowledge_base
    # returns only COMMENT hits for an issue with many comments (its own title+description chunk
    # ranked below them), so the model must fall back to get_issue_details on the key it already
    # knows to actually retrieve the issue's own body — a pile of comment hits is not proof the
    # issue's own description came back.
    details_result = IssueDetailsResult(
        total_count=1,
        details=[{
            "issue_id": 5000932, "issue_key": "ATC-43", "source_id": 5000932,
            "content": "Checkout can create two orders from a double click. During the private beta, "
                       "a slow checkout allowed two Place order clicks.",
        }],
    )
    llm = FakeLLM([
        _tool_use_turn("ATC-43 details"),
        _issue_details_turn({"issue_keys": ["ATC-43"]}),
        _end_turn("ATC-43 is about checkout creating two orders from a double click: during the "
                  "private beta, a slow checkout allowed two Place order clicks (ATC-43)."),
    ])
    retrieval = FakeRetrieval(
        default=[Hit("comment:5003388", issue_key="ATC-43", chunk_type="comment",
                     content="Split the follow-up into ATC-44, ATC-45, and ATC-47.")],
        details_result=details_result,
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("show me the details of ATC-43", space_ids=[7])

    assert retrieval.details_calls == [((7,), ("ATC-43",))]
    assert result.abstained is False
    assert [c.issue_key for c in result.citations] == ["ATC-43"]
    assert "double click" in result.citations[0].content


async def test_chunk_found_by_search_and_by_exact_read_is_cited_once():
    # Seen in the RAGAS cohort: the issue chunk came back from search as `issue:<id>` and from
    # get_issue_details as `issue_details:<id>`. Two copies took two of the issue's citation slots
    # and pushed out the comment that held the rest of the answer.
    body = "ATC-68 Subtask of ATC-50. Tracking link opens a different beta order."
    details_result = IssueDetailsResult(
        total_count=1,
        details=[{"issue_id": 5003207, "issue_key": "ATC-68", "source_id": 5003207, "content": body}],
    )
    llm = FakeLLM([
        _tool_use_turn("tracking email opened a different order"),
        _issue_details_turn({"issue_keys": ["ATC-68"]}),
        _end_turn("The resent tracking link opened a different beta order (ATC-68)."),
    ])
    retrieval = FakeRetrieval(
        default=[
            Hit("issue:5003207", issue_key="ATC-68", chunk_type="issue", content=body),
            Hit("comment:9", issue_key="ATC-68", chunk_type="comment",
                content="Follow-up fix: the first send and the resend now build the same tracking link."),
        ],
        details_result=details_result,
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Why did the tracking email open a different order?", space_ids=[7])

    contents = [c.content for c in result.citations]
    assert contents.count(body) == 1
    assert any("Follow-up fix" in c for c in contents)


async def test_get_issue_attachments_is_dispatched_and_grounds_the_answer_with_citations():
    # Reproduces real live flakiness: search_knowledge_base's semantic/hybrid ranking can only be as
    # good as the words the model guesses, so an exact fact it doesn't already know the wording of
    # (a SKU code) can genuinely miss
    # even though it's in the index — found live, this made "what SKU does ATC-46's attachment use"
    # flip between finding the answer and abstaining across equivalent phrasings. Once the issue key
    # is known, get_issue_attachments makes the lookup exact instead of a ranking gamble.
    attachments_result = IssueAttachmentsResult(
        total_count=1,
        attachments=[{
            "issue_id": 5000935, "issue_key": "ATC-46", "source_id": 5000940,
            "page_number": 2,
            "provenance": {"source_type": "pdf", "page_number": 2},
            "content": "Acceptance criteria table: SKU A-104, order reference BETA-1043, quantity 1.",
        }],
    )
    llm = FakeLLM([
        _tool_use_turn("ATC-46 attachment"),
        _issue_attachments_turn({"issue_keys": ["ATC-46"]}),
        _end_turn("The worked example uses SKU A-104 and order reference BETA-1043 (ATC-46)."),
    ])
    retrieval = FakeRetrieval(
        default=[],
        attachments_result=attachments_result,
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what SKU does ATC-46's attachment use?", space_ids=[7])

    assert retrieval.attachments_calls == [((7,), ("ATC-46",))]
    assert result.abstained is False
    assert [c.issue_key for c in result.citations] == ["ATC-46"]
    assert "A-104" in result.citations[0].content
    assert result.citations[0].page_number == 2
    assert result.citations[0].provenance["source_type"] == "pdf"


async def test_a_cut_off_hit_can_be_expanded_to_its_whole_section():
    # Parent-child retrieval: the hit has steps 4-7 only, lists its section, and the model reads it.
    section_id = "attachment:99:section#0"
    partial = RetrievedChunk(
        id="attachment:99#1", chunk_type="attachment", issue_id=42, issue_key="OPS-7", source_id=99,
        content="Runbook > 2. Rollback\n\nStep 4: drain traffic. ... Step 7: verify.", score=1.0,
        retrievers=["vector"], page_number=2,
        provenance={"source_type": "pdf", "page_number": 2, "section_ids": [section_id]},
    )
    whole = {
        "id": section_id, "issue_id": 42, "issue_key": "OPS-7", "source_id": 99,
        "content": "## 2. Rollback\n\nStep 1: freeze deploys. ... Step 7: verify.",
        "page_number": 2, "page_end": 3,
    }
    llm = FakeLLM([
        _tool_use_turn("rollback steps"),
        _section_turn([section_id]),
        _end_turn("Step 1 is to freeze deploys, and step 7 is to verify (OPS-7)."),
    ])
    retrieval = FakeRetrieval(default=[partial], sections_result=[whole])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("What are all the rollback steps?", space_ids=[7])

    search_result = llm.messages_seen[1][-1]["content"][0]["content"]
    assert f"section_ids={section_id}" in search_result
    # space_ids come from the caller, never from the model.
    assert retrieval.sections_calls == [((7,), (section_id,))]
    assert "get_section" in llm.tools_seen[0]
    assert result.abstained is False
    section_citation = next(c for c in result.citations if c.id == section_id)
    assert "Step 1: freeze deploys" in section_citation.content
    assert section_citation.provenance == {"source_type": "pdf", "page_number": 2, "page_end": 3}


async def test_with_the_section_tool_off_the_agent_works_as_before():
    hit = RetrievedChunk(
        id="attachment:99#1", chunk_type="attachment", issue_id=42, issue_key="OPS-7", source_id=99,
        content="Step 4: drain traffic.", score=1.0, retrievers=["vector"], page_number=2,
        provenance={"source_type": "pdf", "section_ids": ["attachment:99:section#0"]},
    )
    llm = FakeLLM([_tool_use_turn("rollback steps"), _end_turn("Drain traffic (OPS-7).")])
    agent = CragAgent(llm, FakeRetrieval(default=[hit]), max_iterations=4, top_k=5,
                      parent_child_tool_enabled=False)

    await agent.ask("What are the rollback steps?", space_ids=[7])

    assert all("get_section" not in tools for tools in llm.tools_seen)
    assert all("get_section" not in system for system in llm.systems_seen)
    assert "section_ids" not in llm.messages_seen[1][-1]["content"][0]["content"]


async def test_single_explicit_attachment_source_is_cited_when_natural_answer_omits_suffix():
    """A known, retrieved, unambiguous source must not become an abstention over citation style.

    This is the real Product-like natural-answer failure shape: the model obeyed the exact-reader
    route and answered from its result, but omitted the textual ``(ISSUE-N)`` suffix. The API already
    knows the selected issue and the tool returned that same issue, so attaching that one proven
    source is deterministic provenance, not a model guess.
    """
    attachments_result = IssueAttachmentsResult(
        total_count=1,
        attachments=[{
            "issue_id": 5000935,
            "issue_key": "ATC-46",
            "source_id": 5000940,
            "page_number": 2,
            "provenance": {"source_type": "pdf", "page_number": 2},
            "content": "The purchase order should be shipped to Singapore.",
        }],
    )
    llm = FakeLLM([
        _issue_attachments_turn({"issue_keys": ["ATC-46"]}),
        _end_turn("The destination city is Singapore."),
    ])
    retrieval = FakeRetrieval(attachments_result=attachments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("where should the ATC-46 attachment order be shipped?", space_ids=[7])

    assert result.abstained is False
    assert result.text == "The destination city is Singapore (ATC-46)."
    assert [citation.issue_key for citation in result.citations] == ["ATC-46"]


async def test_implicit_attachment_citation_never_uses_a_different_retrieved_key():
    attachments_result = IssueAttachmentsResult(
        total_count=1,
        attachments=[{
            "issue_id": 5000936,
            "issue_key": "ATC-47",
            "source_id": 5000941,
            "page_number": 1,
            "provenance": {"source_type": "pdf", "page_number": 1},
            "content": "A different attachment happens to mention Singapore.",
        }],
    )
    llm = FakeLLM([
        _issue_attachments_turn({"issue_keys": ["ATC-47"]}),
        _end_turn("The destination city is Singapore."),
    ])
    retrieval = FakeRetrieval(attachments_result=attachments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("where should the ATC-46 attachment order be shipped?", space_ids=[7])

    assert result.abstained is True
    assert result.citations == []


async def test_implicit_attachment_citation_ignores_key_shaped_business_identifier():
    attachments_result = IssueAttachmentsResult(
        total_count=1,
        attachments=[{
            "issue_id": 5000935,
            "issue_key": "DVQPLH-2",
            "source_id": 5000940,
            "content": "Invoice INV-3300 has a total due of $37,824.00.",
        }],
    )
    llm = FakeLLM([
        _issue_attachments_turn({"issue_keys": ["DVQPLH-2"]}),
        _end_turn("The total due on invoice INV-3300 is $37,824.00."),
    ])
    retrieval = FakeRetrieval(attachments_result=attachments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(
        "Using only the attachments on DVQPLH-2, what is the total for invoice INV-3300?",
        space_ids=[7],
    )

    assert result.abstained is False
    assert result.text == "The total due on invoice INV-3300 is $37,824.00 (DVQPLH-2)."
    assert [citation.issue_key for citation in result.citations] == ["DVQPLH-2"]


async def test_whole_space_document_identifier_gets_unique_carrier_provenance():
    question = "What availability did Customer Portal achieve in health report HLT-439?"
    llm = FakeLLM([
        _tool_use_turn(question),
        _end_turn("Customer Portal achieved 99.87% availability in health report HLT-439."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit(
            "attachment:7",
            issue_key="DVQPLF2-7",
            chunk_type="attachment",
            content="SERVICE HEALTH REPORT HLT-439 | Customer Portal | Availability 99.87%",
        ),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[7])

    assert result.abstained is False
    assert result.text.endswith("(DVQPLF2-7).")
    assert [citation.issue_key for citation in result.citations] == ["DVQPLF2-7"]


async def test_document_id_in_citation_slot_is_replaced_by_verified_carrier():
    question = "What availability did Customer Portal achieve in health report HLT-439?"
    llm = FakeLLM([
        _tool_use_turn(question),
        _end_turn("Customer Portal achieved 99.87% availability (HLT-439)."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit(
            "attachment:7",
            issue_key="DVQPLF2-7",
            chunk_type="attachment",
            content="SERVICE HEALTH REPORT HLT-439 | Customer Portal | Availability 99.87%",
        ),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[7])

    assert result.abstained is False
    assert result.text == "Customer Portal achieved 99.87% availability (DVQPLF2-7)."
    assert [citation.issue_key for citation in result.citations] == ["DVQPLF2-7"]


async def test_duplicate_document_identifier_does_not_guess_between_carriers():
    question = "What availability is listed in health report HLT-439?"
    llm = FakeLLM([
        _tool_use_turn(question),
        _end_turn("Health report HLT-439 lists 99.87% availability."),
    ])
    retrieval = FakeRetrieval(default=[
        Hit("attachment:7", issue_key="DVQPLF2-7", content="HLT-439 availability 99.87%"),
        Hit("attachment:41", issue_key="DVQPLF2-41", content="Archived HLT-439 availability 99.87%"),
    ])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[7])

    assert result.abstained is True
    assert result.citations == []


_PRA_FILE_HIT = Hit(
    "attachment:5", issue_key="DVQFRESH-5", chunk_type="attachment",
    content="[Attachment: PRA-SS2-21-Outsourcing.pdf — issue DVQFRESH-5 — PDF page 40]\n"
            "9.9 Firms should require the provider to oversee sub-outsourced services.",
)
_OTHER_FILE_HIT = Hit(
    "attachment:6", issue_key="DVQFRESH-6", chunk_type="attachment",
    content="[Attachment: EBA-Outsourcing-Guidelines.pdf — issue DVQFRESH-6 — PDF page 3]\n"
            "Sub-outsourcing of critical functions needs prior notice.",
)


@pytest.mark.parametrize("citation", [
    "(PRA-SS2-21-Outsourcing.pdf, §9.9)",
    "(PRA-SS2-21-Outsourcing)",
])
async def test_an_answer_citing_the_attachment_file_gets_its_issue_key(citation):
    """Seen live: correct answers cited the PDF's file name instead of its issue key and abstained."""
    question = "What must firms require before agreeing to sub-outsourcing?"
    llm = FakeLLM([
        _tool_use_turn(question),
        _end_turn(f"Firms must require the provider to oversee sub-outsourced services {citation}."),
    ])
    retrieval = FakeRetrieval(default=[_PRA_FILE_HIT, _OTHER_FILE_HIT])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[7])

    assert result.abstained is False
    assert result.text.endswith(f"{citation} (DVQFRESH-5).")
    assert [citation.issue_key for citation in result.citations] == ["DVQFRESH-5"]


async def test_a_paragraph_number_alone_is_still_not_a_citation():
    question = "What must firms require before agreeing to sub-outsourcing?"
    llm = FakeLLM([
        _tool_use_turn(question),
        _end_turn("Firms must require the provider to oversee sub-outsourced services (para. 9.9)."),
    ])
    retrieval = FakeRetrieval(default=[_PRA_FILE_HIT, _OTHER_FILE_HIT])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask(question, space_ids=[7])

    assert result.abstained is True


async def test_multi_page_attachment_keeps_every_page_as_a_distinct_citation():
    # Regression for a real bug found live: every page of one multi-page PDF attachment shares the
    # same source_id (one uploaded file, many parsed pages), and the tool-dispatch loop dedupes
    # citations by RetrievedChunk.id. When id was built from source_id alone, every page after the
    # first looked like a duplicate of page 1 and was silently dropped before _ground_answer ever
    # saw it — so a question whose answer lived on page 4 of 4 always cited page 1 instead, even
    # though the model (which sees the full unfiltered multi-page text, not the deduped citations
    # list) answered correctly. id must include page_number so same-source_id pages stay distinct.
    attachments_result = IssueAttachmentsResult(
        total_count=4,
        attachments=[
            {
                "issue_id": 5000935, "issue_key": "ATC-46", "source_id": 5000940,
                "page_number": page,
                "provenance": {"source_type": "pdf", "page_number": page},
                "content": f"page {page} content" if page != 4 else "SKU A-104, order BETA-1043, qty 1",
            }
            for page in (1, 2, 3, 4)
        ],
    )
    llm = FakeLLM([
        _tool_use_turn("ATC-46 attachment"),
        _issue_attachments_turn({"issue_keys": ["ATC-46"]}),
        _end_turn("The worked example uses SKU A-104, order BETA-1043, quantity 1 (ATC-46)."),
    ])
    retrieval = FakeRetrieval(
        default=[],
        attachments_result=attachments_result,
    )
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what SKU does ATC-46's attachment use?", space_ids=[7])

    assert result.abstained is False
    # Before the fix, every page collided on id=f"issue_attachments:{source_id}" and only the first
    # one seen (page 1, containing none of the answer's facts) ever reached _ground_answer's ranking
    # — so the citation was always page 1 regardless of which page the answer actually came from.
    # With page_number included in id, all 4 pages reach the ranking step and the one with real
    # word-overlap against the answer (page 4) correctly wins.
    assert any(c.page_number == 4 and "A-104" in c.content for c in result.citations)
    assert all(c.page_number != 1 for c in result.citations)


async def test_attachment_followup_uses_history_and_exact_reader_without_forced_semantic_search():
    attachments_result = IssueAttachmentsResult(
        total_count=1,
        attachments=[{
            "issue_id": 5001731,
            "issue_key": "ATC-432",
            "source_id": 5001517,
            "content": "The screenshot shows flight BR-857 departing at 09:30.",
            "provenance": {"source_type": "image"},
        }],
    )
    llm = FakeLLM([
        _issue_attachments_turn({"issue_keys": ["ATC-432"]}),
        _end_turn("ATC-432's attachment shows flight BR-857 departing at 09:30 (ATC-432)."),
    ])
    retrieval = FakeRetrieval(attachments_result=attachments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)
    history = [
        {"role": "user", "content": "what is the most recently updated issue?"},
        {"role": "assistant", "content": "ATC-432 is the most recently updated issue."},
    ]

    result = await agent.ask("what does its attachment say?", [5000018], history=history)

    # No keyword/key is present in the follow-up itself, so the AI router uses conversation context
    # to select the exact attachment reader instead of dropping immediately into semantic search.
    assert len(llm.router_calls) == 1
    assert llm.tool_choices_seen[0] == "get_issue_attachments"
    assert retrieval.calls == []
    assert retrieval.attachments_calls == [((5000018,), ("ATC-432",))]
    assert "BR-857" in result.text


async def test_citation_picks_the_chunk_that_actually_supports_the_answer_not_the_first_one():
    # Regression, found live: get_issue_comments can return MANY comments for one issue, all tied at
    # score=0.0 (it's deliberately unranked — see its docstring). The old citation selection took the
    # FIRST chunk seen for a cited issue key, which was a safe shortcut when every citation came from
    # search_knowledge_base's already-relevance-ranked top-K, but breaks badly here: a question about
    # why ATC-30 was reopened got a correct answer grounded in comment #4 of 6, while the UI's
    # "Sources" showed comment #1 — an unrelated story-point-estimate note that merely happened to be
    # chronologically first. Selection must prefer the chunk whose content actually overlaps with
    # what the answer says, not an arbitrary same-issue chunk.
    comments_result = IssueCommentsResult(
        total_count=2,
        comments=[
            {"issue_id": 30, "issue_key": "ATC-30", "source_id": 1,
             "content": "The first estimate missed price snapshots and safe retries. Moving this "
                        "to eight before implementation starts."},
            {"issue_id": 30, "issue_key": "ATC-30", "source_id": 2,
             "content": "Reopening this because a beta incident showed one browser action could "
                        "submit twice. Not complete until the API handles repeated requests safely."},
        ],
    )
    llm = FakeLLM([
        _tool_use_turn("ATC-30 extra work reason"),
        _issue_comments_turn({"issue_keys": ["ATC-30"]}),
        _end_turn("ATC-30 needed the extra work because a beta incident showed one browser action "
                  "could submit twice, and the API needed to handle repeated requests safely (ATC-30)."),
    ])
    retrieval = FakeRetrieval(default=[], comments_result=comments_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("why did ATC-30 need extra work?", space_ids=[7])

    assert result.abstained is False
    assert len(result.citations) == 1
    assert "submit twice" in result.citations[0].content
    assert "price snapshots" not in result.citations[0].content


async def test_field_change_description_is_included_when_it_carries_the_actual_reason():
    # Regression: _format_issue_history used to drop `description` entirely for field_change events,
    # even when it held the real narrative (e.g. a reopen reason an actor typed in) rather than a
    # generic "Updated status" — silently hiding the answer to "why" from the model even though it
    # was right there in the structured tool result it already had.
    result = IssueHistoryResult(
        total_count=1,
        changes=[{
            "issue_key": "ATC-68", "event_type": "field_change", "field_name": "status",
            "from_value": "done", "to_value": "in_progress",
            "description": "Daniel reopened the issue after the resent-email check failed",
            "actor_name": "Daniel Park", "changed_at": "2026-07-02T16:00:00Z",
        }],
    )

    text = _format_issue_history(result, {})

    assert "resent-email check failed" in text

    # A generic, contentless description shouldn't clutter the line with redundant text.
    generic = IssueHistoryResult(
        total_count=1,
        changes=[{
            "issue_key": "ATC-30", "event_type": "field_change", "field_name": "status",
            "from_value": "in_progress", "to_value": "done", "description": "Updated status",
            "actor_name": "Noah Kim", "changed_at": "2026-05-13T21:50:00Z",
        }],
    )
    generic_text = _format_issue_history(generic, {})
    assert generic_text.count("Updated status") == 0


async def test_query_sprints_tool_is_dispatched_and_reports_exact_count():
    result_data = SprintQueryResult(
        total_count=7,
        counts_by_status={"completed": 6, "active": 1},
        sprints=[{"sprint_id": 501, "sprint_name": "Sprint 7", "status": "active",
                  "goal": "Prove the storefront is accessible.", "start_date": "2026-07-13",
                  "end_date": "2026-07-24", "completed_points": 12, "final_scope_points": 20,
                  "initial_committed_points": 18}],
    )
    llm = FakeLLM([_sprint_query_turn({"statuses": ["active"]}), _end_turn("Sprint 7 is active.")])
    retrieval = FakeRetrieval(sprint_result=result_data)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("which sprint is active?", space_ids=[5000014])

    assert len(retrieval.sprint_calls) == 1
    assert retrieval.calls == [] and retrieval.issue_calls == []  # only the sprint tool fired
    assert "Sprint 7" in result.text


async def test_query_sprints_space_ids_injected_and_input_whitelisted():
    llm = FakeLLM([
        _sprint_query_turn({"statuses": ["active"], "space_ids": [999], "evil": "x"}),
        _end_turn("done"),
    ])
    retrieval = FakeRetrieval()
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("which sprint is active?", space_ids=[42, 43])

    # One query per workspace, each one of the caller's, not the model's 999.
    assert [call[0] for call in retrieval.sprint_calls] == [(42,), (43,)]
    for _, filters in retrieval.sprint_calls:
        assert filters == {"statuses": ["active"]}  # space_ids/evil stripped


async def test_verification_still_fires_on_the_forced_stop_path():
    # Regression for eval Finding 2: previously the correction round was implemented by looping
    # back into the SAME iteration counter that bounds the model's own tool-calling, gated by
    # `iteration < max_iterations + 1` — a reserved slot that the model's own research could consume
    # first. With max_iterations=2, a model that keeps calling tools right up to the cap forces the
    # final answer through the tools=[] forced-stop branch, which used to return immediately without
    # ever running the post-generation verifiers. Here the forced answer wrongly presents ATC-46 (a
    # different bug) as an equivalent subtask alongside the real subtask ATC-44 — this must still get
    # caught and corrected even though it came from the forced-stop branch, not a natural end_turn.
    mismatch_text = "ATC-43's follow-ups are ATC-44 and ATC-46."
    corrected_text = "ATC-43's real subtask is ATC-44; ATC-46 is a separate, related bug."
    turns = [_tool_use_turn(f"query {i}", call_id=f"c{i}") for i in range(3)]
    turns.append(_end_turn(mismatch_text))  # the forced final call (tools=[])
    turns.append(_end_turn("EQUIVALENT"))  # _check_subtask_claims' classification call
    turns.append(_end_turn(corrected_text))  # the correction round's own forced call (tools=[])
    llm = FakeLLM(turns)
    issue_result = IssueQueryResult(
        total_count=2, counts_by_type={}, counts_by_status={},
        issues=[
            {"issue_key": "ATC-44", "issue_type": "subtask", "parent_key": "ATC-43"},
            {"issue_key": "ATC-46", "issue_type": "bug", "parent_key": "ATC-4"},
        ],
    )
    retrieval = FakeRetrieval(
        default=[
            Hit("h0", issue_key="ATC-43"), Hit("h1", issue_key="ATC-44"), Hit("h2", issue_key="ATC-46"),
        ],
        issue_result=issue_result,
    )
    agent = CragAgent(llm, retrieval, max_iterations=2, top_k=5)

    result = await agent.ask("Summarize the follow-ups to ATC-43.", space_ids=[7])

    assert result.text == corrected_text
    assert result.abstained is False
    # Regression: found live in the v2 eval — the forced-stop branch called the correction round
    # without ever appending the forced final answer (`final.assistant_message`) to `messages`, so
    # the model doing the correction had no assistant turn in its own history to actually revise;
    # asking it to "revise your answer" against a conversation where it never said that answer is
    # not a real correction request. The LAST call the FakeLLM saw (the correction call) must include
    # the forced-stop text as a real assistant message.
    last_messages_seen = llm.messages_seen[-1]
    assert any(
        m.get("role") == "assistant" and m.get("content") == [{"type": "text", "text": mismatch_text}]
        for m in last_messages_seen
    )


async def test_query_issues_reports_parent_key_for_epic_lookup_questions():
    # Finding 4: the parent_key column was already threaded end-to-end into query_issues' response,
    # but nothing told the model it could use it to answer "which epic is X in" — it had no
    # deterministic way to answer that question at all. The formatted tool result must surface it.
    result_data = IssueQueryResult(
        total_count=1, counts_by_type={}, counts_by_status={},
        issues=[{"issue_key": "ATC-77", "issue_type": "story", "status": "blocked",
                 "sprint_name": "Sprint 7", "parent_key": "ATC-52", "updated_at": "2026-07-27",
                 "title": "Cache the public product catalog safely"}],
    )
    llm = FakeLLM([_issue_query_turn({"issue_keys": ["ATC-77"]}), _end_turn("ATC-77's epic is ATC-52.")])
    retrieval = FakeRetrieval(issue_result=result_data)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("which epic does ATC-77 belong to?", space_ids=[5000014])

    assert "ATC-52" in result.text


async def test_query_issues_reports_assignee_and_story_points():
    # Found live on "Who is assigned to ATC-72, and what is its story point estimate?": the row
    # vectorization-service returns carries assignee_name/story_points/priority, but the renderer
    # dropped all three, so the agent answered "ATC-72 has no assignee recorded" about an issue
    # assigned to Ava Patel. Same class as the parent_key gap above — the data was threaded end to
    # end and then thrown away one line before the model could see it.
    result_data = IssueQueryResult(
        total_count=1, counts_by_type={}, counts_by_status={},
        issues=[{"issue_key": "ATC-72", "issue_type": "story", "status": "blocked",
                 "priority": "high", "assignee_name": "Ava Patel", "story_points": 5,
                 "sprint_name": "AtlasCart Sprint 7", "parent_key": "ATC-51",
                 "updated_at": "2026-08-11", "title": "Index products for catalog search"}],
    )
    rendered = _format_issue_query(result_data, {"issue_keys": ["ATC-72"]})

    assert "assignee=Ava Patel" in rendered
    assert "points=5" in rendered
    assert "priority=high" in rendered


async def test_query_issues_marks_an_empty_assignee_as_a_real_answer():
    # The other half of the same fix: a genuinely unassigned issue must render as an explicit
    # "(unassigned)" rather than vanish from the row, or the model cannot tell "nobody owns this"
    # apart from "this tool doesn't report owners" — which is exactly the confusion that produced
    # the false negative above.
    result_data = IssueQueryResult(
        total_count=1, counts_by_type={}, counts_by_status={},
        issues=[{"issue_key": "ATC-433", "issue_type": "task", "status": "planned",
                 "priority": None, "assignee_name": None, "story_points": None,
                 "sprint_name": "AtlasCart Sprint 7", "parent_key": None,
                 "updated_at": "2026-08-11", "title": "catalog list test1"}],
    )
    rendered = _format_issue_query(result_data, {"issue_keys": ["ATC-433"]})

    assert "assignee=(unassigned)" in rendered
    assert "points=(unestimated)" in rendered


async def test_correction_message_instructs_the_model_not_to_narrate_the_fix():
    # Finding 7: a live run showed the corrected answer leaking "You're right, let me revise..." to
    # the end user, who never saw the flagged/original answer and has no idea a correction happened.
    from app.agent.crag_loop import _verification_correction_message, ClaimMismatch

    message = _verification_correction_message([ClaimMismatch(issue_key="ATC-46", detail="some detail")])

    content = message["content"].lower()
    assert "do not mention this correction" in content
    assert "never saw" in content


async def test_current_state_verifier_corrects_a_stale_status_claim():
    # Finding 1: deep multi-hop synthesis can let an earlier comment's state win out over a later,
    # authoritative status change (ground rule 4 says to prefer current state) — this is the same
    # registry/one-shot-correction architecture already proven for subtask claims,
    # extended to a second claim type instead of adding a new bespoke mechanism.
    stale_text = "ATC-43 is still open and unresolved."
    corrected_text = "ATC-43 is done."
    llm = FakeLLM([
        _tool_use_turn("checkout bug status"),
        _end_turn(stale_text),
        _end_turn('["ATC-43"]'),  # _check_current_state_claims' classification call
        _end_turn(corrected_text),  # the correction round's forced call (tools=[])
    ])
    issue_result = IssueQueryResult(
        total_count=1, counts_by_type={}, counts_by_status={},
        issues=[{"issue_key": "ATC-43", "issue_type": "bug", "status": "done"}],
    )
    retrieval = FakeRetrieval(default=[Hit("issue:43", issue_key="ATC-43")], issue_result=issue_result)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is the status of ATC-43?", space_ids=[7])

    assert result.text == corrected_text


async def test_current_state_verifier_does_not_fire_without_a_status_word():
    # The regex is only a cost-gating precondition, not a whack-a-mole detector — an answer with no
    # status-like vocabulary at all should never trigger the extra query_issues call.
    llm = FakeLLM([_tool_use_turn("checkout bug details"), _end_turn("ATC-43 is about a checkout bug.")])
    retrieval = FakeRetrieval(default=[Hit("issue:43", issue_key="ATC-43")])
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("what is ATC-43 about?", space_ids=[7])

    assert retrieval.issue_calls == []
    assert result.text == "ATC-43 is about a checkout bug."


def _search_tool_turn(tool_input: dict, call_id: str = "s_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="search_knowledge_base", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "search_knowledge_base",
             "input": tool_input}
        ]},
    )


# --- Model-sized retrieval: mode coercion, result_count, alternate phrasings -----------------------
#
# These test the CONTRACT between the tool schema and retrieval, not any particular question. The
# behaviours below are the ones that were either silently broken (an unknown mode reaching the
# retrieval service as a 422 that looks like an empty index) or newly delegated to the model
# (how many results, how many angles), so each needs a guard that does not depend on corpus content.


@pytest.mark.parametrize("raw,expected", [
    ("hybrid", "hybrid"),
    ("vector", "vector"),
    ("bm25", "bm25"),
    # 'lexical' was a real mode until the OpenSearch migration retired it. A model that has seen the
    # older description — or simply guesses — must not take retrieval down with it.
    ("lexical", "hybrid"),
    ("", "hybrid"),
    (None, "hybrid"),
    (7, "hybrid"),
])
def test_unknown_search_mode_falls_back_instead_of_reaching_the_retrieval_service(raw, expected):
    assert crag_loop._coerce_search_mode(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    (None, 5),          # absent -> configured default
    (1, 1),
    (20, 20),
    (0, 1),             # below range -> clamped up, never a zero-result search
    (-3, 1),
    (500, 20),          # above range -> clamped to the budget ceiling
    ("12", 5),          # wrong type -> default rather than a crash
    (True, 5),          # bool is an int subclass in Python; must not read as 1
])
def test_result_count_is_clamped_into_the_budget(raw, expected):
    assert crag_loop._coerce_result_count(raw, 5) == expected

def test_search_tool_advertises_only_modes_the_retrieval_service_accepts():
    # The schema and the coercion table must not drift apart: the schema is what the model reads,
    # the table is what actually reaches the network.
    schema_modes = _SEARCH_TOOL["input_schema"]["properties"]["mode"]["enum"]
    assert tuple(schema_modes) == crag_loop._SEARCH_MODES


def test_search_tool_result_count_ceiling_matches_the_enforced_clamp():
    schema_max = _SEARCH_TOOL["input_schema"]["properties"]["result_count"]["maximum"]
    assert schema_max == crag_loop._MAX_RESULT_COUNT
    assert crag_loop._coerce_result_count(schema_max + 1, 5) == crag_loop._MAX_RESULT_COUNT

def test_omitting_the_new_fields_preserves_the_previous_single_query_behaviour():
    hit = Hit("1", issue_key="A-1")
    retrieval = FakeRetrieval(results_by_query={"just this": [hit]}, default=[])
    llm = FakeLLM([
        _search_tool_turn({"query": "just this"}),
        LLMTurn(stop_reason="end_turn", text="Answer (A-1).", tool_calls=[]),
    ])
    agent = crag_loop.CragAgent(llm=llm, retrieval=retrieval, max_iterations=3, top_k=5)

    result = asyncio.run(agent.ask("q", space_ids=[7]))

    assert [c[0] for c in retrieval.calls] == ["just this"]
    assert {c[2] for c in retrieval.calls} == {5}  # falls back to configured top_k
    assert result.queries_used == ["just this"]

def test_issue_query_tool_documents_the_scan_strategy_and_its_ceiling():
    # The scan path is delivered entirely through this description, so its two load-bearing claims
    # need a guard: that scanning is described at all, and that the limit it tells the model to
    # raise is one the downstream service actually accepts.
    desc = crag_loop._ISSUE_QUERY_TOOL["description"]
    assert "SCANNING" in desc
    limit_desc = crag_loop._ISSUE_QUERY_TOOL["input_schema"]["properties"]["limit"]["description"]
    assert "200" in limit_desc


def test_scan_guidance_forbids_treating_a_truncated_list_as_a_complete_scan():
    # The scan strategy is delivered entirely through this description, and its correctness rests on
    # one thing the model must not get wrong: a scan is only a scan if the whole list arrived. On a
    # workspace larger than the limit, `query_issues` returns an arbitrary slice with a
    # "showing first N of TOTAL" marker, and a model that misses that will answer "these are all the
    # ones that match" about a slice — confidently, and wrongly. That failure is invisible on a small
    # corpus like the eval fixture, which is exactly why it needs a test rather than a manual check.
    desc = crag_loop._ISSUE_QUERY_TOOL["description"]
    assert "showing first N of TOTAL" in desc
    assert "Never present a truncated scan as a complete one." in desc
    # And the narrow-first ordering, which is what keeps the scan bounded at all.
    assert "narrow with the structured filters" in desc.lower() or "narrow the filters" in desc.lower()


# --- ask_clarification: structured "which source?" instead of a question written into the answer --
#
# Ground rule 2b already required the model to ask a clarification question when it finds several
# independently plausible sources with no way to tell them apart. Before this tool existed, that
# question was just the final answer's TEXT — a UI has no way to tell "this is a question" from
# "this is an answer" by looking at the same string, so a user got a well-formed clarification
# rendered exactly like a completed answer, with no picker and no structured way to respond. Found
# live via RAGAS eval, not by design review: two of the lowest-scoring answer_relevancy samples were
# CORRECT clarification questions the metric had no way to recognize as anything but a bad answer.


def _clarification_tool_turn(tool_input: dict, call_id: str = "c_1") -> LLMTurn:
    return LLMTurn(
        stop_reason="tool_use",
        text="",
        tool_calls=[ToolCall(id=call_id, name="ask_clarification", input=tool_input)],
        assistant_message={"role": "assistant", "content": [
            {"type": "tool_use", "id": call_id, "name": "ask_clarification", "input": tool_input}
        ]},
    )


def test_ask_clarification_ends_the_turn_with_structured_options():
    hit = Hit("1", issue_key="INV-1")
    retrieval = FakeRetrieval(default=[hit])
    llm = FakeLLM([
        _tool_use_turn("invoice total due"),
        _clarification_tool_turn({
            "question": "Which invoice do you mean?",
            "options": [
                {"label": "Kestrel invoice (INV-1)", "issue_key": "inv-1"},
                {"label": "Redwood invoice (INV-2)", "issue_key": "INV-2"},
            ],
        }),
    ])
    agent = crag_loop.CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = asyncio.run(agent.ask("What is the total amount due?", space_ids=[7]))

    assert result.needs_clarification is True
    assert result.text == "Which invoice do you mean?"
    assert result.abstained is False  # a clarification is not "the corpus has nothing"
    assert [o.label for o in result.clarification_options] == [
        "Kestrel invoice (INV-1)", "Redwood invoice (INV-2)",
    ]
    # Issue keys are normalized (uppercased) the same way every other tool result's keys are.
    assert [o.issue_key for o in result.clarification_options] == ["INV-1", "INV-2"]
    # The loop must not keep iterating after this — it is a terminal state, not a retrieval step.
    assert llm.calls == 2


@pytest.mark.parametrize("bad_input", [
    {"question": "Which one?", "options": [{"label": "Only one candidate"}]},
    {"question": "   ", "options": [{"label": "A"}, {"label": "B"}]},
    {"question": "   ", "options": []},  # blank question, even with no options, is still malformed
    {"options": [{"label": "A"}, {"label": "B"}]},  # question omitted entirely
])
def test_malformed_clarification_call_is_a_failed_tool_call_not_a_broken_clarification(bad_input):
    # A clarification with exactly one option (not a real choice) or no question would show the user
    # something broken — worse than not asking at all. Treating this as a failed tool call (the same
    # `is_error` path any other bad tool_input takes) gives the model a chance to fix the call or
    # fall back to a normal answer, instead of silently returning a broken clarification. Zero
    # options is NOT in this list — that's the valid "genuinely open question" shape, covered below.
    llm = FakeLLM([
        _tool_use_turn("invoice total due"),
        _clarification_tool_turn(bad_input),
        _end_turn("Fell back to a normal answer (INV-1)."),
    ])
    agent = crag_loop.CragAgent(llm, FakeRetrieval(default=[Hit("1", issue_key="INV-1")]),
                                 max_iterations=4, top_k=5)

    result = asyncio.run(agent.ask("What is the total amount due?", space_ids=[7]))

    assert result.needs_clarification is False
    failed_result = llm.messages_seen[2][-1]["content"][0]
    assert failed_result["is_error"] is True


def test_clarification_options_with_missing_labels_are_dropped_not_crashed_on():
    # `label` is the only required field per-option; a malformed entry (missing it, or not even a
    # dict — a model can emit anything) is filtered rather than raising, as long as at least 2 valid
    # options remain once filtered.
    llm = FakeLLM([
        _tool_use_turn("invoice total due"),
        _clarification_tool_turn({
            "question": "Which one?",
            "options": [
                {"label": "Kestrel (INV-1)"},
                {"issue_key": "INV-2"},  # no label — dropped
                "not even a dict",       # dropped
                {"label": "Redwood (INV-2)", "issue_key": "INV-2"},
            ],
        }),
    ])
    agent = crag_loop.CragAgent(llm, FakeRetrieval(default=[Hit("1")]), max_iterations=4, top_k=5)

    result = asyncio.run(agent.ask("q", space_ids=[7]))

    assert result.needs_clarification is True
    assert [o.label for o in result.clarification_options] == [
        "Kestrel (INV-1)", "Redwood (INV-2)",
    ]


def test_is_answer_cacheable_excludes_a_clarification_turn():
    # A clarification question is not a completed answer — caching it risks handing a later caller
    # (who phrased the same question with a disambiguating detail this run's search happened not to
    # surface) the question they already implicitly answered, since the cache key is question text
    # alone and carries none of the retrieval context that made this run ambiguous.
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=False, needs_clarification=False) is True
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=False, needs_clarification=True) is False
    # Default parameter value preserves every pre-existing call site's behavior unchanged.
    assert is_answer_cacheable(SEARCH_TOOL_NAME, abstained=False) is True


def test_clarification_tool_is_not_a_valid_first_move_routing_target():
    # ask_clarification only makes sense once retrieval has already run and returned candidates to
    # be ambiguous between — the router classifies BEFORE any retrieval, so there is nothing yet to
    # disambiguate. It must never appear as something the capability router or the deterministic
    # keyword layer can select as the FIRST tool of a turn.
    assert crag_loop._CLARIFICATION_TOOL["name"] not in crag_loop._EXECUTABLE_TOOL_NAMES


# --- Clarification option labels must actually tell the candidates apart -------------------------
#
# Reported from the live UI: all three options rendered as "<issue title> — diagnostic log" under the
# question "Which diagnostic log do you mean?". The repeated half carries no information (the
# question already said "diagnostic log"), and three options that end identically are harder to scan
# than three that don't. The same question had returned clean labels through the API minutes
# earlier, so it is model variance — the tool description asks for distinct labels, this enforces it.


def test_shared_suffix_repeated_on_every_option_is_stripped():
    assert crag_loop._strip_shared_label_suffix([
        "Tracking link opens a different beta order — diagnostic log",
        "Edited product remains stale in search — diagnostic log",
        "Cart merge doubles a matching SKU — diagnostic log",
    ]) == [
        "Tracking link opens a different beta order",
        "Edited product remains stale in search",
        "Cart merge doubles a matching SKU",
    ]


def test_a_trailing_phrase_that_actually_differs_is_kept():
    # The mirror case, and the reason this can't just drop everything after a dash: here the
    # document type is exactly what distinguishes the options, so removing it would destroy the
    # disambiguation instead of sharpening it.
    labels = [
        "Upload product images to managed storage — completion record",
        "Show the order confirmation — acceptance report",
    ]
    assert crag_loop._strip_shared_label_suffix(labels) == labels


def test_shared_trailing_word_without_a_separator_is_still_stripped():
    assert crag_loop._strip_shared_label_suffix(
        ["March invoice", "April invoice"]
    ) == ["March", "April"]


def test_a_shared_word_that_is_not_at_the_end_is_left_alone():
    # Only a common SUFFIX is noise-by-construction; a shared word mid-label may well be load-bearing.
    labels = ["Invoice for Kestrel", "Invoice for Redwood"]
    assert crag_loop._strip_shared_label_suffix(labels) == labels


def test_partial_word_overlap_is_not_treated_as_a_shared_suffix():
    # Whole-word comparison: "Holdings" only exists on one side, and character-level matching would
    # wrongly strip a shared "s" or "gs" tail.
    labels = ["Kestrel", "Kestrel Holdings"]
    assert crag_loop._strip_shared_label_suffix(labels) == labels


def test_stripping_never_leaves_an_empty_label():
    # One label consisting only of the shared phrase would otherwise render as a blank button —
    # worse than the redundancy this is fixing, so the whole set is left as the model wrote it.
    labels = ["diagnostic log", "Cart merge doubles a matching SKU diagnostic log"]
    assert crag_loop._strip_shared_label_suffix(labels) == labels


def test_a_single_option_is_returned_untouched():
    assert crag_loop._strip_shared_label_suffix(["Only one"]) == ["Only one"]
    assert crag_loop._strip_shared_label_suffix([]) == []


def test_redundant_suffixes_are_gone_by_the_time_the_answer_leaves_the_agent():
    # End-to-end through the real dispatch path: what the model emitted vs. what a caller receives.
    llm = FakeLLM([
        _tool_use_turn("diagnostic log"),
        _clarification_tool_turn({
            "question": "Which diagnostic log do you mean?",
            "options": [
                {"label": "Tracking link opens a different beta order — diagnostic log",
                 "issue_key": "ATC-68"},
                {"label": "Cart merge doubles a matching SKU — diagnostic log",
                 "issue_key": "ATC-59"},
            ],
        }),
    ])
    agent = crag_loop.CragAgent(llm, FakeRetrieval(default=[Hit("1")]), max_iterations=4, top_k=5)

    result = asyncio.run(agent.ask("What is mentioned in the diagnostic log?", space_ids=[7]))

    assert [o.label for o in result.clarification_options] == [
        "Tracking link opens a different beta order",
        "Cart merge doubles a matching SKU",
    ]
    # The keys still line up with their own labels after the rewrite.
    assert [o.issue_key for o in result.clarification_options] == ["ATC-68", "ATC-59"]


# --- ask_clarification with NO real candidates: the "what does the attachment say" failure mode ---
#
# Reported from the live UI: "What does the attachment say?" — no topic, no issue, no bug named —
# came back with 5 options, each a different unrelated issue's attachment (product images upload,
# checkout feedback, product-list requirements, cart-restore screenshots, header cart count
# screenshot). Those hits only share the literal word "attachment"; none is a real candidate answer
# to anything specific the user asked. Presenting them as a menu misrepresents an unnarrowed search
# as a thoughtful shortlist — a person asked something this vague asks what you mean, they don't
# guess-list five unrelated documents. `options: []` is the tool's way to say exactly that: a real
# clarifying question with nothing to enumerate, still rendered as a clarification turn (not prose).


def test_ask_clarification_with_no_real_candidates_is_a_valid_open_question():
    llm = FakeLLM([
        _tool_use_turn("attachment"),
        _clarification_tool_turn({
            "question": "Which issue or topic's attachment are you asking about?",
            "options": [],
        }),
    ])
    agent = crag_loop.CragAgent(
        llm, FakeRetrieval(default=[Hit("1", issue_key="ATC-71"), Hit("2", issue_key="ATC-31")]),
        max_iterations=4, top_k=5,
    )

    result = asyncio.run(agent.ask("What does the attachment say?", space_ids=[7]))

    assert result.needs_clarification is True
    assert result.text == "Which issue or topic's attachment are you asking about?"
    assert result.clarification_options == []
    # Terminal on the first attempt, same as the option-carrying case — not a failed/retried call.
    assert llm.calls == 2


def test_tool_schema_permits_zero_options_but_not_exactly_one():
    schema = crag_loop._CLARIFICATION_TOOL["input_schema"]["properties"]["options"]
    assert schema["minItems"] == 0


def test_ground_rule_2b_distinguishes_real_candidates_from_a_generic_word_coincidence():
    # The rule must name the exact failure mode reported live, not just the well-behaved case — a
    # rule that only ever shows the "multiple invoices" example gives the model no reason to treat
    # "what does the attachment say" any differently.
    system = render_crag_system_prompt("2026-08-11T00:00:00Z")
    assert "generic word from the question" in system
    assert '"the attachment"' in system
    assert "NONE of them is a real candidate for anything specific" in system
    assert "empty options list" in system


def test_issue_query_tells_the_model_exact_counts_per_assignee_most_first():
    """'Who has the most issues' came out 15 instead of 16 when the model counted rows itself."""
    result = IssueQueryResult(
        total_count=85, counts_by_type={}, counts_by_status={}, issues=[],
        counts_by_assignee={"Noah Kim": 13, "Ava Patel": 16, "(unassigned)": 3},
    )

    rendered = _format_issue_query(result, {})

    assert "by assignee (exact, most first): Ava Patel=16, Noah Kim=13, (unassigned)=3" in rendered


# --- Ask at most once: after a clarification the user's reply is the thing to work from -----------
#
# Seen in the UI: "What is the total amount due?" in a workspace with no invoices got an open
# question; the reply "I don't know, do you have options?" got the same question again.


def test_a_second_clarification_must_be_a_picker_not_an_open_question():
    llm = FakeLLM(
        [
            _tool_use_turn("total amount due"),
            _clarification_tool_turn({"question": "Which document do you mean?", "options": []}, "c_1"),
            _end_turn("There is no information about this in the knowledge base."),
        ],
        router_decisions=[("search_knowledge_base", "high", "ambiguous_single_source")],
    )
    agent = crag_loop.CragAgent(llm, FakeRetrieval(default=[Hit("1", issue_key="ATC-71")]), max_iterations=4, top_k=5)

    result = asyncio.run(agent.ask(
        "I don't know, do you have options?", space_ids=[7],
        history=[
            {"role": "user", "content": "What is the total amount due?"},
            {"role": "assistant", "content": "Which document do you mean?"},
        ],
        clarification_asked=True,
    ))

    # The open question was refused as a failed tool call, so the turn ended in an answer.
    assert result.needs_clarification is False
    assert all("cannot ask an open question again" in system for system in llm.systems_seen)
    assert all("source-scope classifier" not in system for system in llm.systems_seen)


def test_a_second_clarification_with_real_options_is_allowed():
    llm = FakeLLM(
        [
            _tool_use_turn("total amount due"),
            _clarification_tool_turn({
                "question": "Which invoice?",
                "options": [{"label": "Kestrel", "issue_key": "INV-1"}, {"label": "Redwood", "issue_key": "INV-2"}],
            }),
        ],
        router_decisions=[("search_knowledge_base", "high", "ambiguous_single_source")],
    )
    agent = crag_loop.CragAgent(
        llm, FakeRetrieval(default=[Hit("1", issue_key="INV-1"), Hit("2", issue_key="INV-2")]),
        max_iterations=4, top_k=5,
    )

    result = asyncio.run(agent.ask(
        "I don't know, do you have options?", space_ids=[7],
        history=[{"role": "user", "content": "total due?"}, {"role": "assistant", "content": "Which?"}],
        clarification_asked=True,
    ))

    assert result.needs_clarification is True
    assert len(result.clarification_options) == 2


def test_clarification_tool_is_offered_on_a_first_question():
    llm = FakeLLM([_tool_use_turn("total amount due"), _end_turn("x")])
    agent = crag_loop.CragAgent(llm, FakeRetrieval(default=[Hit("1", issue_key="ATC-71")]), max_iterations=4, top_k=5)

    asyncio.run(agent.ask("What is the total amount due?", space_ids=[7]))

    assert all("ask_clarification" in tools for tools in llm.tools_seen)


# --- Two or more workspaces: one structured query per workspace -----------------------------------
#
# A combined query returned one total and the model split the rows by key prefix itself; it read
# ATC 47 / PRM 31 as 42 / 36.


class _PerSpaceRetrieval(FakeRetrieval):
    def __init__(self, issues_by_space, sprints_by_space=None):
        super().__init__()
        self._issues_by_space = issues_by_space
        self._sprints_by_space = sprints_by_space or {}

    async def query_issues(self, space_ids, filters):
        self.issue_calls.append((tuple(space_ids), dict(filters)))
        return self._issues_by_space[space_ids[0]]

    async def query_sprints(self, space_ids, filters):
        self.sprint_calls.append((tuple(space_ids), dict(filters)))
        return self._sprints_by_space[space_ids[0]]


def _issues(total, keys):
    return IssueQueryResult(
        total_count=total, counts_by_type={}, counts_by_status={},
        issues=[{"issue_key": key, "status": "planned"} for key in keys],
    )


async def test_two_workspaces_get_one_issue_query_each_with_their_own_totals():
    llm = FakeLLM([_issue_query_turn({"priorities": ["high"]}), _end_turn("ATC has 47 and PRM has 31.")])
    retrieval = _PerSpaceRetrieval({1: _issues(47, ["ATC-1"]), 2: _issues(31, ["PRM-2"])})
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    answer = await agent.ask(
        "How many high-priority issues does each project have?", space_ids=[1, 2],
        space_labels={1: "ATC", 2: "PRM"},
    )

    assert retrieval.issue_calls == [((1,), {"priorities": ["high"]}), ((2,), {"priorities": ["high"]})]
    evidence = answer.structured_evidence[0]
    assert "combined total_count = 78 (ATC 47 + PRM 31)" in evidence
    assert "=== Workspace ATC ===" in evidence and "total_count = 47" in evidence
    assert "=== Workspace PRM ===" in evidence and "total_count = 31" in evidence


async def test_a_workspace_without_a_label_is_named_by_its_key_prefix_or_its_id():
    llm = FakeLLM([_issue_query_turn({"issue_types": ["bug"]}), _end_turn("ATC has 11 bugs, the other none.")])
    retrieval = _PerSpaceRetrieval({1: _issues(11, ["ATC-5"]), 3: _issues(0, [])})
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    answer = await agent.ask("How many bugs does each project have?", space_ids=[1, 3])

    evidence = answer.structured_evidence[0]
    assert "=== Workspace ATC ===" in evidence
    assert "=== Workspace space 3 ===" in evidence


async def test_one_workspace_keeps_a_single_query():
    llm = FakeLLM([_issue_query_turn({}), _end_turn("There are 47 issues.")])
    retrieval = _PerSpaceRetrieval({1: _issues(47, ["ATC-1"])})
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    answer = await agent.ask("How many issues are there?", space_ids=[1])

    assert retrieval.issue_calls == [((1,), {})]
    assert "Queried each workspace separately" not in answer.structured_evidence[0]


async def test_parent_keys_reach_the_issue_query_but_space_ids_do_not():
    llm = FakeLLM([
        _issue_query_turn({"parent_keys": ["ATC-52"], "space_ids": [999]}),
        _end_turn("ATC-52 has one open child, ATC-77."),
    ])
    retrieval = _PerSpaceRetrieval({1: _issues(1, ["ATC-77"])})
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    await agent.ask("Which children of ATC-52 are open?", space_ids=[1])

    assert retrieval.issue_calls == [((1,), {"parent_keys": ["ATC-52"]})]


async def test_two_workspaces_get_one_sprint_query_each():
    llm = FakeLLM([_sprint_query_turn({"statuses": ["completed"]}), _end_turn("Both have 6 completed sprints.")])
    sprints = {
        1: SprintQueryResult(total_count=6, counts_by_status={"completed": 6}, sprints=[]),
        2: SprintQueryResult(total_count=6, counts_by_status={"completed": 6}, sprints=[]),
    }
    retrieval = _PerSpaceRetrieval({}, sprints)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    answer = await agent.ask(
        "How many completed sprints does each project have?", space_ids=[1, 2],
        space_labels={1: "ATC", 2: "PRM"},
    )

    assert [call[0] for call in retrieval.sprint_calls] == [(1,), (2,)]
    assert "combined total_count = 12 (ATC 6 + PRM 6)" in answer.structured_evidence[0]


class _KeyCheckRetrieval(FakeRetrieval):
    """query_issues answers the agent's own query with `issue_result`, and the key check with
    whichever of the asked keys `existing` holds."""

    def __init__(self, existing, **kwargs):
        super().__init__(**kwargs)
        self.existing = set(existing)

    async def query_issues(self, space_ids, filters):
        self.issue_calls.append((tuple(space_ids), dict(filters)))
        if set(filters) == {"issue_keys", "limit"}:
            found = [{"issue_key": k} for k in filters["issue_keys"] if k in self.existing]
            return IssueQueryResult(total_count=len(found), counts_by_type={}, counts_by_status={}, issues=found)
        return self._issue_result


_EARLIER_LIST = [
    {"role": "user", "content": "List the high priority PRM issues that are not done."},
    {"role": "assistant", "content": "PRM-26 (in progress) and PRM-30 (planned)."},
]
_IN_PROGRESS_ONLY = IssueQueryResult(
    total_count=1, counts_by_type={}, counts_by_status={"in_progress": 1},
    issues=[{"issue_key": "PRM-26", "title": "Cache triage queue", "status": "in_progress"}],
)


async def test_a_follow_up_may_name_a_key_from_the_chat_once_it_is_confirmed():
    llm = FakeLLM([
        _issue_query_turn({"issue_keys": ["PRM-26", "PRM-30"], "statuses": ["in_progress"]}),
        _end_turn("PRM-26 is in progress. PRM-30 is planned, so it is left out."),
    ])
    retrieval = _KeyCheckRetrieval({"PRM-30"}, issue_result=_IN_PROGRESS_ONLY)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Which of those issues are in progress?", [7], history=_EARLIER_LIST)

    assert result.text.startswith("PRM-26 is in progress.")
    assert ((7,), {"issue_keys": ["PRM-30"], "limit": 1}) in retrieval.issue_calls


async def test_a_key_from_the_chat_that_does_not_exist_still_abstains():
    # The client sends history, so a key in it is checked, not trusted.
    llm = FakeLLM([
        _issue_query_turn({"issue_keys": ["PRM-26", "PRM-30"], "statuses": ["in_progress"]}),
        _end_turn("PRM-26 is in progress. PRM-30 is planned, so it is left out."),
    ])
    agent = CragAgent(llm, _KeyCheckRetrieval(set(), issue_result=_IN_PROGRESS_ONLY), max_iterations=4, top_k=5)

    result = await agent.ask("Which of those issues are in progress?", [7], history=_EARLIER_LIST)

    assert result.text == ABSTENTION_PHRASE


async def test_a_key_not_in_the_chat_or_the_results_abstains_without_a_key_check():
    llm = FakeLLM([
        _issue_query_turn({"statuses": ["in_progress"]}),
        _end_turn("PRM-26 is in progress, and so is PRM-99."),
    ])
    retrieval = _KeyCheckRetrieval({"PRM-99"}, issue_result=_IN_PROGRESS_ONLY)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Which of those issues are in progress?", [7], history=_EARLIER_LIST)

    assert result.text == ABSTENTION_PHRASE
    assert len(retrieval.issue_calls) == 1


def test_a_confirmed_chat_key_can_be_named_but_not_cited_as_the_source():
    this_turn = {"PRM-26"}
    ground = lambda text, chat=frozenset(): crag_loop._ground_answer(
        text, [], this_turn, this_turn, False, True, set(), chat,
    )[0]

    named = "PRM-26 is in progress. PRM-30 is planned, so it is left out."
    cited = "PRM-26 is in progress.\n\n(PRM-30)"

    assert ground(named, {"PRM-30"}) == named
    assert ground(named) == ABSTENTION_PHRASE  # not confirmed: still rejected
    assert ground(cited, {"PRM-30"}) == ABSTENTION_PHRASE  # confirmed, but not this turn's source


async def test_each_turn_starts_with_its_own_evidence():
    # Turn 2 returns nothing for PRM-30. Turn 1's evidence is not carried over, only its text.
    llm = FakeLLM([
        _issue_query_turn({"statuses": ["in_progress"]}),
        _end_turn("PRM-26 is in progress.\n\n(PRM-30)"),
    ])
    retrieval = _KeyCheckRetrieval({"PRM-30"}, issue_result=_IN_PROGRESS_ONLY)
    agent = CragAgent(llm, retrieval, max_iterations=4, top_k=5)

    result = await agent.ask("Which of those issues are in progress?", [7], history=_EARLIER_LIST)

    assert result.text == ABSTENTION_PHRASE

