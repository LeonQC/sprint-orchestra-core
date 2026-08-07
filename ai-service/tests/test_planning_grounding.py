"""Tests for retrieval-grounded epic planning (app/planning/grounding.py) — fake retrieval and fake
instructor client, no network.

The properties worth defending are the ones the feature's central claim rests on. That claim is
"this plan was written from your team's own record", and it is only worth making if:

  * only issues that were actually FINISHED AT A SIZE are offered as estimate anchors, and they are
    the ones that survive truncation (an anchor set of unfinished tickets would be a guess anchored
    to a guess);
  * a citation the model produces is verified against what it was actually shown, so a reviewer who
    clicks a key finds the issue the estimate says it was compared to;
  * a plan that could NOT be grounded says so, distinctly, rather than looking identical to one that
    was — a young space, a dead index and a disabled flag are three different sentences;
  * every one of those failures still produces a plan.
"""
from types import SimpleNamespace

import pytest

from app.agent.retrieval_tool import IssueQueryResult, RetrievedChunk, SearchResult
from app.api import routes
from app.planning.grounding import (
    Exemplar,
    GroundingContext,
    apply_estimate_grounding,
    gather_exemplars,
)
from app.planning.prompts import (
    render_critic_prompt,
    render_epic_plan_prompt,
    render_epic_plan_refine_prompt,
)
from app.planning.schemas import EpicDraft, EpicPlanDraft, IssueDraft
from app.planning.service import plan_epic, refine_plan
from tests.test_planning import FakeInstructorClient


class _Details:
    def __init__(self, details):
        self.total_count = len(details)
        self.details = details


class FakeRetrieval:
    """Stands in for RetrievalClient across the calls grounding makes: one semantic search, then two
    structured queries (metadata for what search found, and the recency backfill), then the bodies.

    The two `query_issues` calls are told apart the same way the real endpoint would: one carries
    `issue_keys`, the other carries a status/type/recency filter.
    """

    def __init__(self, hits, rows=None, details=None, recent=None, reranked=False, fail=False):
        self._hits = hits
        self._rows = rows or []
        self._details = details or []
        self._recent = recent or []
        self._reranked = reranked
        self._fail = fail
        self.queries = []
        self.key_lookups = []
        self.recent_filters = []

    async def search_with_meta(self, query, space_ids, limit, mode="hybrid"):
        self.queries.append((query, list(space_ids), mode))
        if self._fail:
            raise RuntimeError("vectorization-service unreachable")
        return SearchResult(hits=self._hits, reranked=self._reranked)

    async def query_issues(self, space_ids, filters):
        if "issue_keys" in filters:
            self.key_lookups.append(filters["issue_keys"])
            wanted = set(filters["issue_keys"])
            issues = [r for r in self._rows if r["issue_key"] in wanted]
        else:
            self.recent_filters.append(filters)
            issues = list(self._recent)
        return IssueQueryResult(
            total_count=len(issues), counts_by_type={}, counts_by_status={}, issues=issues,
        )

    async def get_issue_details(self, space_ids, issue_keys, limit=200):
        wanted = set(issue_keys)
        return _Details([d for d in self._details if d["issue_key"] in wanted])


def _chunk(issue_key, score=0.9, issue_id=1, chunk_type="issue"):
    return RetrievedChunk(
        id=f"c{issue_id}", chunk_type=chunk_type, issue_id=issue_id, issue_key=issue_key,
        source_id=issue_id, content="body", score=score, retrievers=["vector"],
    )


def _row(issue_key, status="done", points=5, issue_type="story", title="Past work"):
    return {
        "issue_key": issue_key, "status": status, "story_points": points,
        "issue_type": issue_type, "title": title,
    }


def _detail(issue_key, content="We shipped the thing. It took a migration."):
    return {"issue_key": issue_key, "content": content}


def _issue(temp_id="1", points=5, references=None):
    return IssueDraft(
        temp_id=temp_id, title=f"Issue {temp_id}", description="d", issue_type="task", labels=[],
        estimate_story_points=points, estimate_rationale="because" if points else None,
        estimate_reference=references or [], depends_on=[],
    )


# --- what gets retrieved, and what it is allowed to be used for -------------------------------


async def test_a_finished_sized_issue_becomes_an_estimate_anchor():
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1")], rows=[_row("ATC-1", status="done", points=8)],
        details=[_detail("ATC-1")],
    )

    context = await gather_exemplars(retrieval, [5], "add a tracking page", limit=8)

    assert context.status == "grounded"
    assert [e.issue_key for e in context.anchors] == ["ATC-1"]
    assert context.estimates_grounded is True


async def test_an_unfinished_issue_is_a_style_example_but_never_an_estimate_anchor():
    """The distinction the whole estimate half of this feature rests on: story points on work that
    hasn't finished are somebody's guess, which is exactly what the planner is producing. Anchoring
    to one would let the feature claim historical evidence it does not have."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-2")], rows=[_row("ATC-2", status="in_progress", points=13)],
        details=[_detail("ATC-2")],
    )

    context = await gather_exemplars(retrieval, [5], "add a tracking page", limit=8)

    assert context.status == "style_only"
    assert [e.issue_key for e in context.exemplars] == ["ATC-2"]  # kept, for how it is written
    assert context.anchors == []                                   # but not for how big it is
    assert context.estimates_grounded is False


async def test_anchors_survive_truncation_ahead_of_closer_unfinished_matches():
    """Ranking decides what survives `limit`, so a space whose nearest matches are all unfinished
    would otherwise arrive with zero anchors while finished work sat one row below the cut."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-9", 0.99, 1), _chunk("ATC-8", 0.95, 2), _chunk("ATC-7", 0.80, 3)],
        rows=[
            _row("ATC-9", status="in_progress", points=None),
            _row("ATC-8", status="in_progress", points=None),
            _row("ATC-7", status="done", points=3),
        ],
        details=[_detail("ATC-9"), _detail("ATC-8"), _detail("ATC-7")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=1)

    assert [e.issue_key for e in context.exemplars] == ["ATC-7"]
    assert context.status == "grounded"


async def test_hits_below_the_confidence_floor_and_non_issue_chunks_are_not_exemplars():
    retrieval = FakeRetrieval(
        hits=[
            _chunk("ATC-1", 0.9, 1),
            _chunk("ATC-2", 0.2, 2),                            # below the floor
            _chunk("ATC-3", 0.99, 3, chunk_type="comment"),     # incidental vocabulary, not substance
            _chunk("ATC-4", 0.99, 4, chunk_type="attachment"),
        ],
        rows=[_row(k) for k in ("ATC-1", "ATC-2", "ATC-3", "ATC-4")],
        details=[_detail(k) for k in ("ATC-1", "ATC-2", "ATC-3", "ATC-4")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-1"]


async def test_the_floor_is_applied_on_the_normalized_scale_not_the_raw_score():
    """With reranking on, `score` is an unbounded cross-encoder logit — a 0.4 logit is a *relevant*
    hit (sigmoid 0.6), and a raw-score floor of 0.5 would silently discard it. This is the same
    two-scale trap app/search/service.py exists to keep callers out of."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1", 0.4)], rows=[_row("ATC-1")], details=[_detail("ATC-1")],
        reranked=True,
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-1"]


async def test_an_issue_with_no_body_is_skipped():
    """A title alone teaches the planner only that this team writes titles."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1", 0.9, 1), _chunk("ATC-2", 0.8, 2)],
        rows=[_row("ATC-1"), _row("ATC-2")],
        details=[_detail("ATC-1", ""), _detail("ATC-2")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-2"]


async def test_html_is_stripped_before_the_text_is_embedded_or_shown():
    """Descriptions are stored as HTML and embedding the markup measurably wrecks retrieval — the
    finding app/precedent/prompts.py::plain_text records. Both the query and the exemplar bodies go
    through it."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1")], rows=[_row("ATC-1")],
        details=[_detail("ATC-1", "<p>Cache the <b>quotes</b></p>")],
    )

    context = await gather_exemplars(retrieval, [5], "<p>Add a <b>tracking</b> page</p>", limit=8)

    assert "<" not in retrieval.queries[0][0]
    assert retrieval.queries[0][0] == "Add a tracking page"
    assert context.exemplars[0].description == "Cache the quotes"


# --- the ways this can come up empty, each of which has to stay distinguishable ---------------


async def test_an_empty_space_reports_no_history_rather_than_a_silent_generic_plan():
    context = await gather_exemplars(FakeRetrieval(hits=[]), [5], "proposal", limit=8)

    assert context.status == "no_history"
    assert context.exemplars == []


async def test_a_dead_index_degrades_to_ungrounded_and_says_so_instead_of_raising():
    context = await gather_exemplars(FakeRetrieval(hits=[], fail=True), [5], "proposal", limit=8)

    assert context.status == "retrieval_failed"
    assert "unreachable" in context.error
    assert context.exemplars == []


async def test_a_blank_proposal_never_reaches_the_index():
    retrieval = FakeRetrieval(hits=[_chunk("ATC-1")])

    context = await gather_exemplars(retrieval, [5], "   ", limit=8)

    assert context.status == "no_history"
    assert retrieval.queries == []


# --- citation verification --------------------------------------------------------------------


def _context(*keys):
    return GroundingContext(
        status="grounded",
        exemplars=[Exemplar(k, "t", "story", "done", 5, "body", 0.9) for k in keys],
    )


def test_a_citation_naming_an_issue_the_model_was_never_shown_is_dropped_and_counted():
    issues = [_issue(references=["ATC-1", "ATC-999"])]

    cleaned, dropped = apply_estimate_grounding(issues, _context("ATC-1"))

    assert cleaned[0].estimate_reference == ["ATC-1"]
    assert dropped == 1


def test_a_failed_citation_costs_the_citation_not_the_estimate():
    """The number is still the planner's best judgement and is still useful; deleting it because its
    justification failed verification would be a worse plan, not a more honest one. What the reviewer
    sees is an estimate with no 'compared to' badge, which reads correctly as a judgement call."""
    cleaned, dropped = apply_estimate_grounding([_issue(points=8, references=["NOPE-1"])], _context("ATC-1"))

    assert cleaned[0].estimate_story_points == 8
    assert cleaned[0].estimate_reference == []
    assert dropped == 1


def test_a_reference_supporting_no_estimate_is_dropped():
    cleaned, dropped = apply_estimate_grounding(
        [_issue(points=None, references=["ATC-1"])], _context("ATC-1"),
    )

    assert cleaned[0].estimate_reference == []
    assert dropped == 1


def test_references_are_deduped_and_capped():
    issues = [_issue(references=["ATC-1", "ATC-1", "ATC-2", "ATC-3", "ATC-4"])]

    cleaned, dropped = apply_estimate_grounding(issues, _context("ATC-1", "ATC-2", "ATC-3", "ATC-4"))

    assert cleaned[0].estimate_reference == ["ATC-1", "ATC-2", "ATC-3"]
    assert dropped == 2


def test_with_no_grounding_at_all_every_reference_is_unverifiable():
    cleaned, dropped = apply_estimate_grounding([_issue(references=["ATC-1"])], None)

    assert cleaned[0].estimate_reference == []
    assert dropped == 1


def test_a_reference_already_on_the_plan_survives_a_later_call_that_retrieved_differently():
    """A refine round retrieves afresh (or, flagged off, not at all). Re-adjudicating citations that
    were verified when the plan was generated would silently strip good ones on every edit."""
    cleaned, dropped = apply_estimate_grounding(
        [_issue(references=["ATC-1"])], None, already_verified={"ATC-1"},
    )

    assert cleaned[0].estimate_reference == ["ATC-1"]
    assert dropped == 0


# --- what actually reaches the model -----------------------------------------------------------


def test_the_prompt_marks_which_past_issues_may_anchor_an_estimate():
    context = GroundingContext(status="grounded", exemplars=[
        Exemplar("ATC-1", "Add quote cache", "task", "done", 5, "Cached for 10 minutes.", 0.9),
        Exemplar("ATC-2", "Flaky test", "bug", "in_progress", None, "Sometimes fails.", 0.7),
    ])

    prompt = render_epic_plan_prompt([], context)

    assert "ESTIMATE BY COMPARISON" in prompt
    anchor_section = prompt[prompt.index("--- ATC-1 ---"):prompt.index("--- ATC-2 ---")]
    assert "SIZED BY OUTCOME" in anchor_section
    assert "not evidence of size" in prompt[prompt.index("--- ATC-2 ---"):]


def test_history_with_no_finished_sizes_grounds_the_style_but_not_the_estimates():
    context = GroundingContext(status="style_only", exemplars=[
        Exemplar("ATC-2", "Flaky test", "bug", "in_progress", None, "Sometimes fails.", 0.7),
    ])

    prompt = render_epic_plan_prompt([], context)

    assert "PAST WORK" in prompt                       # the style half still runs
    assert "ESTIMATE BY COMPARISON" not in prompt      # the estimate half does not
    assert "Leave `estimate_reference` empty" in prompt


def test_an_ungrounded_prompt_is_the_prompt_that_shipped_before_this_feature():
    prompt = render_epic_plan_prompt([], None)

    assert "PAST WORK" not in prompt
    assert "ESTIMATE BY COMPARISON" not in prompt


def test_the_critic_judges_estimates_against_the_same_history_the_planner_saw():
    """A critic holding a generic prior would flag correctly-anchored estimates as unrealistic and
    spend the revision budget pulling them back toward the average of its training data — undoing the
    feature one critique at a time."""
    context = _context("ATC-1")

    prompt = render_critic_prompt(context)

    assert "--- ATC-1 ---" in prompt
    assert "NOT against your own" in prompt
    assert "PAST WORK" not in render_critic_prompt(None)


def test_refine_is_told_to_carry_existing_references_through():
    prompt = render_epic_plan_refine_prompt([], _context("ATC-1"))

    assert "sized against" in prompt
    assert "--- ATC-1 ---" in prompt


# --- end to end through the planning service ----------------------------------------------------


async def test_plan_epic_shows_the_planner_the_history_and_verifies_what_it_cites_back():
    context = _context("ATC-1")
    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]),
        issues=[
            _issue("1", points=5, references=["ATC-1"]),
            _issue("2", points=3, references=["ATC-404"]),
        ],
    ))

    result = await plan_epic(client, "fake-model", "add tracking", [], context)

    assert "--- ATC-1 ---" in client.completions.calls[0]["messages"][0]["content"]
    assert result.plan.issues[0].estimate_reference == ["ATC-1"]
    assert result.plan.issues[1].estimate_reference == []
    assert result.plan.issues[1].estimate_story_points == 3
    assert result.dropped_estimate_references == 1


async def test_a_refine_round_does_not_strip_the_citations_the_plan_arrived_with():
    existing = [_issue("1", points=5, references=["ATC-1"])]
    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]), issues=existing,
    ))

    result = await refine_plan(
        client, "fake-model", EpicDraft(title="T", description="D", goals=[]), existing,
        "add a QA task", [], None,
    )

    assert result.plan.issues[0].estimate_reference == ["ATC-1"]
    assert result.dropped_estimate_references == 0


async def test_the_plan_the_refine_prompt_sees_includes_the_references_it_must_preserve():
    """_render_current_plan is the only view refine has of the current plan — anything it omits is
    silently destroyed by the next edit, which is why that function's docstring exists."""
    existing = [_issue("1", points=5, references=["ATC-1"])]
    client = FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]), issues=existing,
    ))

    await refine_plan(
        client, "fake-model", EpicDraft(title="T", description="D", goals=[]), existing,
        "add a QA task", [], None,
    )

    assert "Estimate sized against: ATC-1" in client.completions.calls[0]["messages"][1]["content"]


# --- the route: permission scope, and the receipts on the way out -------------------------------


def _route_request(*, grounding_enabled, retrieval=None, membership=None, client=None,
                   space_ok=True, pattern_extraction=False):
    from app.stats import ServiceStats

    async def _validate(user_id, username, space_ids):
        if not space_ok:
            raise routes.SpaceMembershipError("u1", list(space_ids))

    return SimpleNamespace(
        headers={"x-user-id": "u1", "x-username": "alice"},
        app=SimpleNamespace(state=SimpleNamespace(
            instructor_client=client, instructor_model="fake-model", planning_graph=None,
            model_provider="Self-hosted", model_deployment="local", stats=ServiceStats(),
            retrieval=retrieval,
            space_membership=membership or SimpleNamespace(validate=_validate),
            settings=SimpleNamespace(
                epic_planning_grounding_enabled=grounding_enabled,
                epic_planning_grounding_exemplars=8,
                epic_planning_requirement_exemplars=3,
                epic_planning_pattern_comments_enabled=False,
                # The history-reading stage is exercised in tests/test_planning_patterns.py; these
                # tests are about retrieval and the receipts, so it stays out of the way.
                epic_planning_pattern_extraction_enabled=pattern_extraction,
            ),
        )),
    )


def _plan_client(references=None):
    return FakeInstructorClient(EpicPlanDraft(
        epic=EpicDraft(title="T", description="D", goals=[]),
        issues=[_issue("1", points=5, references=references or [])],
    ))


async def test_a_grounded_response_carries_the_receipts_the_reviewer_can_check():
    """The trust claim is only worth making if the reader can open what it was based on: the exact
    issues, with the sizes that were offered as anchors."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1")], rows=[_row("ATC-1", status="done", points=5)],
        details=[_detail("ATC-1")],
    )
    request = _route_request(
        grounding_enabled=True, retrieval=retrieval, client=_plan_client(["ATC-1"]),
    )

    response = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(proposal="add tracking", requirements=["Provide shipment tracking"], space_id=5), request,
    )

    assert response.grounding.status == "grounded"
    assert [e.issue_key for e in response.grounding.exemplars] == ["ATC-1"]
    assert response.grounding.exemplars[0].estimate_anchor is True
    assert response.grounding.exemplars[0].story_points == 5
    assert response.grounding.estimate_anchor_count == 1
    assert response.issues[0].estimate_reference == ["ATC-1"]


async def test_a_plan_that_could_not_be_grounded_never_claims_it_was():
    """Same generic plan either way — but `status` is what stops a UI telling someone their team's
    history shaped a plan it never touched."""
    request = _route_request(grounding_enabled=False, client=_plan_client(["ATC-1"]))

    response = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(proposal="add tracking", requirements=["Provide shipment tracking"], space_id=5), request,
    )

    assert response.grounding.status == "disabled"
    assert response.grounding.exemplars == []
    # The flag is off, so nothing was retrieved and nothing licenses a citation.
    assert response.issues[0].estimate_reference == []
    assert response.grounding.dropped_estimate_references == 1


async def test_a_space_the_caller_is_not_a_member_of_is_refused_even_with_the_flag_off():
    """A space id in a request body names somebody's data. "We happened not to read it this time" is
    a coincidence, not an access control — so turning the flag on can never be what starts enforcing
    the boundary."""
    request = _route_request(grounding_enabled=False, client=_plan_client(), space_ok=False)

    with pytest.raises(routes.HTTPException) as exc:
        await routes.plan_epic_endpoint(
            routes.PlanEpicRequest(proposal="add tracking", requirements=["Provide shipment tracking"], space_id=5), request,
        )

    assert exc.value.status_code == 403


async def test_a_request_with_no_space_never_touches_the_membership_check_or_the_index():
    """The pre-existing contract: /plan-epic works without a space, and stays a pure LLM call."""
    retrieval = FakeRetrieval(hits=[_chunk("ATC-1")])
    called = []

    async def _validate(user_id, username, space_ids):
        called.append(space_ids)

    request = _route_request(
        grounding_enabled=True, retrieval=retrieval, client=_plan_client(),
        membership=SimpleNamespace(validate=_validate),
    )

    response = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(proposal="add tracking", requirements=["Provide shipment tracking"]), request,
    )

    assert called == []
    assert retrieval.queries == []
    assert response.grounding.status == "disabled"


async def test_the_refine_response_reports_what_this_round_actually_changed():
    """`change_summary` is computed by diffing the plan against what it was going in — not asserted by
    the model, the same trust move `estimate_reference` makes: a claim that is countable gets counted,
    not described. Run through the actual route (not `refine_plan` directly) so a regression in the
    `IssueDraftIn` -> `IssueDraft` conversion on the way in — see that assignment's docstring on why it
    exists — would also fail this."""
    from app.planning.refine_intent import RefineIntent

    before = [
        IssueDraft(temp_id="1", title="Fix login bug", description="d", issue_type="task", labels=[],
                   estimate_story_points=5, estimate_rationale="because", depends_on=[]),
        IssueDraft(temp_id="2", title="Write tests", description="d", issue_type="task", labels=[],
                   estimate_story_points=3, estimate_rationale="because", depends_on=[]),
        IssueDraft(temp_id="3", title="Deploy to staging", description="d", issue_type="task", labels=[],
                   estimate_story_points=2, estimate_rationale="because", depends_on=[]),
    ]
    after = [
        IssueDraft(temp_id="1", title="Fix login bug (urgent)", description="d", issue_type="task",
                   labels=[], estimate_story_points=8, estimate_rationale="because", depends_on=[]),
        IssueDraft(temp_id="3", title="Deploy to staging", description="d", issue_type="task", labels=[],
                   estimate_story_points=2, estimate_rationale="because", depends_on=[]),
    ]
    client = FakeInstructorClient({
        RefineIntent: [RefineIntent(new_work=[], reasoning="just re-estimating and dropping one")],
        EpicPlanDraft: [EpicPlanDraft(epic=EpicDraft(title="T", description="D", goals=[]), issues=after)],
    })
    request = _route_request(grounding_enabled=False, client=client)
    instruction = "bump issue 1 to 8 points, urgent, and drop the tests issue"

    response = await routes.refine_plan_endpoint(
        routes.RefinePlanRequest(
            epic=routes.EpicDraftIn(title="T", description="D"),
            issues=[routes.IssueDraftIn(**issue.model_dump()) for issue in before],
            instruction=instruction,
        ),
        request,
    )

    summary = response.change_summary
    assert summary is not None
    assert summary.instruction == instruction
    assert summary.added == []
    assert summary.removed == ["Write tests"]
    assert summary.retitled == ["Fix login bug (urgent) (was: Fix login bug)"]
    assert summary.re_estimated == ["Fix login bug (urgent): 5 -> 8 pts"]
    # Issue 3 survived with the same title and points, so it counts as unchanged even though it is
    # also the round that dropped issue 2 — the two are independent per issue, not per round.
    assert summary.unchanged_count == 1


async def test_plan_epic_itself_has_no_change_summary_since_it_has_no_before():
    """`change_summary` is a `/refine`-only concept — `/plan-epic` has nothing to diff a fresh plan
    against, so it stays null rather than reporting a diff against emptiness."""
    request = _route_request(grounding_enabled=False, client=_plan_client())

    response = await routes.plan_epic_endpoint(
        routes.PlanEpicRequest(proposal="add tracking", requirements=["Provide shipment tracking"]), request,
    )

    assert response.change_summary is None


# --- the other two paths that generate a plan --------------------------------------------------


async def test_the_multi_agent_path_shows_the_same_history_to_both_nodes():
    """Retrieved once, read by planner and critic. A critic holding a different sample of history
    than the planner disagrees with every estimate by construction."""
    from app.planning.graph import build_planning_graph, plan_epic_multiagent
    from app.planning.schemas import PlanCritique

    client = FakeInstructorClient({
        EpicPlanDraft: [EpicPlanDraft(
            epic=EpicDraft(title="T", description="D", goals=[]),
            issues=[_issue("1", points=5, references=["ATC-1"])],
        )],
        PlanCritique: [PlanCritique()],
    })
    graph = build_planning_graph(client, "fake-model")

    result = await plan_epic_multiagent(graph, "add tracking", [], _context("ATC-1"))

    planner_prompt, critic_prompt = (c["messages"][0]["content"] for c in client.completions.calls)
    assert "--- ATC-1 ---" in planner_prompt
    assert "--- ATC-1 ---" in critic_prompt
    assert result.plan.issues[0].estimate_reference == ["ATC-1"]


async def test_the_durable_rollout_records_the_history_its_plan_was_written_from():
    """A rollout can sit at pending_approval overnight. The reviewer who finally opens it must see
    the issues the estimates were actually compared against — not a fresh retrieval that would show
    a different set than the plan was written from."""
    from app.planning.rollout_graph import plan_node
    from app.planning.rollout_schemas import initial_rollout_state

    client = _plan_client(["ATC-1"])
    gathered = []

    async def _gatherer(space_id, proposal, requirements=None, on_progress=None):
        gathered.append((space_id, proposal))
        return _context("ATC-1")

    state = initial_rollout_state("add tracking", [], None, None, 5, "u1", "alice")
    update = await plan_node(state, client, "fake-model", lambda: None, _gatherer)

    assert gathered == [(5, "add tracking")]
    assert update["grounding"]["status"] == "grounded"
    assert update["grounding"]["exemplars"][0]["issue_key"] == "ATC-1"
    assert update["grounding"]["estimate_anchor_count"] == 1
    assert update["issues"][0].estimate_reference == ["ATC-1"]


async def test_a_rollout_started_with_the_feature_off_still_produces_a_plan():
    from app.planning.rollout_graph import plan_node
    from app.planning.rollout_schemas import initial_rollout_state

    state = initial_rollout_state("add tracking", [], None, None, 5, "u1", "alice")
    update = await plan_node(state, _plan_client(), "fake-model", lambda: None, None)

    assert update["grounding"]["status"] == "disabled"
    assert update["status"] == "pending_approval"
    assert update["issues"]


async def test_the_leakage_guard_keeps_named_issues_out_of_the_exemplar_set():
    """A planning eval asks the planner to re-derive an epic this team already delivered. Its own
    children must never be among the worked examples, or the eval measures copying."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1", 0.99, 1), _chunk("ATC-2", 0.9, 2)],
        rows=[_row("ATC-1"), _row("ATC-2")],
        details=[_detail("ATC-1"), _detail("ATC-2")],
    )

    context = await gather_exemplars(
        retrieval, [5], "proposal", limit=8, exclude_issue_keys={"atc-1"},
    )

    assert [e.issue_key for e in context.exemplars] == ["ATC-2"]


# --- the recency backfill ------------------------------------------------------------------------


async def test_recent_finished_work_fills_the_slots_semantic_search_could_not():
    """Measured, not hypothetical: on a real board with reranking on, semantic search alone returned
    two exemplars for a typical proposal because the cross-encoder correctly judges most of a board
    irrelevant to any one query. Relevance is what an anchor needs; a style example needs only to be
    something this team wrote."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1", 0.9, 1)],
        rows=[_row("ATC-1")],
        recent=[_row("ATC-20"), _row("ATC-21")],
        details=[_detail(k) for k in ("ATC-1", "ATC-20", "ATC-21")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-1", "ATC-20", "ATC-21"]
    assert [e.source for e in context.exemplars] == ["similar", "recent", "recent"]


async def test_a_backfilled_exemplar_reports_no_relevance_because_it_was_never_scored():
    retrieval = FakeRetrieval(
        hits=[], recent=[_row("ATC-20")], details=[_detail("ATC-20")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert context.exemplars[0].source == "recent"
    assert context.exemplars[0].confidence == 0.0


async def test_a_topically_similar_exemplar_outranks_a_merely_recent_one():
    """Both are anchors here, so the tie is broken on which one can also anchor *comparably*.
    Ordering decides what survives truncation."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-1", 0.9, 1)],
        rows=[_row("ATC-1")],
        recent=[_row("ATC-20")],
        details=[_detail("ATC-1"), _detail("ATC-20")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=1)

    assert [e.issue_key for e in context.exemplars] == ["ATC-1"]


async def test_a_proposal_matching_nothing_on_the_board_still_gets_the_team_style():
    """Genuinely novel work matches nothing semantically. Bailing out there is what the first version
    of this module did, and it is why a real board produced two exemplars."""
    retrieval = FakeRetrieval(
        hits=[], recent=[_row("ATC-20"), _row("ATC-21")],
        details=[_detail("ATC-20"), _detail("ATC-21")],
    )

    context = await gather_exemplars(retrieval, [5], "something nobody here has built", limit=8)

    assert context.status == "grounded"
    assert [e.source for e in context.exemplars] == ["recent", "recent"]


async def test_the_backfill_asks_only_for_finished_sized_work_of_the_right_kind():
    retrieval = FakeRetrieval(hits=[], recent=[], details=[])

    await gather_exemplars(retrieval, [5], "proposal", limit=8)

    (filters,) = retrieval.recent_filters
    assert filters["statuses"] == ["completed", "done"]
    assert filters["issue_types"] == ["story", "task", "bug"]
    assert filters["order_by"] == "updated_at" and filters["order"] == "desc"


async def test_an_unsized_issue_is_never_backfilled():
    """Tier 1 keeps unsized issues as style examples because retrieval said they resemble the
    proposal. Recency says nothing of the sort — an unsized issue nobody asked for teaches the least
    of any row available."""
    retrieval = FakeRetrieval(
        hits=[], recent=[_row("ATC-20", points=None), _row("ATC-21", points=3)],
        details=[_detail("ATC-20"), _detail("ATC-21")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-21"]


async def test_epics_and_subtasks_are_never_exemplars():
    """Genre, not quality: the planner emits stories, tasks and bugs. An epic on this board is an
    outcome statement that is structurally never sized, and a subtask is a granularity below what a
    plan decomposes to."""
    retrieval = FakeRetrieval(
        hits=[_chunk("ATC-E", 0.99, 1), _chunk("ATC-S", 0.95, 2), _chunk("ATC-1", 0.9, 3)],
        rows=[
            _row("ATC-E", issue_type="epic", points=None),
            _row("ATC-S", issue_type="subtask"),
            _row("ATC-1", issue_type="story"),
        ],
        details=[_detail(k) for k in ("ATC-E", "ATC-S", "ATC-1")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert [e.issue_key for e in context.exemplars] == ["ATC-1"]


async def test_the_leakage_guard_covers_the_backfill_too():
    retrieval = FakeRetrieval(
        hits=[], recent=[_row("ATC-20"), _row("ATC-21")],
        details=[_detail("ATC-20"), _detail("ATC-21")],
    )

    context = await gather_exemplars(
        retrieval, [5], "proposal", limit=8, exclude_issue_keys={"ATC-20"},
    )

    assert [e.issue_key for e in context.exemplars] == ["ATC-21"]


async def test_no_semantic_matches_means_no_key_lookup_at_all():
    """An empty `issue_keys` list is falsy in vectorization-service's filter builder, so sending one
    means "no key filter" and returns arbitrary rows — which would then suppress a backfill row via
    the already-seen check. The query has to be skipped, not sent empty."""
    retrieval = FakeRetrieval(
        hits=[], recent=[_row("ATC-20")], details=[_detail("ATC-20")],
    )

    context = await gather_exemplars(retrieval, [5], "proposal", limit=8)

    assert retrieval.key_lookups == []
    assert [e.issue_key for e in context.exemplars] == ["ATC-20"]


async def test_the_verified_citation_is_published_into_the_created_jira_issue():
    """The API response's `grounding` block is gone a week later; the Jira card is where the reviewer
    actually is. The rationale usually names the keys in prose, but prose is the unverified half —
    this list is the one `apply_estimate_grounding` checked."""
    from app.planning.jira_commit_client import JiraCommitClient

    sent = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"id": 7, "issueKey": "ATC-99"}

    class _HTTP:
        async def post(self, url, json, headers):
            sent.update(json)
            return _Resp()

    client = JiraCommitClient("http://jira", "token")
    client._client = _HTTP()

    await client.create_issue(
        5, _issue("1", points=5, references=["ATC-1", "ATC-2"]).model_copy(
            update={"estimate_rationale": "About the size of ATC-1."},
        ),
        parent_id=1, sprint_id=None, user_id="u1", username="alice",
    )

    assert "Estimate rationale: About the size of ATC-1." in sent["description"]
    assert "Sized against: ATC-1, ATC-2" in sent["description"]


# --- the human's edit is the human's, and so is the number they type ------------------------------


def test_a_human_overriding_an_estimate_loses_the_ai_citation_for_it():
    """`Sized against: ATC-1` is published into the Jira card. Next to a number the reviewer typed
    themselves it is a false claim — the comparison justified the model's estimate, not theirs. The
    UI already clears the rationale on a manual point edit; this is the other half of the same claim,
    enforced server-side so it holds for any caller."""
    from app.planning.rollout_graph import _drop_stale_estimate_references

    original = [_issue("1", points=5, references=["ATC-1"])]
    edited = [_issue("1", points=8, references=["ATC-1"])]

    (result,) = _drop_stale_estimate_references(edited, original)

    assert result.estimate_story_points == 8
    assert result.estimate_reference == []


def test_an_untouched_estimate_keeps_its_citation_through_an_edit_of_something_else():
    from app.planning.rollout_graph import _drop_stale_estimate_references

    original = [_issue("1", points=5, references=["ATC-1"])]
    edited = [_issue("1", points=5, references=["ATC-1"]).model_copy(update={"title": "Retitled"})]

    (result,) = _drop_stale_estimate_references(edited, original)

    assert result.estimate_reference == ["ATC-1"]


def test_an_issue_the_human_added_is_left_alone():
    """An addition is the reviewer's own assertion, and this codebase does not make people justify
    their own edits — same stance `drop_unexplained_dependencies` takes."""
    from app.planning.rollout_graph import _drop_stale_estimate_references

    edited = [_issue("99", points=3, references=["ATC-1"])]

    (result,) = _drop_stale_estimate_references(edited, [])

    assert result.estimate_reference == ["ATC-1"]


def test_clearing_an_estimate_entirely_also_clears_the_citation():
    from app.planning.rollout_graph import _drop_stale_estimate_references

    original = [_issue("1", points=5, references=["ATC-1"])]
    edited = [_issue("1", points=None, references=["ATC-1"])]

    (result,) = _drop_stale_estimate_references(edited, original)

    assert result.estimate_reference == []
