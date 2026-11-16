import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from app.agent.crag_loop import ABSTENTION_PHRASE, CragAgent
from app.agent.crag_loop import AgentAnswer
from app.agent.retrieval_tool import RetrievedChunk
from app.api import investigation_routes as routes
from app.investigation.graph import EvidenceRecorder, MultiAgentRag
from app.llm.types import LLMTurn, ToolCall, Usage
from tests.test_crag_loop import FakeRetrieval


def turn(text="", name=None, arguments=None):
    calls = [ToolCall("call", name, arguments or {})] if name else []
    content = ([{"type": "tool_use", "id": "call", "name": name, "input": arguments or {}}]
               if name else [{"type": "text", "text": text}])
    return LLMTurn("tool_use" if name else "end_turn", text, calls,
                   {"role": "assistant", "content": content}, Usage(10, 2))


class ScenarioLLM:
    def __init__(self, *, bad_plan=False, forged=False):
        self.bad_plan, self.forged = bad_plan, forged

    async def next_turn(self, system, messages, tools, tool_choice=None, on_text_delta=None):
        if tool_choice == "review_investigation":
            claims=json.loads(messages[0]["content"])["claims"]
            return turn(name="review_investigation",arguments={"judgments":[{
                "index":c["index"],"supported":True,"relevant":True,"reason":"Supported by attached evidence."}
                for c in claims],"missing_parts":[]})
        if tool_choice == "plan_investigation":
            payload = {"questions": ["Investigate angle A", "Investigate angle B"]}
            if self.bad_plan:
                payload["space_ids"] = [999]
            return turn(name="plan_investigation", arguments=payload)
        if tool_choice == "select_data_source":
            return turn(name="select_data_source", arguments={
                "tool": "search_knowledge_base", "confidence": "high", "source_scope": "identified",
            })
        if tool_choice == "submit_investigation":
            evidence = json.loads(messages[0]["content"])["evidence_catalog"]
            return turn(name="submit_investigation", arguments={"claims":[{
                "text":"Evidence summary. (EVIL-999)" if self.forged else "Evidence summary.",
                "evidence_ids":[e["id"] for e in evidence]}],"limitations":[]})
        if isinstance(messages[-1]["content"], list):
            result = messages[-1]["content"][0]["content"]
            key = "ATC-2" if "ATC-2" in result else "ATC-1"
            return turn(f"Evidence value. ({key})")
        return turn(name="search_knowledge_base", arguments={"query": messages[0]["content"]})


class ParallelRetrieval(FakeRetrieval):
    def __init__(self, slow_b=False):
        super().__init__()
        self.active, self.peak = 0, 0
        self.slow_b = slow_b

    async def search(self, query, space_ids, limit, mode="hybrid"):
        self.calls.append((query, tuple(space_ids), limit, mode))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(.05 if self.slow_b and "B" in query else .002)
            n = 2 if "B" in query else 1
            return [RetrievedChunk(f"chunk-{n}", "attachment", n, f"ATC-{n}", n,
                                   f"Evidence for ATC-{n}", .9, ["vector"], n,
                                   {"page_number": n, "bbox": [1, 2, 3, 4]})]
        finally:
            self.active -= 1


def build(**kwargs):
    retrieval = ParallelRetrieval(slow_b=kwargs.pop("slow_b", False))
    llm = ScenarioLLM(bad_plan=kwargs.pop("bad_plan", False), forged=kwargs.pop("forged", False))
    baseline = CragAgent(llm, retrieval, 4, 5, model="openai-production")
    return baseline, retrieval, MultiAgentRag(baseline, **kwargs)


async def test_real_graph_fans_out_independent_workers_and_accounts_for_all_roles():
    baseline, retrieval, graph = build()
    original = (baseline._llm, baseline._retrieval, baseline._post_generation_verifiers)
    result = await graph.ask("Investigate both angles", [7])
    assert retrieval.peak == 2
    assert len(result.reports) == 2
    assert all(r["space_ids"] == [7] for r in result.reports)
    assert not result.errors
    assert {c.issue_key for c in result.answer.citations} == {"ATC-1", "ATC-2"}
    assert all(c.provenance["bbox"] == [1, 2, 3, 4] for c in result.answer.citations)
    assert {c.page_number for c in result.answer.citations} == {1, 2}
    assert result.answer.usage.input_tokens == sum(c["input_tokens"] for c in result.calls)
    assert result.answer.estimated_cost_usd == pytest.approx(sum(c.cost_usd for c in result.charges))
    assert {c.role for c in result.charges} >= {"planner", "worker-1", "worker-2", "synthesizer"}
    assert original == (baseline._llm, baseline._retrieval, baseline._post_generation_verifiers)


async def test_multiple_spaces_are_assigned_by_code_without_a_model_widening_scope():
    _, retrieval, graph = build()
    result = await graph.ask("Compare evidence", [7, 8, 7])
    assert [t["space_ids"] for t in result.tasks] == [[7], [8]]
    assert {c[1] for c in retrieval.calls} == {(7,), (8,)}
    assert not any(c["role"] == "planner" for c in result.calls)


async def test_three_spaces_spawn_three_independent_workers():
    _, retrieval, graph = build()
    result = await graph.ask("Find all issues across the three workspaces", [7, 8, 9])
    assert [t["id"] for t in result.tasks] == ["worker-1", "worker-2", "worker-3"]
    assert [t["space_ids"] for t in result.tasks] == [[7], [8], [9]]
    assert {c[1] for c in retrieval.calls} == {(7,), (8,), (9,)}
    assert not any(c["role"] == "planner" for c in result.calls)


async def test_invalid_planner_scope_is_rejected_and_fallback_is_explicit():
    _, retrieval, graph = build(bad_plan=True)
    result = await graph.ask("Investigate", [7])
    assert len(result.tasks) == 1
    assert result.errors[0].startswith("planner_failed")
    assert all(call[1] == (7,) for call in retrieval.calls)


async def test_partial_timeout_preserves_successful_worker_and_marks_incomplete_execution():
    _, _, graph = build(slow_b=True, worker_timeout=.02)
    result = await graph.ask("Investigate", [7])
    assert result.reports[0]["result"] is not None
    assert result.reports[1]["error"] == "TimeoutError"
    assert result.errors
    assert result.answer.usage.input_tokens > 0
    assert {c.issue_key for c in result.answer.citations} == {"ATC-1"}
    assert "may cover only part" in result.answer.text


async def test_only_selected_sources_are_returned_in_the_production_answer():
    baseline, _, graph = build()
    inner=baseline._llm
    class SelectOne:
        async def next_turn(self, system, messages, tools, tool_choice=None, **kwargs):
            if tool_choice=="submit_investigation":
                return turn(name="submit_investigation",arguments={"claims":[{
                    "text":"The requested fact.","evidence_ids":["E1"]}],"limitations":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=SelectOne()
    result=await graph.ask("Investigate both angles",[7])
    assert result.evidence_selection["candidate_count"]==2
    assert result.evidence_selection["selected_ids"]==["E1"]
    assert {c.issue_key for c in result.answer.citations}=={"ATC-1"}
    assert "ATC-2" not in result.answer.text


async def test_invalid_evidence_selection_gets_one_metered_repair_then_validates_again():
    baseline, _, graph=build()
    inner=baseline._llm
    class Repair:
        count=0
        async def next_turn(self, system, messages, tools, tool_choice=None, **kwargs):
            if tool_choice=="submit_investigation":
                self.count+=1
                if self.count==2:
                    assert "validation_error" in messages[-1]["content"]
                return turn(name="submit_investigation",arguments={"claims":[{
                    "text":"The requested fact.","evidence_ids":["E999" if self.count==1 else "E1"]}],
                    "limitations":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    repair=Repair()
    baseline._llm=repair
    result=await graph.ask("Investigate both angles",[7])
    assert repair.count==2
    assert not result.errors
    assert result.evidence_selection["validation_failures"]==["claim cites unknown or missing evidence"]
    assert result.evidence_selection["selected_ids"]==["E1"]
    assert len([c for c in result.calls if c["role"]=="synthesizer"])==2
    assert result.answer.usage.input_tokens==sum(c["input_tokens"] for c in result.calls)


@pytest.mark.parametrize("worker_abstained",[False,True])
async def test_zero_counts_keep_distinct_scope_and_do_not_require_a_fabricated_issue(monkeypatch,worker_abstained):
    baseline, _, graph=build()
    async def answer_count(self, question, space_ids):
        return AgentAnswer(text="Zero matching bugs.",abstained=worker_abstained and space_ids==[7],citations=[],retrieval_rounds=1,
                           queries_used=[],structured_evidence=["Structured issue query: total_count = 0"])
    monkeypatch.setattr(CragAgent,"ask",answer_count)
    inner=baseline._llm
    class Counts:
        async def next_turn(self, system, messages, tools, tool_choice=None, **kwargs):
            if tool_choice=="submit_investigation":
                seen_by_model.append(messages[0]["content"])
                pool=json.loads(messages[0]["content"])["evidence_catalog"]
                return turn(name="submit_investigation",arguments={"claims":[{
                    "text":f"{e['workspace']} has zero bugs.","evidence_ids":[e['id']]}
                    for e in pool],"limitations":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    seen_by_model=[]
    baseline._llm=Counts()
    result=await graph.ask("Bug counts per workspace",[7,8],space_labels={7:"ATC",8:"PRM"})
    assert not result.answer.abstained
    assert "ATC has zero bugs." in result.answer.text and "PRM has zero bugs." in result.answer.text
    assert len(result.answer.structured_evidence)==2
    assert result.answer.structured_evidence[0].startswith("Workspace ATC:")
    assert result.answer.structured_evidence[1].startswith("Workspace PRM:")
    assert not result.answer.citations
    # A zero count has no issue key to name its project by; the label is the only way, and the
    # numeric space id must never reach the model (it used to leak into answers).
    assert '"space_ids"' not in seen_by_model[0]


async def test_spaces_without_a_label_are_numbered_not_named_by_id():
    _, _, graph = build()
    result = await graph.ask("Compare evidence", [7, 8], space_labels={7: "ATC"})
    assert [t["workspace"] for t in result.tasks] == ["ATC", "workspace 2"]
    assert "ATC" in result.tasks[0]["question"]


async def test_reviewer_repairs_once_then_drops_claim_still_unsupported():
    baseline,_,graph=build()
    inner=baseline._llm
    class Review:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="submit_investigation":
                return turn(name="submit_investigation",arguments={"claims":[
                    {"text":"Supported requested fact.","evidence_ids":["E1"]},
                    {"text":"Unsupported detail.","evidence_ids":["E2"]}],"limitations":[]})
            if tool_choice=="review_investigation":
                return turn(name="review_investigation",arguments={"judgments":[
                    {"index":0,"supported":True,"relevant":True,"reason":"direct evidence"},
                    {"index":1,"supported":False,"relevant":True,"reason":"missing support"}],"missing_parts":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=Review()
    result=await graph.ask("Investigate both angles",[7])
    assert not result.errors
    assert "Unsupported detail" not in result.answer.text
    assert {c.issue_key for c in result.answer.citations}=={"ATC-1"}
    review=result.evidence_selection["claim_review"]
    assert len(review["attempts"])==2 and len(review["dropped_claims"])==1
    assert len([c for c in result.calls if c["role"]=="claim_reviewer"])==2
    assert len([c for c in result.calls if c["role"]=="synthesizer.revision"])==1


async def test_reviewer_missing_parts_become_one_readable_sentence():
    # Seen in the 2026-11-05 cross-space eval: the labels were pasted into the answer one per line.
    baseline,_,graph=build()
    inner=baseline._llm
    class Missing:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="review_investigation":
                claims=json.loads(messages[0]["content"])["claims"]
                return turn(name="review_investigation",arguments={"judgments":[{
                    "index":c["index"],"supported":True,"relevant":True,"reason":"ok"} for c in claims],
                    "missing_parts":["Whether ATC has a camera check","Issues cited for ATC."]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=Missing()
    result=await graph.ask("Investigate both angles",[7])
    assert result.answer.text.endswith(
        "Not covered by the evidence: Whether ATC has a camera check; Issues cited for ATC.")
    assert "\nWhether ATC" not in result.answer.text
    assert result.evidence_selection["claim_review"]["missing_parts"]==[
        "Whether ATC has a camera check","Issues cited for ATC."]


@pytest.mark.parametrize("note_key,kept", [("ATC-2", True), ("ATC-999", False)])
async def test_a_note_may_name_a_retrieved_but_uncited_issue(note_key, kept):
    # 2026-11-05: "ATC-47 is test-only, not a fix" in a note voided a correct answer 3 times in 3.
    baseline,_,graph=build()
    inner=baseline._llm
    class Note:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="submit_investigation":
                return turn(name="submit_investigation",arguments={"claims":[
                    {"text":"Supported requested fact (ATC-1).","evidence_ids":["E1"]}],
                    "limitations":[f"{note_key} was retrieved but is not a fix."]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=Note()
    result=await graph.ask("Investigate both angles",[7])
    # A note naming a key no tool returned is dropped on its own; the claim stays.
    assert not result.answer.abstained
    assert {c.issue_key for c in result.answer.citations}=={"ATC-1"}
    assert "(ATC-1). (ATC-1)" not in result.answer.text and "(ATC-1)" in result.answer.text
    assert (f"{note_key} was retrieved" in result.answer.text) is kept
    if not kept:
        assert result.answer.text.endswith(
            "One statement was left out because it named an issue no search returned.")
        assert result.evidence_selection["removed_unfound_keys"]==["ATC-999"]


def _claims_by_key(messages, texts):
    """Claims citing the catalog entry for each issue key, e.g. {"ATC-1": "text"}."""
    ids = {e["source"]["issue_key"]: e["id"] for e in json.loads(messages[0]["content"])["evidence_catalog"]}
    return [{"text": text, "evidence_ids": [ids[key]]} for key, text in texts]


async def test_a_claim_naming_an_unfound_issue_is_dropped_and_the_rest_is_kept():
    baseline,_,graph=build()
    inner=baseline._llm
    class OneBadClaim:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="submit_investigation":
                return turn(name="submit_investigation",arguments={"claims":_claims_by_key(messages,[
                    ("ATC-1","Angle A is fixed."), ("ATC-2","Angle B shipped together with ATC-48.")]),
                    "limitations":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=OneBadClaim()
    result=await graph.ask("Investigate both angles",[7])
    assert not result.answer.abstained
    assert "Angle A is fixed. (ATC-1)" in result.answer.text
    assert "ATC-48" not in result.answer.text
    # The dropped claim's evidence is no longer offered as a source.
    assert {c.issue_key for c in result.answer.citations}=={"ATC-1"}
    assert result.evidence_selection["removed_statements"]==1


async def test_when_every_claim_names_an_unfound_issue_the_answer_abstains():
    baseline,_,graph=build()
    inner=baseline._llm
    class AllBad:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="submit_investigation":
                return turn(name="submit_investigation",arguments={"claims":_claims_by_key(messages,[
                    ("ATC-1","Angle A depends on ATC-48.")]),"limitations":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=AllBad()
    result=await graph.ask("Investigate both angles",[7])
    assert result.answer.abstained and result.answer.citations==[]


async def test_malformed_review_does_not_silently_approve_unreviewed_claims():
    baseline,_,graph=build()
    inner=baseline._llm
    class Invalid:
        async def next_turn(self,system,messages,tools,tool_choice=None,**kwargs):
            if tool_choice=="review_investigation":
                return turn(name="review_investigation",arguments={"judgments":[],"missing_parts":[]})
            return await inner.next_turn(system,messages,tools,tool_choice,**kwargs)
    baseline._llm=Invalid()
    result=await graph.ask("Investigate both angles",[7])
    assert result.answer.abstained
    assert "claim_review_failed" in result.errors


async def test_fabricated_synthesis_citation_is_rejected_by_existing_verifier():
    _, _, graph = build(forged=True)
    result = await graph.ask("Investigate", [7])
    assert result.answer.abstained
    assert result.answer.text == ABSTENTION_PHRASE


async def test_recorder_refuses_a_different_scope_before_calling_retrieval():
    inner = FakeRetrieval()
    recorder = EvidenceRecorder(inner, [7])
    with pytest.raises(ValueError, match="scope"):
        await recorder.search("query", [999], 5)
    assert not inner.calls


async def test_worker_clarification_is_surfaced_instead_of_silent_abstention(monkeypatch):
    _, _, graph = build()

    async def clarify(self, question, space_ids):
        return AgentAnswer(
            text="Which document did you mean?",
            abstained=False,
            needs_clarification=True,
            clarification_options=[],
            citations=[],
            retrieval_rounds=1,
            queries_used=["clarify"],
        )

    monkeypatch.setattr(CragAgent, "ask", clarify)
    result = await graph.ask("What does the report say?", [7, 8])
    assert result.answer.needs_clarification is True
    assert result.answer.abstained is False
    assert "Which document" in result.answer.text
    assert result.evidence_selection.get("surfaced_clarification") is True


async def test_surfaced_clarification_counts_each_worker_once(monkeypatch):
    """Regression: the clarification branch used to re-add the non-clarifying workers' rounds."""
    _, _, graph = build()

    async def one_clarifies_one_abstains(self, question, space_ids):
        if list(space_ids) == [7]:
            return AgentAnswer(text="Which document?", needs_clarification=True,
                               retrieval_rounds=1, queries_used=["q7"])
        return AgentAnswer(text=ABSTENTION_PHRASE, abstained=True,
                           retrieval_rounds=2, queries_used=["q8a", "q8b"])

    monkeypatch.setattr(CragAgent, "ask", one_clarifies_one_abstains)
    result = await graph.ask("What does the report say?", [7, 8])
    assert result.answer.needs_clarification is True
    assert result.answer.retrieval_rounds == 3
    assert sorted(result.answer.queries_used) == ["q7", "q8a", "q8b"]


async def test_planner_fallback_does_not_claim_partial_coverage():
    _, _, graph = build(bad_plan=True)
    result = await graph.ask("Investigate", [7])
    assert result.errors[0].startswith("planner_failed")
    assert "may cover only part" not in result.answer.text


async def test_concurrent_graph_invocations_do_not_share_task_scopes():
    _, _, graph = build()
    a, b = await asyncio.gather(graph.ask("First", [7]), graph.ask("Second", [8]))
    assert all(t["space_ids"] == [7] for t in a.tasks)
    assert all(t["space_ids"] == [8] for t in b.tasks)
    assert all(c["arguments"]["space_ids"] == [7] for r in a.reports for c in r["tool_calls"])
    assert all(c["arguments"]["space_ids"] == [8] for r in b.reports for c in r["tool_calls"])


async def test_endpoint_authorizes_before_invoking_investigation():
    investigator = SimpleNamespace(ask=AsyncMock())
    request = SimpleNamespace(headers={}, app=SimpleNamespace(state=SimpleNamespace(investigator=investigator)))
    with patch.object(routes, "_authorize_space_ids", AsyncMock(side_effect=HTTPException(403))), \
         patch.object(routes, "_tag_entrypoint_span"):
        with pytest.raises(HTTPException) as error:
            await routes.investigate(routes.InvestigationRequest(question="Compare", space_ids=[7]), request)
    assert error.value.status_code == 403
    investigator.ask.assert_not_awaited()


@pytest.mark.parametrize("payload", [
    {"question": " ", "spaceIds": [7]}, {"question": "q", "spaceIds": []},
    {"question": "q", "spaceIds": [1, 2, 3, 4]}, {"question": "q", "spaceIds": [-1]},
])
def test_endpoint_rejects_invalid_requests(payload):
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.enforce_user_quota] = lambda: None
    assert TestClient(app).post("/investigate", json=payload).status_code == 422


async def test_one_workspace_gets_no_label_so_answers_do_not_say_in_atc():
    _, _, graph = build()
    result = await graph.ask("Investigate both angles", [7], space_labels={7: "ATC"})
    assert all(t["workspace"] == "" for t in result.tasks)
    assert "ATC" not in " ".join(t["question"] for t in result.tasks)
