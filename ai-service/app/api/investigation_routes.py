"""Explicit multi-agent endpoint. /ask and its cache/streaming contract are untouched."""
from __future__ import annotations

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, StringConstraints

from app import degradation
from app.api.routes import (
    AskResponse, CitationOut, _CamelModel, _authorize_space_ids, _current_traceparent,
    _record_usage, _space_labels, _tag_entrypoint_span,
)
from app.auth.user_quota import enforce_user_quota
from app.llm.usage_ledger import FEATURE_INVESTIGATE

router = APIRouter()


class InvestigationRequest(_CamelModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    space_ids: list[Annotated[int, Field(gt=0)]] = Field(min_length=1, max_length=3)


class InvestigationResponse(AskResponse):
    tasks: list[dict]
    workers: list[dict]
    graph_events: list[dict]
    model_calls: list[dict]
    model_charges: list[dict]
    total_seconds: float
    usage_complete: bool
    evidence_selection: dict
    partial_coverage: bool


@router.post("/investigate", response_model=InvestigationResponse, dependencies=[Depends(enforce_user_quota)])
async def investigate(req: InvestigationRequest, request: Request):
    caller = _tag_entrypoint_span(request, FEATURE_INVESTIGATE)
    await _authorize_space_ids(request, req.space_ids)
    investigator = getattr(request.app.state, "investigator", None)
    if investigator is None:
        raise HTTPException(status_code=503, detail="Multi-agent investigation is unavailable")
    with degradation.collect() as degraded:
        result = await investigator.ask(
            req.question, req.space_ids, space_labels=await _space_labels(request, req.space_ids),
        )
    answer = result.answer
    _record_usage(
        request, FEATURE_INVESTIGATE, result.charges, caller=caller,
        detail={"space_ids": req.space_ids, "workers": len(result.reports),
                "errors": result.errors, "abstained": answer.abstained},
    )
    reasons = list(degraded) + result.errors
    partial = bool(result.errors or result.evidence_selection.get("claim_review", {}).get("missing_parts"))
    if partial and not reasons:
        reasons.append("partial_investigation")
    return InvestigationResponse(
        answer=answer.text, abstained=answer.abstained, retrieval_rounds=answer.retrieval_rounds,
        queries_used=answer.queries_used, citations=[CitationOut(**asdict(c)) for c in answer.citations],
        structured_evidence=answer.structured_evidence,
        input_tokens=answer.usage.input_tokens, output_tokens=answer.usage.output_tokens,
        cached_input_tokens=answer.usage.cache_read_input_tokens,
        estimated_cost_usd=answer.estimated_cost_usd,
        route_source=answer.route_source, route_tool=answer.route_tool,
        degraded=bool(reasons), degraded_reasons=reasons,
        trace_parent=_current_traceparent(), tasks=result.tasks,
        workers=[{
            "id": r["id"], "question": r["question"], "space_ids": r["space_ids"],
            "seconds": r["seconds"], "error": r["error"], "tool_calls": r["tool_calls"],
            "answer": r["result"].text if r["result"] else None,
            "abstained": r["result"].abstained if r["result"] else None,
            "needs_clarification": r["result"].needs_clarification if r["result"] else None,
        } for r in result.reports],
        graph_events=result.events, model_calls=result.calls,
        model_charges=[c.as_dict() for c in result.charges], total_seconds=round(result.seconds, 4),
        # A cancelled/failed call can be billed without usage reaching us; don't call it free.
        usage_complete=not result.errors,
        evidence_selection=result.evidence_selection,
        partial_coverage=partial,
    )
