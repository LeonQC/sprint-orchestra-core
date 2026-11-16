"""MCP server: Sprint Orchestra's AI features as tools for external agents (Claude Code, Cursor,
Claude Desktop), served at POST /mcp, public path /api/ai/mcp.

No tool takes `space_ids`. Scope comes from the authenticated user's memberships, and the
optional `space` argument can only narrow it — the same rule the in-house CRAG loop follows (see
app/agent/retrieval_tool.py), so a prompt-injected client can't widen what it reads.

Stateless Streamable HTTP with JSON responses: every POST is self-contained, so any replica can
serve any call and the load balancer needs no sticky sessions.

No `from __future__ import annotations` here: FastMCP 1.12 inspects tool annotations as live objects
and crashes on string ones.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Annotated, Iterator, List, Optional

from fastapi import FastAPI, HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from opentelemetry import trace
from pydantic import Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from app.api import routes as api_routes
from app.auth.user_quota import enforce_user_quota
from app.observability import MCP_TOOL_CALL_SECONDS, MCP_TOOL_CALLS_TOTAL, get_request_id
from app.sprint_recovery.graph import _build_sprint_snapshot, _detect_risk_signals

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)

# Context budget per get_ticket_context call: the client pays for every token we return.
_MAX_COMMENTS = 50
_MAX_HISTORY = 30
_MAX_ATTACHMENT_CHARS = 12_000
_MAX_FIELD_CHARS = 2_000
_MAX_HISTORY_VALUE_CHARS = 300
_CITATION_EXCERPT_CHARS = 300
_DONE_STATUSES = ("done", "completed")

_INSTRUCTIONS = (
    "Sprint Orchestra: the team's issues, sprints, comments and attachments. Use ask_project for "
    "questions about the project; get_ticket_context before working on a ticket; find_similar_work "
    "to learn how a similar issue was handled; check_sprint_risk for sprint health."
)


@dataclass(slots=True)
class _Caller:
    request: Request
    user_id: str
    spaces: List[dict]


async def _caller(ctx: Context) -> _Caller:
    request = ctx.request_context.request
    user_id = request.headers.get("x-user-id") if request is not None else None
    # Fail closed: unlike the REST routes, there is no caller-supplied scope to validate, so without
    # an identity there is nothing to scope to.
    if not user_id:
        raise ToolError("Not signed in. Connect to /api/ai/mcp with a bearer token.")
    checker = request.app.state.space_membership
    try:
        spaces = await checker.authorized_spaces(user_id, request.headers.get("x-username") or "")
    except Exception as exc:  # noqa: BLE001 - jira-backend unreachable etc.
        logger.error("MCP space membership lookup failed", exc_info=True)
        raise ToolError("Could not check your space memberships right now; try again.") from exc
    if spaces is None:
        raise ToolError("MCP needs AI_SPACE_MEMBERSHIP_CHECK_ENABLED=true: tool scope comes from memberships.")
    if not spaces:
        raise ToolError("You are not a member of any space.")
    return _Caller(request=request, user_id=user_id, spaces=spaces)


def _scope(caller: _Caller, space: Optional[str]) -> List[int]:
    if not space:
        return [s["id"] for s in caller.spaces]
    wanted = space.strip().upper()
    for s in caller.spaces:
        if (s.get("key") or "").upper() == wanted:
            return [s["id"]]
    # Lists only the caller's own spaces, so a wrong guess reveals nothing about other spaces.
    keys = ", ".join(s.get("key") or str(s["id"]) for s in caller.spaces)
    raise ToolError(f"Unknown space '{space}'. Your spaces: {keys}.")


@contextmanager
def _observed(tool: str) -> Iterator[None]:
    start = time.perf_counter()
    outcome = "ok"
    with _tracer.start_as_current_span(f"mcp.{tool}", attributes={"mcp.tool": tool}):
        try:
            yield
        except ToolError:
            outcome = "rejected"
            raise
        except HTTPException as exc:
            outcome = "rejected" if exc.status_code < 500 else "error"
            raise ToolError(str(exc.detail)) from exc
        except Exception as exc:
            outcome = "error"
            logger.exception("MCP tool %s failed", tool)
            # FastMCP would otherwise put the raw exception text in the tool result.
            raise ToolError(f"{tool} failed (request ID: {get_request_id()})") from exc
        finally:
            MCP_TOOL_CALLS_TOTAL.labels(tool, outcome).inc()
            MCP_TOOL_CALL_SECONDS.labels(tool).observe(time.perf_counter() - start)


def _clip(text: Optional[str], limit: int) -> Optional[str]:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + f"… [{len(text) - limit} more chars]"


def _normalize_key(issue_key: str) -> str:
    key = issue_key.strip().upper()
    if not key:
        raise ToolError("issue_key is required, e.g. ATC-77.")
    return key


SpaceArg = Annotated[
    Optional[str],
    Field(description="Space key such as 'ATC' to narrow the scope. Omit to use all your spaces."),
]
IssueKeyArg = Annotated[str, Field(description="Issue key, e.g. 'ATC-77'.")]


async def ask_project(
    question: Annotated[str, Field(description="A natural-language question about the project.")],
    ctx: Context,
    space: SpaceArg = None,
) -> dict:
    """Answer a question about the project's issues, sprints, comments and attachments, with citations.

    Good for "why did checkout break", "how did we fix the HikariCP timeouts", "which bugs are
    blocked", "was ATC-77 ever reopened". The answer is grounded and verified server-side; when the
    evidence is missing it abstains instead of guessing. Prefer this over reading many tickets
    yourself.
    """
    with _observed("ask_project"):
        caller = await _caller(ctx)
        space_ids = _scope(caller, space)
        # Calling the route function directly skips its FastAPI dependencies, quota included.
        await enforce_user_quota(caller.request)
        resp = await api_routes.ask(
            api_routes.AskRequest(question=question, space_ids=space_ids), caller.request
        )
        return {
            "answer": resp.answer,
            "abstained": resp.abstained,
            "needs_clarification": resp.needs_clarification,
            "clarification_options": [
                {"label": o.label, "issue_key": o.issue_key} for o in resp.clarification_options
            ],
            "citations": [
                {
                    "issue_key": c.issue_key,
                    "source": c.chunk_type,
                    "page_number": c.page_number,
                    "excerpt": _clip(c.content, _CITATION_EXCERPT_CHARS),
                }
                for c in resp.citations
            ],
            "answered_from": resp.route_tool,
            "cache_hit": resp.cache_hit,
            "estimated_cost_usd": resp.estimated_cost_usd,
        }


async def _issue_row(retrieval, space_ids: List[int], key: str) -> Optional[dict]:
    result = await retrieval.query_issues(space_ids, {"issue_keys": [key], "limit": 1})
    return result.issues[0] if result.issues else None


async def get_ticket_context(issue_key: IssueKeyArg, ctx: Context) -> dict:
    """Everything about one ticket in a single call: fields, description, comment thread, extracted
    attachment text (PDF, DOCX, XLSX, images) and recent change history.

    Call this before starting work on a ticket. Long attachments are truncated; use ask_project for a
    specific fact inside them.
    """
    with _observed("get_ticket_context"):
        caller = await _caller(ctx)
        key = _normalize_key(issue_key)
        space_ids = _scope(caller, None)
        retrieval = caller.request.app.state.retrieval
        issue, details, comments, attachments, history = await asyncio.gather(
            _issue_row(retrieval, space_ids, key),
            retrieval.get_issue_details(space_ids, [key]),
            retrieval.get_issue_comments(space_ids, [key], limit=_MAX_COMMENTS),
            retrieval.get_issue_attachments(space_ids, [key]),
            retrieval.query_issue_history(space_ids, {"issue_keys": [key], "limit": _MAX_HISTORY}),
        )
        # Same message whether the issue doesn't exist or is in someone else's space.
        if issue is None and not details.details:
            raise ToolError(f"{key} was not found in your spaces.")

        attachment_chars = 0
        attachment_chunks = []
        for chunk in attachments.attachments:
            remaining = _MAX_ATTACHMENT_CHARS - attachment_chars
            if remaining <= 0:
                break
            content = _clip(chunk.get("content") or "", remaining)
            attachment_chars += len(content)
            attachment_chunks.append({
                "attachment_id": chunk.get("source_id"),
                "page_number": chunk.get("page_number"),
                "location": {
                    k: v for k, v in (chunk.get("provenance") or {}).items()
                    if k in ("slide_number", "sheet_name", "cell_range")
                },
                "content": content,
            })

        return {
            "issue": {
                k: (issue or {}).get(k)
                for k in (
                    "issue_key", "title", "issue_type", "status", "priority", "assignee_name",
                    "story_points", "sprint_name", "parent_key", "updated_at",
                )
            } | {"issue_key": key},
            "description": _clip("\n\n".join(d.get("content") or "" for d in details.details), 8_000),
            "comments": [
                {
                    "author": (c.get("provenance") or {}).get("author"),
                    "created_at": (c.get("provenance") or {}).get("created_at"),
                    "content": _clip(c.get("content"), _MAX_FIELD_CHARS),
                }
                for c in comments.comments
            ],
            "comment_count": comments.total_count,
            "attachments": attachment_chunks,
            "attachment_chunk_count": attachments.total_count,
            "history": [
                {
                    "changed_at": h.get("changed_at"),
                    "actor": h.get("actor_name"),
                    "event": h.get("event_type"),
                    "field": h.get("field_name"),
                    "from": _clip(h.get("from_value"), _MAX_HISTORY_VALUE_CHARS),
                    "to": _clip(h.get("to_value"), _MAX_HISTORY_VALUE_CHARS),
                }
                for h in history.changes
            ],
            "history_count": history.total_count,
        }


async def find_similar_work(issue_key: IssueKeyArg, ctx: Context) -> dict:
    """Find earlier issues this team already solved that are relevant to this one, each with why it
    matters and what it teaches.

    Use before starting a bug or feature to reuse a known fix or avoid a known pitfall. A verdict of
    "none_relevant" is a real answer, not a failure.
    """
    with _observed("find_similar_work"):
        caller = await _caller(ctx)
        key = _normalize_key(issue_key)
        space_ids = _scope(caller, None)
        retrieval = caller.request.app.state.retrieval
        issue, details = await asyncio.gather(
            _issue_row(retrieval, space_ids, key),
            retrieval.get_issue_details(space_ids, [key]),
        )
        if issue is None and not details.details:
            raise ToolError(f"{key} was not found in your spaces.")
        await enforce_user_quota(caller.request)
        resp = await api_routes.precedents_endpoint(
            api_routes.PrecedentRequest(
                # Precedents from the issue's own space: another team's fixes rarely transfer.
                space_ids=[issue["space_id"]] if issue and issue.get("space_id") else space_ids,
                issue_key=key,
                title=(issue or {}).get("title") or "",
                description="\n\n".join(d.get("content") or "" for d in details.details),
            ),
            caller.request,
        )
        return {
            "verdict": resp.verdict,
            "precedents": [
                {"issue_key": p.issue_key, "title": p.title, "relevance": p.relevance, "takeaway": p.takeaway}
                for p in resp.precedents
            ],
            "candidates_considered": resp.candidates_considered,
            "degraded": resp.degraded,
        }


def _pick_sprint(sprints: List[dict], name: Optional[str], space_key: str) -> dict:
    if name:
        wanted = name.strip().lower()
        exact = [s for s in sprints if (s.get("sprint_name") or "").lower() == wanted]
        partial = [s for s in sprints if wanted in (s.get("sprint_name") or "").lower()]
        match = exact or partial
        if match:
            return match[0]
        recent = ", ".join(s.get("sprint_name") or "?" for s in sprints[:5])
        raise ToolError(f"No sprint named '{name}' in {space_key}. Recent sprints: {recent}.")
    active = [s for s in sprints if (s.get("status") or "").lower() == "active"]
    if not active:
        recent = ", ".join(s.get("sprint_name") or "?" for s in sprints[:5])
        raise ToolError(f"No active sprint in {space_key}; pass `sprint`. Recent sprints: {recent}.")
    return active[0]


async def check_sprint_risk(
    ctx: Context,
    space: SpaceArg = None,
    sprint: Annotated[
        Optional[str], Field(description="Sprint name. Omit for the space's active sprint.")
    ] = None,
) -> dict:
    """Health check for a sprint: timeline position, progress, and risk signals such as blocked or
    stale issues, an overloaded owner, scope added late, or a completion forecast behind schedule.

    Every signal is computed by code from live sprint data, not judged by a model, so it is safe to
    quote. Explain them to the user in your own words.
    """
    with _observed("check_sprint_risk"):
        caller = await _caller(ctx)
        space_ids = _scope(caller, space)
        if len(space_ids) > 1:
            keys = ", ".join(s.get("key") or str(s["id"]) for s in caller.spaces)
            raise ToolError(f"You belong to several spaces; pass `space` (one of: {keys}).")
        space_id = space_ids[0]
        space_key = next((s.get("key") for s in caller.spaces if s["id"] == space_id), str(space_id))
        retrieval = caller.request.app.state.retrieval
        sprints = await retrieval.query_sprints([space_id], {"limit": 200})
        chosen = _pick_sprint(sprints.sprints, sprint, space_key)
        sprint_id = chosen["sprint_id"]
        snapshot, signals = await asyncio.gather(
            _build_sprint_snapshot(retrieval, space_id, sprint_id),
            _detect_risk_signals(retrieval, space_id, sprint_id),
        )
        issues = snapshot.issues
        done = [i for i in issues if (i.status or "").lower() in _DONE_STATUSES]
        return {
            "sprint": {
                "name": chosen.get("sprint_name"),
                "status": chosen.get("status"),
                "goal": snapshot.goal,
                "start_date": chosen.get("start_date"),
                "end_date": chosen.get("end_date"),
                "days_remaining": snapshot.days_remaining,
                "elapsed_percent": snapshot.elapsed_percent,
            },
            "progress": {
                "issues_total": len(issues),
                "issues_done": len(done),
                "points_total": sum(i.story_points or 0 for i in issues),
                "points_done": sum(i.story_points or 0 for i in done),
                "unestimated_issues": sum(1 for i in issues if i.story_points is None),
            },
            "risk_signals": [s.model_dump() for s in signals],
            "issues": [i.model_dump() for i in issues],
        }


def start_ticket(issue_key: str) -> str:
    """Load a ticket's full context and similar past work, then plan before changing code."""
    return (
        f"I'm starting work on {issue_key}. Call get_ticket_context for {issue_key}, then "
        f"find_similar_work for it. Summarize what is being asked, the acceptance criteria, open "
        "questions raised in the comments, and anything a similar past issue teaches. Then propose a "
        "plan and wait for my go-ahead before changing any code."
    )


def build_mcp_server() -> FastMCP:
    server = FastMCP(
        "sprint-orchestra",
        instructions=_INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
    )
    for tool in (ask_project, get_ticket_context, find_similar_work, check_sprint_risk):
        server.tool()(tool)
    server.prompt()(start_ticket)
    return server


@asynccontextmanager
async def run_mcp_server(app: FastAPI):
    """Run one FastMCP instance for this app's lifespan (a session manager can only run once)."""
    server = build_mcp_server()
    server.streamable_http_app()  # creates the session manager; the returned sub-app is unused
    async with server.session_manager.run():
        app.state.mcp_session_manager = server.session_manager
        try:
            yield
        finally:
            app.state.mcp_session_manager = None


class _MCPEndpoint:
    """A raw ASGI endpoint rather than a mounted Starlette sub-app: a sub-app replaces scope["app"],
    and the tools reach the FastAPI app's state (retrieval client, membership checker) through it."""

    async def __call__(self, scope, receive, send) -> None:
        manager = getattr(scope["app"].state, "mcp_session_manager", None)
        if manager is None:
            await PlainTextResponse("MCP is disabled (AI_MCP_ENABLED=false)", status_code=404)(
                scope, receive, send
            )
            return
        await manager.handle_request(scope, receive, send)


def install_mcp_route(app: FastAPI) -> None:
    app.router.add_route(
        "/mcp", _MCPEndpoint(), methods=["GET", "POST", "DELETE"], include_in_schema=False
    )
