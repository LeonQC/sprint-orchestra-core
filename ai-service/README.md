# ai-service

The AI layer of Sprint Orchestra. It answers natural-language questions about a workspace with an
**agentic RAG loop with corrective retrieval**, and runs the other AI features: task drafting, Plan
Epic, Sprint Pace, Sprint Recovery, issue-level actions and an MCP server for external agents.

The answer model (`gpt-5.6-luna` through the LiteLLM gateway) searches
[`vectorization-service`](../vectorization-service)'s hybrid index with tool calls, judges whether
the results answer the question, and if not reformulates and searches again, up to a bounded number
of rounds, before answering with citations or saying it doesn't have enough information.

```
User question ──▶ ai-service (CragAgent)
                     │
                     │ 1. search_knowledge_base(query, mode?)  ──▶  POST /search (vectorization-service)
                     │ 2. assess: does this answer the question?
                     │      insufficient? → reformulate, go to 1 (bounded by AI_MAX_TOOL_ITERATIONS)
                     │      sufficient?   → answer, with citations, or abstain
                     ▼
              {answer, citations, retrieval_rounds, queries_used, abstained}
```

For the end-to-end picture (routing, Fast/Deep modes, verification, caching) see
[AI_ARCHITECTURE.md](../docs/AI_ARCHITECTURE.md).

## Run locally

```bash
cd ai-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # already points at the LiteLLM gateway (:4000)
uvicorn app.main:app --port 8200
```

Requires `vectorization-service` (`:8100`) with an ingested corpus. The default configuration also
enables the Redis answer cache and both Postgres-checkpointed workflows, so Redis (`:6379`) and
PostgreSQL (`:5432`) must be running. For a minimal standalone process, set `AI_CACHE_ENABLED=false`,
`AI_EPIC_ROLLOUT_ENABLED=false` and `AI_SPRINT_RECOVERY_ENABLED=false`.

`scripts/dev_up.sh` at the repository root starts the whole stack.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz` | Liveness |
| GET | `/healthz/cache` | Cache configuration and the current code version (internal) |
| GET | `/stats` | Cumulative token usage and estimated cost since process start |
| GET | `/quota` | The signed-in user's limits and today's spend |
| GET | `/metrics` | Prometheus metrics |
| POST | `/ask` | `{question, spaceIds, history?, historySummary?, attachmentIssueKeys?, mode?: fast\|deep}` → `{answer, abstained, retrieval_rounds, queries_used, citations, input_tokens, output_tokens, estimated_cost_usd, cache_hit, degraded, degradedReasons, historySummary?}` |
| POST | `/ask/stream` | The same result over SSE, with progress events and streamed answer tokens |
| POST | `/investigate` | Bounded multi-agent investigation; returns worker reports, citations, tokens and cost |
| POST | `/search` | Space-authorized semantic search for duplicate detection |
| POST | `/draft-task` | `{description, existing_labels?}` → `{draft: {title, issue_type, labels, estimate_story_points, dependencies}, degraded, latency_seconds}` |
| POST | `/plan-epic`, `/plan-epic/refine` | Stateless plan generation and refinement |
| POST/GET | `/plan-epic/rollout/**` | Plan Epic lifecycle: generate, review and edit, publish sprints/issues/dependencies, resume and retry |
| POST | `/sprint-pace` | Pace signals computed in code, plus a grounded explanation |
| POST/GET | `/sprint-recovery/**` | Sprint Recovery: start, clarify, decide, retry, re-evaluate, stop, history, time travel |
| POST | `/summarize-comments`, `/description-review`, `/link-similar-work-items` | Issue-level actions |
| POST | `/mcp` | MCP server (Streamable HTTP); public path `/api/ai/mcp` |

Request and response models are in the generated OpenAPI document (`/docs`).

## Ask AI

### Tools

| Tool | Reads | For |
|------|-------|-----|
| `search_knowledge_base` | Hybrid search over issue, comment and attachment text (`POST /search`) | "Why did X break", "how did we fix Y", "find issues about T" |
| `query_issues` | Issue metadata from jira-backend's tables (`POST /issues/query`) | Counting, listing, filtering, recency: "how many bugs", "which are blocked" |
| `get_issue_history` | The change log (`POST /issues/history`) | Transitions: "which issues were reopened", "who moved X to blocked" |
| `query_sprints` | Sprint state, dates and counts | Questions about an active or named sprint |
| `get_issue_comments`, `get_issue_details`, `get_issue_attachments` | Complete source text for known issue keys | When top-k search would return only part of the source |
| `get_section` | The whole PDF section a search hit was cut from (`POST /sections`) | A relevant hit that is incomplete, e.g. steps 4-7 of a procedure |

Semantic search returns its best few matches, never a complete set, so counting and listing questions
go to the structured tools. A count of 0 from `query_issues` is a real answer, not a reason to abstain.
The system prompt is built per request with the current UTC time, so the model can turn "this week"
or "just now" into concrete date filters.

### Authorization: the model never controls `space_ids`

No tool schema has a `space_ids` field. The loop injects the caller's authorized spaces into every
tool call and strips any field that isn't on the tool's allow-list, so a confused or prompt-injected
model cannot widen its own scope. Covered by
`tests/test_crag_loop.py::test_space_ids_are_never_taken_from_the_model`.

### Routing and modes

The first tool is chosen by keyword rules, then by one small LLM call when no rule matches; low
confidence falls back to semantic search. The router can run on a different, smaller model:

```bash
AI_ROUTER_LLM_PROVIDER=openai_compatible     # default: inherit (use the answer model)
AI_ROUTER_MODEL=<gateway route alias>
AI_ROUTER_OPENAI_BASE_URL=http://localhost:4000
AI_ROUTER_OPENAI_API_KEY=sk-ai-service-local
AI_ROUTER_MAX_OUTPUT_TOKENS=128
AI_ROUTER_REASONING_EFFORT=none
```

`mode: fast` (the default) always runs one agent. `mode: deep` allows a multi-agent run (one worker
per workspace, then synthesis, claim review and verification) for questions that compare two or
three selected workspaces.

### Verification

Every answer is checked in code before it is returned: each issue key must come from this request's
tool results, an answer built from search must cite one of its sources, and status and subtask
claims are re-checked against live data (one rewrite on a mismatch). Thin evidence gets a fixed
abstention. An optional LLM faithfulness check (`AI_FAITHFULNESS_CHECK_ENABLED`) is off by default.

### Long chats

The UI sends the whole conversation with every question. When the turns before the latest question
and answer exceed `AI_HISTORY_SUMMARY_OVER_TOKENS` (default 8,000, estimated as characters / 4), the
answer model replaces them with a summary of at most 300 words that keeps issue keys with their status
and priority, numbers, dates, sprints and names. The latest question and answer are always sent word
for word.

The summary is returned as `historySummary` (`text`, `coversTurns`); the client sends the latest one
back with each later question, so the same turns are never summarized twice. If summarizing fails,
the full chat is sent. The summary call is charged to the request.

A follow-up may mention an issue key from the chat that this turn's tools did not return; the key is
looked up in the request's spaces and, if it exists, may be named but not cited.

## LLM access

Every model call goes through the LiteLLM gateway (`../litellm-gateway/`, `:4000`), which holds the
provider keys and enforces per-service budgets, rate limits and routing. Services refer to models by
gateway route alias:

```
AI_LLM_PROVIDER=openai_compatible
AI_AGENT_MODEL=openai-production          # gateway route alias
AI_OPENAI_BASE_URL=http://localhost:4000
AI_OPENAI_API_KEY=sk-ai-service-local     # this service's LiteLLM virtual key
```

| `AI_LLM_PROVIDER` | Implementation | Notes |
|---|---|---|
| `openai_compatible` (default) | `openai` SDK Responses API against `AI_OPENAI_BASE_URL` | Goes through the gateway |
| `anthropic` | `anthropic` SDK, Messages API | Bypasses the gateway. To keep Claude behind the gateway, stay on `openai_compatible` with `AI_AGENT_MODEL=claude-production` |

The agent loop is written once against the `LLMClient` interface (`app/llm/`);
`openai_compatible_client.py` translates between the two message and tool-call shapes.

### When the answer model is down

- **Retries and provider fallback happen only in the gateway.** Both OpenAI clients run with
  `max_retries=0`.
- **A circuit breaker fails fast** (`app/llm/circuit_breaker.py`): after 3 consecutive upstream
  failures on a model, calls fail without being sent for 30 s, then one probe request decides whether
  it closes. `llm_circuit_open{model}` is 1 while it is open.
- **Ask AI returns sources instead of an error.** If the agent fails, `/ask` runs one retrieval and
  returns the most relevant issues and citations with a notice that this is not a generated answer,
  `degraded: true` and `degradedReasons: ["answer_model_unavailable"]`. Only if retrieval also fails
  is the response a 502.
- **Plan Epic and Sprint Recovery say the AI is unavailable** (503, or an SSE `error` frame). A Plan
  Epic run ends as `failed`; a Sprint Recovery run stays resumable from the failed step.
- **BM25-only search is flagged** with `degradedReasons: ["retrieval_bm25_only"]`.
- **Degraded answers are never cached.** Both reasons are counted in
  `agent_ask_degraded_total{reason}`.

## Answer cache

`/ask` checks a two-tier Redis Stack cache (`app/cache/semantic_cache.py`) before running the agent:

- **Exact tier:** one `GET` on the normalized question.
- **Semantic tier:** a RediSearch KNN query over question embeddings (from vectorization-service's
  `POST /embed`), used only above `AI_CACHE_SIMILARITY_THRESHOLD` (default `0.97`) and only when both
  questions contain the same identifiers (words with a digit: part numbers, issue keys, versions,
  dates). A semantic hit is promoted to the exact tier.

Rules:

- **Never crosses `space_ids` scopes**
  (`tests/test_semantic_cache.py::test_space_ids_isolation_never_leaks_across_permission_scopes`).
- **The key follows the index.** Every entry carries a `data_version`: a hash of vectorization-service's
  `GET /index/watermark` token (which changes when searchable text changes) and of `code_version()`,
  a hash of every `.py` file and prompt template under `app/`. An upload or an edited description
  makes the space's old answers unreachable; so does a deploy that changes code or prompts. If the
  token can't be fetched, the request skips the cache.
- **Only answers that don't depend on live Jira state are written:** the request routed to semantic
  search or a multi-agent run, no structured result (issue, sprint, history) went into the answer,
  and the agent did not abstain. Follow-up questions read only the exact tier and are never written
  under their bare text. Attachment-scoped questions skip the cache.
- TTL 10 minutes.

`agent_cache_writes_total{outcome}` (`stored`, `skipped_route`, `skipped_abstained`,
`skipped_request`, `skipped_degraded`) shows why answers are or aren't stored;
`AnswerCacheNeverHits` in `observability/rules/slo.yml` alerts when the cache stops hitting.

### Gateway response cache (`AI_LLM_RESPONSE_CACHE_ENABLED`)

Under the answer cache, each Ask AI model call opts in to LiteLLM's exact-match response cache
(`litellm-gateway/config.yaml`, 1 hour). Its key is the whole request, including the retrieved
evidence, so it needs no freshness signal: if retrieval returns the same chunks, the call replays; if
new content is retrieved, it is a real call. The answer prompt carries the time rounded to 5 minutes.
The end user travels in the `x-litellm-end-user-id` header rather than the request body, so entries
are shared across users who retrieved the same evidence. Only Ask AI opts in; a "regenerate" in other
features must produce a new result.

## Cost tracking and per-user limits

Token usage from every model call is summed per request (`app/llm/types.py::Usage`) and priced with
`app/llm/pricing.py` into `estimated_cost_usd`; `GET /stats` accumulates it. Gateway aliases resolve
back to real model names for pricing (`_GATEWAY_ALIASES`).

`app/auth/user_quota.py` adds two limits per signed-in user, counted in Redis so every replica
shares them:

| Limit | Default | When it's hit |
|-------|---------|---------------|
| Requests per minute | 20 (`AI_USER_QUOTA_REQUESTS_PER_MINUTE`) | 429 with `Retry-After` until the next minute |
| Dollars per UTC day | $2.00 (`AI_USER_QUOTA_DAILY_BUDGET_USD`) | 429 with `Retry-After` until midnight UTC |

- Only endpoints that start new AI work check the limits. Continuing a workflow (approving a plan,
  answering a clarification) is never blocked, though its spend counts. `tests/test_user_quota.py`
  fails if a new POST endpoint isn't classified. The MCP tools that call a model check too.
- Spend comes from the same usage ledger that feeds tracing, including workflow steps that finish
  after the HTTP request.
- Internal-token callers (other services, scripts) are not limited.
- If Redis is down, each process counts on its own until it is back; the gateway's per-service budget
  still caps the total. `ai_user_quota_local_fallback_total` shows when this happens.
- Refusals are counted in `ai_user_quota_rejections_total{reason}`.

## MCP server

`POST /mcp` ([`app/mcp_server/server.py`](app/mcp_server/server.py)) exposes the product's AI features
to MCP clients such as Claude Code, Claude Desktop and Cursor.

| Tool | Returns | Backed by |
|------|---------|-----------|
| `ask_project(question, space?)` | A cited, verified answer, or an abstention | The `/ask` handler |
| `get_ticket_context(issue_key)` | Fields, description, comments, attachment text, change history | Five structured reads in parallel |
| `find_similar_work(issue_key)` | Earlier issues in the same space, with why each matters | The `/link-similar-work-items` handler |
| `check_sprint_risk(space?, sprint?)` | Timeline, progress and risk signals | Sprint Recovery's signal detector; no model call |

There is one prompt, `start_ticket(issue_key)`.

- **Authorization.** No tool has a `space_ids` field. The JWT is verified and every tool derives its
  scope from that user's memberships; `space` can only narrow it. Calls without an identity are
  refused. An issue in another user's space reads as "not found". Covered over the real MCP protocol
  by `tests/test_mcp_server.py`.
- **Transport.** Stateless Streamable HTTP with JSON responses, so any replica can serve any call.
  Metrics: `mcp_tool_calls_total{tool,outcome}`, `mcp_tool_call_seconds`.
- **Configuration.** `AI_MCP_ENABLED` (default `true`); requires
  `AI_SPACE_MEMBERSHIP_CHECK_ENABLED=true`.

Connect Claude Code (token from `POST /api/auth/token`, valid 120 minutes):

```bash
TOKEN=$(curl -s localhost:5173/api/auth/token -H 'Content-Type: application/json' \
  -d '{"username":"atlascart.maya","password":"123"}' | jq -r .accessToken)
claude mcp add --transport http sprint-orchestra http://localhost:5173/api/ai/mcp \
  --header "Authorization: Bearer $TOKEN"
```

`python -m scripts.mcp_smoke` exercises every tool through the public path and checks the refusals.

## Task drafting (`POST /draft-task`)

One structured-output call turns a rough description into a validated `TaskDraft` (title, issue type,
labels, estimate, dependencies), using [Instructor](https://python.useinstructor.com/) with Pydantic
(`app/drafting/`). The prompt is a Jinja2 template
(`app/drafting/templates/task_draft_system.jinja`). `Mode.JSON_SCHEMA` enforces the schema on the
OpenAI-compatible path, `Mode.ANTHROPIC_TOOLS` on the native Anthropic path. If validation retries are
exhausted, the endpoint returns a minimal draft built from the description with `degraded: true`
instead of an error.

## Plan Epic

`/plan-epic/rollout/**` is a LangGraph workflow with a Postgres checkpointer: generate a plan, pause
for human review and edits, then create the epic, sprints, issues and dependencies one at a time with
idempotency keys. A plan needs a proposal and at least one stated requirement; requirements are kept
through every refine round.

With a `spaceId`, planning is grounded in the space's own history:

- **Retrieval** (`app/planning/grounding.py`) finds similar issues, backfilled with the space's most
  recent finished work, and each exemplar says which tier it came from (`similar` or `recent`).
- **Pattern extraction** (`app/planning/patterns.py`, `AI_EPIC_PLANNING_PATTERN_EXTRACTION_ENABLED`)
  reads the retrieved issues once, rates how well they apply (`strength`: strong, partial or none)
  and extracts patterns that recur across at least two issues. A `none` verdict withholds the
  exemplars from the planner. `verify_assessment` drops evidence keys that were never shown and
  downgrades `strength` when the surviving evidence no longer supports it.
- **Estimates** come from code: the model says whether an issue is simpler, similar or more complex
  than the finished issues it cites, and the points are taken from what those issues actually cost.
  Each issue reports `estimateReference`, `historicalBaselinePoints` and `estimateConfidence`; cited
  keys are verified against the exemplar set.
- **House format** (`app/planning/house_format.py`) detects the section labels the board's own issue
  descriptions use (for example Context / Acceptance criteria / Estimate reasoning) and makes them
  part of the output contract. `formatCompliance`, `coverage` and `sampleSize` are on the response.
- The response's `grounding` block lists the exact issues the planner was shown. If a reviewer
  overrides a story point, the AI's estimate citation for it is dropped before publishing
  (`rollout_graph.py::_drop_stale_estimate_references`).

Every grounding step degrades to planning from the proposal alone if it fails.

## Tracing and prompt versioning

With `AI_PHOENIX_ENABLED=true` (Phoenix) and Langfuse configured, one `/ask` is a single trace across
ai-service, the gateway and vectorization-service: `POST /ask` → `CragAgent.ask` → each LLM round →
`litellm_proxy_request` (real model, cost, calling key) and `POST /search` → `search.embed`. W3C
`traceparent` is propagated by `HTTPXClientInstrumentor`. LLM spans are named by step
(`ask.router`, `ask.turn_2 → search_knowledge_base`). See `../observability/README.md`.

```bash
docker run -d -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
AI_PHOENIX_ENABLED=true uvicorn app.main:app --port 8200     # UI: http://localhost:6006
```

`AnthropicClient` is not auto-instrumented (the pinned `anthropic` version is older than the
OpenInference instrumentor supports); calls through the gateway are traced on the gateway side.

**Prompts.** Every prompt is a `.jinja` file under `app/<feature>/templates/`, loaded through
`app/llm/prompt_registry.py`; `tests/test_prompt_registry.py` fails if a prompt is written as a Python
string literal. A prompt's version is `sha256(template)[:12]`, recorded as a span attribute
(`prompt.<name>.version`) and in eval results. `python -m scripts.push_prompt_to_phoenix` pushes changed
templates to Phoenix's prompt registry (`--dry-run` prints the fingerprints); rendering always happens
in-process.

## Metrics

`GET /metrics` exposes request latency plus `agent_llm_call_seconds`, `agent_retrieval_call_seconds`,
`agent_cache_lookup_seconds`, the `agent_retrieval_rounds` distribution,
`agent_cache_hits_total{hit_type}` and `agent_llm_response_cache_hits_total{model}`. Every request gets
an `x-request-id` header and the same id on its structured log lines.

## Eval gate

A pass/fail check before merging an AI change. Floors are in
[`eval/eval_gate.json`](eval/eval_gate.json).

| Check | Script | Where it runs |
|---|---|---|
| Retrieval hit@5 and recall@10 on labelled AtlasCart questions | `eval/retrieval_hit.py` | Locally (`smoke`, `gate`); needs vectorization-service and the corpus |
| Sprint Recovery prompt-injection probe | `eval/sprint_recovery_injection_probe.py` | CI on every PR (`ai-evals`) and locally |
| Sprint Recovery scenarios | `eval/sprint_recovery_scenarios_eval.py` | CI and locally |
| RAGAS on the AtlasCart question set | `eval/ragas_eval.py` | Locally; needs the whole stack and the corpus |

The Sprint Recovery evals use a real model with fake retrieval and Jira, so CI needs only the
`OPENAI_API_KEY` repository secret (the job is skipped with a warning without it).

```bash
AI_CACHE_ENABLED=false AI_LLM_RESPONSE_CACHE_ENABLED=false scripts/dev_up.sh restart ai-service
scripts/run_rag_eval.sh smoke    # retrieval + injection probe, no LLM judge, under a minute
scripts/run_rag_eval.sh gate     # everything, ~15 minutes; refuses to run with a cache on
```

## Tests

```bash
pytest   # agent control flow, the space_ids boundary, pricing, cache isolation and failure handling,
         # workflows, MCP — against fakes, no network
```

Durable-workflow integration tests use PostgreSQL when it is reachable; CI starts one.

## Deployment

Route only the public `/api/ai/**` paths to this service (CloudFront → internal ALB). It verifies every request's JWT itself;
`/metrics`, `/healthz*` and `/stats` must stay internal.

The service uses one PostgreSQL instance: business records in `pocdb.public` (read through
jira-backend), LangGraph checkpoints in the same database.

## Known limitations

- **CORS is open (`allow_origins=["*"]`).** Space authorization is enforced on every route, but a
  deployment needs an origin allow-list.
- **The runtime faithfulness check is off by default**: it adds a model call per answer and flags
  some correct answers.
- **Retrieval grading is the agent's own judgment**, not a trained relevance classifier.
- **Internal-token callers without a user** (scripts, smoke tests) are trusted: space membership is
  checked only when `X-User-Id` is present.
- **The circuit breaker is per process.** Several replicas each keep their own state.
