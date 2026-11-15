# observability

Optional Prometheus + Alertmanager + Grafana + Phoenix stack — the signals this project's own code
exposes: metrics (`GET /metrics` in `vectorization-service` and `ai-service`, `app/observability.py` in
each; `/actuator/prometheus` in `jira-backend`), and distributed traces — LLM calls plus retrieval, across `ai-service` and
`vectorization-service`, joined by W3C `traceparent` (see each service's `app/tracing.py`), with the
LiteLLM proxy exporting its own gateway-side spans when traffic routes through it. Kept out of the main
`docker-compose.yml`, so a plain `docker compose up` doesn't start five extra containers.
`scripts/dev_up.sh` does start it; the commands below run it on its own.

On AWS the same containers run on ECS: `aws/` has the images (the official ones with this
directory's config copied in) and Prometheus's AWS targets.

## Run it

```bash
# 1. Start jira-backend, vectorization-service and ai-service normally (see their own READMEs) —
#    Prometheus scrapes them on :8081 / :8100 / :8200.
#    For traces, ai-service also needs AI_PHOENIX_ENABLED=true (see ai-service/.env.example).

# 2. From the repo root:
docker compose -f observability/docker-compose.observability.yml up -d

# 3. Open Grafana (metrics) and Phoenix (traces)
open http://localhost:3001   # Grafana — anonymous admin access locally, see Security note
open http://localhost:6006   # Phoenix — trace explorer
open http://localhost:9090/alerts   # Prometheus — SLO alert state (see "Service-level objectives")
open http://localhost:9093          # Alertmanager — what was sent to Slack, and silences
```

## Traces (Phoenix)

Metrics answer "is this slow, and which stage" — traces answer "show me the actual input/output for
*this one* `/ask` call, and every LLM call a multi-round CragAgent conversation made along the way."
`ai-service` sends OTLP spans to Phoenix's collector (`:4317`) whenever `AI_PHOENIX_ENABLED=true`,
`vectorization-service` when `VEC_TRACING_ENABLED=true`; browse them at `:6006`. See
`ai-service/app/tracing.py` for which LLM client paths are instrumented.

One `/ask` is a single connected trace across **three processes** (~40 spans):
`POST /ask` (ai-service) → `CragAgent.ask` → each `Response` (LLM round) → `litellm_proxy_request`
(the LiteLLM gateway, real model, `key=ai-service`) → and `POST /search` (vectorization-service) →
`search.embed` → `litellm_proxy_request` (gateway, `key=vectorization-service`) /
`search.retrieval` / `search.rerank`. W3C `traceparent` is injected by `HTTPXClientInstrumentor` in
both services and read by the gateway and by vectorization-service. All three need tracing on
(`AI_PHOENIX_ENABLED` / `VEC_TRACING_ENABLED` / the gateway's `arize_phoenix` callback, on by
default) for the whole waterfall. (Both services' `instrument_app` calls run at *import* time, not in
`lifespan` — Starlette freezes the middleware stack before the lifespan body runs, so instrumenting
there produced no server span. See each `app/main.py`.)

### Cost attribution: who spent what, on which AI feature

Every AI feature call also writes a **usage ledger** onto its span — the authenticated user, the
conversation, the feature, and the cost itemized as `tokens x published rate` per billed bucket. See
`ai-service/app/llm/usage_ledger.py`.

In the Phoenix UI this data is filterable — `ai.feature`, `ai.usage.cost_usd` and friends on the
span, the full itemized record under `metadata`, and Phoenix's own Sessions view grouping a chat's
turns via `session.id`.

### Langfuse: the same spans, in a cost dashboard

Phoenix answers "what did this one call do"; Langfuse Cloud gives the aggregate view ("who spent
what this month") from the **same spans**: `ai-service/app/tracing.py` adds a second OTLP exporter.

Create an API key in the Langfuse project (Settings → API Keys), then in `ai-service/.env` set
`AI_LANGFUSE_ENABLED=true`, `AI_LANGFUSE_BASE_URL` (`https://us.cloud.langfuse.com` or
`https://cloud.langfuse.com` for EU) and the two keys, and restart.

Every LLM span is named by step, in both Phoenix and Langfuse, instead of the SDK's generic
`Response` / `ChatCompletion`: `ask.router`, `ask.turn_2 → search_knowledge_base`,
`ask.turn_3 → answer`, `ask.verify.faithfulness`, `plan_epic.generate`, `plan_epic.critique_round_1`,
`sprint_recovery.diagnose`, and so on. The prefix is the usage ledger's feature name. A new LLM call
gets its name from `with llm_step("<feature>.<step>"):` around it (`ai-service/app/tracing.py`).

How to find what happened to one request:

- **Start from the Traces list, not Observations.** One row per request, named after the endpoint
  (`POST /ask`, `POST /plan-epic`), with the request body as Input and the response (for streaming
  endpoints, the final `result` or `error` event) as Output. Filter by tag (the feature, e.g.
  `plan_epic`), user or session. The Observations list is every span flattened; concurrent requests
  interleave there, so use it to filter by step name or Type = Generation across traces.
- **Open the trace and read the tree.** Spans without cost (the request, `CragAgent.ask`, graph nodes
  such as `sprint_recovery.diagnose`) are containers; the LLM calls are the generations under them,
  with the full prompt and response.
- **Errors.** Filter Level = Error. A paused approval (`sprint_recovery.approval`) is not an error;
  it carries `sprint_recovery.paused = true`. `APIConnectionError` on LLM calls means the gateway at
  :4000 was unreachable.
- **From a log line to its trace and back.** Every JSON log line written during a request carries
  `trace_id`, the id Langfuse and Phoenix show, in both ai-service and vectorization-service.
  `grep <trace_id> /tmp/sprint-orchestra-dev/logs/*.log` gives that request's log lines across both
  services. Langfuse holds LLM-relevant spans only; Phoenix holds every span, including retrieval
  and HTTP timing.

Two details make the cost view work:

- **The caller is on every span.** Token counts live on the provider-SDK spans, so
  `CallerAttributeMiddleware` puts the caller in the OTel context and every downstream span inherits
  it.
- **Langfuse can price the model name it sees.** LiteLLM reports the gateway route
  alias (`openai-production`) for some calls, which Langfuse cannot price. The Langfuse exporter
  rewrites known aliases to the real model name (`_normalize_langfuse_model` in `tracing.py`) and
  keeps the alias as `ai.model.route_alias`. A model Langfuse still does not know can be added under
  Settings → Models. Pricing is applied **at ingestion**, so traces recorded before are not
  retroactively priced.

### Storage: Postgres, not the default SQLite

Phoenix keeps its traces in the `phoenix` database on the `postgres` container from the root
`docker-compose.yml` (`PHOENIX_SQL_DATABASE_URL`), the same one the LiteLLM gateway uses for its own
metadata. `scripts/dev_up.sh` creates the database; by hand it is:

```bash
docker exec orchestra-postgres psql -U poc -d pocdb -c "CREATE DATABASE phoenix;"
```

Phoenix holds the cost-attribution history, so it lives in an ordinary database that can be backed
up, rather than the default SQLite inside the container:

```bash
docker exec orchestra-postgres pg_dump -U poc -d phoenix > phoenix-backup.sql
```

In a deployed environment, point `PHOENIX_SQL_DATABASE_URL` at RDS and trace history follows that
instance's snapshot policy.

Two Phoenix projects, two viewpoints:

- **`sprint-orchestra-ai-service`** — the per-request view: one `/ask`, every LLM call it made
  (with the gateway's spans and the `vectorization-service` `search.*` spans nested in). Phoenix
  files the whole cross-process trace under this one project — the root span's project wins.
- **`litellm-gateway`** — collects gateway calls that arrive *without* a `traceparent`: direct
  curls, scripts, any other caller. (A call from `ai-service` carries a
  `traceparent`, so its gateway spans join the trace above instead.) Every gateway span, in either
  project, carries the real model name, real cost, and `metadata.user_api_key_alias`
  (`ai-service` / `vectorization-service`) — see `litellm-gateway/README.md`.

The dashboard **"Sprint Orchestra — RAG Platform"** is auto-provisioned (no manual "add data source" /
"import dashboard" clicking) — it should already be there under the "RAG Platform" folder the moment
Grafana starts, showing:

- Request rate + P95 latency, both services
- 5xx error rate
- `vectorization-service` `/search` stage breakdown (embed / retrieval / rerank latency)
- `ai-service` `/ask` stage breakdown (LLM call / retrieval tool call / cache lookup latency)
- Semantic cache hit rate
- Average corrective-retrieval rounds per question
- Cumulative estimated $ spent

Generate some traffic to see it move — hit `/ask` and `/search` a few times.

## Service-level objectives and alerts (`rules/slo.yml`)

Prometheus evaluates these every 15 seconds:

| Objective | Target |
|---|---|
| vectorization-service `/search` P95 | **< 1s** |
| ai-service `/ask` P95 | **< 8s** (hosted LLM endpoint) |
| 5xx rate, any service | **< 1%** |
| Answer cache hit rate | **≥ 5%** with real traffic |

Targets leave headroom above normal values, so normal jitter doesn't page anyone.

### Cache alerts

`AnswerCacheNeverHits` fires when the answer cache serves no hits despite traffic: a cache that never
hits is broken whatever the reason, and nothing else would show it.

`AnswerCacheFreshnessCheckUnavailable` fires when the index freshness check fails, which makes every
request bypass the cache. From the hit rate alone it looks the same as a cold cache, but the fix is in
vectorization-service, not Redis.

### Outages that don't look like errors

Four alerts cover dependencies whose failure users never see as an error, because the system
degrades instead of failing. Without them, the only symptom is slightly worse answers.

| Alert | Fires when | What users get meanwhile |
|---|---|---|
| `LLMFallbackActive` | `llm_fallbacks_total` has risen for 5 min | Gemini answers instead of OpenAI, at a higher cost per call (about 5x per /ask) |
| `AskAnswerModelsUnavailable` (critical) | `agent_ask_degraded_total{reason="answer_model_unavailable"}` has risen for 2 min | Ask AI returns sources with no generated answer |
| `SearchDegradedToBm25` | `search_degraded_total` has risen for 5 min | Search runs on BM25 only; ingestion is paused on the same outage |

`llm_circuit_open{model}` (ai-service) shows whether calls to a gateway model are currently being
failed fast, and `ingest_vlm_retries_total{outcome}` (vectorization-service) shows whether attachments
indexed during a VLM outage are being caught up.

```bash
# Rules loaded, and their current state:
open http://localhost:9090/rules
open http://localhost:9090/alerts        # inactive / pending / firing
```

### Getting alerts to Slack (Alertmanager)

Prometheus only decides that an alert is firing. Alertmanager (`alertmanager/alertmanager.yml`) sends
it to Slack, groups alerts from the same incident into one message, repeats it every 4 hours while it
keeps firing, sends a "resolved" message when it stops, and drops a warning when a critical alert for
the same job is already firing.

The Slack webhook URL is a secret, so it is read from a gitignored file:

```bash
# Slack: Apps -> Incoming Webhooks -> Add to a channel, copy the URL
echo -n 'https://hooks.slack.com/services/...' > observability/alertmanager/slack_webhook_url
docker restart orchestra-alertmanager
```

Without that file, alerts still show at `:9093`; only the Slack send fails (in the container log).
`ServiceDown` covers the case no other alert can: a service that stopped answering Prometheus at all.

## Redis metrics (`redis-exporter`)

ai-service measures its hit rate; only Redis can say whether it is evicting entries under memory
pressure. The exporter adds:

- **Redis: memory used by cached answers + HNSW index** — the application's own FIFO/`max_entries`
  eviction bounds entries *per space*; this is the eviction it cannot see.
- **Redis: keys evicted + expired** — expiry is the TTL working as designed; eviction is Redis running
  out of room. Two very different lines that a single "keys removed" number would have merged.

## What's provisioned automatically vs. not

- `observability/grafana/provisioning/datasources/prometheus.yml` — the Prometheus data source, no
  manual setup.
- `observability/grafana/provisioning/dashboards/dashboards.yml` — tells Grafana to load any
  dashboard JSON in `grafana/dashboards/` on startup.
- `observability/prometheus.yml` — scrape config. Targets `host.docker.internal:8100` /
  `:8200`, since both services run directly on the host via `uvicorn`, not inside this compose
  network. Works out of the box on Docker Desktop (macOS/Windows); on Linux, add
  `extra_hosts: ["host.docker.internal:host-gateway"]` under the `prometheus` service in
  `docker-compose.observability.yml`.

## Security note

`GF_AUTH_ANONYMOUS_ENABLED=true` (admin role, no login) is set in
`docker-compose.observability.yml` for local use only. A deployment sets `GF_SECURITY_ADMIN_PASSWORD`
and turns anonymous access off.

## Stop it

```bash
docker compose -f observability/docker-compose.observability.yml down
```


## PostgreSQL

The local stack has one PostgreSQL container, `orchestra-postgres` on 5432: business data in
`pocdb.public`, search data in `pocdb.rag`, plus separate `phoenix` and `litellm` databases on the same
instance. Watch combined load and pool waits when sizing it; schemas don't isolate CPU or I/O.
