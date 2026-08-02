# litellm-gateway

A [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) instance: the single OpenAI-compatible
endpoint that every LLM and embedding call from `ai-service` and `vectorization-service` goes
through. It holds the provider keys, enforces a budget and rate limit per calling service, owns the
model routing, caches Ask AI responses, and exports a span for every call to Phoenix.

`ai-service/app/llm/` decides how one process calls a model; the gateway handles what has to be
shared across services: per-caller limits, routing that changes without a redeploy, and one place
where keys, spend and traces live.

## Run it

Starts with `docker compose up` from the repository root. `scripts/dev_up.sh` does the setup steps
automatically; by hand:

```bash
# 1. One-time: the container restarts until its database exists.
docker exec orchestra-postgres psql -U poc -d pocdb -c "CREATE DATABASE litellm;"

# 2. Provider keys go in the root ./.env (gitignored; docker compose reads it).
cat >> .env <<'EOF'
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
EOF

docker compose up -d litellm
curl http://localhost:4000/health/liveliness

# 3. One virtual key per service (idempotent).
scripts/litellm_provision_keys.sh
```

## Routes

Services reference the alias, never a provider model id:

| `model_name` | Used by | Backend |
|---|---|---|
| `openai-production` | ai-service: agent, router, structured features (`AI_AGENT_MODEL`) | OpenAI `gpt-5.6-luna` |
| `claude-production` | Fallback for `openai-production`; ai-service with `AI_LLM_PROVIDER=anthropic` | Anthropic `claude-sonnet-5` |
| `openai-embedding-3-small` | vectorization-service embeddings (`VEC_EMBEDDING_MODEL`) | OpenAI `text-embedding-3-small` |
| `gemini-vlm` | vectorization-service VLM fallback (`VEC_VLM_OPENAI_MODEL`) | Google `gemini-3.8-flash` |
| `claude-vlm` | Alternative VLM route | Anthropic `claude-sonnet-5` |
| `claude-context` | Contextual retrieval (`VEC_CONTEXTUAL_OPENAI_MODEL`, off by default) | Anthropic `claude-haiku-4-5` |
| `claude-judge`, `gemini-judge` | Eval grading | Anthropic `claude-sonnet-5`, Google `gemini-3.5-flash-lite` |

## Provider fallback

`router_settings.fallbacks` sends a failed `openai-production` call (5xx, timeout, 429, connection
error, after `num_retries: 2`) to `claude-production`. Retries happen only here; ai-service's clients
run with `max_retries=0`.

Only the answer model fails over. Embeddings must not switch models (query vectors would stop matching
the index), and judges must stay fixed.

Every route has one deployment, so LiteLLM's cooldown (which rotates between deployments of the same
route) never applies. Failing fast when a provider is down is ai-service's circuit breaker
(`ai-service/app/llm/circuit_breaker.py`). A second deployment of the same model would enable cooldown.

How a caller can tell a fallback happened:

| | Normal | After fallback |
|---|---|---|
| Response body `model` | `openai-production` | `claude-sonnet-5` |
| `x-litellm-model-group` header | `openai-production` | `claude-production` |
| `x-litellm-attempted-fallbacks` header | `0` | `1` |

`ai-service/app/llm/fallback.py` reads the body's `model` to price fallback tokens correctly and count
them in `llm_fallbacks_total`; `LLMFallbackActive` in `observability/rules/slo.yml` alerts on it.

## Response cache

`litellm_settings.cache` is an exact-match cache in Redis keyed on the whole request (model,
instructions, input, tools), with a 1-hour TTL. It is opt-in per request
(`cache: {"use-cache": true}`); only ai-service's Ask AI calls opt in. See
[ai-service README](../ai-service/README.md#gateway-response-cache-ai_llm_response_cache_enabled).

## Per-service virtual keys

`scripts/litellm_provision_keys.sh` creates one key per service through the admin API, each with its
own model allow-list, budget and rate limit:

| Key | Models | Limits |
|---|---|---|
| `sk-ai-service-local` | `openai-production`, `claude-production` | `max_budget: $100`, `rpm_limit: 4000`, `tpm_limit: 1600000` |
| `sk-vec-service-local` | `openai-embedding-3-small`, `claude-vlm`, `gemini-vlm`, `claude-context` | `rpm_limit: 600`, `max_parallel_requests: 32` |

A key calling a model outside its allow-list gets a 403; over its rate limit or budget, a 429. Edit the
`specs=(...)` array in the script and re-run it to change limits.

## Tracing and spend

With `litellm_settings.callbacks: ["arize_phoenix"]` and `HTTPXClientInstrumentor` in both services,
the gateway reads the caller's W3C `traceparent`, so its spans (`litellm_proxy_request`, with the real
model, `llm.token_count.*`, `llm.cost.total` and `metadata.user_api_key_alias`) nest inside the
caller's trace. Calls without a `traceparent` (curl, scripts) land in the `litellm-gateway` Phoenix
project. Export runs off the request path; if Phoenix is down, spans are dropped and calls are
unaffected.

| Where | Scope |
|---|---|
| Phoenix, `sprint-orchestra-ai-service` project | One request with all its LLM and retrieval calls |
| Phoenix, `litellm-gateway` project | Gateway calls without a parent trace |
| `GET :4000/spend/logs`, `:4000/ui` | Per-key spend from the gateway's database (authoritative) |
| `GET :8200/stats`, Grafana | ai-service's own estimate, for cross-checking |

Set `phoenix_project_name` in a virtual key's metadata to send that key's root traces to its own
project.

## Deployment notes

- LiteLLM keeps keys, budgets and spend in a `litellm` database on the shared PostgreSQL instance.
- The master key defaults to `sk-litellm-local-master-key` in `docker-compose.yml` (override with
  `LITELLM_MASTER_KEY`). The virtual keys are fixed local values in `.env.example`. Outside local
  development, all of them and the provider keys come from a secrets manager.
- The gateway is a hard dependency: if it is down, every AI feature fails. The container uses
  `restart: unless-stopped`, and `scripts/dev_up.sh` health-checks it before starting the services.
- `PHOENIX_COLLECTOR_ENDPOINT` points at `host.docker.internal:4317` because Phoenix runs in the
  separate `observability/` compose project.
