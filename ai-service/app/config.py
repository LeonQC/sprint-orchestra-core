"""Runtime configuration for the ai-service (query-time agent layer)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fine on a laptop, unsafe anywhere else. Kept as defaults so a checkout runs with just
# `docker compose up`; the real risk is a deploy that forgets to override one and starts anyway.
# The `environment` check below makes that a startup failure.
_LOCAL_ONLY_DEFAULTS = {
    "internal_service_token": "dev-internal-service-token-min-32-chars-long!!",
    "epic_rollout_checkpoint_db_url": "postgresql://poc:poc123@localhost:5432/pocdb",
    "sprint_recovery_checkpoint_db_url": "postgresql://poc:poc123@localhost:5432/pocdb",
    "openai_api_key": "sk-ai-service-local",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")

    service_name: str = "ai-service"
    log_level: str = "INFO"
    log_json: bool = True

    # Anything other than "local" makes the dev-credential check below fatal.
    environment: str = "local"  # local | staging | production

    # --- Downstream: vectorization-service's retrieval API ---
    vectorization_service_url: str = "http://localhost:8100"

    # --- LLM provider ---
    # openai_compatible = the LiteLLM gateway, which holds the real provider keys; `agent_model` is a
    # gateway route alias (openai-production -> gpt-5.6-luna). anthropic = the native SDK, bypassing
    # the gateway. Same agent code either way.
    llm_provider: str = "openai_compatible"  # openai_compatible | anthropic
    anthropic_api_key: str = ""
    agent_model: str = "openai-production"
    openai_base_url: str = "http://localhost:4000"
    # This service's LiteLLM virtual key (scripts/litellm_provision_keys.sh).
    openai_api_key: str = "sk-ai-service-local"
    # openai_compatible only. Claude's extended thinking is a token budget, not an effort level.
    reasoning_effort: str = "low"

    # --- Capability router LLM ---
    # The router only picks one of seven tools, so it can run on a small model. "inherit" reuses the
    # main client.
    router_llm_provider: str = "inherit"  # inherit | anthropic | openai_compatible
    router_model: str = ""
    router_openai_base_url: str = "http://localhost:4000"
    router_openai_api_key: str = ""
    router_anthropic_api_key: str = ""
    router_max_output_tokens: int = 128
    router_request_timeout_seconds: float = 30.0
    router_reasoning_effort: str = "none"

    # --- Agentic / Corrective RAG loop ---
    max_tool_iterations: int = 4  # hard cap on retrieve-and-reassess rounds — bounds cost/latency
    retrieval_top_k: int = 5
    max_output_tokens: int = 2048
    # Generous because a timeout isn't retried (a retry would only double the wait), and a long
    # reasoning generation can take over a minute.
    llm_request_timeout_seconds: float = 300.0

    # --- Runtime faithfulness check (crag_loop.py `_check_faithfulness`) ---
    # RAGAS measures faithfulness offline; this checks each live answer against what was retrieved
    # before the user sees it. Off by default: one extra LLM call per answer.
    faithfulness_check_enabled: bool = False

    # --- Parent-child retrieval (crag_loop.py get_section) ---
    # Lets the agent read the whole PDF section a search hit was cut from. Off = the agent as before.
    parent_child_tool_enabled: bool = True

    # --- Adaptive Ask AI: single CRAG vs multi-agent investigation (app/agent/mode_router.py) ---
    # When on, POST /ask first chooses an execution mode. Multi is reserved for cross-workspace or
    # clearly decomposable compares (see mode_router); ordinary QA stays on CragAgent. Off keeps
    # the historical /ask = CRAG-only contract. /investigate remains an explicit opt-in endpoint.
    adaptive_rag_routing_enabled: bool = True

    # --- Ask AI chat history (app/agent/history_summary.py) ---
    # The turns before the latest answer are summarized once they pass this many tokens (estimated).
    history_summary_over_tokens: int = 8_000

    # --- Multi-agent epic planning (app/planning/graph.py) ---
    # Planner -> critic loop instead of one call; same response shape. Off because on 8 proposals it
    # cost 2-4x the calls for +0.13/+0.26 (of 5), below the pre-registered bar
    #. Kept since that eval is small.
    epic_planning_multiagent_enabled: bool = False

    # --- Retrieval-grounded planning (app/planning/grounding.py) ---
    # Show the planner issues this space already shipped, so wording and story points follow the
    # team's own history instead of the model's guesses.
    # On by measurement: story-point distribution distance to the real board 0.683 -> 0.477, better on
    # 7 of 8 epics, p = 0.0078. An earlier total-points eval said
    # "worse" because grounding also splits work into fewer cards — it measured the wrong thing.
    # Not shown: that it writes "like this team" (blind style test 13-11, p=0.84). Also costs prompt
    # caching (~13x fresh input tokens).
    # Does nothing without a spaceId in the request.
    epic_planning_grounding_enabled: bool = True
    # Enough examples to show a house style, few enough (~5.6k chars) not to crowd out the proposal.
    epic_planning_grounding_exemplars: int = 8
    # Extra past issues per stated requirement. Small so 10 requirements don't become 80 cards of
    # prompt; 3 is enough to judge "bigger or smaller than these".
    epic_planning_requirement_exemplars: int = 3
    # Before planning, judge whether the retrieved history is relevant at all and pull out patterns
    # (app/planning/patterns.py). A "none" verdict hides the examples from the planner — the only thing
    # here that stops irrelevant history from shaping a plan. Needs grounding on; one extra LLM call.
    epic_planning_pattern_extraction_enabled: bool = True
    # Let pattern extraction (never the planner) read the exemplars' comments, where "what went wrong
    # last time" usually lives. Off = cheaper but blinder.
    epic_planning_pattern_comments_enabled: bool = True

    # --- Semantic query cache (app/cache/semantic_cache.py, Redis) ---
    # On by default: a hit saves an LLM call.
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"  # same `redis` container docker-compose.yml already runs
    cache_ttl_seconds: float = 600.0  # TTL-only invalidation; see module docstring
    cache_similarity_threshold: float = 0.97  # deliberately high — see module docstring
    cache_max_entries: int = 500
    # Must equal vectorization-service's VEC_EMBEDDING_DIM; the Redis vector index has a fixed size.
    cache_embedding_dim: int = 1024
    # How long to reuse the index freshness token. 0 = fetch on every lookup (a few ms), because any
    # memo window lets stale answers back in for that long.
    cache_watermark_ttl_seconds: float = 0.0
    # Ask AI's LLM calls opt in to LiteLLM's exact-match response cache (litellm-gateway/config.yaml).
    # Keyed on the whole request, retrieved evidence included, so it needs no watermark. Turn off for
    # eval runs that measure run-to-run variance, or repeats would be replays. openai_compatible only.
    llm_response_cache_enabled: bool = True

    # --- Request authentication (app/auth/request_auth.py) ---
    # Every request carries a user JWT, verified with jira-backend's public keys, or the internal
    # service token. To trust an external issuer (Cognito, Auth0), point both at it.
    jwt_jwks_url: str = "http://localhost:8081/.well-known/jwks.json"
    jwt_issuer: str = "jira-agentic-ai"
    # Shared secret for service-to-service calls (X-Internal-Token), both ways with jira-backend.
    # Also signs AI feedback receipts, so jira-backend must hold the same value.
    internal_service_token: str = "dev-internal-service-token-min-32-chars-long!!"

    # --- Per-user AI limits (app/auth/user_quota.py), counted in Redis at redis_url ---
    # Checked only on endpoints that start new AI work, only for JWT users.
    user_quota_enabled: bool = True
    user_quota_requests_per_minute: int = 20
    # About 290 Ask AI questions (~$0.007 each on gpt-5.6-luna, measured 2026-10-29).
    user_quota_daily_budget_usd: float = 2.0

    # --- Space-membership check (auth boundary) ---
    # Don't trust space_ids from the request body: check them against the user's real memberships
    # (X-User-Id from the verified JWT + jira-backend's membership endpoint). Skipped when X-User-Id
    # is absent, which only happens for internal-token callers such as eval scripts.
    space_membership_check_enabled: bool = True
    jira_backend_url: str = "http://localhost:8081"
    space_membership_cache_ttl_seconds: float = 60.0

    # --- MCP server (app/mcp_server/server.py): POST /mcp, public path /api/ai/mcp ---
    mcp_enabled: bool = True

    # --- Epic rollout: human-approved, exactly-once commit to Jira (app/planning/rollout_graph.py) ---
    # Controls what happens after a plan is approved (durable pause, then write once), independent of
    # how the plan was generated.
    epic_rollout_enabled: bool = True
    # Reuses jira-backend's Postgres in its own schema; not worth a second container.
    epic_rollout_checkpoint_db_url: str = "postgresql://poc:poc123@localhost:5432/pocdb"

    # --- Sprint recovery (app/sprint_recovery/graph.py) ---
    sprint_recovery_enabled: bool = True
    # Same checkpoint store as epic rollout.
    sprint_recovery_checkpoint_db_url: str = "postgresql://poc:poc123@localhost:5432/pocdb"
    sprint_recovery_max_clarification_rounds: int = 2
    sprint_recovery_max_escalation_rounds: int = 2
    # Caps the "human rejected with feedback -> replan" loop, like every other loop in the graph.
    sprint_recovery_max_plan_revision_rounds: int = 2
    sprint_recovery_max_token_budget: int = 200_000
    # Optional: escalations also POST {"text": ...} here (works as a Slack incoming webhook).
    # A WARNING log line is always written either way. See app/sprint_recovery/notifications.py.
    sprint_recovery_escalation_webhook_url: Optional[str] = None

    # --- Phoenix (Arize) tracing + prompt management ---
    # Off by default: needs a running Phoenix container. See app/tracing.py.
    phoenix_enabled: bool = False
    phoenix_collector_endpoint: str = "http://localhost:4317"
    phoenix_project_name: str = "sprint-orchestra-ai-service"
    # Phoenix's UI/API port (spans go in on 4317, come out on 6006). Only scripts read it.
    phoenix_base_url: str = "http://localhost:6006"

    # --- Langfuse (optional second trace backend) ---
    # Same spans as Phoenix, exported to both (that's the payoff of using OpenTelemetry, not a vendor
    # SDK). Langfuse adds per-user / per-feature cost dashboards. Its OTLP endpoint is HTTP + Basic auth.
    langfuse_enabled: bool = False
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    @model_validator(mode="after")
    def _reject_local_defaults_outside_local(self) -> "Settings":
        """Fail startup if a deployed environment still uses a dev credential.

        Fail instead of warn: nobody reads container warnings. Lists every offender at once so one
        fix covers them all.
        """
        if self.environment.strip().lower() == "local":
            return self
        offenders = [
            f"AI_{name.upper()}"
            for name, dev_value in _LOCAL_ONLY_DEFAULTS.items()
            if getattr(self, name, None) == dev_value
        ]
        if offenders:
            raise ValueError(
                f"environment={self.environment!r} but these still hold their local development "
                f"defaults: {', '.join(offenders)}. Set them from the deployment's secret store "
                f"(see README 'Deploying'), or set AI_ENVIRONMENT=local if this really is a laptop."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
