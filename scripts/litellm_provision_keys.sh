#!/usr/bin/env bash
# Provision one LiteLLM virtual key per service, idempotently.
#
# Why this exists: the gateway's whole value over a plain routing proxy is *per-caller* governance —
# each service authenticates with its own key that carries its own model allow-list, spend budget,
# and rate limit, instead of every service sharing one raw provider key with no isolation. Those
# keys are rows in the gateway's Postgres (they survive container restarts), created via the admin
# API, not the config file. This script is the "infra as a script" stand-in for what Terraform or a
# controller would do in a real deployment.
#
# Idempotent: run it as many times as you like. If a key exists it is updated to match the spec
# below; if not, it is created with the given key string.
#
# Locally the keys default to fixed values (sk-ai-service-local, sk-vec-service-local) so they can sit
# in the committed .env.example; they only work against your own gateway. Anywhere else, pass random
# values from the secret store, the same values the services get as their API keys:
#
#   LITELLM_URL=http://localhost:14000 LITELLM_MASTER_KEY=<from the secret store> \
#   LITELLM_AI_SERVICE_KEY=<from the secret store> LITELLM_VEC_SERVICE_KEY=<from the secret store> \
#   scripts/litellm_provision_keys.sh
#
# Giving a key a new value rotates it: the old one stops working at once.
#
# Run by scripts/dev_up.sh automatically; also fine to run by hand after editing the specs here.
set -euo pipefail

LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"
MASTER_KEY="${LITELLM_MASTER_KEY:-sk-litellm-local-master-key}"
AI_SERVICE_KEY="${LITELLM_AI_SERVICE_KEY:-sk-ai-service-local}"
VEC_SERVICE_KEY="${LITELLM_VEC_SERVICE_KEY:-sk-vec-service-local}"

# The local defaults are public (they are in this repo), so refuse them on any gateway other than the
# local docker one on :4000. A port-forward to a deployed gateway uses another local port, so it is
# refused too.
case "${LITELLM_URL}" in
  http://localhost:4000|http://127.0.0.1:4000) ;;
  *)
    for value in "${MASTER_KEY}" "${AI_SERVICE_KEY}" "${VEC_SERVICE_KEY}"; do
      case "${value}" in
        sk-litellm-local-master-key|sk-ai-service-local|sk-vec-service-local)
          echo "  ERROR: ${LITELLM_URL} is not local, but a key is still its public local default (${value})." >&2
          echo "         Set LITELLM_MASTER_KEY, LITELLM_AI_SERVICE_KEY and LITELLM_VEC_SERVICE_KEY from the secret store." >&2
          exit 1
          ;;
      esac
    done
    ;;
esac

auth=(-H "Authorization: Bearer ${MASTER_KEY}" -H "Content-Type: application/json")

# --- key specs: <key> | <comma-separated models> | <json overrides merged into the body> ---
# ai-service: interactive agent + structured-output features. Real $ (gpt-5.6-luna via
#   openai-production), so a real monthly budget plus rate caps. Caps sit at ~80% of OpenAI's own
#   limits for luna on this account (5,000 RPM / 2M TPM, read from x-ratelimit-* 2026-10-24), so the
#   gateway sheds load before the provider does. TPM is the one that binds first: a question is ~19k
#   tokens over ~5 calls, so 1.6M TPM ≈ 85 questions/min. The old 240 RPM throttled at 16 concurrent users
#   in a load test.
# vec-service: ingestion-time embeddings (text-embedding-3-small), VLM fallback (claude-vlm or gemini-vlm) and
#   contextualization (claude-context, off by default). Guarded on throughput (rpm + parallelism);
#   higher rpm because ingestion arrives in bursts. No spend budget yet.
specs=(
  "${AI_SERVICE_KEY}|openai-production,gemini-production,claude-production|{\"max_budget\": 100, \"rpm_limit\": 4000, \"tpm_limit\": 1600000, \"key_alias\": \"ai-service\"}"
  "${VEC_SERVICE_KEY}|openai-embedding-3-small,claude-vlm,gemini-vlm,claude-context|{\"rpm_limit\": 600, \"max_parallel_requests\": 32, \"key_alias\": \"vectorization-service\"}"
)

echo "litellm: provisioning virtual keys against ${LITELLM_URL}"

# Fail fast with a clear message if the gateway or master key is wrong, rather than a wall of curl errors.
if ! curl -sf "${auth[@]}" "${LITELLM_URL}/health/liveliness" >/dev/null; then
  echo "  ERROR: ${LITELLM_URL} not reachable — is the litellm container up? (docker compose up -d litellm)" >&2
  exit 1
fi

for spec in "${specs[@]}"; do
  IFS='|' read -r key models overrides <<<"${spec}"
  models_json=$(printf '%s' "${models}" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read().split(",")))')
  body=$(python3 - "$key" "$models_json" "$overrides" <<'PY'
import json, sys
key, models_json, overrides = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"key": key, "models": json.loads(models_json)}
body.update(json.loads(overrides))
print(json.dumps(body))
PY
)

  if curl -sf "${auth[@]}" "${LITELLM_URL}/key/info?key=${key}" >/dev/null 2>&1; then
    curl -sf -X POST "${auth[@]}" "${LITELLM_URL}/key/update" -d "${body}" >/dev/null
    echo "  updated ${key:0:6}...  (models: ${models})"
  else
    # Aliases must be unique, so an older key under the same alias (a rotation) is deleted first.
    # From here on only the new key works; give the service its new value before relying on it.
    alias=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1]).get("key_alias", ""))' "${overrides}")
    if [ -n "${alias}" ]; then
      curl -s -X POST "${auth[@]}" "${LITELLM_URL}/key/delete" -d "{\"key_aliases\": [\"${alias}\"]}" >/dev/null || true
    fi
    curl -sf -X POST "${auth[@]}" "${LITELLM_URL}/key/generate" -d "${body}" >/dev/null
    echo "  created ${key:0:6}...  (models: ${models})"
  fi
done

echo "litellm: done. Verify with:  curl -s \"${LITELLM_URL}/key/info?key=<key>\" -H \"Authorization: Bearer <master key>\" | python3 -m json.tool"
