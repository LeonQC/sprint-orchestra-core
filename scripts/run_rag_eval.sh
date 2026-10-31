#!/usr/bin/env bash
# The local half of the AI eval gate. Needs the whole stack running (scripts/dev_up.sh): PostgreSQL,
# OpenSearch, the LiteLLM gateway, vectorization-service and ai-service. CI runs only the part that
# needs no stack (Sprint Recovery scenarios + injection probe, see .github/workflows/ci.yml).
#
# Usage:
#   scripts/run_rag_eval.sh ragas    RAGAS on the AtlasCart question set (~10 min)
#   scripts/run_rag_eval.sh gate     everything below, checked against ai-service/eval/eval_gate.json
#   scripts/run_rag_eval.sh smoke    the deterministic checks only: retrieval hit@5 + injection probe
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-gate}"
failed=0

c_ok=$'\033[32m'; c_fail=$'\033[31m'; c_off=$'\033[0m'

_reachable() { curl -fsS --max-time 2 "$1" >/dev/null 2>&1; }

# Both caches must be off, or repeated questions replay old answers.
_check_stack() {
    local cache
    cache=$(curl -fsS --max-time 2 http://localhost:8200/healthz/cache 2>/dev/null) || {
        echo "${c_fail}[FAIL]${c_off} ai-service (:8200) not reachable"; return 1; }
    if ! echo "$cache" | grep -q '"cache_enabled":false' || ! echo "$cache" | grep -q '"llm_response_cache_enabled":false'; then
        echo "${c_fail}[FAIL]${c_off} restart ai-service with both caches off first:"
        echo "    AI_CACHE_ENABLED=false AI_LLM_RESPONSE_CACHE_ENABLED=false scripts/dev_up.sh restart ai-service"
        return 1
    fi
    if ! _reachable "http://localhost:4000/health/liveliness"; then
        echo "${c_fail}[FAIL]${c_off} LiteLLM gateway (:4000) not reachable"; return 1
    fi
}

run_ragas() {
    echo "--- RAGAS, AtlasCart 50 questions ---"
    _check_stack || { failed=1; return; }
    (cd "$ROOT/ai-service" && source .venv/bin/activate && python -m eval.ragas_eval) \
        && echo "${c_ok}[ OK ]${c_off} RAGAS" \
        || { echo "${c_fail}[FAIL]${c_off} RAGAS"; failed=1; }
}

# Pass/fail check before merging an AI change: retrieval hit@k, RAGAS, the Sprint Recovery scenarios
# and the injection probe. ~15 minutes, roughly $1 of model calls.
run_gate() {
    echo "--- eval gate ---"
    _check_stack || { failed=1; return; }
    local out="$ROOT/ai-service/eval/results/gate"
    mkdir -p "$out"
    (
        cd "$ROOT/ai-service" && source .venv/bin/activate || exit 1
        # The probe exits 1 when an attack succeeds; keep going so the gate reports everything.
        python -m eval.sprint_recovery_injection_probe --out "$out/injection.json"
        # Issue-level hit@5 straight from /search; deterministic, seconds.
        python -m eval.retrieval_hit --out "$out/retrieval.json" || true
        python -m eval.sprint_recovery_scenarios_eval --out "$out/scenarios.json" \
            && python -m eval.ragas_eval --out "$out/ragas.json" \
            && python -m eval.eval_gate --ragas "$out/ragas.json" --scenarios "$out/scenarios.json" \
                --injection "$out/injection.json" --retrieval "$out/retrieval.json"
    ) && echo "${c_ok}[ OK ]${c_off} eval gate" \
      || { echo "${c_fail}[FAIL]${c_off} eval gate"; failed=1; }
}

# The fast half of the gate: everything that needs no LLM judge and gives the same answer twice.
# Retrieval hit@5 on the 32 labelled AtlasCart questions (vectorization-service, ~10 s) and the
# Sprint Recovery injection probe (fakes, no live model). Cheap enough to run before every AI change;
# the judged checks (RAGAS) stay in `gate`.
run_smoke() {
    echo "--- eval smoke (deterministic checks) ---"
    if ! _reachable "http://localhost:8100/healthz"; then
        echo "${c_fail}[FAIL]${c_off} vectorization-service (:8100) not reachable"; failed=1; return
    fi
    local out="$ROOT/ai-service/eval/results/gate"
    mkdir -p "$out"
    (
        cd "$ROOT/ai-service" && source .venv/bin/activate || exit 1
        python -m eval.retrieval_hit --out "$out/retrieval.json" \
            && python -m eval.sprint_recovery_injection_probe --out "$out/injection.json" \
            && python - "$out/retrieval.json" "$out/injection.json" <<'PY'
import json, sys
from eval import eval_gate
config = json.loads(eval_gate.DEFAULT_CONFIG.read_text())
retrieval, injection = (json.loads(open(p).read()) for p in sys.argv[1:])
checks = [c for c in eval_gate.checks(config, None, {"diagnosis": {"passed": 0}, "plan": {"passed": 0}}, injection, retrieval=retrieval)
          if c.name.startswith(("retrieval.", "sprint_recovery.injection"))]
for c in checks:
    print(f"  {'PASS' if c.ok else 'FAIL'}  {c.name:38s} {c.value:>7.3f}  (floor {c.floor:.3f})")
sys.exit(0 if all(c.ok for c in checks) else 1)
PY
    ) && echo "${c_ok}[ OK ]${c_off} eval smoke" \
      || { echo "${c_fail}[FAIL]${c_off} eval smoke"; failed=1; }
}

case "$target" in
    ragas) run_ragas ;;
    gate) run_gate ;;
    smoke) run_smoke ;;
    *)
        echo "Usage: $0 [ragas|gate|smoke]" >&2
        exit 2
        ;;
esac

exit "$failed"
