#!/usr/bin/env bash
# One-click local dev stack for AtlasCart / Ask-AI.
#
# Problem this solves: the full "Ask AI works in the browser" path is SIX things that all have to
# be up at once (frontend -> jira-backend + ai-service -> vectorization-service -> DBs, plus the
# LiteLLM gateway). After any one of them gets killed/restarted, the UI just says "server unavailable" with
# no clue which of the six is the culprit. This script starts everything that isn't already running,
# in the right order, and prints one clear status table at the end. That includes the observability
# overlay (Phoenix, Prometheus, Grafana), so local runs are traced and scraped like production.
#
# Safe to re-run any time: every step checks its port FIRST and skips anything already up, so this
# never double-starts a service or fights with one you started by hand.
#
# Usage:
#   scripts/dev_up.sh              start everything that isn't already running
#   scripts/dev_up.sh status       just print the status table, start nothing
#   scripts/dev_up.sh down         stop the processes THIS script started (PIDs in $PIDS_DIR)
#   scripts/dev_up.sh restart [service...]
#                                  stop the named services (default: all of them), then start
#                                  everything that isn't up. Use this after changing code — a plain
#                                  re-run will NOT pick your changes up (see below).
#
# Why `restart` exists: "skip anything already up" is the right default for starting a stack and
# exactly the wrong one after an edit. A plain re-run prints [ OK ] for a service that is still
# happily serving the OLD code, which looks like success. Only vite reloads itself (HMR) — the
# uvicorn services here run without --reload, and the Spring services obviously don't watch source
# either, so every Python or Java change needs the process replaced.
#
# The dependency chain, so a failure is easy to place:
#   browser -> vite :5173 (dev proxy) -> jira-backend :8081   (issues/sprints CRUD, Postgres :5432)
#                                     \-> ai-service   :8200   (Ask-AI agent; /api/ai/* prefix stripped)
#                                            -> vectorization-service :8100  (retrieval, pocdb.rag :5432)
#                                            -> LiteLLM :4000                (LLM + embeddings, hosted)
#                                            -> Redis :6379                  (answer cache)
#   Each service verifies the user's JWT itself; ai-service fetches jira-backend's public key.
#
# Logs for anything THIS script starts land in $LOG_DIR (printed at the end) — `tail -f` one if a
# service comes up unhealthy. Anything already running when the script starts is left completely
# alone (not restarted, no log redirect), since that's usually a process you're actively working with.
#
# Each launcher below ends with `</dev/null >/dev/null 2>&1` on the *subshell*, not just on the
# service inside it. The service's own output already goes to $LOG_DIR, but the short-lived shell
# that forks it inherits the caller's stdout and outlives the script as the started process's parent
# — so it holds that file descriptor open for as long as the service runs. On a terminal that is
# invisible; through a pipe it is a hang. `scripts/dev_up.sh restart ai-service | tee out.log` never
# returned, because `tee` sat waiting for an EOF that the still-running uvicorn's parent shell was
# holding.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDIO_DIR="$(cd "$ROOT/../sprint-orchestra-studio" 2>/dev/null && pwd || true)"
RUN_DIR="/tmp/sprint-orchestra-dev"
LOG_DIR="$RUN_DIR/logs"
PIDS_DIR="$RUN_DIR/pids"
mkdir -p "$LOG_DIR" "$PIDS_DIR"

# --- tiny output helpers ---------------------------------------------------------------------
c_ok=$'\033[32m'; c_warn=$'\033[33m'; c_start=$'\033[36m'; c_off=$'\033[0m'
say_ok()    { printf "  ${c_ok}[ OK ]${c_off}    %s\n" "$1"; }
say_start() { printf "  ${c_start}[START]${c_off}    %s\n" "$1"; }
say_warn()  { printf "  ${c_warn}[WARN]${c_off}    %s\n" "$1"; }
say_skip()  { printf "  ${c_warn}[SKIP]${c_off}    %s\n" "$1"; }

port_open() { # port_open <port>
    lsof -tiTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

# wait_for_port <port> <seconds> <label> — poll, don't sleep-and-hope.
wait_for_port() {
    local port="$1" timeout="$2" label="$3" waited=0
    while ! port_open "$port"; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge "$timeout" ]; then
            say_warn "$label still not listening on :$port after ${timeout}s — check its log"
            return 1
        fi
    done
    [ "$waited" -gt 0 ] && printf "           (took ~%ss)\n" "$waited"
    return 0
}

# --- action: status ---------------------------------------------------------------------------
print_status() {
    echo ""
    echo "=== Status ==="
    printf "  %-28s %-8s %s\n" "SERVICE" "PORT" "STATE"
    for row in \
        "PostgreSQL (public + rag):5432" \
        "Redis (answer cache):6379" \
        "jira-backend:8081" \
        "vectorization-service:8100" \
        "ai-service:8200" \
        "litellm gateway:4000" \
        "frontend (vite):5173" \
        "Phoenix (traces):6006" \
        "Prometheus (metrics):9090" \
        "Grafana (dashboards):3001" \
        "Alertmanager (alerts):9093" \
    ; do
        name="${row%%:*}"; port="${row##*:}"
        if port_open "$port"; then state="${c_ok}up${c_off}"; else state="${c_warn}down${c_off}"; fi
        printf "  %-28s %-8s %b\n" "$name" "$port" "$state"
    done
    echo ""
}

if [ "${1:-}" = "status" ]; then
    print_status
    exit 0
fi

# Port for a service name, so `restart` can find a process this script never started. A `case` and
# not an associative array on purpose: macOS still ships bash 3.2, where `declare -A` doesn't exist.
service_port() {
    case "$1" in
        jira-backend)           echo 8081 ;;
        vectorization-service)  echo 8100 ;;
        ai-service)             echo 8200 ;;
        frontend)               echo 5173 ;;
        *)                      echo "" ;;
    esac
}
ALL_SERVICES="jira-backend vectorization-service ai-service frontend"

# Stop one service by BOTH handles: the pid this script recorded, and whatever is actually holding
# the port. The second is the one that matters in practice — a service you started by hand in your
# own terminal has no pid file here, and stopping only the recorded pid would leave the old process
# serving stale code while the start phase below reports the port as already up.
stop_service() {
    local name="$1" port="$2" stopped="" pid pids waited=0
    if [ -e "$PIDS_DIR/$name.pid" ]; then
        pid="$(cat "$PIDS_DIR/$name.pid")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && stopped="pid $pid"
        fi
        rm -f "$PIDS_DIR/$name.pid"
    fi
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)"
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086 - word splitting is wanted, lsof may return several pids
        kill $pids 2>/dev/null
        stopped="${stopped:+$stopped, }:$port"
    fi
    if [ -z "$stopped" ]; then
        say_skip "$name was not running"
        return 0
    fi
    while port_open "$port" && [ "$waited" -lt 20 ]; do sleep 1; waited=$((waited + 1)); done
    if port_open "$port"; then
        say_warn "$name still holding :$port after ${waited}s — 'kill -9 \$(lsof -tiTCP:$port)' if it's wedged"
    else
        say_ok "stopped $name ($stopped)"
    fi
}

if [ "${1:-}" = "restart" ]; then
    shift
    targets="$*"
    [ -z "$targets" ] && targets="$ALL_SERVICES"
    echo "Restarting: $targets"
    for name in $targets; do
        port="$(service_port "$name")"
        if [ -z "$port" ]; then
            say_warn "unknown service '$name' — known: $ALL_SERVICES"
            continue
        fi
        stop_service "$name" "$port"
    done
    echo ""
    # No exit: fall through to the normal start path, which now sees the stopped ports as free and
    # brings them back with the current code. Anything else that was already up stays untouched.
fi

if [ "${1:-}" = "down" ]; then
    echo "Stopping processes this script started (recorded in $PIDS_DIR)..."
    for f in "$PIDS_DIR"/*.pid; do
        [ -e "$f" ] || continue
        pid="$(cat "$f")"
        name="$(basename "$f" .pid)"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && say_ok "stopped $name (pid $pid)"
        else
            say_skip "$name (pid $pid) already gone"
        fi
        rm -f "$f"
    done
    exit 0
fi

echo "Starting the AtlasCart / Ask-AI local dev stack..."
echo "(safe to re-run — anything already up is left alone)"
echo ""

# --- 1. Docker infra: Postgres (public + rag), Redis, Kafka+ZK, MinIO ---------------------------------
echo "--- Docker infra ---"
if port_open 5432 && port_open 6379; then
    say_ok "Postgres (public + rag) / Redis already up"
else
    say_start "docker compose up -d (postgres, redis, kafka, minio...)"
    (cd "$ROOT" && docker compose up -d) >"$LOG_DIR/docker-compose.log" 2>&1
    wait_for_port 5432 60 "Postgres"
    wait_for_port 6379 30 "Redis"
    # OpenSearch owns all text search — jira-backend's search bar and RAG's keyword matching both fail
    # closed without it, so wait for the port rather than letting the services start into a
    # dependency that is not there yet. It boots a JVM, so it is the slowest thing here.
    wait_for_port 9200 120 "OpenSearch"
fi

# --- 2b. litellm gateway (:4000): every LLM/embedding call from ai-service and vectorization-service
#         egresses through this. Started by `docker compose up -d` in step 1 now (no profile), but it
#         needs its own database and one virtual key per service. Both are idempotent. ------------
echo "--- litellm gateway :4000 ---"
if curl -sf -m 3 http://localhost:4000/health/liveliness >/dev/null 2>&1; then
    say_ok "reachable on :4000"
else
    # One-time: the litellm container crash-loops until this database exists.
    docker exec orchestra-postgres psql -U poc -d pocdb -tc \
        "SELECT 1 FROM pg_database WHERE datname='litellm'" 2>/dev/null | grep -q 1 \
        || docker exec orchestra-postgres psql -U poc -d pocdb -c "CREATE DATABASE litellm" >/dev/null 2>&1
    say_start "docker compose up -d litellm"
    (cd "$ROOT" && docker compose up -d litellm) >>"$LOG_DIR/docker-compose.log" 2>&1
    waited=0
    while ! curl -sf -m 3 http://localhost:4000/health/liveliness >/dev/null 2>&1; do
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge 40 ]; then
            say_warn "litellm not up on :4000 after ${waited}s — check 'docker logs orchestra-litellm'."
            say_warn "ai-service / vectorization-service will fail their LLM calls until it is."
            say_warn "(offline? each .env.example has an escape-hatch block to bypass the gateway.)"
            break
        fi
    done
    curl -sf -m 3 http://localhost:4000/health/liveliness >/dev/null 2>&1 && say_ok "up (took ~${waited}s)"
fi
if curl -sf -m 3 http://localhost:4000/health/liveliness >/dev/null 2>&1; then
    bash "$ROOT/scripts/litellm_provision_keys.sh" 2>&1 | sed 's/^/           /' || \
        say_warn "key provisioning failed — run scripts/litellm_provision_keys.sh by hand"
fi

# --- 2c. observability overlay: Phoenix :6006/:4317, Prometheus :9090, Alertmanager :9093, Grafana :3001
#         Before ai-service, so its first spans have somewhere to go. A separate compose file, so a
#         plain `docker compose up` still doesn't start it. --------------------------------------
echo "--- observability (Phoenix / Prometheus / Alertmanager / Grafana) ---"
if port_open 4317 && port_open 9090 && port_open 9093 && port_open 3001; then
    say_ok "already up"
else
    # One-time: Phoenix keeps its traces (and the per-user AI cost ledger) in this Postgres database,
    # and crash-loops on boot if it is missing.
    docker exec orchestra-postgres psql -U poc -d pocdb -tc \
        "SELECT 1 FROM pg_database WHERE datname='phoenix'" 2>/dev/null | grep -q 1 \
        || docker exec orchestra-postgres psql -U poc -d pocdb -c "CREATE DATABASE phoenix" >/dev/null 2>&1
    say_start "docker compose -f observability/docker-compose.observability.yml up -d"
    (cd "$ROOT" && docker compose -f observability/docker-compose.observability.yml up -d) \
        >>"$LOG_DIR/docker-compose.log" 2>&1
    wait_for_port 4317 60 "Phoenix collector"
    wait_for_port 9090 30 "Prometheus"
    wait_for_port 9093 30 "Alertmanager"
    wait_for_port 3001 30 "Grafana"
fi

# --- 3. jira-backend (Spring, :8081) ------------------------------------------------------------
echo "--- jira-backend :8081 ---"
if port_open 8081; then
    say_ok "already running"
else
    say_start "./gradlew :jira-backend:bootRun  (first run compiles — can take ~1-2 min)"
    (cd "$ROOT" && nohup ./gradlew :jira-backend:bootRun \
        > "$LOG_DIR/jira-backend.log" 2>&1 & echo $! > "$PIDS_DIR/jira-backend.pid") </dev/null >/dev/null 2>&1
    wait_for_port 8081 150 "jira-backend"
fi

# --- 4. vectorization-service (Python, :8100) — retrieval over pocdb.rag ---------------------------
echo "--- vectorization-service :8100 ---"
# Config comes from vectorization-service/.env (app/config.py loads it with env_prefix="VEC_"), the
# same way the ai-service block below already works. This script used to hardcode
# VEC_EMBEDDING_PROVIDER/MODEL/DIM + VEC_PG_DSN + VEC_KAFKA_BOOTSTRAP_SERVERS inline instead, which
# was actively dangerous: a shell env var OVERRIDES the .env file, so those literals silently won
# over whatever the developer had actually configured — and they drifted. The pin here named a
# smaller embedding model long after local .env files had moved to a larger one. Starting the service through this script would then embed
# queries with a DIFFERENT model than the one that built the vectors already stored in rag.chunks, so
# query and index vectors land in unrelated spaces. That fails silently — no error, no warning, just
# uniformly bad retrieval — and it is very hard to trace back to a launcher script. Re-pinning the
# literal would only re-arm the same trap for the next model change, so the pin is gone: .env
# is the single source of truth, and the resolved model is echoed below so a mismatch is visible at
# start-up rather than three debugging hours later.
if port_open 8100; then
    say_ok "already running"
elif [ ! -f "$ROOT/vectorization-service/.env" ]; then
    say_warn "no vectorization-service/.env — copy .env.example and set your embedding provider first"
    say_warn "(.env.example already points at the LiteLLM gateway's hosted routes)"
else
    vec_model="$(grep -E '^VEC_EMBEDDING_MODEL=' "$ROOT/vectorization-service/.env" | tail -1 | cut -d= -f2-)"
    say_start "uvicorn (vectorization-service venv, config from vectorization-service/.env)"
    printf "           embedding model: %s\n" "${vec_model:-<unset — check .env>}"
    printf "           (must match the model that built the vectors in rag.chunks, or retrieval breaks silently)\n"
    (cd "$ROOT/vectorization-service" && source .venv/bin/activate && \
        nohup python -m uvicorn app.main:app --port 8100 \
        > "$LOG_DIR/vectorization-service.log" 2>&1 & echo $! > "$PIDS_DIR/vectorization-service.pid") </dev/null >/dev/null 2>&1
    wait_for_port 8100 60 "vectorization-service"
fi

# --- 5. ai-service (Python, :8200) — the Ask-AI agent, config from ai-service/.env -------------
echo "--- ai-service :8200 ---"
if port_open 8200; then
    say_ok "already running"
else
    say_start "uvicorn (ai-service venv, config from ai-service/.env)"
    (cd "$ROOT/ai-service" && source .venv/bin/activate && \
        nohup python -m uvicorn app.main:app --host 127.0.0.1 --port 8200 \
        > "$LOG_DIR/ai-service.log" 2>&1 & echo $! > "$PIDS_DIR/ai-service.pid") </dev/null >/dev/null 2>&1
    wait_for_port 8200 60 "ai-service"
fi

# --- 6. frontend (Vite dev server, also the API proxy) --------------------------------------------------------------
echo "--- frontend (vite) ---"
if port_open 5173; then
    say_ok "already running on :5173"
elif [ -z "$STUDIO_DIR" ]; then
    say_warn "sprint-orchestra-studio not found next to this repo — start it manually (npm run dev)"
else
    say_start "npm run dev (sprint-orchestra-studio)"
    (cd "$STUDIO_DIR" && nohup npm run dev \
        > "$LOG_DIR/frontend.log" 2>&1 & echo $! > "$PIDS_DIR/frontend.pid") </dev/null >/dev/null 2>&1
    wait_for_port 5173 30 "frontend"
fi

print_status
echo "Logs for anything started just now: $LOG_DIR/"
echo "Open the app: http://localhost:5173  (log in, then click 'Ask AI')"
echo "Traces: http://localhost:6006 (Phoenix)   Dashboards: http://localhost:3001 (Grafana)"
echo ""
echo "Run 'scripts/dev_up.sh status' any time to re-check without starting anything."
echo "Run 'scripts/dev_up.sh down' to stop only what this script itself started."
