#!/usr/bin/env bash
# Re-anchors the AtlasCart M3 fixture's "current" sprint to real calendar time, end to end: jira-backend
# reseed -> vectorization-service backfill (issues/comments/history/attachments) -> issue-level AI
# demo fixtures -> Redis cache flush.
#
# The AI demo fixtures (step 5) are additive and separate from the golden, exactly-validated M3
# dataset in m3_data.py on purpose: m3_seed.py's validate() hardcodes exact counts and statistical
# properties for the 35-issue M3 extension (AC distribution, subtask count, verifier balance, ...),
# so it is not a place to casually append a few more issues — every assertion in validate() would
# need updating in lockstep. seed_issue_ai_demo_fixtures.py is the same additive-fixture pattern
# seed_recovery_test_fixtures.py already uses for the sprint-recovery workflow: its own script,
# its own idempotency marker (the "ai-demo" label), run as a step here so one command produces
# everything a demo needs instead of two.
#
# Run this whenever the demo has been sitting long enough that the active sprint feels stale (its
# end_date has passed, or is about to). A plain reseed alone isn't enough: space_id/sprint_ids change,
# and vectorization-service needs its own explicit backfill since the reseed writes raw SQL and never
# publishes the Kafka events that would normally keep it in sync.
#
# Why AS_OF = today + 7 days, not just "today" (the generator's own default): seed-data/atlascart/m3_data.py pins
# the active sprint's window to [that date's ISO week Monday - 7, + 4] relative to whatever
# ATLASCART_AS_OF resolves to — meaning "today" always lands in the window's FINAL week, anywhere from
# day 0 (if today is a Monday) to day 4 = the very LAST day (if today is a Friday). Reseeding on a
# Friday with no override leaves you 0 days of runway before the sprint's own end_date. Setting AS_OF
# to (today + 7) shifts the *window*, not the meaning of "today": the resulting sprint always starts
# on THIS week's Monday and runs 11 days, so "today" (whichever real weekday you run this on) always
# lands within the window's first 5 days — guaranteeing at least ~7 days of runway before end_date,
# every time, regardless of which day of the week you happen to run this script.
#
# Usage: bash seed-data/atlascart/refresh_demo_data.sh
# Requires: docker compose services already up (orchestra-postgres (public + rag), orchestra-minio, orchestra-kafka,
# orchestra-litellm). Embedding settings come from vectorization-service/.env, the same file the running
# service reads, so the backfill can't embed with a different model than the one serving queries.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root, regardless of where this is invoked from

AS_OF=$(python3 -c "from datetime import date, timedelta; print((date.today() + timedelta(days=7)).isoformat())")
echo "=== Re-anchoring AtlasCart M3 to ATLASCART_AS_OF=${AS_OF} (today + 7 days) ==="

export ATLASCART_AS_OF="$AS_OF"

echo -e "\n--- 1/6: validate (no writes) ---"
python3 seed-data/atlascart/m3_seed.py validate --attachment-ratio 100 --link-ratio 20

echo -e "\n--- 2/6: check-db (transactional dry-run, rolled back) ---"
python3 seed-data/atlascart/m3_seed.py check-db --attachment-ratio 100 --link-ratio 20

echo -e "\n--- 3/6: inject (real write — safely replaces the existing ATC fixture) ---"
python3 seed-data/atlascart/m3_seed.py inject --attachment-ratio 100 --link-ratio 20 --yes

echo -e "\n--- 4/6: vectorization-service backfill (issues + comments + history + attachments) ---"
SPACE_ID=$(docker exec orchestra-postgres psql -U poc -d pocdb -At -c "SELECT id FROM spaces WHERE space_key='ATC'")
echo "new space_id: ${SPACE_ID}"
(
  cd vectorization-service
  source .venv/bin/activate
  JIRA_BACKEND_PG_DSN="postgresql://poc:poc123@localhost:5432/pocdb" \
  python -m scripts.backfill_jira_backend --space-ids "${SPACE_ID}" --truncate-first --include-attachments
)

echo -e "\n--- 5/6: seed issue-level AI demo fixtures (Improve description + Link similar work items) ---"
(
  cd ai-service
  source .venv/bin/activate
  python3 ../seed-data/atlascart/seed_issue_ai_demo_fixtures.py --space-id "${SPACE_ID}"
)

echo -e "\n--- 6/6: clear cached AI answers in Redis (stale pre-reseed answers look exactly like a real"
echo "    regression otherwise) ---"
# Cached answers only: the same Redis holds login sessions, per-user AI quotas and LiteLLM spend.
docker exec orchestra-redis sh -c 'for p in "aicache:*" "litellm.response_cache:*"; do
  redis-cli --scan --pattern "$p" | grep -E "^aicache:|:[0-9a-f]{64}\$" | xargs -r redis-cli DEL >/dev/null; done'

echo -e "\n=== Done. New identifiers (update eval script defaults if they hardcode the old ones): ==="
docker exec orchestra-postgres psql -U poc -d pocdb -c \
  "SELECT s.id AS space_id, sp.id AS sprint_id, sp.name, sp.start_date, sp.end_date, sp.status
   FROM spaces s JOIN sprints sp ON sp.space_id = s.id
   WHERE s.space_key = 'ATC' ORDER BY sp.id;"
