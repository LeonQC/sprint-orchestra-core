# Seed data

Scripts that load demo workspaces into a local or staging environment. Each one is idempotent and
touches only the rows it created, so it is safe to re-run.

| Workspace | Key | Script | Contents |
|-----------|-----|--------|----------|
| AtlasCart | `ATC` | `atlascart/m3_seed.py` | E-commerce product team: 83 issues, 7 sprints, comments, issue links, history and attachments (PDF, DOCX, XLSX, images) |
| PulseRoom | `PRM` | `pulseroom/m3_seed.py` | Telehealth product team: 75 issues, 9 sprints, comments and history (no attachments) |

Both write straight to jira-backend's PostgreSQL (`postgresql://poc:poc123@localhost:5432/pocdb`);
AtlasCart also uploads its attachments to MinIO. Because they bypass jira-backend's API, no Kafka events are published; run a vectorization
backfill afterwards (below) so the workspace is searchable in Ask AI.

## AtlasCart

```bash
docker compose up -d
python3 seed-data/atlascart/m3_seed.py validate --attachment-ratio 30 --link-ratio 20
python3 seed-data/atlascart/m3_seed.py inject --attachment-ratio 30 --link-ratio 20 --yes
python3 seed-data/atlascart/m3_seed.py status
python3 seed-data/atlascart/m3_seed.py cleanup --yes
```

`validate` checks the generated data without touching the database. `generate` writes the SQL and
attachment files to `atlascart/generated/` (ignored by git) for review. `m1_seed.py` loads only the
first month.

Two additive fixture sets sit on top of the main dataset. Each marks what it creates (a label or a
sprint-goal tag) and deletes its own previous rows before re-creating them:

- `seed_issue_ai_demo_fixtures.py` — issues for the "Improve description" and "Link similar work
  items" actions.
- `seed_recovery_test_fixtures.py` — at-risk sprints for Sprint Recovery.

`refresh_demo_data.sh` runs the whole sequence: reseed with the active sprint anchored to the current
week, backfill vectorization-service, add the issue-level AI fixtures, flush the answer cache.

## PulseRoom

```bash
python3 seed-data/pulseroom/m3_seed.py    # replaces an existing PRM workspace's issues and sprints
```

## Making a workspace searchable

```bash
cd vectorization-service && source .venv/bin/activate
python -m scripts.backfill_jira_backend --space-ids <space id> --include-attachments
```

Space ids are assigned by the database on insert:

```bash
docker exec orchestra-postgres psql -U poc -d pocdb -At -c "SELECT id FROM spaces WHERE space_key='ATC'"
```
