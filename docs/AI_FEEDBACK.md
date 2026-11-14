# AI Feedback

The Studio sidebar's **AI Feedback** page is the review inbox for Ask AI answers. Users submit
feedback below a completed answer; owners review the original question, answer and evidence,
classify the problem and record what the correct answer should have been. Langfuse
is not required for collecting or reviewing feedback.

## How it works

1. Ask a question and select the helpful or unhelpful button. Helpful feedback is recorded in
   one click. Unhelpful feedback opens optional reason and comment fields. Submitting either signal
   shares the conversation and returned sources with space owners.
2. Open `/ai-feedback` as a space owner. The quality summary shows the aggregate helpful rate and
   rating volumes. The inbox contains only reported issues; search by question, reporter, or
   comment, and filter by review status.
3. Select an item to inspect the answer, earlier conversation, citations and structured evidence.
   Use the trace reference under **Answer information** for deeper engineering investigation.
4. Save a status, root cause and review notes. Closing an item requires a note; resolving it also
   requires a root cause. **Resolved** is the reviewer's assessment, not an automatically verified
   test result. Include the fix, version and verification performed in the review note.
5. Enter the verified expected answer and, optionally, the issue keys a correct answer must cite.
   Fix the problem, then resolve the item with a note on what changed.

Review visibility requires ownership of **all** spaces included in the answer. Owning only one
space in a multi-space answer is insufficient. Ownership and active membership are checked again
at API access time. A user who no longer has access to the answer's spaces cannot submit it.

## What a feedback item records

The original request (including history and attachment scope), the captured response and evidence,
answer time, data watermark and prompt versions, plus the review: status, root cause, notes,
expected answer and required issue keys.

Captured citations and a watermark are **not** a complete corpus snapshot, and sprint state or
relative dates may have changed since the answer. A prompt's `servingPromptVersions` records files
present when the receipt was issued; on a cache hit these may differ from the prompt that originally
generated the answer.

## API and persistence

All browser endpoints take the user from the verified JWT. No browser-supplied user ID is trusted.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ai-feedback` | Submit `{receipt,rating,reason?,comment?}` |
| GET | `/api/ai-feedback` | Owner-scoped issue list; `status`, `query`, zero-based `page` |
| GET | `/api/ai-feedback/{id}` | Question, snapshot, user feedback and review |
| PUT | `/api/ai-feedback/{id}` | Save `{status,rootCause,reviewNote,expectedAnswer,requiredIssueKeys,revision}` |

The list returns 25 issue rows per page plus aggregate `signals` counts. Ratings are `up`/`down`;
review states are `new`/`reviewing`/`resolved`/`dismissed`.
Submission is idempotent by `(answer_id, reporter_id)`; retries return the original report without
rewriting reviewed feedback. Updating a stale review
revision returns 409 instead of overwriting another reviewer's edits.

Helpful ratings are stored as aggregate quality signals and do not appear as individual inbox
items or expose a review form. They support helpful-rate monitoring and later version or experiment
comparisons. They are not treated as verified ground truth and do not train or personalize a model
by themselves. Unhelpful ratings create the actionable review cases used for investigation.

`/ask` and `/ask/stream` return optional `answerId` and `feedbackReceipt`. A domain-scoped
HMAC-SHA256 receipt binds the authenticated user to the request, answer and returned evidence.
The browser carries it until feedback submission; jira-backend verifies it before saving the
snapshot in PostgreSQL, so a browser-edited answer is rejected and the answer path needs no extra
network call. The receipt is signed with the shared internal service token, expires after
seven days, and is limited to 1 MB. Internal anonymous eval traffic and answers exceeding that
receipt limit have no feedback control.

Receipts and answer IDs never enter the shared answer cache. Each cache delivery has a fresh
answer ID and the current recipient's identity. Only submitted feedback is stored in this table;
no automated Langfuse synchronization, source modification, model training, issue creation, or
test execution is performed by the feedback page.

## Verification

```bash
cd ai-service
.venv/bin/python -m pytest tests/test_feedback_receipts.py tests/test_routes_cache_safety.py \
  tests/test_usage_ledger_routes.py tests/test_trace_tag_coverage.py -q
```

From the repository root, against local PostgreSQL:

```bash
RUN_FEEDBACK_DB_TESTS=true ./gradlew :jira-backend:test --tests '*Feedback*' --offline
```

The database tests use and remove a unique temporary schema. They do not migrate or reset the
application's data. `FEEDBACK_IT_JDBC_URL`, `FEEDBACK_IT_DB_USER`, and `FEEDBACK_IT_DB_PASSWORD`
can point them at another local test database.
