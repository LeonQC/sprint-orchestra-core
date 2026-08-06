"""Tests for AI-assisted task creation (app/drafting/) — a fake instructor-shaped client, no network.

Covers the two things that actually matter here: (1) a successful call returns the validated
TaskDraft untouched, and (2) any failure — model down, validation retries exhausted, whatever —
degrades to a safe, always-valid fallback instead of raising, since this is a user-facing "help me
create a task" call that must never block someone from creating *something*.
"""
import pytest

from app.drafting.prompts import render_task_draft_prompt
from app.drafting.schemas import TaskDraft
from app.drafting.service import draft_task


class _FakeUsage:
    def __init__(self, prompt_tokens=120, completion_tokens=45):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletion:
    def __init__(self, usage=None):
        self.usage = usage or _FakeUsage()


class _FakeCompletions:
    def __init__(self, result, completion=None):
        self._result = result
        self._completion = completion or _FakeCompletion()
        self.calls = []

    async def create_with_completion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result, self._completion


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeInstructorClient:
    def __init__(self, result, completion=None):
        self.completions = _FakeCompletions(result, completion)
        self.chat = _FakeChat(self.completions)


async def test_draft_task_returns_validated_draft_on_success():
    expected = TaskDraft(
        title="Add rate limiting to login endpoint", issue_type="bug",
        labels=["security"], estimate_story_points=3,
        estimate_rationale="One endpoint, well-understood middleware, no schema changes.",
        dependencies=["gateway auth work"],
    )
    client = FakeInstructorClient(expected)

    result = await draft_task(client, "fake-model", "someone was hammering the login endpoint")

    assert result.draft == expected
    assert result.degraded is False
    assert result.error is None
    assert client.completions.calls[0]["response_model"] is TaskDraft


async def test_draft_task_reports_the_tokens_and_cost_it_spent():
    """This feature used to call `create` (no completion object, so no usage at all) while every
    other structured-output feature reported tokens — making it the one paid call the service could
    not account for when reconciling an invoice."""
    client = FakeInstructorClient(TaskDraft(title="x"), _FakeCompletion(_FakeUsage(1_000_000, 1_000_000)))

    result = await draft_task(client, "gpt-5.6-luna", "desc")

    assert result.input_tokens == 1_000_000
    assert result.output_tokens == 1_000_000
    # gpt-5.6-luna: $0.20/1M input + $1.20/1M output.
    assert result.estimated_cost_usd == pytest.approx(1.40)


async def test_draft_task_reports_no_spend_when_the_call_never_reached_the_model():
    client = FakeInstructorClient(RuntimeError("model unreachable"))

    result = await draft_task(client, "gpt-5.6-luna", "desc")

    assert result.estimated_cost_usd == 0.0
    assert result.input_tokens == 0


async def test_draft_task_degrades_safely_when_the_model_call_fails():
    client = FakeInstructorClient(RuntimeError("model unreachable"))

    result = await draft_task(client, "fake-model", "some task\nmore detail nobody will read")

    assert result.degraded is True
    assert result.draft.title == "some task"  # first line of the raw description, not invented
    assert result.draft.labels == []
    assert result.draft.estimate_story_points is None
    assert result.draft.estimate_rationale is None
    assert result.error == "model unreachable"


async def test_draft_task_fallback_handles_blank_description():
    client = FakeInstructorClient(RuntimeError("boom"))

    result = await draft_task(client, "fake-model", "   ")

    assert result.draft.title == "Untitled task"


async def test_draft_task_passes_existing_labels_into_the_rendered_system_prompt():
    client = FakeInstructorClient(TaskDraft(title="x"))

    await draft_task(client, "fake-model", "desc", existing_labels=["urgent-fix"])

    system_message = client.completions.calls[0]["messages"][0]["content"]
    assert "urgent-fix" in system_message


def test_rationale_is_dropped_when_the_model_gives_one_without_an_estimate():
    """The UI renders the rationale as an explanation of the number in the points field, so a
    rationale with no number would be an explanation of nothing — normalized away rather than
    raised, which would burn one of instructor's re-prompt retries on a harmless mismatch."""
    draft = TaskDraft(title="x", estimate_rationale="Feels smallish, hard to say.")

    assert draft.estimate_rationale is None


def test_rationale_survives_when_it_accompanies_an_estimate():
    draft = TaskDraft(title="x", estimate_story_points=5, estimate_rationale="  Two services touched.  ")

    assert draft.estimate_rationale == "Two services touched."


def test_the_draft_schema_inherits_the_every_field_required_rule():
    """The rule itself is tested in test_structured_output.py — this only pins that TaskDraft is
    still on it, since dropping back to a plain BaseModel would silently restore the bug where a
    schema-constrained model answered with a title and nothing else."""
    assert set(TaskDraft.model_json_schema()["required"]) == set(TaskDraft.model_json_schema()["properties"])


def test_render_prompt_asks_for_a_rationale_alongside_the_estimate():
    assert "rationale" in render_task_draft_prompt()


def test_render_without_existing_labels_omits_the_reuse_instruction():
    prompt = render_task_draft_prompt()
    assert "Prefer reusing" not in prompt


def test_render_with_existing_labels_includes_them_joined():
    prompt = render_task_draft_prompt(["backend", "urgent"])
    assert "backend, urgent" in prompt
