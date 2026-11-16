"""Tests for reading a refine instruction's intent (app/planning/refine_intent.py).

A refine instruction is one of two different things and they deserve different treatment: one asks for
work that is not in the plan yet, the other rearranges work that already is. Only the first has
anything to find a precedent for. Getting this wrong is cheap in one direction and expensive in the
other, which is why the failure path is asymmetric.
"""
from app.planning.refine_intent import RefineIntent, read_refine_intent
from tests.test_planning import FakeInstructorClient


def _intent(*work, reasoning="because"):
    return RefineIntent(new_work=list(work), reasoning=reasoning)


async def test_an_instruction_that_adds_work_reports_each_piece_separately():
    """Each entry becomes its own search query, so "one for backend and one for frontend" has to come
    back as two — a single blended query for both would retrieve neither well."""
    client = FakeInstructorClient(
        _intent("a task covering backend testing", "a task covering frontend testing")
    )

    result = await read_refine_intent(
        client, "fake-model", "we need one task for backend testing and one for frontend testing",
    )

    assert result.adds_work is True
    assert result.intent.new_work == [
        "a task covering backend testing", "a task covering frontend testing",
    ]


async def test_an_instruction_that_only_edits_reports_nothing_to_look_up():
    """The common case, and not a failure: re-estimating an issue specifies no work that isn't
    already in the plan, so there is nothing to find a precedent FOR."""
    client = FakeInstructorClient(_intent(reasoning="Re-estimating an issue that already exists."))

    result = await read_refine_intent(client, "fake-model", "make issue 3 five points")

    assert result.adds_work is False
    assert result.intent.new_work == []
    assert result.degraded is False


async def test_a_blank_instruction_costs_no_call():
    client = FakeInstructorClient(RuntimeError("must not be called"))

    result = await read_refine_intent(client, "fake-model", "   ")

    assert result.adds_work is False
    assert client.completions.calls == []


async def test_a_failed_read_falls_back_to_consulting_history():
    """Asymmetric on purpose. Reading history when it wasn't needed costs time; skipping it when it
    was costs an ungrounded issue, so an unreadable instruction is treated as if it adds work."""
    client = FakeInstructorClient(RuntimeError("model exploded"))

    result = await read_refine_intent(client, "fake-model", "add a QA task")

    assert result.degraded is True
    assert result.adds_work is False        # nothing was extracted…
    assert "model exploded" in result.error
    # …but the caller keys on `adds_work or degraded`, so history is still consulted.


async def test_blank_entries_are_dropped_rather_than_searched_for():
    client = FakeInstructorClient(_intent("a task covering backend testing", "   ", ""))

    result = await read_refine_intent(client, "fake-model", "add backend testing")

    assert result.intent.new_work == ["a task covering backend testing"]


async def test_the_instruction_is_all_the_model_is_given():
    """Deliberately not the plan: this is a cheap classification, and feeding it the whole plan would
    make it as expensive as the call it exists to avoid."""
    client = FakeInstructorClient(_intent())

    await read_refine_intent(client, "fake-model", "drop the inventory one for MVP")

    (call,) = client.completions.calls
    assert call["messages"][1]["content"] == "drop the inventory one for MVP"
    assert call["max_tokens"] <= 1000
