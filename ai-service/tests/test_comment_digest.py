"""Tests for AI comment-thread digests (app/comment_digest/) — same fake-instructor-client,
no-network approach as tests/test_sprint_pace.py.

Covers: (1) too few comments skips the model call entirely; (2) a successful call returns the
digest with its comment_ids grounded against the real thread; (3) an open-question/action-item
whose ids don't resolve is dropped rather than shown unattributed, while `outcome` (a required
field, not a droppable list item) keeps its text and just strips the bad ids; (4) a model/provider
failure degrades to an empty digest, never a raised exception; (5) more comments than the cap get
truncated rather than silently dropped from the caller's view.
"""
from app.comment_digest.prompts import render_comment_digest_system_prompt, render_comment_thread_message
from app.comment_digest.schemas import ActionItemOut, AttributedPoint, CommentDigest, CommentIn, Outcome
from app.comment_digest.service import summarize_comments


class _FakeUsage:
    def __init__(self, prompt_tokens=200, completion_tokens=60):
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


def _comment(id, author="Ava Patel", created_at="2026-08-11T10:00:00Z", content="some text"):
    return CommentIn(id=id, author_name=author, created_at=created_at, content=content)


async def test_fewer_than_two_comments_skips_the_model_call():
    client = FakeInstructorClient(RuntimeError("should never be called"))

    result = await summarize_comments(client, "fake-model", [_comment(1)])

    assert result.skipped is True
    assert result.degraded is False
    assert result.digest.context == ""
    assert result.digest.discussion == ""
    assert result.digest.outcome.text == ""
    assert client.completions.calls == []


async def test_zero_comments_also_skips():
    client = FakeInstructorClient(RuntimeError("should never be called"))

    result = await summarize_comments(client, "fake-model", [])

    assert result.skipped is True
    assert client.completions.calls == []


async def test_successful_digest_is_returned_with_grounded_ids():
    expected = CommentDigest(
        context="Ava raised a question about rollout risk before a planned release.",
        discussion="The team debated shipping immediately vs. behind a flag; the flag won out once "
                    "Sam pointed out partial-rollout risk to existing carts.",
        outcome=Outcome(text="The team decided to ship behind a feature flag.", comment_ids=[2, 3]),
        open_questions=[AttributedPoint(text="Who owns the flag cleanup?", comment_ids=[3])],
        action_items=[ActionItemOut(text="Add the flag.", owner="Ava Patel", comment_ids=[2])],
    )
    client = FakeInstructorClient(expected)
    comments = [_comment(1), _comment(2), _comment(3)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.skipped is False
    assert result.degraded is False
    assert result.digest.context == expected.context
    assert result.digest.discussion == expected.discussion
    assert result.digest.outcome.text == expected.outcome.text
    assert result.digest.outcome.comment_ids == [2, 3]
    assert result.digest.action_items[0].owner == "Ava Patel"
    assert result.input_tokens == 200
    assert result.output_tokens == 60
    assert client.completions.calls[0]["response_model"] is CommentDigest


async def test_a_point_with_no_real_comment_id_is_dropped_not_shown_unattributed():
    # The model claimed comment id 99, which was never in the thread it was given — a fabricated
    # citation is worse than no citation, so the whole point is dropped rather than shown trusting it.
    expected = CommentDigest(
        context="ok", discussion="ok", outcome=Outcome(text="ok", comment_ids=[1]),
        open_questions=[AttributedPoint(text="Invented question.", comment_ids=[99])],
        action_items=[],
    )
    client = FakeInstructorClient(expected)
    comments = [_comment(1), _comment(2)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.digest.open_questions == []


async def test_a_point_with_a_mix_of_real_and_fake_ids_keeps_only_the_real_ones():
    expected = CommentDigest(
        context="ok", discussion="ok", outcome=Outcome(text="ok", comment_ids=[1]),
        open_questions=[AttributedPoint(text="Partially grounded.", comment_ids=[1, 99])],
        action_items=[],
    )
    client = FakeInstructorClient(expected)
    comments = [_comment(1), _comment(2)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.digest.open_questions[0].comment_ids == [1]


async def test_outcome_with_no_real_comment_id_keeps_its_text_but_drops_the_id():
    # Unlike a list item, `outcome` is a required top-level field — there's no "drop it" option, so
    # a fabricated id just gets stripped, leaving the (still possibly useful) text in place.
    expected = CommentDigest(
        context="ok", discussion="ok",
        outcome=Outcome(text="The team decided to ship behind a flag.", comment_ids=[99]),
        open_questions=[], action_items=[],
    )
    client = FakeInstructorClient(expected)
    comments = [_comment(1), _comment(2)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.digest.outcome.text == "The team decided to ship behind a flag."
    assert result.digest.outcome.comment_ids == []


async def test_model_failure_degrades_to_an_empty_digest():
    client = FakeInstructorClient(RuntimeError("model unreachable"))
    comments = [_comment(1), _comment(2)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.degraded is True
    assert result.skipped is False
    assert result.digest.context == ""
    assert result.digest.discussion == ""
    assert result.digest.outcome.text == ""
    assert result.digest.outcome.comment_ids == []
    assert result.error == "model unreachable"
    assert result.input_tokens == 0


async def test_more_comments_than_the_cap_are_truncated_to_the_most_recent():
    from app.comment_digest.service import _MAX_COMMENTS

    expected = CommentDigest(
        context="ok", discussion="ok", outcome=Outcome(text="ok", comment_ids=[]),
        open_questions=[], action_items=[],
    )
    client = FakeInstructorClient(expected)
    comments = [_comment(i) for i in range(1, _MAX_COMMENTS + 11)]

    result = await summarize_comments(client, "fake-model", comments)

    assert result.truncated_count == 10
    sent_message = client.completions.calls[0]["messages"][1]["content"]
    assert f"comment id={_MAX_COMMENTS + 10}" in sent_message  # most recent kept
    assert "comment id=1 " not in sent_message  # oldest dropped


def test_render_thread_message_strips_html_and_includes_ids():
    comments = [_comment(1, content="<p>Ship it <strong>now</strong></p>")]

    message = render_comment_thread_message(comments, truncated_count=0)

    assert "comment id=1" in message
    assert "Ship it" in message
    assert "<p>" not in message
    assert "<strong>" not in message


def test_render_thread_message_notes_truncation():
    message = render_comment_thread_message([_comment(5)], truncated_count=4)

    assert "4 earlier comment" in message


def test_system_prompt_forbids_inline_citation_markers_in_the_prose_fields():
    # Found live: without this instruction, the model put bracketed comment ids like "[5004629]"
    # directly inside context/discussion/outcome, mimicking the decisions/questions citation style —
    # but those numbers aren't clickable there, just noise. Citations belong only in comment_ids.
    prompt = render_comment_digest_system_prompt()

    assert "bracketed reference" in prompt
    assert "comment_ids" in prompt


def test_system_prompt_prioritizes_plain_spoken_wording_over_a_word_count():
    # Round-tripped on live feedback: an earlier version pushed brevity as the top priority ("a
    # digest whose length approaches the thread's own has failed"), which produced technically
    # shorter but confusing output — details got cut that a reader actually needed. The current
    # framing puts sounding natural first and treats brevity as "short when short still makes
    # sense," not a target to hit regardless of clarity.
    prompt = render_comment_digest_system_prompt()

    assert "SOUND LIKE A PERSON, NOT A REPORT" in prompt
    assert "clarity always wins over length" in prompt
    assert "done" in prompt and "complete" in prompt  # the plain-word substitution examples


def test_the_prose_fields_have_a_generous_but_bounded_character_ceiling():
    # Not a tight budget — see the module docstring's note on why a hard low ceiling backfired
    # (it forced cuts to whatever the model was mid-sentence on). This just pins that a ceiling
    # still exists at all, as a backstop against a genuinely runaway response.
    schema = CommentDigest.model_json_schema()
    defs = schema["$defs"]

    assert 200 <= schema["properties"]["context"]["maxLength"] <= 400
    assert 300 <= schema["properties"]["discussion"]["maxLength"] <= 600
    assert 200 <= defs["Outcome"]["properties"]["text"]["maxLength"] <= 400
    assert 150 <= defs["AttributedPoint"]["properties"]["text"]["maxLength"] <= 300
