"""Tests for AI description review (app/description_review/) — same fake-instructor-client,
no-network approach as tests/test_comment_digest.py.

The two properties worth defending here are the ones that make this feature safe to ship:
(1) the model's findings are APPLIED by us, not trusted — a quote that isn't really in the
description is dropped instead of guessed at; and (2) the description's HTML and its inline
attachment chips survive, because replacement never touches anything inside a tag.
"""
import pytest

from app.description_review.prompts import (
    render_description_review_system_prompt,
    strip_description_html,
)
from app.description_review.schemas import DescriptionReview, Finding
from app.description_review.service import review_description


class _FakeUsage:
    def __init__(self, prompt_tokens=300, completion_tokens=80):
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


_DESCRIPTION = (
    "<h3>Context</h3><p>During the private beta, a slow checkout allowed two Place order clicks. "
    "Two order records were created and stock was reduced twice.</p>"
)


def _finding(quote, suggestion, kind="unclear", why="reads ambiguously"):
    return Finding(kind=kind, quote=quote, suggestion=suggestion, why=why)


async def test_a_clean_description_comes_back_untouched():
    """The common, expected path — and the one that makes the feature trustworthy at all."""
    client = FakeInstructorClient(DescriptionReview(verdict="fine_as_is", findings=[]))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert result.review.verdict == "fine_as_is"
    assert result.improved_description is None
    assert result.applied_findings == []
    assert result.degraded is False


async def test_a_verified_finding_is_applied_to_the_original_text():
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes",
        findings=[_finding("stock was reduced twice", "stock was reduced two times")],
    ))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert result.review.verdict == "needs_changes"
    assert result.improved_description is not None
    assert "stock was reduced two times" in result.improved_description
    assert len(result.applied_findings) == 1
    assert result.input_tokens == 300


async def test_a_finding_quoting_text_that_isnt_there_is_dropped():
    # The grounding half: a model can quote something it paraphrased or imagined. Applying that
    # would either silently do nothing or, worse, hit the wrong span — so it's dropped instead.
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes",
        findings=[_finding("the payment gateway timed out", "the gateway timed out")],
    ))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert result.applied_findings == []
    assert len(result.dropped_findings) == 1
    # Nothing applied means there is no improvement to offer, so the verdict is normalized down
    # rather than left claiming a change the UI cannot show.
    assert result.review.verdict == "fine_as_is"
    assert result.improved_description is None


async def test_replacement_never_edits_inside_an_html_tag():
    # A quote that also occurs inside markup (here, "order" appears in a class attribute) must only
    # ever hit the visible prose. Rewriting an attribute would corrupt the description's structure.
    description = (
        '<p class="order-note">Two order records were created during the private beta checkout, '
        "and the stock count was reduced twice.</p>"
    )
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes", findings=[_finding("order records", "order rows")],
    ))

    result = await review_description(client, "fake-model", description)

    assert 'class="order-note"' in result.improved_description
    assert "Two order rows were created" in result.improved_description


async def test_an_inline_attachment_chip_survives_a_replacement():
    # Descriptions carry attachment chips as markup (see the frontend's commentHtml.ts). Because we
    # only ever replace inside text segments, the chip is untouched by construction.
    chip = '<span data-attachment-name="trace.log">[file] trace.log</span>'
    description = f"<p>The checkout was slow during the private beta and stock got reduced twice. {chip}</p>"
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes", findings=[_finding("was slow", "responded slowly")],
    ))

    result = await review_description(client, "fake-model", description)

    assert chip in result.improved_description
    assert "responded slowly" in result.improved_description


async def test_a_no_op_finding_is_not_reported_as_a_change():
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes",
        findings=[_finding("stock was reduced twice", "stock was reduced twice")],
    ))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert result.applied_findings == []
    assert result.review.verdict == "fine_as_is"


async def test_overlapping_findings_do_not_corrupt_the_result():
    # The second finding quotes text the first one already rewrote away. It must simply fail to
    # match and be dropped, leaving the first edit intact.
    client = FakeInstructorClient(DescriptionReview(
        verdict="needs_changes",
        findings=[
            _finding("stock was reduced twice", "stock dropped twice"),
            _finding("stock was reduced", "inventory was reduced"),
        ],
    ))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert "stock dropped twice" in result.improved_description
    assert len(result.applied_findings) == 1
    assert len(result.dropped_findings) == 1


async def test_a_description_with_too_little_prose_skips_the_model_call():
    client = FakeInstructorClient(RuntimeError("should never be called"))

    result = await review_description(client, "fake-model", "<p>Fix it.</p>")

    assert result.skipped is True
    assert result.improved_description is None
    assert client.completions.calls == []


async def test_model_failure_degrades_without_touching_the_description():
    client = FakeInstructorClient(RuntimeError("model unreachable"))

    result = await review_description(client, "fake-model", _DESCRIPTION)

    assert result.degraded is True
    assert result.review.verdict == "fine_as_is"
    assert result.improved_description is None
    assert result.error == "model unreachable"
    assert result.input_tokens == 0


async def test_the_title_is_passed_as_context_but_marked_unreviewable():
    client = FakeInstructorClient(DescriptionReview(verdict="fine_as_is", findings=[]))

    await review_description(client, "fake-model", _DESCRIPTION, title="Checkout double-charge")

    sent = client.completions.calls[0]["messages"][1]["content"]
    assert "Checkout double-charge" in sent
    assert "do not review or quote this" in sent


def test_strip_keeps_block_structure_as_newlines():
    # Flattening blocks into one line invites the model to quote across visually separate lines,
    # producing a quote that matches no single text node and gets discarded.
    text = strip_description_html("<h3>Context</h3><p>First line.</p><p>Second line.</p>")

    assert "Context\nFirst line.\nSecond line." == text


def test_strip_removes_tags_without_splitting_words():
    assert strip_description_html("<p>Ship it <strong>now</strong>.</p>") == "Ship it now."


def test_strip_decodes_entities_so_the_model_sees_what_a_reader_sees():
    # Found in an abstention eval: an undecoded &#x27; reached the model, which
    # reported it as a typo — a false positive created by our own preprocessing, since a browser
    # renders that as an apostrophe and nobody ever sees the entity.
    assert strip_description_html("<p>the file&#x27;s name &amp; size</p>") == "the file's name & size"


@pytest.mark.parametrize("phrase", ["inventing", "fine_as_is", "acceptance criteria"])
def test_system_prompt_states_the_scope_limits(phrase):
    """Pins the three instructions that keep this a copy-edit and not a rewrite: don't invent, prefer
    abstaining, and specifically don't add missing acceptance criteria."""
    assert phrase in render_description_review_system_prompt()
