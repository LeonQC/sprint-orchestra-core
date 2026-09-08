"""Per-request billing records (app/llm/usage_ledger.py).

What is worth testing here is not "does it set attributes" but the properties that decide whether the
data can be *used* for the thing it exists for — reconciling a provider invoice:

  * itemized components must add up to the headline total (a total its own line items contradict is
    worse than no data);
  * a gateway route alias must record the model that actually bills, since an invoice itemizes the
    underlying model and never this project's alias;
  * a conversation's turns must land in one session, or per-chat cost cannot be computed at all;
  * a partially-priced request must not report itself as fully priced.
"""
import json

import pytest

from app.llm.pricing import estimate_cost_usd, itemize_cost
from app.llm.types import Usage
from app.llm.usage_ledger import (
    FEATURE_SUMMARIZE_COMMENTS,
    USAGE_BILLED_MODELS,
    USAGE_COST_USD,
    USAGE_PRICING_STATUS,
    USAGE_TOTAL_TOKENS,
    Caller,
    UsageLedger,
    caller_from_headers,
    charge_for,
)


class _RecordingSpan:
    """Captures `set_attribute` calls the way an exporter would see them."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _ledger(*charges, feature="ask_ai", detail=None):
    return UsageLedger(
        feature=feature,
        caller=Caller(user_id="42", username="michael", session_id="chat-abc"),
        request_id="req-1",
        charges=list(charges),
        detail=detail or {},
    )


# --- the arithmetic a dispute rests on -------------------------------------------------------


def test_itemized_components_sum_to_the_reported_total():
    """The invariant the whole feature depends on: `estimate_cost_usd` is *defined* as this sum, so
    a breakdown can never be shown next to a total it does not explain."""
    components = itemize_cost("gpt-5.6-luna", 900_000, 300_000, 40_000, 120_000)

    assert sum(c.cost_usd for c in components) == estimate_cost_usd(
        "gpt-5.6-luna", 900_000, 300_000, 40_000, 120_000
    )


def test_each_component_carries_the_rate_it_was_priced_at():
    """A total is unarguable; tokens x published rate is what an invoice is checked against."""
    by_bucket = {c.bucket: c for c in itemize_cost("gpt-5.6-luna", 1_000_000, 1_000_000)}

    assert by_bucket["input"].rate_usd_per_million == 0.20
    assert by_bucket["input"].cost_usd == pytest.approx(0.20)
    assert by_bucket["output"].rate_usd_per_million == 1.20
    assert by_bucket["output"].cost_usd == pytest.approx(1.20)


def test_output_and_input_are_priced_separately_rather_than_at_one_blended_rate():
    """gpt-5.6-luna bills output at 6x input. A ledger that averaged them would misattribute the
    cost of a long answer to the size of the question that provoked it."""
    input_heavy = estimate_cost_usd("gpt-5.6-luna", 1_000_000, 0)
    output_heavy = estimate_cost_usd("gpt-5.6-luna", 0, 1_000_000)

    assert output_heavy == pytest.approx(input_heavy * 6)


def test_buckets_with_no_tokens_are_left_out_rather_than_reported_as_zero_rows():
    buckets = {c.bucket for c in itemize_cost("gpt-5.6-luna", 100, 50)}

    assert buckets == {"input", "output"}


def test_a_cache_read_is_priced_at_a_tenth_of_fresh_input():
    """The single largest reason a real bill comes in under a naive token-count estimate."""
    fresh = estimate_cost_usd("gpt-5.6-luna", 1_000_000, 0)
    cached = estimate_cost_usd("gpt-5.6-luna", 0, 0, cache_read_input_tokens=1_000_000)

    assert cached == pytest.approx(fresh * 0.1)


# --- attribution -----------------------------------------------------------------------------


def test_a_gateway_alias_records_the_model_that_actually_bills():
    """`openai-production` is this project's route name. An OpenAI invoice says `gpt-5.6-luna`;
    without both recorded, the two cannot be lined up."""
    charge = charge_for(
        "answer_agent", "litellm gateway", "openai-production", "hosted",
        Usage(input_tokens=1_000_000, output_tokens=0),
    )

    assert charge.model == "openai-production"
    assert charge.billed_model == "gpt-5.6-luna"
    assert charge.cost_usd == pytest.approx(0.20)


def test_a_local_model_is_free_and_says_so_rather_than_being_priced_by_name():
    """A self-hosted model has no metered bill even if its name collides with a priced one."""
    charge = charge_for(
        "answer_agent", "Self-hosted", "gpt-5.6-luna", "local",
        Usage(input_tokens=1_000_000, output_tokens=1_000_000),
    )

    assert charge.cost_usd == 0.0
    assert charge.pricing_status == "free_local"
    assert charge.total_tokens == 2_000_000  # still counted: token volume is a capacity signal


def test_an_unpriceable_hosted_model_is_flagged_rather_than_silently_costing_nothing():
    """The one status that makes a total an undercount instead of an estimate — a hosted call this
    project bills for but cannot price, which is what a missing price-table row looks like."""
    charge = charge_for(
        "answer_agent", "OpenAI", "gpt-6-unreleased", "hosted", Usage(input_tokens=5000),
    )

    assert charge.pricing_status == "unavailable"
    assert charge.cost_usd == 0.0


def test_a_request_is_only_as_priced_as_its_least_priced_model():
    """A router priced and an answer model not must not report as fully "estimated"."""
    ledger = _ledger(
        charge_for("router", "Self-hosted", "self-hosted-router", "local", Usage(input_tokens=10)),
        charge_for("answer_agent", "OpenAI", "gpt-6-unreleased", "hosted", Usage(input_tokens=10)),
    )

    assert ledger.pricing_status == "unavailable"


def test_the_ledger_total_is_the_sum_of_its_models():
    ledger = _ledger(
        charge_for("router", "litellm gateway", "gpt-5.6-luna", "hosted", Usage(input_tokens=1_000_000)),
        charge_for("answer_agent", "litellm gateway", "gpt-5.6-luna", "hosted", Usage(output_tokens=1_000_000)),
    )

    assert ledger.cost_usd == pytest.approx(0.20 + 1.20)
    assert ledger.total_tokens == 2_000_000


# --- conversation grouping -------------------------------------------------------------------


def test_follow_up_turns_of_one_chat_share_a_session_without_the_caller_sending_an_id():
    """The frontend holds conversation state and sends no session id, so the opening question is the
    only thing every turn of one chat has in common. Without this, per-conversation cost — the number
    a user actually recognizes — cannot be computed at all."""
    headers = {"x-user-id": "42"}

    turn_two = caller_from_headers(headers, history_key="how many issues are blocked?")
    turn_five = caller_from_headers(headers, history_key="how many issues are blocked?")

    assert turn_two.session_id == turn_five.session_id


def test_a_different_conversation_gets_a_different_session():
    headers = {"x-user-id": "42"}

    assert (
        caller_from_headers(headers, history_key="question A").session_id
        != caller_from_headers(headers, history_key="question B").session_id
    )


def test_two_users_asking_the_same_opening_question_are_not_merged_into_one_chat():
    """The hash is over (user, question) — one user's spend must never be attributed to another's."""
    same_question = "what is the sprint status?"

    assert (
        caller_from_headers({"x-user-id": "42"}, same_question).session_id
        != caller_from_headers({"x-user-id": "99"}, same_question).session_id
    )


def test_the_session_id_does_not_leak_the_question_text():
    caller = caller_from_headers({"x-user-id": "42"}, "our acquisition of NewCo closes friday")

    assert "acquisition" not in caller.session_id and "NewCo" not in caller.session_id


def test_an_explicit_session_header_wins_over_the_derived_one():
    caller = caller_from_headers({"x-user-id": "42", "x-session-id": "ui-session-7"}, "anything")

    assert caller.session_id == "ui-session-7"


def test_a_first_question_starts_a_new_session_each_time():
    """No history means nothing to group with — two people opening a fresh chat with the same
    question are two conversations, not one."""
    first = caller_from_headers({"x-user-id": "42"}, history_key="")
    second = caller_from_headers({"x-user-id": "42"}, history_key="")

    assert first.session_id != second.session_id


def test_an_internal_caller_has_no_user_rather_than_a_placeholder_one():
    """Eval and demo scripts call with the internal token and no user. Charging them to a fabricated
    user would put this project's own traffic on a real person's bill."""
    caller = caller_from_headers({})

    assert caller.user_id == "" and caller.username == ""


# --- what lands on the span ------------------------------------------------------------------


def test_the_span_carries_both_a_queryable_total_and_the_itemized_record():
    span = _RecordingSpan()
    _ledger(
        charge_for("answer_agent", "litellm gateway", "openai-production", "hosted",
                   Usage(input_tokens=1_000_000, output_tokens=1_000_000)),
        detail={"question": "why is ATC-77 blocked?"},
    ).attach(span)

    assert span.attributes[USAGE_COST_USD] == pytest.approx(1.40)
    assert span.attributes[USAGE_TOTAL_TOKENS] == 2_000_000
    assert span.attributes[USAGE_PRICING_STATUS] == "estimated"
    assert span.attributes[USAGE_BILLED_MODELS] == "gpt-5.6-luna"

    record = json.loads(span.attributes["metadata"])
    assert record["detail"]["question"] == "why is ATC-77 blocked?"
    assert record["totals"]["cost_usd"] == pytest.approx(1.40)
    assert [c["bucket"] for c in record["charges"][0]["components"]] == ["input", "output"]


def test_the_json_record_reproduces_the_flat_total_it_is_filtered_by():
    """The flat attribute is a duplicate of the blob for filtering. If they can disagree, a query
    that filters on one and reports the other is silently wrong."""
    span = _RecordingSpan()
    ledger = _ledger(
        charge_for("router", "litellm gateway", "gpt-5.6-luna", "hosted", Usage(input_tokens=333_333)),
        charge_for("answer_agent", "litellm gateway", "gpt-5.6-luna", "hosted", Usage(output_tokens=777_777)),
    )
    ledger.attach(span)

    record = json.loads(span.attributes["metadata"])
    assert record["totals"]["cost_usd"] == pytest.approx(span.attributes[USAGE_COST_USD])
    assert record["totals"]["total_tokens"] == span.attributes[USAGE_TOTAL_TOKENS]


def test_an_unknown_user_is_left_off_the_span_rather_than_recorded_as_empty():
    """An empty `user.id` would make Phoenix's user filter match every internal call as one
    anonymous person — worse than the attribute being absent."""
    span = _RecordingSpan()
    UsageLedger(
        feature="ask_ai",
        caller=Caller(user_id="", username="", session_id="chat-x"),
        request_id="req-1",
        charges=[],
    ).attach(span)

    assert "user.id" not in span.attributes
    assert span.attributes["session.id"] == "chat-x"


def test_a_cache_hit_is_recorded_as_a_real_zero_cost_answer():
    """A question served from cache still happened. Dropping it would leave a hole in a user's
    session and make a cheap month look like a quiet one."""
    span = _RecordingSpan()
    _ledger(detail={"question": "same question again", "cache_hit": True}).attach(span)

    assert span.attributes[USAGE_COST_USD] == 0.0
    assert json.loads(span.attributes["metadata"])["detail"]["cache_hit"] is True


def test_the_feature_is_recorded_as_both_an_attribute_and_a_tag():
    span = _RecordingSpan()
    _ledger(feature=FEATURE_SUMMARIZE_COMMENTS).attach(span)

    assert span.attributes["ai.feature"] == FEATURE_SUMMARIZE_COMMENTS
    assert span.attributes["tag.tags"] == [FEATURE_SUMMARIZE_COMMENTS]


def test_the_record_is_json_serializable_as_one_attribute_value():
    """OTel attribute values are scalars — the nested breakdown has exactly one legal encoding, and
    a non-serializable `detail` would raise at request time rather than at review time."""
    span = _RecordingSpan()
    _ledger(detail={"space_ids": [1, 2], "abstained": False, "route_tool": None}).attach(span)

    assert isinstance(span.attributes["metadata"], str)
    assert json.loads(span.attributes["metadata"])["detail"]["space_ids"] == [1, 2]


# --- the feature vocabulary is a contract with the UI ------------------------------------------


def test_no_endpoint_passes_a_bare_string_as_its_feature_name():
    """The dimension the whole cost report groups by must come from the shared vocabulary.

    A literal at a call site is how the names drifted from the product in the first place — three
    features were labelled after their implementation (`comment_digest`, `description_review`,
    `precedents`), so the dashboard row for "Summarize comments" read as though someone had left a
    comment. A literal also fails silently: a typo does not raise, it splits one feature into two
    rows and understates both.

    Checked against the source rather than by calling the endpoints, because the failure this guards
    is someone adding the *next* endpoint, which no existing test would exercise.
    """
    import ast
    import pathlib

    import app.api.routes as routes_module
    import app.llm.usage_ledger as ledger_module

    vocabulary = {
        name for name in dir(ledger_module) if name.startswith("FEATURE_")
    }
    tree = ast.parse(pathlib.Path(routes_module.__file__).read_text())

    literals = [
        node.args[1].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_usage"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
    ]
    assert not literals, f"pass a FEATURE_* constant, not a literal: {literals}"

    used = {
        node.args[1].id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_usage"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Name)
    }
    assert used and used <= vocabulary, f"unknown feature constants: {used - vocabulary}"


def test_the_vocabulary_has_no_duplicate_values():
    """Two constants sharing a value would merge two products into one dashboard row."""
    import app.llm.usage_ledger as ledger_module

    values = [
        getattr(ledger_module, name)
        for name in dir(ledger_module)
        if name.startswith("FEATURE_") and name != "FEATURE"
    ]
    assert len(values) == len(set(values))
