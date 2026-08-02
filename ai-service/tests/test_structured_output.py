"""Tests for the wire-schema rule every instructor response_model inherits.

The bug this guards against isn't a crash — it's a model quietly answering less than it was asked.
A field with a Python default lands outside Pydantic's `required` list, and a schema-constrained
server reads that as permission to stop early: LM Studio's qwen3.6-27b returned a bare title for a
detailed description, every single time, until the schema said otherwise.
"""
from typing import List, Optional

from pydantic import Field

from app.comment_digest.schemas import CommentDigest, CommentIn
from app.drafting.schemas import TaskDraft
from app.llm.structured_output import StructuredOutput
from app.planning.schemas import EpicPlanDraft, IssueDraft
from app.sprint_pace.schemas import FlaggedIssue, SprintPaceInsight, SprintStats
from app.sprint_recovery.schemas import (
    DiagnosisResult,
    EvidenceItem,
    RecoveryPlanSet,
    RiskSignal,
    SprintSnapshot,
)


class _Sample(StructuredOutput):
    required_field: str
    with_default: List[str] = Field(default_factory=list)
    nullable: Optional[int] = None


def test_fields_with_defaults_are_still_required_on_the_wire():
    schema = _Sample.model_json_schema()

    assert set(schema["required"]) == {"required_field", "with_default", "nullable"}
    assert schema["additionalProperties"] is False


def test_python_construction_keeps_its_defaults():
    """The strictness is a property of the emitted schema only — the degraded fallback paths build
    partial objects by hand and must keep working."""
    sample = _Sample(required_field="x")

    assert sample.with_default == []
    assert sample.nullable is None


def test_nullability_is_how_a_model_declines_to_answer():
    """Required-ness would be a trap without this: the model has to emit the key, so it needs a
    legal way to say "no value" other than omitting it."""
    schema = _Sample.model_json_schema()

    assert {"type": "null"} in schema["properties"]["nullable"]["anyOf"]


def test_nested_models_carry_the_rule_into_their_own_subschema():
    """A strict parent containing a permissive child leaves the child's fields skippable — and for
    plan-epic the child (IssueDraft) is where the estimates and dependencies live."""
    defs = EpicPlanDraft.model_json_schema()["$defs"]

    assert set(defs["IssueDraft"]["required"]) == set(defs["IssueDraft"]["properties"])
    assert set(defs["EpicDraft"]["required"]) == set(defs["EpicDraft"]["properties"])


def test_the_fields_that_were_actually_being_dropped_are_covered():
    """Named explicitly rather than left to the set comparison above: these are the ones observed
    missing from real responses, and the ones the UI has nothing to show without."""
    for field in ("labels", "estimate_story_points", "estimate_rationale", "dependencies"):
        assert field in TaskDraft.model_json_schema()["required"]
    for field in ("labels", "estimate_story_points", "estimate_rationale", "depends_on"):
        assert field in IssueDraft.model_json_schema()["required"]


def test_a_declared_insufficient_diagnosis_must_carry_its_clarifying_question():
    """The sharpest case, and the reason this rule isn't cosmetic: `diagnose_node` stores
    `clarifying_question` verbatim and `_after_diagnose` routes to the human-clarification round only
    when it is truthy. While the field was skippable, the local model declared "insufficient" and
    omitted the question in 7-8 of 8 runs — so the round could never fire, silently."""
    schema = DiagnosisResult.model_json_schema()

    assert "clarifying_question" in schema["required"]
    assert set(schema["$defs"]["RootCauseHypothesis"]["required"]) == {
        "statement", "confidence", "supporting_evidence_ids",
    }


def test_every_model_nested_in_a_recovery_plan_is_strict_too():
    defs = RecoveryPlanSet.model_json_schema()["$defs"]

    for name in ("RecoveryPlan", "RecoveryAction"):
        assert set(defs[name]["required"]) == set(defs[name]["properties"]), name


def test_the_pace_insight_cannot_drop_its_recommendations():
    assert "recommendations" in SprintPaceInsight.model_json_schema()["required"]


def test_models_that_never_come_from_an_llm_stay_plain():
    """Request payloads and code-built state deliberately keep Pydantic's normal optionality — the
    wire schema never reaches a model for these, and requiring every key would only make the API
    stricter on callers for no benefit."""
    for model in (SprintStats, FlaggedIssue, EvidenceItem, RiskSignal, SprintSnapshot, CommentIn):
        assert not issubclass(model, StructuredOutput), model.__name__


def test_the_comment_digest_and_its_nested_points_are_strict_too():
    """`comment_ids` is the field this whole schema exists for — grounding a claim only works if the
    model can't quietly omit which comments back it (see app/comment_digest/schemas.py)."""
    schema = CommentDigest.model_json_schema()
    defs = schema["$defs"]

    assert set(schema["required"]) == set(schema["properties"])
    for name in ("AttributedPoint", "ActionItemOut", "Outcome"):
        assert "comment_ids" in defs[name]["required"], name
