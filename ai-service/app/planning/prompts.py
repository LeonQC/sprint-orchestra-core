"""Prompt rendering for epic planning (same setup as app/drafting/prompts.py).

All three templates (plan, critic, refine) get the same optional grounding block. A critic that
doesn't see the team's history would flag every grounded estimate as unrealistic and pull it back to
a generic average, and a refine call without it would write added issues in the generic voice.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from app.llm.prompt_registry import template_loader
from app.planning.grounding import GroundingContext
from app.planning.pattern_schemas import MIN_PATTERN_EVIDENCE, EvidenceAssessment

_prompts = template_loader(__file__)


def render_pattern_extraction_prompt() -> str:
    """The evidence-evaluation / pattern-extraction stage (`app/planning/patterns.py`)."""
    return _prompts.render(
        "pattern_extraction_system.jinja", min_pattern_evidence=MIN_PATTERN_EVIDENCE,
    )


def _grounding_vars(
    grounding: Optional[GroundingContext], assessment: Optional[EvidenceAssessment] = None,
) -> Dict[str, Any]:
    """`exemplars`, `estimates_grounded` and the assessment are separate on purpose: having history
    isn't having SIZED history (only that turns on estimate-by-comparison), and sized history isn't
    RELEVANT history — with `evidence_strength == "none"` the template hides it from the planner.
    """
    if grounding is None:
        return {
            "exemplars": [], "estimates_grounded": False, "patterns": [],
            "evidence_strength": "none", "evidence_reasoning": "",
            "house_format": [], "house_epic_format": [], "house_sample": 0,
            "house_item_marker": None,
            "house_item_low": None, "house_item_high": None, "house_item_max": None,
            "requirements": [],
        }
    strength = assessment.strength if assessment is not None else "unassessed"
    # "unassessed" is not "none": it means nobody looked. The exemplars still go in (the measured
    # behaviour); only a real "none" verdict withholds them.
    withhold = strength == "none"
    return {
        # NOT withheld by a "none" verdict, unlike everything else here. Relevance decides what the
        # history TEACHES about breaking work down and sizing it; it has nothing to say about how this
        # board lays out a description, which is a convention that holds whatever the epic is about.
        "house_format": grounding.house_format.sections,
        "house_sample": grounding.house_format.sample_size,
        "house_epic_format": grounding.house_format.epic_sections,
        "house_item_marker": grounding.house_format.item_marker,
        "house_item_low": grounding.house_format.item_count_low,
        "house_item_high": grounding.house_format.item_count_high,
        "house_item_max": grounding.house_format.item_count_max,
        # A history verdict can withhold examples, never the user's requirements themselves.
        "requirements": [replace(item, exemplars=[]) for item in grounding.requirements]
        if withhold else grounding.requirements,
        "exemplars": [] if withhold else grounding.exemplars,
        "estimates_grounded": bool(grounding.estimates_grounded and not withhold),
        "patterns": assessment.patterns if assessment is not None else [],
        "evidence_strength": strength,
        "evidence_reasoning": assessment.reasoning if assessment is not None else "",
    }


def render_epic_plan_prompt(
    existing_labels: Optional[List[str]] = None,
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> str:
    return _prompts.render(
        "epic_plan_system.jinja", existing_labels=existing_labels or [], **_grounding_vars(grounding, assessment)
    )


def render_epic_plan_refine_prompt(
    existing_labels: Optional[List[str]] = None,
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> str:
    return _prompts.render(
        "epic_plan_refine_system.jinja", existing_labels=existing_labels or [],
        **_grounding_vars(grounding, assessment),
    )


def render_critic_prompt(
    grounding: Optional[GroundingContext] = None,
    assessment: Optional[EvidenceAssessment] = None,
) -> str:
    """No `existing_labels`: the critic only reports and never edits. It does get the exemplars (see
    the module docstring).
    """
    return _prompts.render("critic_system.jinja", **_grounding_vars(grounding, assessment))
