"""Base model for every schema passed to instructor as `response_model`.

It marks every field `required` in the JSON Schema sent to the model. Pydantic only requires fields
without defaults, and the model happily skips optional ones: on a smaller model that enforces the
schema literally, TaskDraft came back as just a title; with all fields required it
returned labels, an 8-point estimate and a rationale. Hosted gpt-5.6-luna skipped the estimate on 2 of
8 runs.

"No answer" is expressed as null instead, which is also what OpenAI's strict mode requires. Python
construction is unchanged (defaults still apply), so fallback code that builds partial objects works.
"""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict


def every_field_required(schema: Dict[str, Any]) -> None:
    """Pydantic `json_schema_extra` hook — see the module docstring for why."""
    schema["required"] = list(schema.get("properties", {}))
    schema["additionalProperties"] = False


class StructuredOutput(BaseModel):
    """Inherit this instead of `BaseModel` for anything used as an `instructor` `response_model`.

    Nested models need it too: each one contributes its own subschema, with its own `required` list,
    so a strict parent containing a permissive child leaves the child's fields skippable.
    """

    model_config = ConfigDict(json_schema_extra=every_field_required)
