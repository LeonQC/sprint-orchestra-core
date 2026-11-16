"""Tests for the shared prompt loader and its content-addressed versions (app/llm/prompt_registry.py).

The properties worth pinning are the ones the whole prompt-management story rests on: a version is
the content (so it can never be forgotten or bumped by hand), every prompt in the service is covered
(so "which prompt version answered this?" has no exceptions), and rendering leaves the fingerprint on
the trace (so the answer is discoverable without re-deriving it).
"""
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.llm.prompt_registry import (
    PromptTemplates,
    all_templates,
    all_versions,
    fingerprint,
    template_loader,
    versions_for,
)

_APP = Path(__file__).resolve().parent.parent / "app"


def test_version_is_the_content_so_an_edit_can_never_be_forgotten(tmp_path):
    """The point of hashing instead of a version field: you cannot change a prompt without changing
    its version, and changing it back restores the old one — which is what makes two runs comparable
    by fingerprint rather than by trusting a changelog."""
    (tmp_path / "templates").mkdir()
    template = tmp_path / "templates" / "t.jinja"
    template.write_text("hello {{ name }}")
    prompts = PromptTemplates(tmp_path / "templates")

    original = prompts.version("t.jinja")
    template.write_text("hello there {{ name }}")
    edited = prompts.version("t.jinja")
    template.write_text("hello {{ name }}")

    assert edited != original
    assert prompts.version("t.jinja") == original


def test_version_tracks_the_source_not_the_rendered_output(tmp_path):
    """A prompt with an interpolated section renders differently per request; fingerprinting the
    output would mint a new "version" per call and identify nothing."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "t.jinja").write_text("labels: {{ labels }}")
    prompts = PromptTemplates(tmp_path / "templates")

    assert prompts.render("t.jinja", labels="a") != prompts.render("t.jinja", labels="b")
    assert prompts.version("t.jinja") == prompts.version("t.jinja")


def test_rendering_stamps_the_version_on_the_current_span(tmp_path):
    """What makes a Phoenix trace self-describing: the fingerprint of every template that went into
    the request is on the span, so a trace never has to be matched to a prompt by timestamp."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "greet.jinja").write_text("hi")
    prompts = PromptTemplates(tmp_path / "templates")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with trace.use_span(provider.get_tracer(__name__).start_span("request"), end_on_exit=True):
        prompts.render("greet.jinja")

    (span,) = exporter.get_finished_spans()
    assert span.attributes["prompt.greet.version"] == fingerprint("hi")


def test_rendering_without_tracing_is_a_no_op_not_a_crash(tmp_path):
    """Tracing is off by default (`AI_PHOENIX_ENABLED=false`), which must cost nothing and break
    nothing — OTel's non-recording span swallows the attribute."""
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "t.jinja").write_text("plain")
    assert PromptTemplates(tmp_path / "templates").render("t.jinja") == "plain"


def test_every_prompt_in_the_service_is_a_versioned_template():
    """No feature may reintroduce an f-string prompt: `crag_loop.py` and `sprint_recovery/graph.py`
    were the last two holdouts, and an unversioned prompt is invisible to Phoenix, to the eval result
    files and to the trace attribute above. Catching a new one here is cheaper than noticing months
    later that one prompt's history is missing."""
    features_with_a_model_call = {
        "agent", "comment_digest", "description_review", "drafting", "planning", "precedent",
        "sprint_pace", "sprint_recovery",
    }
    covered = {path.relative_to(_APP).parts[0] for path in all_templates().values()}
    assert features_with_a_model_call - covered == set()


def test_every_feature_prompts_module_uses_the_shared_loader():
    """Six copies of the same `Environment(...)` block is what this module replaced; a seventh would
    quietly reintroduce the drift (a different autoescape setting, a missing `trim_blocks`) that made
    the templates stop rendering identically."""
    offenders = [
        str(p.relative_to(_APP)) for p in _APP.rglob("prompts.py")
        if "FileSystemLoader" in p.read_text()
    ]
    assert offenders == []


def test_all_versions_covers_every_template_file():
    assert set(all_versions()) == set(all_templates())
    assert all(len(v) == 12 for v in all_versions().values())


def test_versions_for_rejects_an_unknown_name_instead_of_recording_nothing():
    """An eval that records `{}` for a renamed template looks like it ran against no prompt at all."""
    with pytest.raises(KeyError):
        versions_for("task_draft_system", "a_template_that_does_not_exist")


def test_template_loader_resolves_the_directory_next_to_the_calling_module():
    prompts = template_loader(str(_APP / "drafting" / "prompts.py"))
    assert prompts.directory == _APP / "drafting" / "templates"
    assert "task_draft_system" in {p.stem for p in prompts.names()}
