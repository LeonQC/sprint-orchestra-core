"""One Jinja2 environment for every prompt, plus a content-based version for each template.

1. Six feature modules each built an identical Environment; `template_loader(__file__)` replaces
   them.
2. Nothing recorded which prompt produced an eval score or a trace. Now the version IS the content:
   `sha256(template source)[:12]`. Nothing to bump or forget, and two runs share a version exactly
   when they used the same prompt text.

Rendering stamps `prompt.<name>.version` on the current span (one key per template, since a request
can render several). With tracing off it's a no-op. scripts/push_prompt_to_phoenix.py uses the same
versions.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape
from opentelemetry import trace

TEMPLATE_SUFFIX = ".jinja"
#: 12 hex chars = 48 bits. Long enough that a collision across the handful of templates this service
#: has is not a thing that happens, short enough to read out loud in a demo or eyeball in a trace.
_VERSION_CHARS = 12

_APP_DIR = Path(__file__).resolve().parent.parent


def fingerprint(text: str) -> str:
    """The version of a prompt is the hash of its content — see the module docstring."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_VERSION_CHARS]


class PromptTemplates:
    """The `.jinja` files in one feature's `templates/` directory.

    Read from disk on every render (no cache): files are small, and the version can then never claim
    a template the process isn't actually sending. Edits also show up without a restart.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self._env = Environment(
            loader=FileSystemLoader(directory),
            # No HTML escaping: prompts are plain text for a model, and escaping would break their
            # quoting.
            autoescape=select_autoescape(disabled_extensions=("jinja",)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def path(self, name: str) -> Path:
        return self.directory / name

    def version(self, name: str) -> str:
        """Version of the template SOURCE, not the rendered text — the rendered text differs per
        request (interpolated data), so it would give a new "version" every time.
        """
        return fingerprint(self.path(name).read_text(encoding="utf-8"))

    def names(self) -> List[Path]:
        return sorted(self.directory.glob(f"*{TEMPLATE_SUFFIX}"))

    def render(self, name: str, **context) -> str:
        rendered = self._env.get_template(name).render(**context)
        _record_version(Path(name).stem, self.version(name))
        return rendered


def template_loader(module_file: str) -> PromptTemplates:
    """`PromptTemplates` for the `templates/` folder next to the calling module
    (`template_loader(__file__)`), so prompts stay beside the code that renders them.
    """
    return PromptTemplates(Path(module_file).resolve().parent / "templates")


def all_templates() -> Dict[str, Path]:
    """Every prompt template in the service, `{stem: path}`, found by scanning
    `app/**/templates/*.jinja` (no list to maintain). Stems must be unique (asserted), or two prompts
    would share one Phoenix name.
    """
    found: Dict[str, Path] = {}
    for path in sorted(_APP_DIR.rglob(f"templates/*{TEMPLATE_SUFFIX}")):
        if path.stem in found:
            raise RuntimeError(
                f"duplicate prompt template stem {path.stem!r}: {found[path.stem]} and {path}. "
                "Template stems are used as Phoenix prompt names and must be unique across the service."
            )
        found[path.stem] = path
    return found


def all_versions() -> Dict[str, str]:
    """`{stem: version}` for every template in the service — recorded in eval result files so a score
    can be traced back to the exact prompt text that produced it."""
    return {stem: fingerprint(path.read_text(encoding="utf-8")) for stem, path in all_templates().items()}


def versions_for(*stems: str) -> Dict[str, str]:
    """`{stem: version}` for the named templates, recorded next to an eval score. Only the named ones,
    so the result doesn't list prompts it never used. An unknown name raises, so a rename breaks the
    eval instead of recording nothing.
    """
    templates = all_templates()
    missing = [s for s in stems if s not in templates]
    if missing:
        raise KeyError(f"no such prompt template(s): {missing}. Known: {sorted(templates)}")
    return {s: fingerprint(templates[s].read_text(encoding="utf-8")) for s in stems}


def _record_version(stem: str, version: str) -> None:
    span = trace.get_current_span()
    # No `is_recording()` guard needed: OTel's NonRecordingSpan.set_attribute is already a no-op, and
    # the guard would only duplicate that check.
    span.set_attribute(f"prompt.{stem}.version", version)
