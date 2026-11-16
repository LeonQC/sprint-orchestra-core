"""Publish every prompt template in this service to Phoenix, one Phoenix prompt per `.jinja` file.

Run it when you deliberately change a prompt and want the change on record — not on every service
start. A version per meaningful edit, not per process restart.

**What "changed" means is decided by content, not by the operator.** Each template's version is
`sha256(file)[:12]` (`app/llm/prompt_registry.py`), and this script pushes a template only when that
fingerprint differs from the fingerprint of what Phoenix already holds. So running it twice in a row
is a no-op, running it after editing one template pushes exactly that one, and there is no way to
forget to bump a version number — there isn't one to bump.

That fingerprint is the join key for the whole prompt-management story: the same string appears
  * in the Phoenix prompt version's description (`prompt version: <fp>`), pushed from here,
  * on every trace, as the `prompt.<name>.version` span attribute stamped at render time,
  * in each eval result file, under `prompt_versions` (`eval/ragas_eval.py`).
Which is what makes "which prompt produced this score / this answer / this trace?" a lookup instead
of an exercise in remembering what the file said last Tuesday.

**Scope, stated precisely rather than implied:** Phoenix's native template formats (`MUSTACHE`,
`F_STRING`) don't support Jinja2's `{% if %}` / `{% for %}` control flow, so this pushes the raw
`.jinja` source with `template_format="NONE"` — Phoenix stores, versions and diffs the *content* for
you to browse in its UI, but does **not** render it. Rendering still happens in-process via
`app/llm/prompt_registry.py`. This is a version-tracking side-channel, not a runtime dependency: every
feature works identically whether or not this has ever been run.

Usage:
    docker run -d -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest   # if not already running
    python -m scripts.push_prompt_to_phoenix                 # push whatever changed
    python -m scripts.push_prompt_to_phoenix --dry-run       # fingerprint report, no Phoenix needed
    python -m scripts.push_prompt_to_phoenix --only crag_answer_system
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.llm.prompt_registry import all_templates, fingerprint

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _published_fingerprint(client, name: str) -> Optional[str]:
    """Fingerprint of the newest version Phoenix holds for `name`, or None if it holds none.

    Read back through `PromptVersion.format()` rather than trusted from a local record of what was
    pushed before: a local record would go stale the moment anyone pushed from another machine, and
    the point of the comparison is "does Phoenix already have this text", which only Phoenix can
    answer. A prompt that doesn't exist yet raises rather than returning empty, so the not-found case
    is a caught exception rather than a special value.
    """
    try:
        version = client.prompts.get(prompt_identifier=name)
    except Exception:  # noqa: BLE001 - phoenix raises a client-specific NotFound; any failure to read
        return None    # an existing version is correctly treated as "not published yet" and pushed.
    messages = version.format().messages
    if not messages:
        return None
    content = messages[0].get("content")
    return fingerprint(content) if isinstance(content, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="print each template's fingerprint and stop — never contacts Phoenix")
    parser.add_argument("--only", metavar="NAME", default=None,
                        help="push just this template (its filename without .jinja)")
    args = parser.parse_args()

    templates = all_templates()
    if args.only:
        if args.only not in templates:
            print(f"no such template: {args.only!r}. Known: {', '.join(sorted(templates))}", file=sys.stderr)
            return 2
        templates = {args.only: templates[args.only]}

    if args.dry_run:
        print(f"{'PROMPT':<28} {'VERSION':<14} SOURCE")
        for name, path in sorted(templates.items()):
            source = path.read_text(encoding="utf-8")
            print(f"{name:<28} {fingerprint(source):<14} {path.relative_to(_REPO_ROOT)}")
        return 0

    from phoenix.client import Client
    from phoenix.client.types import PromptVersion

    settings = Settings()
    client = Client()  # defaults to PHOENIX_COLLECTOR_ENDPOINT / http://localhost:6006
    pushed = skipped = 0
    for name, path in sorted(templates.items()):
        source = path.read_text(encoding="utf-8")
        local = fingerprint(source)
        if _published_fingerprint(client, name) == local:
            print(f"  unchanged  {name:<28} {local}")
            skipped += 1
            continue
        client.prompts.create(
            name=name,
            version=PromptVersion(
                [{"role": "system", "content": source}],
                model_name=settings.agent_model,
                model_provider="ANTHROPIC" if settings.llm_provider == "anthropic" else "OPENAI",
                template_format="NONE",  # see module docstring — Jinja2 control flow isn't representable
            ),
            prompt_description=(
                f"prompt version: {local} — Jinja2 source of {path.relative_to(_REPO_ROOT)}. Stored "
                "as-is for version tracking/diffing in the Phoenix UI; rendering happens in-process "
                "via app/llm/prompt_registry.py, not by Phoenix. The same fingerprint appears on "
                f"traces as prompt.{name}.version and in eval results under prompt_versions."
            ),
        )
        print(f"  PUSHED     {name:<28} {local}")
        pushed += 1

    print(f"\n{pushed} pushed, {skipped} unchanged. Browse them at http://localhost:6006/prompts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
