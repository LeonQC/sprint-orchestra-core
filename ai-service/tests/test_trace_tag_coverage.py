"""Every AI endpoint must tag its own entrypoint span with the feature it serves.

Why a source-level test rather than a behavioural one: the defect this guards against is invisible in
responses and only shows up in Langfuse, hours later, as a trace with `tags=[]` that nobody can
attribute to a feature. Measured live before this was fixed — `POST /plan-epic/rollout/stream` and
all 13 sprint-recovery endpoints produced traces with no tags at all, while the very same plan
generation through `POST /plan-epic` carried `['plan_epic']`:

    POST /plan-epic/refine        ['plan_epic_refine']   ChatCompletion  ['plan_epic_refine']
    POST /plan-epic               ['plan_epic']          ChatCompletion  ['plan_epic']
    POST /plan-epic/rollout/stream  []                   ChatCompletion  []          <- same work
    POST /sprint-recovery/start/stream  []               sprint_recovery.diagnose []

Parenting and trace ids were already correct in that data, so the tag is the whole gap: the child
span sat under the right trace, it just carried nothing to group by.

`_tag_entrypoint_span` has to be called explicitly in each handler — `FastAPIInstrumentor` opens the
server span outside every Starlette middleware, so no context-based mechanism can reach back and
attach to it (see that function's docstring). "Explicit in every handler" is exactly the kind of rule
that decays as endpoints are added, which is what this test exists to prevent.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROUTE_FILES = ("app/api/routes.py", "app/api/sprint_recovery_routes.py")

# Health probes carry no feature; /stats and /quota are reports that make no provider call.
EXEMPT = {"/healthz", "/healthz/cache", "/stats", "/quota"}


def _route_handlers() -> list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    found = []
    root = pathlib.Path(__file__).resolve().parents[1]
    for relative in ROUTE_FILES:
        tree = ast.parse((root / relative).read_text())
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "router"
                    and func.attr in ("get", "post", "put", "delete", "patch")
                    and decorator.args
                    and isinstance(decorator.args[0], ast.Constant)
                ):
                    found.append((relative, decorator.args[0].value, node))
    return found


def test_the_route_scan_actually_finds_the_endpoints():
    """Guard the guard: an AST walk that silently matched nothing would make every assertion vacuous."""
    handlers = _route_handlers()
    paths = {path for _, path, _ in handlers}
    assert len(handlers) >= 30
    assert "/ask" in paths
    assert "/plan-epic/rollout/stream" in paths
    assert "/start/stream" in paths  # the sprint-recovery router mounts under its own prefix


@pytest.mark.parametrize(
    "relative,path,handler",
    [pytest.param(f, p, h, id=f"{f.split('/')[-1]}:{p}") for f, p, h in _route_handlers()],
)
def test_every_ai_endpoint_tags_its_entrypoint_span(relative, path, handler):
    if path in EXEMPT:
        return
    calls = [
        node.func.id
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "_tag_entrypoint_span" in calls, (
        f"{relative} {path} never calls _tag_entrypoint_span, so its Langfuse trace will carry "
        f"tags=[] and cannot be attributed to a feature. Add it as the first statement in the "
        f"handler, or add the path to EXEMPT if it genuinely serves no AI feature."
    )
