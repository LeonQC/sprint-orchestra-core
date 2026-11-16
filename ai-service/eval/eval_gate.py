"""Pass/fail gate over one run of the AI evals: RAGAS on the 50-question AtlasCart cohort, the 12
Sprint Recovery scenarios and the Sprint Recovery prompt-injection probe.
With --retrieval, the issue-level hit@5 / recall@10 from eval/retrieval_hit.py are checked too.

Two places run it. CI (.github/workflows/ci.yml, `ai-evals`) runs the two Sprint Recovery evals on
every push: they need only a model API key, since retrieval and Jira are fakes. RAGAS needs the whole
stack with the corpus, so `scripts/run_rag_eval.sh gate` runs all three locally before merging an
AI change; without --ragas the RAGAS checks are skipped.

Floors live in eval/eval_gate.json. RAGAS floors are the lowest of three runs with no known defect,
minus 0.03: unchanged code moves up to ~0.06 from one run to the next on this cohort, so a floor taken
from a single run fails on noise.
The injection probe has no slack: one successful attack fails the gate.

Usage: python -m eval.eval_gate --scenarios S.json --injection I.json [--ragas R.json] [--retrieval H.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_CONFIG = Path(__file__).parent / "eval_gate.json"


@dataclass
class Check:
    name: str
    value: float
    floor: float

    @property
    def ok(self) -> bool:
        return self.value >= self.floor


def checks(
    config: dict,
    ragas: Optional[dict],
    scenarios: dict,
    injection: dict,
    retrieval: Optional[dict] = None,
) -> List[Check]:
    out = []
    if retrieval is not None:
        out += [Check(f"retrieval.{metric}", retrieval["means"][metric], floor) for metric, floor in config["retrieval"]["min"].items()]
    if ragas is not None:
        out += [Check(f"ragas.{metric}", ragas["means"][metric], floor) for metric, floor in config["ragas"]["min"].items()]
        out.append(Check("ragas.abstention_refused", ragas["abstention"]["refused"], config["ragas"]["abstention_refused_min"]))
    sc = config["sprint_recovery_scenarios"]
    out.append(Check("sprint_recovery.diagnosis_correct", scenarios["diagnosis"]["passed"], sc["diagnosis_min"]))
    out.append(Check("sprint_recovery.plan_correct", scenarios["plan"]["passed"], sc["plan_min"]))
    out.append(Check("sprint_recovery.injection_blocked", injection["passed"], injection["total"]))
    return out


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragas", type=Path, default=None)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--injection", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, default=None, help="eval/retrieval_hit.py output")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    results = checks(
        config,
        json.loads(args.ragas.read_text()) if args.ragas else None,
        json.loads(args.scenarios.read_text()),
        json.loads(args.injection.read_text()),
        json.loads(args.retrieval.read_text()) if args.retrieval else None,
    )
    print(f"eval gate (floors from {args.config.name}, baseline {config['baseline']})")
    if args.ragas is None:
        print("  skip  ragas.*                                  (no --ragas: needs the live stack)")
    if args.retrieval is None:
        print("  skip  retrieval.*                              (no --retrieval: needs vectorization-service)")
    for c in results:
        print(f"  {'PASS' if c.ok else 'FAIL'}  {c.name:38s} {c.value:>7.3f}  (floor {c.floor:.3f})")
    failed = [c.name for c in results if not c.ok]
    print("GATE: " + ("PASS" if not failed else "FAIL — " + ", ".join(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
