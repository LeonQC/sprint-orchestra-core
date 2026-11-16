import json

from eval import eval_gate

CONFIG = {
    "baseline": "ragas_50q_test.json",
    "ragas": {"min": {"faithfulness": 0.89, "context_recall": 0.82}, "abstention_refused_min": 3},
    "sprint_recovery_scenarios": {"diagnosis_min": 11, "plan_min": 6},
}


def _results(faithfulness=0.94, recall=0.87, refused=3, diagnosis=12, plan=7, blocked=3):
    ragas = {"means": {"faithfulness": faithfulness, "context_recall": recall}, "abstention": {"refused": refused, "total": 3}}
    scenarios = {"diagnosis": {"passed": diagnosis, "total": 12}, "plan": {"passed": plan, "total": 7}}
    injection = {"passed": blocked, "total": 3}
    return ragas, scenarios, injection


def _failed(**kwargs):
    return [c.name for c in eval_gate.checks(CONFIG, *_results(**kwargs)) if not c.ok]


def test_a_run_at_the_baseline_passes():
    assert _failed() == []


def test_a_metric_under_its_floor_fails_only_that_check():
    assert _failed(recall=0.80) == ["ragas.context_recall"]


def test_one_successful_injection_fails_the_gate():
    assert _failed(blocked=2) == ["sprint_recovery.injection_blocked"]


def test_a_question_answered_instead_of_refused_fails_the_gate():
    assert _failed(refused=2) == ["ragas.abstention_refused"]


def test_main_exits_nonzero_on_a_regression(tmp_path):
    ragas, scenarios, injection = _results(plan=5)
    paths = {}
    for name, data in {"config": CONFIG, "ragas": ragas, "scenarios": scenarios, "injection": injection}.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(data))
    argv = [f"--{k}={v}" for k, v in paths.items()]
    assert eval_gate.main(argv) == 1


def test_without_ragas_only_the_sprint_recovery_checks_run():
    _, scenarios, injection = _results()
    names = [c.name for c in eval_gate.checks(CONFIG, None, scenarios, injection)]
    assert names == ["sprint_recovery.diagnosis_correct", "sprint_recovery.plan_correct", "sprint_recovery.injection_blocked"]



def test_retrieval_checks_use_their_own_floors():
    config = {**CONFIG, "retrieval": {"min": {"hit@5": 0.93, "recall@10": 0.95}}}
    retrieval = {"means": {"hit@5": 0.90, "recall@10": 0.98}}
    failed = [c.name for c in eval_gate.checks(config, *_results(), retrieval=retrieval) if not c.ok]
    assert failed == ["retrieval.hit@5"]


def test_without_retrieval_no_retrieval_checks_run():
    names = [c.name for c in eval_gate.checks(CONFIG, *_results())]
    assert not any(n.startswith("retrieval.") for n in names)
