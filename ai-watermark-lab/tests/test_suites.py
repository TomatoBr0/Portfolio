from lab.experiment import Settings
from lab.suites import analyze_baseline, make_plan, wilson


def test_calibration_and_test_use_disjoint_seeds_and_labels():
    plan = make_plan(Settings(kind="baseline", calibration_samples=20, evaluation_pairs=5))
    calibration = [job for job in plan if job["phase"] == "calibration"]
    evaluation = [job for job in plan if job["phase"] == "evaluation"]
    assert len(calibration) == 20 and len(evaluation) == 10
    assert not any(job["marked"] for job in calibration)
    assert not {job["seed"] for job in calibration} & {job["seed"] for job in evaluation}
    for pair in range(1, 6):
        off, on = [job for job in evaluation if job["pair"] == pair]
        assert off["seed"] == on["seed"] and off["prompt"] == on["prompt"]


def test_threshold_uses_calibration_only_and_evaluation_rates_are_separate():
    rows = [{"phase": "calibration", "marked": False, "z": float(i)} for i in range(40)]
    rows += [{"phase": "evaluation", "marked": False, "z": 38.0},
             {"phase": "evaluation", "marked": False, "z": 1000.0},
             {"phase": "evaluation", "marked": True, "z": 39.0}]
    result = analyze_baseline(rows, 0.05)
    assert result["threshold"] == 38.0
    assert result["false_alarms"]["rate"] == 0.5
    assert result["detection"]["rate"] == 1
    assert analyze_baseline(rows[:5], 0.01)["threshold"] is None


def test_zero_observed_false_alarms_does_not_imply_zero_uncertainty():
    interval = wilson(0, 10)
    assert interval["rate"] == 0
    assert interval["interval95"][1] > 0.25
    assert wilson(0, 0)["rate"] is None


def test_strength_sweep_includes_zero_bias_control_and_shared_seeds():
    plan = make_plan(Settings(kind="strength", pairs=2))
    assert len(plan) == 10
    assert {job["bias"] for job in plan if job["marked"]} == {0, 0.75, 1.5, 2.5}
    assert len({job["seed"] for job in plan if job["pair"] == 1}) == 1
