import json

from scripts.aggregate_gradient_mpc import read_run, wilson_interval


def test_target_level_mpc_aggregation_keeps_last_cumulative_success(tmp_path):
    run = tmp_path / "case"
    attempt = run / "attempt-1"
    attempt.mkdir(parents=True)
    (run / "done.txt").write_text(str(attempt) + "\n")
    records = [
        {"sample_idx": 0, "step": 1, "success": False},
        {"sample_idx": 0, "step": 2, "success": True},
        {"sample_idx": 1, "step": 1, "success": False},
        {"sample_idx": 1, "step": 2, "success": False},
    ]
    (attempt / "per_iter_logs.json").write_text("".join(json.dumps(r) + "\n" for r in records))
    winners, last = read_run(run, total=2)
    assert winners == {0}
    assert last[0]["step"] == 2
    low, high = wilson_interval(1, 2)
    assert low < 50 < high
