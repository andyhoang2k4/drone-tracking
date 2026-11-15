"""Test scripts/diagnose_mm_hijack.py tren log/metrics/manifest DINH DANG THAT (sinh boi
run_occlusion_ablation.py voi YOLO gia) + ham phan loai."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.diagnose_mm_hijack as dg  # noqa: E402
import src.evaluate as ev  # noqa: E402
from tests.test_tune_motion_memory import run_source, tune_env  # noqa: E402,F401


def test_classify_frame():
    assert dg.classify_frame({"state": "LOST", "track_id": 1}, 1) == "lost_extrapolated"
    assert dg.classify_frame({"state": "TRACKING", "track_id": 1}, 1) == "tracking_original_id"
    assert dg.classify_frame({"state": "TRACKING", "track_id": 7}, 1) == "tracking_other_id"
    assert dg.classify_frame({"state": "REIDENTIFYING", "track_id": 7}, 1) == "other"


def test_diagnose_matches_scenario_metrics_and_writes_json(tune_env, capsys):
    tmp, _, weights = tune_env
    run_source(tmp, weights)
    exps = tmp / "experiments"
    names = [f"{d}_botsort_mm_occl_dev_v2_001" for d in ("yolo11", "yolo26")]
    out = tmp / "diag.json"
    assert dg.main(sum((["--experiment", n] for n in names), [])
                   + ["--output-root", str(exps), "--scenario-root", str(tmp / "scn"), "--output", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["kind"] == dg.KIND and len(res["metrics_config"]["config_sha256"]) == 64
    for n in names:
        e = res["experiments"][n]
        agg = json.loads((exps / n / "metrics.json").read_text())["overall"]
        assert e["scenarios_locked"] == agg["scenarios_evaluated"] == 2
        assert e["scenarios_not_locked_skipped"] == 1
        # cong theo loai == so lieu metrics v2 cua cung lan chay (cung dinh nghia)
        assert e["overall"]["total"] == {"frames_with_box": agg["occluded_frames_with_output"],
                                         "correct": agg["correct_occluded_frames"]}
        assert set(e["overall"]) == set(dg.CATEGORIES) | {"total"}
        assert list(e["by_duration_frames"]) == ["5"]
        assert e["hijacked_before_occlusion_scenarios"] == 0
        assert "motion_gate" in e["recover_reason_frames"] or "same_id" in e["recover_reason_frames"]
    assert "bi cuop truoc khi che" in capsys.readouterr().out


def test_refuses_baseline_experiment(tune_env):
    tmp, _, weights = tune_env
    run_source(tmp, weights)
    with pytest.raises(ev.EvaluationError, match="khong phai lan chay Motion Memory"):
        dg.main(["--experiment", "yolo11_botsort_occl_dev_v2_001", "--output-root", str(tmp / "experiments"),
                 "--scenario-root", str(tmp / "scn"), "--output", str(tmp / "x.json")])
