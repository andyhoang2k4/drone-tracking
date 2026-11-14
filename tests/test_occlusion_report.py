"""Test src/occlusion_report.py (ham thuan) bang metrics gia."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import occlusion_report as rep  # noqa: E402

DETS = ("yolo11", "yolo26")


def scen(seq, dur, correct, occ=10, out=8, reid=True, locked=True, sid=None):
    return {"scenario_id": sid or f"dev/{seq}__d{dur}__{correct}",
            "scenario": {"sequence": seq, "duration_frames": dur},
            "metrics": {"target_locked_at_occlusion": locked, "reidentified": reid, "occlusion_frames": occ,
                        "correct_occluded_frames": correct, "occluded_frames_with_output": out}}


def items(bonus=0):
    """3 sequence x 2 duration; `bonus` them frame dung o sequence s1."""
    out = []
    for i, seq in enumerate(("s1", "s2", "s3")):
        for dur in (5, 15):
            out.append(scen(seq, dur, correct=2 + i + (bonus if seq == "s1" else 0), sid=f"dev/{seq}__d{dur}"))
    out.append(scen("s3", 15, correct=0, locked=False, sid="dev/s3__d15__x"))   # chua khoa -> bo qua
    return out


def test_box_precision_and_pooled_sum_counts_not_mean():
    assert rep.box_precision({"correct_occluded_frames": 3, "occluded_frames_with_output": 12}) == 0.25
    assert rep.box_precision({"correct_occluded_frames": 0, "occluded_frames_with_output": 0}) is None
    aggs = {"yolo11": {"overall": {"correct_occluded_frames": 10, "occlusion_frames": 100,
                                   "occluded_frames_with_output": 20, "reidentified": 1, "scenarios_evaluated": 2},
                       "by_duration_frames": {}},
            "yolo26": {"overall": {"correct_occluded_frames": 30, "occlusion_frames": 50,
                                   "occluded_frames_with_output": 60, "reidentified": 3, "scenarios_evaluated": 4},
                       "by_duration_frames": {}}}
    p = rep.pooled_across_detectors(aggs, DETS)["overall"]
    assert p["correct_prediction_ratio"] == 40 / 150 and p["box_precision"] == 40 / 80
    assert p["reidentification_rate"] == 4 / 6


def test_scenario_counts_skip_not_locked_and_require_v2():
    assert rep.scenario_counts(scen("s", 5, 1, locked=False)) is None
    with pytest.raises(rep.ReportError, match="v2"):
        rep.scenario_counts({"scenario_id": "x", "metrics": {"reidentified": True}})


def run(cols, **kw):
    args = dict(resamples=300, seed=7, ci=0.95, paired_vs=["no_mm"],
                metrics=["correct_prediction_ratio", "box_precision", "reidentification_rate"], durations=[5, 15])
    args.update(kw)
    return rep.bootstrap(cols, DETS, **args)


def test_bootstrap_point_is_full_sample_ratio_and_deterministic():
    cols = {"no_mm": {d: items() for d in DETS}, "mm": {d: items(bonus=3) for d in DETS}}
    a, b = run(cols), run(cols)
    assert a == b                                                     # seed co dinh
    est = a["estimates"]["yolo11"]["overall"]["mm"]["correct_prediction_ratio"]
    assert est["point"] == (2 + 3 + 2 + 3 + 3 + 3 + 4 + 4) / 60      # cong count tren kich ban da khoa
    assert est["ci_low"] <= est["point"] <= est["ci_high"] and est["n_valid"] == 300
    pooled = a["estimates"]["pooled"]["d5"]["no_mm"]["correct_prediction_ratio"]["point"]
    assert pooled == (2 + 3 + 4) * 2 / 60
    assert a["sequences"] == ["s1", "s2", "s3"] and "KHONG phai kiem dinh" in a["note"]


def test_paired_difference_uses_same_sample():
    cols = {"no_mm": {d: items() for d in DETS}, "same": {d: items() for d in DETS},
            "better": {d: items(bonus=1) for d in DETS}}
    res = run(cols)
    same = res["paired_differences"]["yolo26"]["overall"]["same - no_mm"]["correct_prediction_ratio"]
    assert same["point"] == same["ci_low"] == same["ci_high"] == 0.0      # cung mau -> hieu dung 0
    better = res["paired_differences"]["yolo26"]["overall"]["better - no_mm"]["correct_prediction_ratio"]
    assert better["ci_low"] >= 0.0 and better["point"] > 0                 # hieu khong am tren MOI mau


def test_bootstrap_refuses_different_locked_sets_and_bad_args():
    other = items()
    other[0]["metrics"]["target_locked_at_occlusion"] = False
    with pytest.raises(rep.ReportError, match="khong cung tap"):
        run({"no_mm": {d: items() for d in DETS}, "mm": {"yolo11": other, "yolo26": items()}})
    with pytest.raises(rep.ReportError, match="tham chieu"):
        run({"mm": {d: items() for d in DETS}})
    with pytest.raises(rep.ReportError, match="ho tro"):
        run({"no_mm": {d: items() for d in DETS}}, metrics=["mean_position_error_px"])
