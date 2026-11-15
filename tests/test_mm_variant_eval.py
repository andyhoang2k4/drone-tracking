"""Test src/mm_variant_eval.py (ham thuan) va file khai bao configs/motion_memory_anchor_dev.yaml."""

from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import mm_variant_eval as ve  # noqa: E402
from src.motion_memory import MotionMemoryConfig  # noqa: E402

PLAN_PATH = REPO_ROOT / "configs" / "motion_memory_anchor_dev.yaml"
DETS = ("yolo11", "yolo26")


def plan_dict() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))


def plan(**changes) -> ve.VariantPlan:
    data = plan_dict()
    data.update(changes)
    return ve.VariantPlan.from_mapping(data)


# ---------------- file khai bao that ----------------

def test_repo_plan_loads_with_fixed_variants_and_rules():
    from src.utils.motion_memory_settings import resolve_motion_memory_config

    p = plan()
    assert p.version == 1 and p.group == "dev" and p.detectors == DETS and p.tag == "anchor_v1"
    assert [(v.id, v.role) for v in p.variants] == [
        ("default_regression", "regression"), ("anchor", "decision"), ("anchor_lost25", "descriptive")]
    assert p.variant("default_regression").overrides == {}
    assert p.variant("anchor").overrides == {"anchor_recovery": True}
    assert p.variant("anchor_lost25").overrides == {"lost_timeout_frames": 25, "anchor_recovery": True}
    assert p.variant("anchor_lost25").check_file == "configs/motion_memory_tuned.yaml"
    assert p.decision["reid_not_below"] == [{"reference": "no_mm", "durations": [15, 30]},
                                            {"reference": "mm_default", "durations": [5]}]
    assert p.decision["cpr_max_drop"] == {"reference": "mm_default", "value": "0.01"}
    assert p.regression == {"variant": "default_regression", "reference": "mm_default",
                            "sections": ["overall", "by_duration_frames"]}
    assert p.reference_experiment("no_mm", "yolo26") == "yolo26_botsort_occl_dev_v2_001"
    assert p.reference_experiment("mm_default", "yolo11") == "yolo11_botsort_mm_occl_dev_tune_default"
    base = asdict(resolve_motion_memory_config(None, REPO_ROOT / p.base_motion_memory_config).config)
    ve.check_defaults(p, base)
    for v in p.variants:
        MotionMemoryConfig(**{**base, **v.overrides})          # moi bien the hop le


def test_plan_validation_errors():
    with pytest.raises(ve.VariantPlanError, match="khong co trong MotionMemoryConfig"):
        data = plan_dict()
        data["variants"][1]["overrides"] = {"anchor": True}
        ve.VariantPlan.from_mapping(data)
    with pytest.raises(ve.VariantPlanError, match="sai kieu"):
        data = plan_dict()
        data["variants"][1]["overrides"] = {"anchor_recovery": 1}
        ve.VariantPlan.from_mapping(data)
    with pytest.raises(ve.VariantPlanError, match="role=decision"):
        data = plan_dict()
        data["variants"][2]["role"] = "decision"
        ve.VariantPlan.from_mapping(data)
    with pytest.raises(ve.VariantPlanError, match="trung"):
        data = plan_dict()
        data["variants"][2]["id"] = "anchor"
        ve.VariantPlan.from_mapping(data)
    with pytest.raises(ve.VariantPlanError, match="reid_not_below"):
        data = plan_dict()
        data["decision"]["reid_not_below"][0]["reference"] = "khong_co"
        ve.VariantPlan.from_mapping(data)
    with pytest.raises(ve.VariantPlanError, match="version"):
        plan(version=2)


def test_check_defaults_and_check_file():
    p = plan()
    base = asdict(MotionMemoryConfig())
    with pytest.raises(ve.VariantPlanError, match="expected_defaults"):
        ve.check_defaults(p, {**base, "lost_timeout_frames": 25})
    v = p.variant("anchor_lost25")
    variant_values = {**base, **v.overrides}
    tuned = {**base, "lost_timeout_frames": 25}
    ve.check_file_values(v, variant_values, tuned)
    with pytest.raises(ve.VariantPlanError, match="khac configs/motion_memory_tuned.yaml"):
        ve.check_file_values(v, variant_values, {**tuned, "reid_timeout_frames": 60})


# ---------------- hoi quy / quyet dinh / mo ta (du lieu gia) ----------------

def pool(reid=10, n=20, correct=100, occ=1000, out=400, err=5.0) -> dict:
    return {"scenarios_evaluated": n, "reidentified": reid, "correct_occluded_frames": correct,
            "occlusion_frames": occ, "occluded_frames_with_output": out, "mean_position_error_px": err,
            "reidentified_with_new_track_id": 0, "mean_time_to_reidentify_frames": 3.0,
            "longest_tracking_gap_max": 40, "longest_tracking_gap_mean": 20.0}


def agg(complete=True, overall=None, **by_d) -> dict:
    durs = {str(d): pool() for d in (5, 15, 30, 60)}
    for k, v in by_d.items():
        durs[k[1:]] = pool(**v)
    return {"overall": pool(**(overall or {})), "by_duration_frames": durs,
            "completeness": {"complete": complete}}


def both(**kw) -> dict:
    return {d: agg(**kw) for d in DETS}


def refs(no_mm=None, mm_default=None) -> dict:
    return {"no_mm": no_mm or both(), "mm_default": mm_default or both(), "mm_tuned": both()}


def test_decide_passes_when_all_conditions_hold_on_both_detectors():
    res = ve.decide(plan(), both(), refs())
    assert res["passed"] and res["variant"] == "anchor"
    conds = [c["condition"] for c in res["per_detector"]["yolo11"]["conditions"]]
    assert conds == ["reid(d15) >= no_mm", "reid(d30) >= no_mm", "reid(d5) >= mm_default",
                     "cpr(overall) >= mm_default - 0.01"]


def test_decide_reid_d15_below_no_mm_fails():
    res = ve.decide(plan(), both(d15={"reid": 9}), refs())
    assert not res["passed"] and "KHONG DAT" in res["conclusion"]
    assert [c["ok"] for c in res["per_detector"]["yolo26"]["conditions"]] == [False, True, True, True]


def test_decide_cpr_drop_boundary_exact_fraction():
    ok = {d: agg(overall={"correct": 90}) for d in DETS}             # 0.09 vs 0.10: drop dung 0.01 -> dat
    assert ve.decide(plan(), ok, refs())["passed"]
    bad = {d: agg(overall={"correct": 899, "occ": 10000}) for d in DETS}
    ref_mm = {d: agg(overall={"correct": 1000, "occ": 10000}) for d in DETS}
    assert not ve.decide(plan(), bad, refs(mm_default=ref_mm))["passed"]   # drop 0.0101


def test_decide_d5_reid_below_mm_default_fails_and_one_detector_enough_to_fail():
    variant = {"yolo11": agg(), "yolo26": agg(d5={"reid": 9})}
    res = ve.decide(plan(), variant, refs())
    assert res["per_detector"]["yolo11"]["passed"] and not res["per_detector"]["yolo26"]["passed"]
    assert not res["passed"]


def test_decide_refuses_partial_and_different_locked_sets():
    with pytest.raises(ve.VariantPlanError, match="partial"):
        ve.decide(plan(), both(complete=False), refs())
    with pytest.raises(ve.VariantPlanError, match="partial"):
        ve.decide(plan(), both(), refs(no_mm={"yolo11": agg(), "yolo26": agg(complete=False)}))
    with pytest.raises(ve.VariantPlanError, match="khong cung tap"):
        ve.decide(plan(), both(d15={"n": 19}), refs())


def test_regression_check_exact_equality_with_paths():
    p = plan()
    assert ve.regression_check(p, both(), both())["ok"]
    res = ve.regression_check(p, both(), {"yolo11": agg(), "yolo26": agg(d30={"correct": 101})})
    assert not res["ok"] and res["differences"] == [
        "yolo26.by_duration_frames.30.correct_occluded_frames: 100 != 101"]
    assert not ve.regression_check(p, both(), {"yolo11": agg()})["ok"]           # thieu detector


def test_box_precision_and_describe():
    assert ve.box_precision(pool(correct=100, out=400)) == 0.25
    assert ve.box_precision(pool(correct=0, out=0)) is None
    d = ve.describe(both(), DETS)
    assert d["yolo11"]["overall"]["box_precision"] == 0.25
    assert list(d["yolo11"]["by_duration_frames"]) == ["5", "15", "30", "60"]
    assert d["yolo11"]["overall"]["reidentification_rate"] == 0.5
