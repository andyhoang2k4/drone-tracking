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
