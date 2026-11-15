"""Test src/motion_memory_tuning.py (ham thuan) bang metrics gia."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import motion_memory_tuning as mt  # noqa: E402
from src.motion_memory import MotionMemoryConfig  # noqa: E402
from dataclasses import asdict  # noqa: E402

DETS = ("yolo11", "yolo26")
RULES_DICT = {
    "version": 1,
    "name": "test",
    "group": "dev",
    "detectors": list(DETS),
    "metrics_config_version": 2,
    "source_tag": "v2_001",
    "grid": {
        "velocity_decay": [0.90, 0.95, 0.98],
        "lost_timeout_frames": [10, 15, 25],
        "match_iou": [0.2, 0.3, 0.4],
        "reid_timeout_frames": [45, 75],
    },
    "expected_defaults": {"velocity_decay": 0.95, "lost_timeout_frames": 15, "match_iou": 0.3,
                          "reid_timeout_frames": 45},
    "selection": {"min_improvement": 0.01},
    "max_rounds": 2,
}
R1_IDS = [
    "default",
    "r1_velocity_decay_0p9", "r1_velocity_decay_0p98",
    "r1_lost_timeout_frames_10", "r1_lost_timeout_frames_25",
    "r1_match_iou_0p2", "r1_match_iou_0p4",
    "r1_reid_timeout_frames_75",
]


def rules(**changes) -> mt.TuningRules:
    data = copy.deepcopy(RULES_DICT)
    data.update(changes)
    return mt.TuningRules.from_mapping(data)


def overall(correct=100, occ=1000, reid=10, n=20, err=5.0, with_output=800) -> dict:
    return {"scenarios_evaluated": n, "occlusion_frames": occ, "correct_occluded_frames": correct,
            "reidentified": reid, "occluded_frames_with_output": with_output, "mean_position_error_px": err}


def entry(complete=True, **per_det) -> dict:
    return {"complete": complete, "detectors": {d: overall(**per_det.get(d, {})) for d in DETS}}


def all_default_results() -> dict:
    """Moi cau hinh vong 1 giong het mac dinh."""
    return {c["id"]: entry() for c in mt.round1_configs(rules())}


# ---------------- config + luoi ----------------

def test_round1_is_default_plus_one_change_per_config_in_fixed_order():
    cfgs = mt.round1_configs(rules())
    assert [c["id"] for c in cfgs] == R1_IDS
    non_default = [c for c in cfgs if c["id"] != mt.DEFAULT_ID]
    # 2 (velocity_decay) + 2 (lost_timeout_frames) + 2 (match_iou) + 1 (reid_timeout_frames)
    assert len(non_default) == 7
    for c in non_default:
        assert len(c["overrides"]) == 1 and c["overrides"][c["param"]] == c["value"]
        assert c["value"] != rules().defaults[c["param"]]


def test_rules_reject_unknown_param_and_missing_default():
    with pytest.raises(mt.TuningError, match="khong co trong MotionMemoryConfig"):
        rules(grid={**RULES_DICT["grid"], "lost_frames": [10, 15]},
              expected_defaults={**RULES_DICT["expected_defaults"], "lost_frames": 15})
    with pytest.raises(mt.TuningError, match="chua gia tri mac dinh"):
        rules(expected_defaults={**RULES_DICT["expected_defaults"], "match_iou": 0.35})
    with pytest.raises(mt.TuningError, match="so nguyen"):
        rules(grid={**RULES_DICT["grid"], "lost_timeout_frames": [10, 15.5]})
    with pytest.raises(mt.TuningError, match="vong 3"):
        rules(max_rounds=3)
    with pytest.raises(mt.TuningError, match="dung cac tham so"):
        rules(expected_defaults={"velocity_decay": 0.95})


def test_check_defaults_against_motion_memory_yaml_values():
    base = asdict(MotionMemoryConfig())
    mt.check_defaults(rules(), base)
    with pytest.raises(mt.TuningError, match="expected_defaults"):
        mt.check_defaults(rules(), {**base, "velocity_decay": 0.9})


def test_config_values_validated_by_motion_memory_config():
    base = asdict(MotionMemoryConfig())
    assert mt.config_values(base, {"match_iou": 0.4})["match_iou"] == 0.4
    with pytest.raises(mt.TuningError, match="khong hop le"):
        mt.config_values(base, {"match_iou": 0.0})
    with pytest.raises(mt.TuningError, match="khong ton tai"):
        mt.config_values(base, {"lost_frames": 3})


def test_value_token():
    assert mt.value_token(0.9) == "0p9" and mt.value_token(25) == "25" and mt.value_token(0.98) == "0p98"


# ---------------- diem + hop le ----------------

def test_score_pools_numerators_and_denominators_across_detectors():
    e = {"complete": True, "detectors": {"yolo11": overall(correct=10, occ=100),
                                         "yolo26": overall(correct=30, occ=50)}}
    assert mt.score(rules(), e) == mt.Fraction(40, 150)          # khong phai (0.1 + 0.6) / 2


def test_one_detector_better_other_worse_is_invalid():
    default = entry()
    cfg = entry(yolo11={"correct": 200}, yolo26={"correct": 90})  # diem gop tang manh nhung yolo26 xau di
    a = mt.assess(rules(), "x", cfg, default)
    assert a["delta_vs_default"] > 0.01 and not a["valid"]
    assert a["validity"]["per_detector"]["yolo26"]["cpr_better"] is False


def test_reid_rate_drop_makes_invalid_and_equal_cpr_is_not_better():
    default = entry()
    worse_reid = entry(yolo11={"correct": 150, "reid": 9}, yolo26={"correct": 150})
    assert not mt.assess(rules(), "x", worse_reid, default)["valid"]
    same_cpr = entry(yolo11={"correct": 150})                    # yolo26 bang mac dinh -> khong "hon"
    assert not mt.assess(rules(), "x", same_cpr, default)["valid"]
    ok = entry(yolo11={"correct": 101}, yolo26={"correct": 101, "reid": 10})
    assert mt.assess(rules(), "x", ok, default)["valid"]


def test_incomplete_or_mismatched_results_refused():
    default = entry()
    with pytest.raises(mt.TuningError, match="partial"):
        mt.assess(rules(), "x", entry(complete=False), default)
    with pytest.raises(mt.TuningError, match="khong cung tap kich ban"):
        mt.assess(rules(), "x", entry(yolo11={"n": 19}), default)
    with pytest.raises(mt.TuningError, match="detector"):
        mt.assess(rules(), "x", {"complete": True, "detectors": {"yolo11": overall()}}, default)
    with pytest.raises(mt.TuningError, match="thieu"):
        bad = entry()
        del bad["detectors"]["yolo11"]["reidentified"]
        mt.assess(rules(), "x", bad, default)


def test_round1_refuses_when_a_config_is_missing_or_partial():
    res = all_default_results()
    del res["r1_match_iou_0p4"]
    with pytest.raises(mt.TuningError, match="Chua co ket qua"):
        mt.round1_winners(rules(), res)
    res = all_default_results()
    res["default"]["complete"] = False
    with pytest.raises(mt.TuningError, match="partial"):
        mt.round1_winners(rules(), res)


# ---------------- chon gia tri thang + cau hinh cuoi ----------------

def test_no_valid_config_keeps_default_and_round2_needs_no_run():
    res = all_default_results()
    res["r1_velocity_decay_0p9"] = entry(yolo11={"correct": 300}, yolo26={"correct": 99})
    winners = mt.round1_winners(rules(), res)
    assert all(w["value"] is None and w["config_id"] is None for w in winners.values())
    r2 = mt.round2_config(rules(), winners)
    assert r2["overrides"] == {} and r2["same_as"] == mt.DEFAULT_ID
    sel = mt.final_selection(rules(), res, winners)
    assert sel["conclusion"] == mt.CONCLUSION_KEEP_DEFAULT and sel["selected_config_id"] is None
    assert "giu cau hinh mac dinh" in sel["reason"]


def test_valid_beats_higher_scoring_invalid():
    res = all_default_results()
    res["r1_match_iou_0p2"] = entry(yolo11={"correct": 400}, yolo26={"correct": 99})   # khong hop le
    res["r1_match_iou_0p4"] = entry(yolo11={"correct": 120}, yolo26={"correct": 120})
    w = mt.round1_winners(rules(), res)["match_iou"]
    assert w["value"] == 0.4 and w["config_id"] == "r1_match_iou_0p4"


def test_score_tie_broken_by_lower_pooled_error_then_default():
    res = all_default_results()
    res["r1_lost_timeout_frames_10"] = entry(yolo11={"correct": 130, "err": 4.0}, yolo26={"correct": 130, "err": 4.0})
    res["r1_lost_timeout_frames_25"] = entry(yolo11={"correct": 130, "err": 3.0}, yolo26={"correct": 130, "err": 3.0})
    w = mt.round1_winners(rules(), res)["lost_timeout_frames"]
    assert w["value"] == 25 and "sai so" in w["reason"]

    res["r1_lost_timeout_frames_25"] = entry(yolo11={"correct": 130, "err": 4.0}, yolo26={"correct": 130, "err": 4.0})
    w = mt.round1_winners(rules(), res)["lost_timeout_frames"]
    assert w["value"] is None and "giu mac dinh" in w["reason"]


def test_improvement_below_margin_keeps_default_exact_margin_selected():
    res = all_default_results()   # mac dinh: 200/2000 = 0.10
    res["r1_velocity_decay_0p98"] = entry(yolo11={"correct": 109}, yolo26={"correct": 110})  # +0.0095
    winners = mt.round1_winners(rules(), res)
    assert winners["velocity_decay"]["value"] == 0.98            # hop le -> van la gia tri thang vong 1
    sel = mt.final_selection(rules(), res, winners)              # nhung duoi bien 0.01
    assert sel["conclusion"] == mt.CONCLUSION_KEEP_DEFAULT and sel["eligible"] == []

    res["r1_velocity_decay_0p98"] = entry(yolo11={"correct": 110}, yolo26={"correct": 110})  # dung +0.01
    winners = mt.round1_winners(rules(), res)
    sel = mt.final_selection(rules(), res, winners)
    assert sel["conclusion"] == mt.CONCLUSION_TUNED
    assert sel["selected_config_id"] == "r1_velocity_decay_0p98" and sel["overrides"] == {"velocity_decay": 0.98}
    assert sel["round2"]["same_as"] == "r1_velocity_decay_0p98"   # mot tham so thang -> vong 2 khong chay


def test_round2_combined_required_then_selected_when_best():
    res = all_default_results()
    res["r1_velocity_decay_0p9"] = entry(yolo11={"correct": 120}, yolo26={"correct": 120})
    res["r1_reid_timeout_frames_75"] = entry(yolo11={"correct": 115, "reid": 12}, yolo26={"correct": 115})
    winners = mt.round1_winners(rules(), res)
    r2 = mt.round2_config(rules(), winners)
    assert r2["same_as"] is None
    assert r2["overrides"] == {"velocity_decay": 0.9, "reid_timeout_frames": 75}
    with pytest.raises(mt.TuningError, match="vong 2"):
        mt.final_selection(rules(), res, winners)

    res[mt.ROUND2_ID] = entry(yolo11={"correct": 130}, yolo26={"correct": 125})
    sel = mt.final_selection(rules(), res, winners)
    assert sel["selected_config_id"] == mt.ROUND2_ID
    assert sel["overrides"] == {"velocity_decay": 0.9, "reid_timeout_frames": 75}

    res[mt.ROUND2_ID] = entry(yolo11={"correct": 130}, yolo26={"correct": 99})    # vong 2 khong hop le
    sel = mt.final_selection(rules(), res, winners)
    assert sel["selected_config_id"] == "r1_velocity_decay_0p9"
    assert [c["config_id"] for c in sel["candidates"]] == ["r1_velocity_decay_0p9", "r1_reid_timeout_frames_75",
                                                           mt.ROUND2_ID]


def test_final_tie_between_round1_and_round2_uses_error_then_default():
    res = all_default_results()
    res["r1_velocity_decay_0p9"] = entry(yolo11={"correct": 120, "err": 5.0}, yolo26={"correct": 120, "err": 5.0})
    res["r1_match_iou_0p4"] = entry(yolo11={"correct": 115}, yolo26={"correct": 115})
    winners = mt.round1_winners(rules(), res)
    res[mt.ROUND2_ID] = entry(yolo11={"correct": 120, "err": 5.0}, yolo26={"correct": 120, "err": 5.0})
    sel = mt.final_selection(rules(), res, winners)
    assert sel["conclusion"] == mt.CONCLUSION_KEEP_DEFAULT and "hoa ca diem va sai so" in sel["reason"]


# ---------------- config that cua repo ----------------

def test_repo_tuning_config_loads_and_matches_motion_memory_yaml():
    import yaml

    from src.utils.motion_memory_settings import resolve_motion_memory_config

    data = yaml.safe_load((REPO_ROOT / "configs" / "motion_memory_tuning_dev.yaml").read_text(encoding="utf-8"))
    r = mt.TuningRules.from_mapping(data)
    assert r.group == "dev" and r.detectors == DETS and r.metrics_config_version == 2
    assert r.min_improvement == mt.Fraction(1, 100)
    base = asdict(resolve_motion_memory_config(None, REPO_ROOT / data["base_motion_memory_config"]).config)
    mt.check_defaults(r, base)
    assert [c["id"] for c in mt.round1_configs(r)] == R1_IDS
    for c in mt.round1_configs(r):
        mt.config_values(base, c["overrides"])          # moi cau hinh hop le voi MotionMemoryConfig
