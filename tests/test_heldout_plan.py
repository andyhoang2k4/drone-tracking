"""Test src/heldout_plan.py va file ke hoach that configs/heldout_report_plan.yaml."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.heldout_plan import HeldoutPlan, HeldoutPlanError  # noqa: E402

PLAN_PATH = REPO_ROOT / "configs" / "heldout_report_plan.yaml"


def plan_dict() -> dict:
    return yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))


def test_repo_plan_columns_and_hashes_match_files():
    from src.occlusion_scenarios import config_sha256

    p = HeldoutPlan.from_mapping(plan_dict())
    assert p.group == "heldout" and p.tag == "v2_001" and p.detectors == ("yolo11", "yolo26")
    assert [(c.name, c.kind, c.role) for c in p.columns] == [
        ("no_mm", "baseline", "reference"), ("mm_default", "motion_memory", "reference"),
        ("mm_tuned", "extra", "descriptive"), ("mm_anchor", "extra", "decision"),
        ("mm_anchor_lost25", "extra", "descriptive")]
    assert p.column("mm_tuned").sha256 == "80cd5182f4f66a69d24c91a091c89703e7407cbe472db02dc9bd0a8461eceb8a"
    assert set(p.extras) == {"tuned", "anchor", "anchor_lost25"}
    assert p.decision["column"] == "mm_anchor"
    assert p.decision["reference_columns"] == {"no_mm": "no_mm", "mm_default": "mm_default"}
    assert p.bootstrap == {"cluster": "sequence", "resamples": 2000, "seed": 20261115, "ci": 0.95,
                           "paired_vs": ["no_mm", "mm_default"],
                           "metrics": ["correct_prediction_ratio", "box_precision", "reidentification_rate"]}
    assert p.weights_must_match == ("dev", "heldout")
    for path, sha in p.referenced_files().items():          # moi sha trong plan = file trong repo
        assert config_sha256((REPO_ROOT / path).read_bytes()) == sha, path
    assert p.experiment_id("no_mm", "yolo11") == "yolo11_botsort_occl_heldout_v2_001"
    assert p.experiment_id("mm_default", "yolo11") == "yolo11_botsort_mm_occl_heldout_v2_001"
    assert p.experiment_id("mm_anchor", "yolo26") == "yolo26_botsort_mm_anchor_occl_heldout_v2_001"


def test_rules_from_is_the_anchor_dev_plan_with_three_conditions():
    from src import mm_variant_eval as ve

    p = HeldoutPlan.from_mapping(plan_dict())
    dev = ve.VariantPlan.from_mapping(yaml.safe_load((REPO_ROOT / p.decision["rules_from"]["path"]).read_text()))
    assert dev.decision["variant"] == "anchor"
    assert set(r["reference"] for r in dev.decision["reid_not_below"]) | {dev.decision["cpr_max_drop"]["reference"]} \
        == set(p.decision["reference_columns"])


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(group="dev"), "heldout"),
    (lambda d: d["columns"][2].pop("sha256"), "sha256"),
    (lambda d: d["columns"][3].update(role="descriptive"), "role=decision"),
    (lambda d: d["columns"][4].update(extra_name="anchor"), "trung"),
    (lambda d: d["decision"]["reference_columns"].update(no_mm="khong_co"), "reference_columns"),
    (lambda d: d["bootstrap"].update(paired_vs=["x"]), "paired_vs"),
    (lambda d: d["report"].update(diagnosis_columns=["no_mm"]), "diagnosis_columns"),
    (lambda d: d["columns"].append({"name": "mm2", "kind": "motion_memory", "role": "descriptive",
                                    "config": "a", "sha256": "0" * 64}), "kind=motion_memory"),
])
def test_plan_validation(mutate, match):
    data = copy.deepcopy(plan_dict())
    mutate(data)
    with pytest.raises(HeldoutPlanError, match=match):
        HeldoutPlan.from_mapping(data)
