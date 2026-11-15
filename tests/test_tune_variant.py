"""Test che do --variant cua scripts/tune_motion_memory.py (bo chay GIA, khong YOLO).

Nguon: ablation v2_001 + tuning vong 1 (tham chieu tune_default, tune_r1_lost_timeout_frames_25)
dung bang chinh cac script tren kich ban v2 gia. File khai bao la BAN SAO cua
configs/motion_memory_anchor_dev.yaml, chi doi: durations cua quy tac (i) -> [5] (kich ban gia
chi co d=5) va check_file -> file tuned tam (file that chi co tren cloud).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.tune_motion_memory as tune  # noqa: E402
from tests.test_run_occlusion_ablation import V2_CASES, FakeRunner  # noqa: E402
from tests.test_tune_motion_memory import run_source, tune_env  # noqa: E402,F401

DETS = ("yolo11", "yolo26")
PLAN = REPO_ROOT / "configs" / "motion_memory_anchor_dev.yaml"


def write_plan(tmp: Path, tuned_lost: int = 25) -> Path:
    data = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    data["decision"]["reid_not_below"][0]["durations"] = [5]
    tuned = tmp / "motion_memory_tuned.yaml"
    tuned.write_text(f"motion_memory:\n  lost_timeout_frames: {tuned_lost}\n", encoding="utf-8")
    data["variants"][2]["check_file"] = str(tuned)
    path = tmp / "plan.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def variant_env(tune_env):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    assert tune.main(base + ["--round", "1"], FakeRunner()) == 0      # tham chieu tune_default, ..._25
    return tmp, base


def summary(tmp: Path, name: str = "motion_memory_anchor_dev") -> dict:
    return json.loads((tmp / "experiments" / name / "summary.json").read_text())


def test_dry_run_plans_three_variants_writes_nothing(variant_env, capsys):
    tmp, base = variant_env
    plan = write_plan(tmp)
    runner = FakeRunner()
    assert tune.main(base + ["--variant", str(plan), "--dry-run"], runner) == 0
    out = capsys.readouterr().out
    assert f"run_pipeline {3 * len(DETS) * len(V2_CASES)}" in out
    for vid in ("default_regression", "anchor", "anchor_lost25"):
        assert f"yolo11_botsort_mm_occl_dev_anchor_v1_{vid}" in out
    assert runner.calls == [] and not (tmp / "experiments" / "motion_memory_anchor_dev").exists()


def test_variant_run_regression_ok_then_decision(variant_env, capsys):
    tmp, base = variant_env
    plan = write_plan(tmp)
    code = tune.main(base + ["--variant", str(plan)], FakeRunner())
    s = summary(tmp)
    assert s["status"] == "complete" and code == 0
    assert s["regression"]["ok"] is True                                   # cung code path, co tat -> trung
    assert s["plan"]["version"] == 1 and len(s["plan"]["sha256"]) == 64
    assert set(s["decision"]["per_detector"]) == set(DETS)
    assert s["decision"]["passed"] in (True, False) and s["decision"]["conclusion"]
    assert s["check_files"]["anchor_lost25"]["sha256"]
    assert [v["id"] for v in s["variants"]] == ["default_regression", "anchor", "anchor_lost25"]
    for col in ("ref:no_mm", "ref:mm_default", "ref:mm_tuned", "variant:anchor", "variant:anchor_lost25"):
        assert s["descriptive"][col]["yolo11"]["overall"]["scenarios_evaluated"] == 2
    assert s["recover_reason_scenarios"]["variant:anchor"]["yolo26"]["overall"]["scenarios"] == 2
    mm = yaml.safe_load((tmp / "experiments" / "motion_memory_anchor_dev" / "mm_configs" / "anchor.yaml")
                        .read_text())["motion_memory"]
    assert mm["anchor_recovery"] is True and mm["lost_timeout_frames"] == 15
    mm25 = yaml.safe_load((tmp / "experiments" / "motion_memory_anchor_dev" / "mm_configs" / "anchor_lost25.yaml")
                          .read_text())["motion_memory"]
    assert mm25["anchor_recovery"] is True and mm25["lost_timeout_frames"] == 25
    reg = yaml.safe_load((tmp / "experiments" / "motion_memory_anchor_dev" / "mm_configs"
                          / "default_regression.yaml").read_text())["motion_memory"]
    assert reg["anchor_recovery"] is False
    assert "KET LUAN" in capsys.readouterr().out


def test_regression_mismatch_blocks_decision(variant_env, capsys):
    tmp, base = variant_env
    plan = write_plan(tmp)
    ref = tmp / "experiments" / "yolo26_botsort_mm_occl_dev_tune_default" / "metrics.json"
    data = json.loads(ref.read_text())
    data["by_duration_frames"]["5"]["correct_occluded_frames"] += 1     # gia lap lech hoi quy
    ref.write_text(json.dumps(data))
    assert tune.main(base + ["--variant", str(plan)], FakeRunner()) == 1
    s = summary(tmp)
    assert s["status"] == "blocked_by_regression" and s["decision"] is None
    assert any("yolo26.by_duration_frames.5.correct_occluded_frames" in d for d in s["regression"]["differences"])
    assert "LOI HOI QUY" in capsys.readouterr().out


def test_check_file_mismatch_refused_before_running(variant_env):
    tmp, base = variant_env
    plan = write_plan(tmp, tuned_lost=30)
    runner = FakeRunner()
    with pytest.raises(SystemExit, match="anchor_lost25"):
        tune.main(base + ["--variant", str(plan)], runner)
    assert runner.calls == []


def test_limit_run_is_partial_with_numbered_tag(variant_env, capsys):
    tmp, base = variant_env
    plan = write_plan(tmp)
    assert tune.main(base + ["--variant", str(plan), "--limit-scenarios", "1"], FakeRunner()) == 0
    s = summary(tmp, "motion_memory_anchor_dev_anchor_v1_limit1_run1")
    assert s["partial"] is True and s["status"] == "not_available" and s["decision"] is None
    assert not (tmp / "experiments" / "motion_memory_anchor_dev").exists()
    with pytest.raises(SystemExit):
        tune.parse_args(base + ["--variant", str(plan), "--limit-scenarios", "1", "--tag", "anchor_v1"])
