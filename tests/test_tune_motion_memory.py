"""Test scripts/tune_motion_memory.py voi bo chay GIA (khong YOLO, khong subprocess).

Ablation nguon v2_001 duoc dung bang scripts/run_occlusion_ablation.py tren
kich ban v2 gia (cung helper voi tests/test_run_occlusion_ablation.py), roi
tuning chi replay log tracker do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
import scripts.run_pipeline as rp  # noqa: E402
import scripts.tune_motion_memory as tune  # noqa: E402
import src.track as tr  # noqa: E402
from src import motion_memory_tuning as mt  # noqa: E402
from tests.test_run_occlusion_ablation import V2_CASES, FakeRunner, fake_tracks_v2, make_group_v2  # noqa: E402

N_R1 = 8          # default + 7 cau hinh
DETS = ("yolo11", "yolo26")
PREFLIGHT = len(DETS)   # moi vong: 1 lenh replay that moi detector truoc khi chay


@pytest.fixture
def tune_env(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks_v2(s)))

    def no_yolo(*a, **k):
        raise AssertionError("khong duoc chay YOLO khi replay")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    weights = {}
    for name in DETS:
        weights[name] = tmp_path / f"{name}.pt"
        weights[name].write_bytes(f"weights-{name}".encode())
    scn, exps = tmp_path / "scn", tmp_path / "experiments"
    make_group_v2(scn, weights)
    base = ["--scenario-root", str(scn), "--output-root", str(exps), "--tuned-output", str(tmp_path / "tuned.yaml")]
    return tmp_path, base, weights


def run_source(tmp: Path, weights: dict) -> None:
    args = ["--group", "dev", "--tag", "v2_001", "--scenario-root", str(tmp / "scn"),
            "--output-root", str(tmp / "experiments"),
            "--detector", f"yolo11={weights['yolo11']}", "--detector", f"yolo26={weights['yolo26']}"]
    assert abl.main(args, FakeRunner()) == 0


def summary(tmp: Path, name: str = "motion_memory_tuning_dev") -> dict:
    return json.loads((tmp / "experiments" / name / "summary.json").read_text())


def test_refuses_when_source_ablation_missing(tune_env):
    tmp, base, _ = tune_env
    with pytest.raises(SystemExit, match="nguon chua san sang"):
        tune.main(base + ["--round", "1", "--dry-run"], FakeRunner())


def test_dry_run_plans_round1_and_writes_nothing(tune_env, capsys):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    before = sorted(p.name for p in (tmp / "experiments").iterdir())
    runner = FakeRunner()
    assert tune.main(base + ["--round", "1", "--dry-run", "--jobs", "3"], runner) == 0
    out = capsys.readouterr().out
    n = N_R1 * len(DETS) * len(V2_CASES)
    assert f"run_pipeline {n}" in out and f"evaluate occlusion {n}" in out
    assert f"evaluate aggregate {N_R1 * len(DETS)}" in out
    assert f"evaluate compare {N_R1 * len(DETS)}" in out            # moi cau hinh vs baseline nguon
    assert "yolo11_botsort_mm_occl_dev_tune_r1_match_iou_0p4" in out
    assert runner.calls == []
    assert sorted(p.name for p in (tmp / "experiments").iterdir()) == before


def test_round2_refused_before_round1(tune_env):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    with pytest.raises(SystemExit, match="Vong 1 chua du"):
        tune.main(base + ["--round", "2"], FakeRunner())


def test_full_flow_round1_resume_round2_select(tune_env, capsys):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    assert tune.main(base + ["--round", "1", "--jobs", "4"], FakeRunner()) == 0
    exps = tmp / "experiments"
    s = summary(tmp)
    assert s["partial"] is False and s["round1"]["status"] == "complete"
    assert s["tuning_config"]["version"] == 1 and len(s["tuning_config"]["sha256"]) == 64
    assert s["metrics_config"]["version"] == 2
    assert [c["id"] for c in s["configs"]][:N_R1] == [c["id"] for c in mt.round1_configs(tune.load_tuning(
        tune.DEFAULT_TUNING_CONFIG)[0])]
    for det in DETS:
        assert s["source"]["tracks"][det]["logs"] == len(V2_CASES)
        assert len(s["source"]["tracks"][det]["combined_sha256"]) == 64
        assert s["default_reproduces_source_motion_memory"][det] is True   # default chay lai == MM v2_001
    for c in s["configs"][:N_R1]:
        assert c["mm_config_sha256"] and (exps / "motion_memory_tuning_dev" / c["mm_config_file"]).is_file()
        for det in DETS:
            d = c["detectors"][det]
            assert d["complete"] and d["overall"]["scenarios_evaluated"] == 2   # unseen chua khoa
            assert "per_sequence" in d["overall"] and d["compare_vs_source_baseline"]
        assert "score" in c["assessment"]
    import yaml

    conf = yaml.safe_load((exps / "yolo26_botsort_mm_occl_dev_tune_r1_lost_timeout_frames_25" / "config.yaml")
                          .read_text())
    assert conf["tuning"]["overrides"] == {"lost_timeout_frames": 25}
    assert conf["tuning"]["config"]["sha256"] == s["tuning_config"]["sha256"]
    assert conf["tracker_input"]["tracks_experiment"] == "yolo26_botsort_tracks_occl_dev_v2_001"
    mm = yaml.safe_load((exps / "motion_memory_tuning_dev" / "mm_configs" / "r1_lost_timeout_frames_25.yaml")
                        .read_text())["motion_memory"]
    assert mm["lost_timeout_frames"] == 25 and mm["velocity_decay"] == 0.95

    runner = FakeRunner()                                     # resume: khong replay lai
    assert tune.main(base + ["--round", "1", "--jobs", "2"], runner) == 0
    # resume: chi pre-flight (1 replay that moi detector, thu muc tam), khong replay lai kich ban
    assert runner.count("pipeline") == PREFLIGHT and runner.count("occlusion") == 0

    capsys.readouterr()
    assert tune.main(base + ["--round", "2"], FakeRunner()) == 0
    assert tune.main(base + ["--select"], FakeRunner()) == 0
    s = summary(tmp)
    assert s["selection"]["status"] == "complete"
    # Ket qua phai trung ham thuan tinh lai tu cung du lieu
    assert s["selection"]["conclusion"] in (mt.CONCLUSION_KEEP_DEFAULT, mt.CONCLUSION_TUNED)
    assert (tmp / "tuned.yaml").is_file() == (s["selection"]["conclusion"] == mt.CONCLUSION_TUNED)
    assert s["conclusion"]


def test_round2_runs_combined_and_select_writes_tuned_only_when_chosen(tune_env, monkeypatch, capsys):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    assert tune.main(base + ["--round", "1"], FakeRunner()) == 0
    real_winners = mt.round1_winners

    def fake_winners(rules, results):     # ep hai tham so "thang" de co cau hinh ket hop moi
        w = real_winners(rules, results)
        for p, v in (("velocity_decay", 0.9), ("match_iou", 0.4)):
            w[p] = {**w[p], "value": v, "config_id": mt.round1_config_id(p, v)}
        return w

    monkeypatch.setattr(mt, "round1_winners", fake_winners)
    runner = FakeRunner()
    assert tune.main(base + ["--round", "2"], runner) == 0
    assert runner.count("pipeline") == len(DETS) * len(V2_CASES) + PREFLIGHT
    s = summary(tmp)
    assert s["round2"]["status"] == "complete"
    assert s["round2"]["config"]["overrides"] == {"velocity_decay": 0.9, "match_iou": 0.4}
    assert (tmp / "experiments" / "yolo11_botsort_mm_occl_dev_tune_r2_combined" / "metrics.json").is_file()

    real_final = mt.final_selection

    def chosen(rules, results, winners):
        out = real_final(rules, results, winners)
        out.update({"selected_config_id": mt.ROUND2_ID, "overrides": {"velocity_decay": 0.9, "match_iou": 0.4},
                    "conclusion": mt.CONCLUSION_TUNED, "reason": "ep trong test"})
        return out

    monkeypatch.setattr(mt, "final_selection", chosen)
    assert tune.main(base + ["--select"], FakeRunner()) == 0
    import yaml

    tuned = yaml.safe_load((tmp / "tuned.yaml").read_text())["motion_memory"]
    assert tuned["velocity_decay"] == 0.9 and tuned["match_iou"] == 0.4 and tuned["lost_timeout_frames"] == 15
    assert tune.main(base + ["--select"], FakeRunner()) == 0         # chay lai cung noi dung -> OK
    (tmp / "tuned.yaml").write_text("motion_memory: {}\n")
    with pytest.raises(SystemExit, match="khong ghi de"):
        tune.main(base + ["--select"], FakeRunner())

    monkeypatch.setattr(mt, "final_selection", real_final)
    (tmp / "tuned.yaml").unlink()
    assert tune.main(base + ["--select"], FakeRunner()) == 0
    if summary(tmp)["selection"]["conclusion"] == mt.CONCLUSION_KEEP_DEFAULT:
        assert not (tmp / "tuned.yaml").exists()


def test_never_overwrites_default_motion_memory_yaml(tune_env):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    args = [a if a != str(tmp / "tuned.yaml") else str(tune.PROTECTED_MM_CONFIG) for a in base]
    ctx = tune.build_context(tune.parse_args(args + ["--select"]))
    with pytest.raises(SystemExit, match="Khong ghi de"):
        tune.write_tuned_config(ctx, {"selection": {"conclusion": mt.CONCLUSION_TUNED, "selected_config_id": "x",
                                                    "overrides": {"match_iou": 0.4}}})


def test_limit_scenarios_is_partial_separate_and_cannot_select(tune_env, capsys):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    runner = FakeRunner()
    assert tune.main(base + ["--round", "1", "--limit-scenarios", "1"], runner) == 0
    assert runner.count("compare") == 0                      # khong cung tap kich ban voi baseline -> khong so sanh
    assert runner.count("pipeline") == N_R1 * len(DETS) + PREFLIGHT
    s = summary(tmp, "motion_memory_tuning_dev_tune_limit1")
    assert s["partial"] is True and s["round1"]["status"] == "incomplete"
    assert s["selection"]["status"] == "not_available" and s["conclusion"] is None
    assert (tmp / "experiments" / "yolo11_botsort_mm_occl_dev_tune_limit1_default" / "metrics.json").is_file()
    assert not (tmp / "experiments" / "motion_memory_tuning_dev").exists()
    for extra in (["--round", "2"], ["--select"]):
        with pytest.raises(SystemExit):
            tune.main(base + extra + ["--limit-scenarios", "1"], FakeRunner())
    with pytest.raises(SystemExit):
        tune.main(base + ["--round", "1", "--limit-scenarios", "1", "--tag", "tune"], FakeRunner())


def test_changed_defaults_or_rules_refused(tune_env, tmp_path):
    tmp, base, weights = tune_env
    run_source(tmp, weights)
    text = tune.DEFAULT_TUNING_CONFIG.read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace("  velocity_decay: 0.95\n", "  velocity_decay: 0.98\n"), encoding="utf-8")
    with pytest.raises(mt.TuningError, match="expected_defaults"):
        tune.main(base + ["--round", "1", "--dry-run", "--config", str(bad)], FakeRunner())

    assert tune.main(base + ["--round", "1", "--limit-scenarios", "1"], FakeRunner()) == 0
    edited = tmp_path / "edited.yaml"
    edited.write_text(text + "# sua sau khi chay\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cau hinh khac"):
        tune.main(base + ["--round", "1", "--limit-scenarios", "1", "--config", str(edited)], FakeRunner())


def test_default_jobs_leaves_two_cores(monkeypatch):
    monkeypatch.setattr(tune.os, "cpu_count", lambda: 18)
    assert tune.default_jobs() == 16
    monkeypatch.setattr(tune.os, "cpu_count", lambda: 2)
    assert tune.default_jobs() == 1
    with pytest.raises(SystemExit):
        tune.parse_args(["--round", "1", "--jobs", "0"])


def test_subprocess_env_single_thread_and_no_gpu():
    env = tune.subprocess_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == "1" and env["MKL_NUM_THREADS"] == "1"
