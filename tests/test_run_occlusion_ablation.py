"""Test scripts/run_occlusion_ablation.py voi bo chay GIA (khong YOLO, khong subprocess).

Bo chay gia goi src.track.main / run_pipeline.main / src.evaluate.main trong
cung tien trinh; src.track.iter_tracks duoc thay bang track gia theo ten kich
ban, con run_pipeline.iter_tracks bi cam (replay khong duoc chay YOLO).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
import scripts.run_pipeline as rp  # noqa: E402
import src.evaluate as ev  # noqa: E402
import src.track as tr  # noqa: E402
from src.motion_memory import BBox  # noqa: E402

N, OCC_LO, OCC_HI, GT_ID = 20, 6, 10, 42
CASES = ("new_id", "never", "same_id")
METRICS_V1 = REPO_ROOT / "configs" / "occlusion_metrics.yaml"


def gt_box(k: int) -> BBox:
    cx = 100.0 + 3 * k
    return BBox(cx - 20, 90, cx + 20, 110)


def make_group(root: Path, group: str = "dev", partial: bool = False) -> list[str]:
    names = []
    for case in CASES:
        name = f"seqA__t{GT_ID}__d005__{case}"
        scen = root / group / name
        (scen / "images").mkdir(parents=True)
        for k in range(N):
            (scen / "images" / f"{k + 1:07d}.jpg").write_bytes(b"")
        lines, rows = [], []
        for k in range(N):
            b = gt_box(k)
            lines.append(f"{k + 1},{GT_ID},{int(b.x1)},{int(b.y1)},{int(b.width)},{int(b.height)},1,4,0,0")
            rows.append({"frame_idx": k, "source_frame": k + 1, "bbox": list(b.as_tuple()),
                         "occluded": OCC_LO <= k <= OCC_HI, "occluder_bbox": None})
        (scen / "annotations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (scen / "manifest.json").write_text(json.dumps({
            "scenario_id": f"{group}/{name}", "name": name, "group": group,
            "source": {"split_dir": "VAL", "sequence": "seqA", "first_source_frame": 1,
                       "last_source_frame": N},
            "target": {"gt_target_id": GT_ID, "class": 3, "category": 4},
            "frames": N, "first_frame": 1,
            "occlusion": {"start_frame_idx": OCC_LO, "end_frame_idx": OCC_HI, "duration_frames": 5,
                          "occluder": "full_solid", "coverage": "full", "texture": "solid"},
            "ground_truth": rows, "seed": 1, "config_version": 1, "config_sha256": "abc",
        }), encoding="utf-8")
        names.append(name)
    (root / group / "index.json").write_text(json.dumps({
        "group": group, "scenarios": names, "partial": partial, "seed": 1, "config_sha256": "abc",
    }), encoding="utf-8")
    return names


def fake_tracks(source: str):
    case = next(c for c in CASES if c in Path(source).parent.name)
    out = []
    for k in range(N):
        tracks = [{"track_id": 9, "bbox": [500, 400, 540, 420], "conf": 0.9, "cls": 3}]
        tid = 1 if k <= 5 else {"new_id": 2, "same_id": 1}.get(case) if k > OCC_HI else None
        if tid is not None:
            tracks.append({"track_id": tid, "bbox": list(gt_box(k).as_tuple()), "conf": 0.8, "cls": 3})
        out.append({"frame_idx": k, "path": str(Path(source) / f"{k + 1:07d}.jpg"), "tracks": tracks})
    return out


class FakeRunner:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> int:
        self.calls.append(cmd)
        if cmd[1:3] == ["-m", "src.track"]:
            return tr.main(cmd[3:] + ["--imgsz", "960"])
        if cmd[1].endswith("run_pipeline.py"):
            return rp.main(cmd[2:] + ["--imgsz", "960"])
        if cmd[1:3] == ["-m", "src.evaluate"]:
            try:
                return ev.main(cmd[3:])
            except ev.EvaluationError:
                return 1
        raise AssertionError(cmd)

    def count(self, what: str) -> int:
        if what == "pipeline":
            return sum(1 for c in self.calls if c[1].endswith("run_pipeline.py"))
        if what == "track":
            return sum(1 for c in self.calls if c[1:3] == ["-m", "src.track"])
        return sum(1 for c in self.calls if len(c) > 3 and c[3] == what)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks(s)))

    def no_yolo(*a, **k):
        raise AssertionError("run_pipeline khong duoc chay YOLO khi replay")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    scen_root = tmp_path / "scn"
    make_group(scen_root)
    base = ["--group", "dev", "--tag", "001", "--scenario-root", str(scen_root),
            "--output-root", str(tmp_path / "experiments"),
            "--detector", "yolo11=w11.pt", "--detector", "yolo26=w26.pt",
            # kich ban gia o day la v1 (khong screening) -> metrics v1 ro rang (mac dinh runner la v2)
            "--metrics-config", str(METRICS_V1)]
    return tmp_path, base


def test_experiment_naming():
    assert abl.experiment_id("yolo11", False, "dev", "001") == "yolo11_botsort_occl_dev_001"
    assert abl.experiment_id("yolo26", True, "heldout", "002") == "yolo26_botsort_mm_occl_heldout_002"


def test_dry_run_prints_commands_and_writes_nothing(env, capsys):
    tmp, base = env
    runner = FakeRunner()
    assert abl.main(base + ["--dry-run"], runner) == 0
    assert runner.calls == []
    assert not (tmp / "experiments").exists()
    out = capsys.readouterr().out
    assert out.count("-m src.track") == 2 * len(CASES)  # mot lan moi detector x kich ban
    assert out.count("run_pipeline.py") == 2 * 2 * len(CASES)
    assert out.count("--replay-tracks") == 2 * 2 * len(CASES)
    assert out.count("--motion-memory ") == 2 * len(CASES)
    assert "src.evaluate compare" in out


def test_full_run_then_rerun_skips(env):
    tmp, base = env
    runner = FakeRunner()
    assert abl.main(base, runner) == 0
    assert runner.count("track") == 2 * len(CASES)
    assert runner.count("pipeline") == 2 * 2 * len(CASES)
    exps = tmp / "experiments"
    for exp in ("yolo11_botsort_occl_dev_001", "yolo11_botsort_mm_occl_dev_001",
                "yolo26_botsort_occl_dev_001", "yolo26_botsort_mm_occl_dev_001"):
        assert (exps / exp / "config.yaml").is_file()
        agg = json.loads((exps / exp / "metrics.json").read_text())
        assert agg["completeness"]["complete"] is True and agg["overall"]["scenarios"] == len(CASES)
    cmp_ = json.loads((exps / "yolo11_botsort_mm_occl_dev_001"
                       / "ablation_vs_yolo11_botsort_occl_dev_001.json").read_text())
    assert cmp_["baseline"]["overall"]["reidentified"] == 1
    assert cmp_["memory"]["overall"]["reidentified"] == 2

    # chay lai: khong chay pipeline nao, khong ghi de ket qua kich ban
    before = (exps / "yolo11_botsort_occl_dev_001" / "scenarios" / f"seqA__t{GT_ID}__d005__never"
              / "metrics.json").read_bytes()
    runner2 = FakeRunner()
    assert abl.main(base, runner2) == 0
    assert runner2.count("track") == 0
    assert runner2.count("pipeline") == 0 and runner2.count("occlusion") == 0
    after = (exps / "yolo11_botsort_occl_dev_001" / "scenarios" / f"seqA__t{GT_ID}__d005__never"
             / "metrics.json").read_bytes()
    assert before == after


def test_interrupted_run_is_reevaluated_not_rerun(env):
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    run_dir = (tmp / "experiments" / "yolo11_botsort_mm_occl_dev_001" / "scenarios"
               / f"seqA__t{GT_ID}__d005__new_id")
    (run_dir / "metrics.json").unlink()
    runner = FakeRunner()
    assert abl.main(base, runner) == 0
    assert runner.count("track") == 0
    assert runner.count("pipeline") == 0 and runner.count("occlusion") == 1
    assert (run_dir / "metrics.json").is_file()


def test_broken_log_reported_not_deleted(env, capsys):
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    run_dir = (tmp / "experiments" / "yolo26_botsort_occl_dev_001" / "scenarios"
               / f"seqA__t{GT_ID}__d005__same_id")
    (run_dir / "metrics.json").unlink()
    log = run_dir / "logs" / "pipeline.jsonl"
    log.write_text("\n".join(log.read_text().splitlines()[:5]) + "\n")  # log dang do
    capsys.readouterr()  # bo output cua lan chay dau
    runner = FakeRunner()
    assert abl.main(base, runner) == 1
    assert log.is_file() and runner.count("pipeline") == 0
    out = capsys.readouterr().out
    assert "can xu ly tay" in out
    assert "yolo26_botsort_occl_dev_001" not in out.split("Da gop:")[1].split("\n")[0]


def test_config_mismatch_refused(env):
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    changed = [a.replace("w11.pt", "other.pt") for a in base]
    with pytest.raises(ValueError, match="cau hinh khac"):
        abl.main(changed, FakeRunner())


def test_partial_index_refused(tmp_path):
    root = tmp_path / "scn"
    make_group(root, partial=True)
    args = ["--group", "dev", "--tag", "x", "--scenario-root", str(root),
            "--output-root", str(tmp_path / "e"), "--detector", "y=w.pt", "--dry-run",
            "--metrics-config", str(METRICS_V1)]
    with pytest.raises(ValueError, match="partial"):
        abl.main(args, FakeRunner())
    assert abl.main(args + ["--allow-partial-index"], FakeRunner()) == 0


def test_bad_detector_arg():
    with pytest.raises(SystemExit):
        abl.parse_args(["--group", "dev", "--tag", "1", "--detector", "noequals"])
    with pytest.raises(SystemExit):
        abl.parse_args(["--group", "dev", "--tag", "1", "--detector", "a=x", "--detector", "a=y"])


def _context(exps: Path, exp: str, scen: str) -> dict:
    return json.loads((exps / exp / "scenarios" / scen / "run_context.json").read_text())


def test_both_configs_replay_exactly_same_tracker_output(env):
    """Hai cau hinh doc CUNG mot file log track (cung SHA-256), khong chay YOLO."""
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    exps = tmp / "experiments"
    for det in ("yolo11", "yolo26"):
        for case in CASES:
            scen = f"seqA__t{GT_ID}__d005__{case}"
            log = exps / f"{det}_botsort_tracks_occl_dev_001" / "scenarios" / scen / "logs" / "tracks.jsonl"
            sha = hashlib.sha256(log.read_bytes()).hexdigest()
            b = _context(exps, f"{det}_botsort_occl_dev_001", scen)
            m = _context(exps, f"{det}_botsort_mm_occl_dev_001", scen)
            assert b["replay"]["tracks_sha256"] == m["replay"]["tracks_sha256"] == sha
            assert Path(b["replay"]["tracks_log"]) == Path(m["replay"]["tracks_log"]) == log
            assert b["config"]["motion_memory"] is False and m["config"]["motion_memory"] is True


def test_rerun_reuses_tracks_log(env):
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    run_dir = (tmp / "experiments" / "yolo11_botsort_mm_occl_dev_001" / "scenarios"
               / f"seqA__t{GT_ID}__d005__never")
    (run_dir / "metrics.json").unlink()
    (run_dir / "logs" / "pipeline.jsonl").unlink()
    runner = FakeRunner()
    assert abl.main(base, runner) == 0
    assert runner.count("track") == 0 and runner.count("pipeline") == 1


def test_broken_tracks_log_reported_not_deleted(env, capsys):
    tmp, base = env
    assert abl.main(base, FakeRunner()) == 0
    scen = f"seqA__t{GT_ID}__d005__same_id"
    exps = tmp / "experiments"
    log = exps / "yolo11_botsort_tracks_occl_dev_001" / "scenarios" / scen / "logs" / "tracks.jsonl"
    log.write_text("\n".join(log.read_text().splitlines()[:5]) + "\n")  # log dang do
    run_dir = exps / "yolo11_botsort_occl_dev_001" / "scenarios" / scen
    (run_dir / "metrics.json").unlink()
    (run_dir / "logs" / "pipeline.jsonl").unlink()
    capsys.readouterr()
    runner = FakeRunner()
    assert abl.main(base, runner) == 1
    assert log.is_file() and runner.count("track") == 0 and runner.count("pipeline") == 0
    out = capsys.readouterr().out
    assert "log track cu khong hop le" in out
    assert "yolo11_botsort_occl_dev_001" not in out.split("Da gop:")[1].split("\n")[0]


def test_dry_run_prints_counts(env, capsys):
    _, base = env
    assert abl.main(base + ["--dry-run"], FakeRunner()) == 0
    out = capsys.readouterr().out
    n = len(CASES)
    assert f"Nhom dev: {n}/{n} kich ban, 2 detector" in out
    assert (f"src.track {2 * n}, run_pipeline {4 * n}, evaluate occlusion {4 * n}, "
            f"evaluate aggregate 4, evaluate compare 2 (tong {2 * n + 8 * n + 6} lenh)") in out
    assert "bo qua vi da co metrics: 0" in out


def test_dry_run_counts_reflect_existing_results(env, capsys):
    _, base = env
    assert abl.main(base, FakeRunner()) == 0
    capsys.readouterr()
    assert abl.main(base + ["--dry-run"], FakeRunner()) == 0
    out = capsys.readouterr().out
    assert "src.track 0, run_pipeline 0, evaluate occlusion 0" in out
    assert f"bo qua vi da co metrics: {4 * len(CASES)}" in out


def test_limit_scenarios_runs_first_n_marked_incomplete(env, capsys):
    tmp, base = env
    args = [a.replace("001", "smoke") for a in base] + ["--limit-scenarios", "1"]
    runner = FakeRunner()
    assert abl.main(args, runner) == 0
    assert runner.count("track") == 2 and runner.count("pipeline") == 4
    out = capsys.readouterr().out
    assert "Thoi gian (giay):" in out and "Uoc tinh ca nhom (3 kich ban x 2 detector)" in out
    assert "KHONG phai benchmark" in out
    exp = tmp / "experiments" / "yolo11_botsort_mm_occl_dev_smoke"
    agg = json.loads((exp / "metrics.json").read_text())
    assert agg["completeness"]["complete"] is False and agg["overall"]["scenarios"] == 1
    assert agg["completeness"]["missing"] == [f"seqA__t{GT_ID}__d005__{c}" for c in CASES[1:]]
    import yaml
    assert yaml.safe_load((exp / "config.yaml").read_text())["scenarios"]["limit_scenarios"] == 1


def test_limit_scenarios_not_mixed_with_full_run(env):
    _, base = env
    assert abl.main(base + ["--limit-scenarios", "1"], FakeRunner()) == 0
    with pytest.raises(ValueError, match="cau hinh khac"):
        abl.main(base, FakeRunner())


def test_limit_scenarios_must_be_positive():
    with pytest.raises(SystemExit):
        abl.parse_args(["--group", "dev", "--tag", "1", "--detector", "a=x", "--limit-scenarios", "0"])


# ---------------- kich ban v2 (screening) + metrics v2 ----------------

V2_CASES = ("new_id", "same_id", "unseen")   # unseen: detector khong thay muc tieu -> chua khoa
V2_LOCK = {"match_iou": 0.5, "max_search_frames": 4, "lock_confirm_frames": 3, "lock_iou": 0.5}


def fake_tracks_v2(source: str):
    case = next(c for c in V2_CASES if c in Path(source).parent.name)
    out = []
    for k in range(N):
        tracks = [{"track_id": 9, "bbox": [500, 400, 540, 420], "conf": 0.9, "cls": 3}]
        if case != "unseen":
            tid = 1 if k <= 5 else {"new_id": 2, "same_id": 1}[case] if k > OCC_HI else None
            if tid is not None:
                tracks.append({"track_id": tid, "bbox": list(gt_box(k).as_tuple()), "conf": 0.8, "cls": 3})
        out.append({"frame_idx": k, "path": str(Path(source) / f"{k + 1:07d}.jpg"), "tracks": tracks})
    return out


def make_group_v2(root: Path, weights: dict, group: str = "dev") -> list[str]:
    from src.occlusion_scenarios import file_sha256

    screening = {"file": "screening.json", "sha256": "scr",
                 "detectors": {n: {"weights_sha256": file_sha256(p)} for n, p in weights.items()}}
    names = []
    for case in V2_CASES:
        name = f"seqA__t{GT_ID}__d005__{case}"
        scen = root / group / name
        (scen / "images").mkdir(parents=True)
        lines, rows = [], []
        for k in range(N):
            (scen / "images" / f"{k + 1:07d}.jpg").write_bytes(b"")
            b = gt_box(k)
            lines.append(f"{k + 1},{GT_ID},{int(b.x1)},{int(b.y1)},{int(b.width)},{int(b.height)},1,4,0,0")
            rows.append({"frame_idx": k, "source_frame": k + 1, "bbox": list(b.as_tuple()),
                         "occluded": OCC_LO <= k <= OCC_HI, "occluder_bbox": None})
        (scen / "annotations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (scen / "manifest.json").write_text(json.dumps({
            "scenario_id": f"{group}/{name}", "name": name, "group": group,
            "source": {"split_dir": "VAL", "sequence": "seqA", "first_source_frame": 1, "last_source_frame": N},
            "target": {"gt_target_id": GT_ID, "class": 3, "category": 4},
            "frames": N, "first_frame": 1,
            "occlusion": {"start_frame_idx": OCC_LO, "end_frame_idx": OCC_HI, "duration_frames": 5,
                          "occluder": "full_solid", "coverage": "full", "texture": "solid"},
            "ground_truth": rows, "seed": 1, "config_version": 2, "config_sha256": "v2sha",
            "target_lock": {**V2_LOCK, "required_detectors": sorted(weights), "occlusion_start_frame_idx": OCC_LO},
            "screening": screening,
        }), encoding="utf-8")
        names.append(name)
    (root / group / "index.json").write_text(json.dumps({
        "group": group, "scenarios": names, "partial": False, "seed": 1, "config_sha256": "v2sha",
        "config_version": 2, "screening": screening,
        "target_lock": {**V2_LOCK, "required_detectors": sorted(weights)},
    }), encoding="utf-8")
    return names


@pytest.fixture
def env_v2(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks_v2(s)))

    def no_yolo(*a, **k):
        raise AssertionError("run_pipeline khong duoc chay YOLO khi replay")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    weights = {}
    for name in ("yolo11", "yolo26"):
        weights[name] = tmp_path / f"{name}.pt"
        weights[name].write_bytes(f"weights-{name}".encode())
    make_group_v2(tmp_path / "scn", weights)
    base = ["--group", "dev", "--tag", "v2", "--scenario-root", str(tmp_path / "scn"),
            "--output-root", str(tmp_path / "experiments"),
            "--detector", f"yolo11={weights['yolo11']}", "--detector", f"yolo26={weights['yolo26']}"]
    return tmp_path, base, weights


def test_v2_pipeline_gets_lock_args_from_manifest(env_v2, capsys):
    _, base, _ = env_v2
    assert abl.main(base + ["--dry-run"], FakeRunner()) == 0
    out = capsys.readouterr().out
    n_pipeline = 2 * 2 * len(V2_CASES)
    assert out.count("--gt-match-iou 0.5 --gt-max-search-frames 4") == n_pipeline
    assert out.count("occlusion_metrics_v2.yaml") == n_pipeline   # evaluate occlusion dung metrics v2


def test_v2_full_run_counts_not_locked_separately_and_resumes(env_v2, capsys):
    tmp, base, _ = env_v2
    assert abl.main(base, FakeRunner()) == 0
    exps = tmp / "experiments"
    import yaml

    for det in ("yolo11", "yolo26"):
        for mm in ("", "mm_"):
            exp = exps / f"{det}_botsort_{mm}occl_dev_v2"
            agg = json.loads((exp / "metrics.json").read_text())
            o = agg["overall"]
            assert agg["metrics_config"]["version"] == 2
            assert o["scenarios_total"] == 3 and o["scenarios_evaluated"] == 2 and o["scenarios_not_locked"] == 1
            assert o["not_locked_scenario_ids"] == [f"dev/seqA__t{GT_ID}__d005__unseen"]
            assert o["independent_targets"] == 1 and o["independent_sequences"] == 1
            conf = yaml.safe_load((exp / "config.yaml").read_text())
            assert conf["metrics_config"]["version"] == 2 and len(conf["metrics_config"]["config_sha256"]) == 64
            assert conf["scenarios"]["screening_sha256"] == "scr" and conf["scenarios"]["config_version"] == 2
        cmp_ = json.loads((exps / f"{det}_botsort_mm_occl_dev_v2" / f"ablation_vs_{det}_botsort_occl_dev_v2.json")
                          .read_text())
        assert cmp_["scenarios_evaluated"] == 2 and cmp_["scenarios_not_locked"] == 1
        assert cmp_["baseline"]["overall"]["reidentification_rate"] == 0.5    # same_id / (same_id + new_id)
        assert cmp_["memory"]["overall"]["reidentification_rate"] == 1.0
    out = capsys.readouterr().out
    assert "Thoi gian (giay):" in out

    runner = FakeRunner()                                   # resume: khong chay lai gi
    assert abl.main(base, runner) == 0
    assert runner.count("track") == 0 and runner.count("pipeline") == 0 and runner.count("occlusion") == 0


def test_v2_refuses_weights_not_matching_screening(env_v2):
    tmp, base, weights = env_v2
    weights["yolo26"].write_bytes(b"weights da doi")
    with pytest.raises(ValueError, match="sha256 khac"):
        abl.main(base + ["--dry-run"], FakeRunner())


def test_v2_refuses_detector_not_in_screening(env_v2):
    tmp, base, weights = env_v2
    args = [a.replace("yolo26=", "rtdetr=") for a in base]
    with pytest.raises(ValueError, match="khong co trong screening"):
        abl.main(args + ["--dry-run"], FakeRunner())


def test_metrics_version_must_match_scenario_version(env_v2, tmp_path):
    _, base, _ = env_v2
    with pytest.raises(ValueError, match="metrics v2"):
        abl.main(base + ["--dry-run", "--metrics-config", str(METRICS_V1)], FakeRunner())
    root = tmp_path / "scn_v1"
    make_group(root)
    args = ["--group", "dev", "--tag", "x", "--scenario-root", str(root),
            "--output-root", str(tmp_path / "e"), "--detector", "y=w.pt", "--dry-run"]
    with pytest.raises(ValueError, match="kich ban v1"):
        abl.main(args, FakeRunner())                        # mac dinh metrics v2 tren kich ban v1
