"""Test scripts/run_occlusion_ablation.py voi bo chay GIA (khong YOLO, khong subprocess).

Bo chay gia goi run_pipeline.main / src.evaluate.main trong cung tien trinh;
iter_tracks duoc thay bang track gia theo ten kich ban.
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
import src.evaluate as ev  # noqa: E402
from src.motion_memory import BBox  # noqa: E402

N, OCC_LO, OCC_HI, GT_ID = 20, 6, 10, 42
CASES = ("new_id", "never", "same_id")


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
        if cmd[1].endswith("run_pipeline.py"):
            return rp.main(cmd[2:] + ["--imgsz", "960"])
        if cmd[1:3] == ["-m", "src.evaluate"]:
            try:
                return ev.main(cmd[3:])
            except ev.EvaluationError:
                return 1
        raise AssertionError(cmd)

    def count(self, what: str) -> int:
        return sum(1 for c in self.calls if (c[1].endswith("run_pipeline.py") if what == "pipeline"
                                              else len(c) > 3 and c[3] == what))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "iter_tracks", lambda w, s, **k: iter(fake_tracks(s)))
    scen_root = tmp_path / "scn"
    make_group(scen_root)
    base = ["--group", "dev", "--tag", "001", "--scenario-root", str(scen_root),
            "--output-root", str(tmp_path / "experiments"),
            "--detector", "yolo11=w11.pt", "--detector", "yolo26=w26.pt"]
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
    assert out.count("run_pipeline.py") == 2 * 2 * len(CASES)
    assert out.count("--motion-memory ") == 2 * len(CASES)
    assert "src.evaluate compare" in out


def test_full_run_then_rerun_skips(env):
    tmp, base = env
    runner = FakeRunner()
    assert abl.main(base, runner) == 0
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
            "--output-root", str(tmp_path / "e"), "--detector", "y=w.pt", "--dry-run"]
    with pytest.raises(ValueError, match="partial"):
        abl.main(args, FakeRunner())
    assert abl.main(args + ["--allow-partial-index"], FakeRunner()) == 0


def test_bad_detector_arg():
    with pytest.raises(SystemExit):
        abl.parse_args(["--group", "dev", "--tag", "1", "--detector", "noequals"])
    with pytest.raises(SystemExit):
        abl.parse_args(["--group", "dev", "--tag", "1", "--detector", "a=x", "--detector", "a=y"])
