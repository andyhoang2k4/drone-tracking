"""Test screening khoa muc tieu (src/occlusion_screening.py, scripts/screen_occlusion_targets.py).

Sequence VisDrone GIA rat nho; runner gia goi src.track.main trong cung tien
trinh voi iter_tracks gia (khong YOLO, khong subprocess). Hanh vi detector gia
lap theo (detector, ung vien).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

import scripts.screen_occlusion_targets as scr  # noqa: E402
import src.track as tr  # noqa: E402
from src.occlusion_scenarios import (  # noqa: E402
    ScreeningError,
    file_sha256,
    generate_group,
    load_scenario_config,
)
from src.occlusion_screening import size_bin  # noqa: E402
from src.utils.gt_target import load_target_boxes  # noqa: E402

N_FRAMES, W, H, PRE = 14, 96, 64, 3
DATASET_CONFIG = (REPO_ROOT / "configs" / "dataset_visdrone.yaml").as_posix()
CATEGORY_MAP = {c: c - 1 for c in range(1, 11)}


def make_sequence(split_dir: Path, name: str, n_targets: int = 4) -> None:
    seq = split_dir / "sequences" / name
    seq.mkdir(parents=True)
    rng = np.random.default_rng(sum(map(ord, name)))
    for f in range(1, N_FRAMES + 1):
        cv2.imwrite(str(seq / f"{f:07d}.png"), rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
    lines = []
    for f in range(1, N_FRAMES + 1):
        for t in range(1, n_targets + 1):
            w = 8 + 4 * t  # kich thuoc khac nhau de test phan bo kich thuoc
            lines.append(f"{f},{t},{2 + 2 * (f - 1)},{2 + 14 * (t - 1)},{w},6,1,4,0,0")
    (split_dir / "annotations").mkdir(exist_ok=True)
    (split_dir / "annotations" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


CONFIG = """
version: 2
seed: 11
raw_root: {raw}
dataset_config: {dataset}
output_root: {out}
groups:
  dev: {{split_dir: VAL}}
  heldout: {{split_dir: TEST}}
target_rules:
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  min_box_side_px: 4
  max_gt_truncation: 0
  max_gt_occlusion: 1
  max_ignore_ioa: 0.0
  pre_occlusion_frames: {pre}
  post_occlusion_frames: 2
  candidates_per_sequence: 3
  targets_per_sequence: 2
target_lock:
  match_iou: 0.5
  max_search_frames: 2
  lock_confirm_frames: 2
  lock_iou: 0.5
  required_detectors: [a, b]
screening:
  work_root: {work}
  size_bins_px: [4, 6, 8]
durations_frames: [1, 3]
start_policy: seeded_shared
occluders:
  - {{name: full_solid, coverage: full, texture: solid}}
occluder_params:
  margin_ratio: 0.25
  partial_fraction: 0.5
  solid_color_bgr: [10, 200, 30]
image:
  format: png
"""

# Hanh vi detector gia: (detector, candidate_id) -> locked | unseen | lost | other_class
BEHAVIOR: dict[tuple[str, str], str] = {}


def fake_iter_tracks(weights, source, **kw):
    clip = Path(source).parent
    info = json.loads((clip / "clip.json").read_text())
    boxes = load_target_boxes(clip / "annotations.txt", info["gt_target_id"], CATEGORY_MAP)
    mode = BEHAVIOR.get((Path(weights).stem, info["candidate_id"]), "locked")
    names = sorted(p.name for p in Path(source).iterdir())
    for k, name in enumerate(names):
        tracks = [{"track_id": 99, "bbox": [80, 50, 90, 60], "conf": 0.9, "cls": 0}]
        visible = mode in ("locked", "other_class") or (mode == "lost" and k == 0)
        if visible:
            b = boxes[k + 1]
            tracks.append({"track_id": 1, "bbox": [b.x1, b.y1, b.x2, b.y2], "conf": 0.8,
                           "cls": 7 if mode == "other_class" else 3})
        yield {"frame_idx": k, "path": str(Path(source) / name), "tracks": tracks}


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, cmd):
        self.calls.append(cmd)
        assert cmd[1:3] == ["-m", "src.track"], cmd
        return tr.main(cmd[3:] + ["--imgsz", "960"])


@pytest.fixture
def env(tmp_path, monkeypatch):
    BEHAVIOR.clear()
    monkeypatch.setattr(tr, "iter_tracks", fake_iter_tracks)
    raw = tmp_path / "raw"
    make_sequence(raw / "VAL", "seqA")
    make_sequence(raw / "VAL", "seqB")
    make_sequence(raw / "TEST", "seqC")
    cfg_path = tmp_path / "v2.yaml"
    cfg_path.write_text(CONFIG.format(raw=raw.as_posix(), dataset=DATASET_CONFIG, pre=PRE,
                                      out=(tmp_path / "out").as_posix(),
                                      work=(tmp_path / "work").as_posix()), encoding="utf-8")
    weights = {}
    for name in ("a", "b"):
        weights[name] = tmp_path / f"{name}.pt"
        weights[name].write_bytes(f"w-{name}".encode())
    args = ["--config", str(cfg_path), "--group", "dev",
            "--detector", f"a={weights['a']}", "--detector", f"b={weights['b']}"]
    return tmp_path, cfg_path, weights, args


def screening_of(tmp_path):
    return json.loads((tmp_path / "out" / "dev" / "screening.json").read_text())


def test_dry_run_counts_and_writes_nothing(env, capsys):
    tmp, _, _, args = env
    runner = FakeRunner()
    assert scr.main(args + ["--dry-run"], runner) == 0
    assert runner.calls == [] and not (tmp / "out").exists() and not (tmp / "work").exists()
    out = capsys.readouterr().out
    assert out.count("-m src.track") == 6 * 2          # 2 sequence x 3 ung vien x 2 detector
    assert "6/6 ung vien x 2 detector; du kien src.track 12 lan" in out
    assert scr.main(args + ["--dry-run", "--limit", "2"], FakeRunner()) == 0
    assert "2/6 ung vien" in capsys.readouterr().out


def test_full_screening_statuses_and_summary(env, capsys):
    tmp, cfg_path, weights, args = env
    cfg = load_scenario_config(cfg_path)
    from src.occlusion_screening import plan_group_candidates

    cids = [f"{p.sequence}__t{p.target_id}" for p, _ in plan_group_candidates(cfg, "dev")]
    BEHAVIOR.update({("b", cids[0]): "unseen", ("a", cids[1]): "lost", ("a", cids[2]): "other_class",
                     ("a", cids[3]): "unseen", ("b", cids[3]): "unseen"})
    runner = FakeRunner()
    assert scr.main(args, runner) == 0
    assert len(runner.calls) == 12
    s = screening_of(tmp)
    assert s["partial"] is False and s["errors"] == [] and s["config_sha256"] == cfg.config_sha256
    assert s["detectors"]["a"]["weights_sha256"] == file_sha256(weights["a"])
    by_id = {e["candidate_id"]: e for e in s["candidates"]}
    assert [e["candidate_id"] for e in s["candidates"]] == cids
    e0, e1, e2, e3 = (by_id[c] for c in cids[:4])
    assert e0["detectors"]["a"]["status"] == "locked" and e0["detectors"]["b"]["status"] == "not_selected"
    assert e0["eligible"] is False and e0["detectors"]["b"]["best_iou"] == 0.0
    assert e1["detectors"]["a"]["status"] == "lost_before_occlusion" and not e1["eligible"]
    # khop khong xet class: class khac van khoa duoc, class duoc ghi lai
    assert e2["eligible"] and e2["detectors"]["a"]["selected_class"] == 7
    assert e2["detectors"]["a"]["class_matches_gt"] is False and e2["detectors"]["b"]["class_matches_gt"] is True
    assert e3["detectors"]["a"]["frames_with_match"] == 0
    assert all(by_id[c]["eligible"] for c in cids[4:])

    summ = s["summary"]
    assert summ["screened"] == 6 and summ["eligible"] == 3
    # a khoa e0, e2, e4, e5; b khoa e1, e2, e4, e5 -> dat (giao) e2, e4, e5
    assert summ["per_detector"]["a"]["locked"] == 4 and summ["per_detector"]["a"]["pass_rate"] == pytest.approx(4 / 6)
    assert summ["per_detector"]["a"]["by_status"] == {"locked": 4, "lost_before_occlusion": 1, "not_selected": 1}
    assert summ["per_detector"]["b"]["locked"] == 4
    assert summ["per_detector"]["a"]["locked_with_other_class"] == 1
    assert sum(r["eligible"] for r in summ["per_sequence"].values()) == 3
    assert summ["candidate_distribution"]["count"] == 6 and summ["eligible_distribution"]["count"] == 3
    assert sum(summ["candidate_distribution"]["by_class"].values()) == 6
    assert sum(summ["candidate_distribution"]["by_min_side_px"].values()) == 6
    assert "selection_bias_note" in summ
    assert "Da ghi:" in capsys.readouterr().out

    # screening day du -> generator dung duoc, chi sinh cho ung vien dat
    index = generate_group(cfg, "dev", weights_sha256={k: file_sha256(v) for k, v in weights.items()},
                           dry_run=True)
    selected = {f"{seq}__t{t}" for seq, ts in index["selected_targets"].items() for t in ts}
    assert selected == {c for c in cids if by_id[c]["eligible"]}


def test_limit_marks_partial_then_resume_reuses_logs(env):
    tmp, cfg_path, weights, args = env
    runner = FakeRunner()
    assert scr.main(args + ["--limit", "2"], runner) == 0
    assert len(runner.calls) == 4
    s = screening_of(tmp)
    assert s["partial"] is True and s["limit"] == 2
    assert [e["status"] for e in s["candidates"]] == ["screened"] * 2 + ["not_run"] * 4
    cfg = load_scenario_config(cfg_path)
    shas = {k: file_sha256(v) for k, v in weights.items()}
    with pytest.raises(ScreeningError, match="partial"):
        generate_group(cfg, "dev", weights_sha256=shas, dry_run=True)

    runner2 = FakeRunner()
    assert scr.main(args, runner2) == 0
    assert len(runner2.calls) == 8                      # chi 4 ung vien con lai x 2 detector
    assert screening_of(tmp)["partial"] is False
    generate_group(cfg, "dev", weights_sha256=shas, dry_run=True)


def test_broken_log_reported_not_deleted(env, capsys):
    tmp, cfg_path, _, args = env
    assert scr.main(args, FakeRunner()) == 0
    log = next((tmp / "work" / "dev").glob("*/logs/a/tracks.jsonl"))
    log.write_text(log.read_text().splitlines()[0] + "\n")   # log dang do
    capsys.readouterr()
    assert scr.main(args, FakeRunner()) == 1
    assert log.is_file()
    s = screening_of(tmp)
    assert s["partial"] is True and len(s["errors"]) == 1 and "xu ly tay" in s["errors"][0]
    assert sum(1 for e in s["candidates"] if e["status"] == "error") == 1


def test_changed_weights_refuse_old_logs(env):
    tmp, _, weights, args = env
    assert scr.main(args, FakeRunner()) == 0
    weights["b"].write_bytes(b"weights moi")
    runner = FakeRunner()
    assert scr.main(args, runner) == 1
    assert runner.calls == []
    assert all("weights khac" in e for e in screening_of(tmp)["errors"])


def test_detector_names_must_match_required(env):
    _, cfg_path, weights, _ = env
    with pytest.raises(SystemExit):
        scr.main(["--config", str(cfg_path), "--group", "dev", "--detector", f"a={weights['a']}"], FakeRunner())
    with pytest.raises(SystemExit):
        scr.main(["--config", str(cfg_path), "--group", "dev", "--detector", f"a={weights['a']}",
                  "--detector", f"c={weights['b']}"], FakeRunner())


def test_v1_config_refused(tmp_path):
    with pytest.raises(SystemExit):
        scr.main(["--config", str(REPO_ROOT / "configs" / "occlusion_scenarios.yaml"), "--group", "dev",
                  "--detector", "yolo11=x.pt", "--dry-run"], FakeRunner())


def test_clip_identical_to_scenario_prefix(env):
    """Clip screening = dung frame 0..pre-1 cua kich ban (cung byte anh, cung GT)."""
    tmp, cfg_path, weights, args = env
    assert scr.main(args, FakeRunner()) == 0
    cfg = load_scenario_config(cfg_path)
    index = generate_group(cfg, "dev", weights_sha256={k: file_sha256(v) for k, v in weights.items()})
    name = index["scenarios"][0]
    seq, tid = name.split("__")[0], int(name.split("__")[1][1:])
    clip = tmp / "work" / "dev" / f"{seq}__t{tid}"
    scen = cfg.output_root / "dev" / name
    clip_images = sorted((clip / "images").iterdir())
    assert len(clip_images) == PRE
    for p in clip_images:
        assert p.read_bytes() == (scen / "images" / p.name).read_bytes()
    clip_gt = load_target_boxes(clip / "annotations.txt", tid, CATEGORY_MAP)
    scen_gt = load_target_boxes(scen / "annotations.txt", tid, CATEGORY_MAP)
    assert all(clip_gt[f] == scen_gt[f] for f in clip_gt)


def test_size_bin_labels():
    edges = [16, 24, 32]
    assert size_bin(10, edges) == "<16" and size_bin(16, edges) == "[16,24)"
    assert size_bin(31.9, edges) == "[24,32)" and size_bin(500, edges) == ">=32"
