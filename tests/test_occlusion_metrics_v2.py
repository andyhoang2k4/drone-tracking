"""Test metrics occlusion v2 (configs/occlusion_metrics_v2.yaml, src/evaluate.py).

- Gop: kich ban CHUA khoa truoc khi che khong lam loang ti le, van duoc dem rieng.
- Danh gia mot lan chay: 'khoa duoc' tinh tren log track replay dung chung.
- compare: tu choi khi tap kich ban da khoa khac nhau.

Log/manifest gia lap; src.track va run_pipeline chay voi iter_tracks gia (khong YOLO).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_pipeline as rp  # noqa: E402
import src.evaluate as ev  # noqa: E402
import src.track as tr  # noqa: E402
from src.evaluate import EvaluationError, aggregate_occlusion_metrics  # noqa: E402
from src.motion_memory import BBox  # noqa: E402

V1 = {"version": 1, "correct_iou": 0.5, "reid_min_consecutive_frames": 3, "path": "v1",
      "config_sha256": "s1"}
V2 = {"version": 2, "correct_iou": 0.5, "reid_min_consecutive_frames": 3, "path": "v2",
      "config_sha256": "s2"}


# ---------------- gop (ham thuan) ----------------


def item(name, *, locked, duration=5, occ=5, correct=0, reid=False, ttr=None, with_output=None,
         err_sum=0.0, gap=None, new_id=False, acquired=True, seq="seqA", tid=1,
         occluder="full_solid", cfg=V2):
    metrics = {
        "occlusion_frames": occ, "correct_occluded_frames": correct,
        "occluded_frames_with_output": correct if with_output is None else with_output,
        "position_error_sum_px": err_sum, "reidentified": reid, "time_to_reidentify_frames": ttr,
        "reidentified_with_new_track_id": new_id,
        "longest_tracking_gap": occ - correct if gap is None else gap,
        "target_acquired_before_occlusion": acquired,
    }
    if locked is not None:
        metrics["target_locked_at_occlusion"] = locked
    return {
        "scenario_id": f"dev/{name}",
        "scenario": {"name": name, "group": "dev", "duration_frames": duration, "occluder": occluder,
                     "sequence": seq, "gt_target_id": tid, "config_sha256": "scn"},
        "run": {"weights": "w", "tracker": "t", "conf": 0.1, "imgsz": 960, "classes": None,
                "motion_memory": False, "motion_memory_config": None},
        "metrics_config": dict(cfg),
        "metrics": metrics,
    }


def mixed_items():
    return [
        item("a", locked=True, occ=5, correct=5, reid=True, ttr=2, err_sum=10.0, seq="seqA", tid=1),
        item("b", locked=True, occ=5, correct=0, reid=False, gap=40, seq="seqA", tid=2,
             occluder="partial_noise"),
        # chua khoa: thoi luong 60, khong output, gap = ca kich ban (150)
        item("c", locked=False, duration=60, occ=60, correct=0, gap=150, acquired=False, seq="seqB", tid=7),
        # chua khoa nhung co output sai vi tri (chon nham track): sai so lon, ID moi
        item("d", locked=False, duration=60, occ=60, correct=0, with_output=60, err_sum=6000.0,
             gap=150, reid=True, ttr=9, new_id=True, seq="seqB", tid=8),
    ]


def test_v1_dilutes_v2_does_not():
    """Cung du lieu: v1 tinh ca kich ban chua khoa (loang), v2 chi tinh kich ban da khoa."""
    v1_items = [dict(it, metrics_config=dict(V1)) for it in mixed_items()]
    o1 = aggregate_occlusion_metrics(v1_items)["overall"]
    assert o1["correct_prediction_ratio"] == pytest.approx(5 / 130)
    assert o1["reidentification_rate"] == pytest.approx(2 / 4)

    o2 = aggregate_occlusion_metrics(mixed_items())["overall"]
    assert o2["correct_prediction_ratio"] == pytest.approx(5 / 10)   # mau so chi a, b
    assert o2["occlusion_frames"] == 10 and o2["correct_occluded_frames"] == 5


def test_v2_reidentification_fields_exclude_not_locked():
    o = aggregate_occlusion_metrics(mixed_items())["overall"]
    assert o["reidentified"] == 1 and o["reidentification_failed"] == 1
    assert o["reidentification_rate"] == pytest.approx(0.5)
    assert o["reidentified_with_new_track_id"] == 0            # d bi loai
    assert o["mean_time_to_reidentify_frames"] == 2.0 and o["max_time_to_reidentify_frames"] == 2


def test_v2_gap_and_position_error_exclude_not_locked():
    o = aggregate_occlusion_metrics(mixed_items())["overall"]
    assert o["longest_tracking_gap_max"] == 40                 # khong phai 150
    assert o["longest_tracking_gap_mean"] == pytest.approx((0 + 40) / 2)
    assert o["occluded_frames_with_output"] == 5
    assert o["mean_position_error_px"] == pytest.approx(10.0 / 5)   # khong cong 6000 px cua d
    assert o["target_not_acquired_before_occlusion"] == 0


def test_v2_not_locked_counted_separately():
    agg = aggregate_occlusion_metrics(mixed_items())
    o = agg["overall"]
    assert o["scenarios_total"] == 4 and o["scenarios_evaluated"] == 2
    assert o["scenarios_not_locked"] == 2
    assert o["not_locked_scenario_ids"] == ["dev/c", "dev/d"]
    assert "scenarios" not in o
    assert "note" in agg and "thien lech" in agg["note"]


def test_v2_groups_by_duration_and_occluder():
    agg = aggregate_occlusion_metrics(mixed_items())
    d60 = agg["by_duration_frames"]["60"]
    assert d60["scenarios_total"] == 2 and d60["scenarios_evaluated"] == 0
    assert d60["not_locked_scenario_ids"] == ["dev/c", "dev/d"]
    assert d60["correct_prediction_ratio"] is None and d60["reidentification_rate"] is None
    assert d60["longest_tracking_gap_max"] is None
    d5 = agg["by_duration_frames"]["5"]
    assert d5["scenarios_not_locked"] == 0 and d5["correct_prediction_ratio"] == pytest.approx(0.5)
    occ = agg["by_occluder"]
    assert occ["full_solid"]["scenarios_total"] == 3 and occ["full_solid"]["scenarios_not_locked"] == 2
    assert occ["partial_noise"]["reidentification_rate"] == 0.0


def test_v2_independent_targets_sequences_and_per_sequence():
    o = aggregate_occlusion_metrics(mixed_items())["overall"]
    assert o["independent_targets"] == 2      # seqA t1, seqA t2 (seqB chua khoa)
    assert o["independent_sequences"] == 1
    assert o["per_sequence"] == {
        "seqA": {"scenarios_total": 2, "scenarios_evaluated": 2, "targets_total": 2, "targets_evaluated": 2},
        "seqB": {"scenarios_total": 2, "scenarios_evaluated": 0, "targets_total": 2, "targets_evaluated": 0},
    }


def test_v2_all_not_locked_gives_none_ratios():
    o = aggregate_occlusion_metrics([item("c", locked=False, gap=150)])["overall"]
    assert o["scenarios_evaluated"] == 0 and o["scenarios_not_locked"] == 1
    assert o["correct_prediction_ratio"] is None and o["reidentification_rate"] is None
    assert o["mean_position_error_px"] is None and o["independent_targets"] == 0


def test_v2_refuses_items_without_lock_flag():
    with pytest.raises(EvaluationError, match="target_locked_at_occlusion"):
        aggregate_occlusion_metrics([item("a", locked=True), item("b", locked=None)])


def test_refuses_mixing_metrics_versions_or_config_sha():
    with pytest.raises(EvaluationError, match="dinh nghia do do"):
        aggregate_occlusion_metrics([item("a", locked=True), item("b", locked=True, cfg=V1)])
    other = dict(V2, config_sha256="khac")
    with pytest.raises(EvaluationError, match="dinh nghia do do"):
        aggregate_occlusion_metrics([item("a", locked=True), item("b", locked=True, cfg=other)])


def test_provenance_records_config_hashes():
    prov = aggregate_occlusion_metrics(mixed_items())["provenance"]
    assert prov["metrics_config_sha256"] == ["s2"] and prov["scenario_config_sha256"] == ["scn"]


def test_metrics_config_files_and_hash():
    v1 = ev.load_occlusion_metrics_config(REPO_ROOT / "configs" / "occlusion_metrics.yaml")
    v2 = ev.load_occlusion_metrics_config(REPO_ROOT / "configs" / "occlusion_metrics_v2.yaml")
    assert v1["version"] == 1 and v2["version"] == 2
    assert len(v2["config_sha256"]) == 64 and v1["config_sha256"] != v2["config_sha256"]
    assert (v1["correct_iou"], v1["reid_min_consecutive_frames"]) == (
        v2["correct_iou"], v2["reid_min_consecutive_frames"])


# ---------------- danh gia lan chay (replay) ----------------

N, OCC_LO, OCC_HI, GT_ID = 20, 6, 10, 42
LOCK = {"match_iou": 0.5, "max_search_frames": 4, "lock_confirm_frames": 3, "lock_iou": 0.5}


def gt_box(k: int) -> BBox:
    cx = 100.0 + 3 * k
    return BBox(cx - 20, 90, cx + 20, 110)


def make_scenario(root: Path, name: str, *, with_lock=True, duration=5, seq="seqA") -> Path:
    scen = root / "dev" / name
    (scen / "images").mkdir(parents=True)
    lines, rows = [], []
    for k in range(N):
        (scen / "images" / f"{k + 1:07d}.jpg").write_bytes(b"")
        b = gt_box(k)
        lines.append(f"{k + 1},{GT_ID},{int(b.x1)},{int(b.y1)},{int(b.width)},{int(b.height)},1,4,0,0")
        rows.append({"frame_idx": k, "source_frame": k + 1, "bbox": list(b.as_tuple()),
                     "occluded": OCC_LO <= k <= OCC_HI, "occluder_bbox": None})
    (scen / "annotations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "scenario_id": f"dev/{name}", "name": name, "group": "dev",
        "source": {"split_dir": "VAL", "sequence": seq, "first_source_frame": 1, "last_source_frame": N},
        "target": {"gt_target_id": GT_ID, "class": 3, "category": 4},
        "frames": N, "first_frame": 1,
        "occlusion": {"start_frame_idx": OCC_LO, "end_frame_idx": OCC_HI, "duration_frames": duration,
                      "occluder": "full_solid", "coverage": "full", "texture": "solid"},
        "ground_truth": rows, "seed": 1, "config_version": 2 if with_lock else 1, "config_sha256": "abc",
    }
    if with_lock:
        manifest["target_lock"] = {**LOCK, "required_detectors": ["y"], "occlusion_start_frame_idx": OCC_LO}
    (scen / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scen


def visible_frames(case: str) -> set[int]:
    return {
        "same_id": set(range(OCC_LO)) | set(range(OCC_HI + 1, N)),
        "unseen": set(),                                     # detector khong thay -> khong chon duoc
        "lost": {0, 1, 2} | set(range(OCC_HI + 1, N)),       # chon duoc, mat truoc khi che
    }[case]


def fake_tracks(case: str, source: str):
    out = []
    for k in range(N):
        tracks = [{"track_id": 9, "bbox": [500, 400, 540, 420], "conf": 0.9, "cls": 3}]
        if k in visible_frames(case):
            tracks.append({"track_id": 1, "bbox": list(gt_box(k).as_tuple()), "conf": 0.8, "cls": 3})
        out.append({"frame_idx": k, "path": str(Path(source) / f"{k + 1:07d}.jpg"), "tracks": tracks})
    return out


def write_tracks(monkeypatch, scen: Path, case: str, out: Path) -> Path:
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks(case, s)))
    log = out / "tracks.jsonl"
    assert tr.main(["--weights", "w.pt", "--source", str(scen / "images"), "--output", str(log),
                    "--imgsz", "960"]) == 0
    return log


def run(monkeypatch, scen: Path, exp: Path, tracks_log: Path, *extra: str, lock_args=True) -> Path:
    def no_yolo(*a, **k):
        raise AssertionError("replay khong duoc chay YOLO")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    argv = ["--weights", "w.pt", "--source", str(scen / "images"), "--replay-tracks", str(tracks_log),
            "--experiment-id", scen.name, "--output-root", str(exp / "scenarios"),
            "--gt-target-id", str(GT_ID), "--gt-annotation", str(scen / "annotations.txt"), *extra]
    if lock_args:
        argv += ["--gt-match-iou", str(LOCK["match_iou"]),
                 "--gt-max-search-frames", str(LOCK["max_search_frames"])]
    assert rp.main(argv) == 0
    return exp / "scenarios" / scen.name


@pytest.mark.parametrize("case, status", [("same_id", "locked"), ("unseen", "not_selected"),
                                          ("lost", "lost_before_occlusion")])
def test_run_lock_status_same_for_both_configs(tmp_path, monkeypatch, case, status):
    scen = make_scenario(tmp_path / "scn", f"s_{case}")
    log = write_tracks(monkeypatch, scen, case, tmp_path / "tracks" / case)
    res = {}
    for exp, extra in (("base", ()), ("mm", ("--motion-memory",))):
        res[exp] = ev.evaluate_occlusion_run(scen, run(monkeypatch, scen, tmp_path / exp, log, *extra), V2)
    for r in res.values():
        assert r["target_lock"]["status"] == status
        assert r["metrics"]["target_locked_at_occlusion"] is (status == "locked")
        assert r["scenario"]["config_version"] == 2


def test_v2_refuses_v1_manifest(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1", with_lock=False)
    log = write_tracks(monkeypatch, scen, "same_id", tmp_path / "t")
    run_dir = run(monkeypatch, scen, tmp_path / "e", log)
    with pytest.raises(EvaluationError, match="target_lock"):
        ev.evaluate_occlusion_run(scen, run_dir, V2)
    assert "target_locked_at_occlusion" not in ev.evaluate_occlusion_run(scen, run_dir, V1)["metrics"]


def test_v2_refuses_run_without_replay(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    monkeypatch.setattr(rp, "iter_tracks", lambda w, s, **k: iter(fake_tracks("same_id", s)))
    argv = ["--weights", "w.pt", "--source", str(scen / "images"), "--experiment-id", "s1",
            "--output-root", str(tmp_path / "e" / "scenarios"), "--gt-target-id", str(GT_ID),
            "--gt-annotation", str(scen / "annotations.txt"), "--imgsz", "960",
            "--gt-match-iou", "0.5", "--gt-max-search-frames", "4"]
    assert rp.main(argv) == 0
    with pytest.raises(EvaluationError, match="replay"):
        ev.evaluate_occlusion_run(scen, tmp_path / "e" / "scenarios" / "s1", V2)


def test_v2_refuses_selection_params_different_from_manifest(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    log = write_tracks(monkeypatch, scen, "same_id", tmp_path / "t")
    run_dir = run(monkeypatch, scen, tmp_path / "e", log, lock_args=False)  # 0.5 / 10 tu eval_visdrone
    with pytest.raises(EvaluationError, match="khac manifest.target_lock"):
        ev.evaluate_occlusion_run(scen, run_dir, V2)


def test_v2_refuses_modified_tracks_log(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    log = write_tracks(monkeypatch, scen, "same_id", tmp_path / "t")
    run_dir = run(monkeypatch, scen, tmp_path / "e", log)
    log.write_text(log.read_text() + "\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="SHA-256"):
        ev.evaluate_occlusion_run(scen, run_dir, V2)


# ---------------- CLI: aggregate + compare ----------------


def build(tmp_path, monkeypatch, cases=("same_id", "unseen", "lost")):
    names = []
    for i, case in enumerate(cases):
        scen = make_scenario(tmp_path / "scn", f"s{i}_{case}", duration=5 if i % 2 == 0 else 15,
                             seq=f"seq{i % 2}")
        names.append(scen.name)
        log = write_tracks(monkeypatch, scen, case, tmp_path / "tracks" / scen.name)
        for exp, extra in (("base", ()), ("mm", ("--motion-memory",))):
            run_dir = run(monkeypatch, scen, tmp_path / exp, log, *extra)
            # khong truyen --metrics-config: manifest co target_lock -> tu chon v2
            assert ev.main(["occlusion", "--scenario-dir", str(scen), "--run-dir", str(run_dir)]) == 0
    index = tmp_path / "scn" / "dev" / "index.json"
    index.write_text(json.dumps({"scenarios": names, "partial": False}))
    return index


def test_cli_v2_aggregate_and_compare(tmp_path, monkeypatch, capsys):
    index = build(tmp_path, monkeypatch)
    for exp in ("base", "mm"):
        assert ev.main(["aggregate", "--exp-dir", str(tmp_path / exp), "--index", str(index)]) == 0
    agg = json.loads((tmp_path / "base" / "metrics.json").read_text())
    assert agg["metrics_config"]["version"] == 2
    o = agg["overall"]
    assert o["scenarios_total"] == 3 and o["scenarios_evaluated"] == 1 and o["scenarios_not_locked"] == 2
    assert o["reidentification_rate"] == 1.0          # same_id duoc bat lai; 2 kich ban chua khoa khong tinh
    assert "2 chua khoa" in capsys.readouterr().out

    out = tmp_path / "cmp.json"
    assert ev.main(["compare", "--baseline", str(tmp_path / "base"), "--memory", str(tmp_path / "mm"),
                    "--output", str(out)]) == 0
    cmp_ = json.loads(out.read_text())
    assert cmp_["scenarios"] == 3 and cmp_["scenarios_evaluated"] == 1 and cmp_["scenarios_not_locked"] == 2


def test_compare_refuses_different_locked_sets(tmp_path, monkeypatch):
    build(tmp_path, monkeypatch, cases=("same_id", "unseen"))
    path = tmp_path / "mm" / "scenarios" / "s1_unseen" / "metrics.json"
    data = json.loads(path.read_text())
    data["metrics"]["target_locked_at_occlusion"] = True   # gia lap lech tap da khoa
    path.write_text(json.dumps(data))
    with pytest.raises(EvaluationError, match="Tap kich ban da khoa khac nhau"):
        ev.main(["compare", "--baseline", str(tmp_path / "base"), "--memory", str(tmp_path / "mm"),
                 "--output", str(tmp_path / "c.json")])
