"""Test do do occlusion (src/evaluate.py): ham thuan, danh gia log run_pipeline
+ manifest, gop nhieu kich ban, so sanh ablation.

Log va manifest deu gia lap; run_pipeline chay voi iter_tracks gia (khong YOLO).
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
from src.evaluate import (  # noqa: E402
    EvaluationError,
    aggregate_occlusion_metrics,
    compute_occlusion_metrics,
    evaluate_occlusion_run,
)
from src.motion_memory import BBox  # noqa: E402

METRICS_CFG = {"version": 1, "correct_iou": 0.5, "reid_min_consecutive_frames": 3, "path": "x"}


def box(cx: float, w: float = 40.0, h: float = 20.0, cy: float = 100.0) -> BBox:
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


GT12 = [box(100.0 + 3 * k) for k in range(12)]
OCC12 = [4 <= k <= 6 for k in range(12)]


# ---------------- ham thuan ----------------


def test_basic_gap_and_reid():
    preds = [None if 4 <= k <= 6 else GT12[k] for k in range(12)]
    m = compute_occlusion_metrics(preds, GT12, OCC12)
    assert m.occlusion_frames == 3 and m.correct_prediction_ratio == 0.0
    assert m.longest_tracking_gap == 3
    assert m.reidentified and m.time_to_reidentify_frames == 1
    assert m.mean_position_error_px is None and m.occluded_frames_with_output == 0
    assert m.per_frame_errors == [None, None, None]
    assert m.target_acquired_before_occlusion and m.first_correct_frame == 0


def test_correct_predictions_during_occlusion():
    m = compute_occlusion_metrics(list(GT12), GT12, OCC12)
    assert m.correct_prediction_ratio == 1.0 and m.longest_tracking_gap == 0
    assert m.mean_position_error_px == 0.0 and m.occluded_frames_with_output == 3


def test_never_reidentified():
    preds = [GT12[k] if k < 4 else None for k in range(12)]
    m = compute_occlusion_metrics(preds, GT12, OCC12)
    assert not m.reidentified and m.time_to_reidentify_frames is None
    assert m.longest_tracking_gap == 8  # frame 4..11


def test_reid_needs_consecutive_frames():
    ok = {0, 1, 2, 3, 7, 9, 10, 11}
    preds = [GT12[k] if k in ok else None for k in range(12)]
    m = compute_occlusion_metrics(preds, GT12, OCC12, reid_min_consecutive_frames=3)
    assert m.time_to_reidentify_frames == 3  # k = 9, b = 6
    m1 = compute_occlusion_metrics(preds, GT12, OCC12, reid_min_consecutive_frames=1)
    assert m1.time_to_reidentify_frames == 1


def test_reid_window_must_fit_in_scenario():
    preds = [GT12[k] if k < 4 or k >= 10 else None for k in range(12)]
    assert not compute_occlusion_metrics(preds, GT12, OCC12, reid_min_consecutive_frames=3).reidentified


def test_wrong_position_is_not_correct_but_has_error():
    preds = list(GT12)
    for k in (4, 5, 6):
        preds[k] = box(100.0 + 3 * k + 30.0)  # lech 30 px -> IoU < 0.5
    m = compute_occlusion_metrics(preds, GT12, OCC12)
    assert m.correct_prediction_ratio == 0.0
    assert m.mean_position_error_px == pytest.approx(30.0)
    assert m.position_error_sum_px == pytest.approx(90.0)


def test_extrapolation_after_occlusion_is_not_reidentification():
    """Output dung vi tri nhung la ngoai suy (khong xac nhan) -> khong tinh la bat lai."""
    preds = list(GT12)
    confirmed = [k < 4 for k in range(12)]  # sau frame 3 chi con ngoai suy
    m = compute_occlusion_metrics(preds, GT12, OCC12, confirmed=confirmed)
    assert m.correct_prediction_ratio == 1.0  # van tinh vao do do vi tri
    assert not m.reidentified and m.time_to_reidentify_frames is None
    confirmed2 = [k < 4 or k >= 8 for k in range(12)]
    assert compute_occlusion_metrics(preds, GT12, OCC12, confirmed=confirmed2).time_to_reidentify_frames == 2


def test_target_never_acquired():
    m = compute_occlusion_metrics([None] * 12, GT12, OCC12)
    assert m.longest_tracking_gap == 12 and not m.target_acquired_before_occlusion
    assert m.first_correct_frame is None


@pytest.mark.parametrize(
    "preds, gt, occ",
    [
        ([None] * 11, GT12, OCC12),                                  # thieu frame
        ([None] * 12, GT12[:11] + [None], OCC12),                    # GT thieu
        ([None] * 12, GT12, [False] * 12),                           # khong che
        ([None] * 12, GT12, [k in (2, 5) for k in range(12)]),       # khong lien tuc
    ],
)
def test_invalid_inputs(preds, gt, occ):
    with pytest.raises(ValueError):
        compute_occlusion_metrics(preds, gt, occ)


# ---------------- log run_pipeline + manifest ----------------

N, OCC_LO, OCC_HI = 20, 6, 10
GT_ID = 42


def gt_box(k: int) -> BBox:
    return box(100.0 + 3 * k)


def make_scenario(root: Path, name: str, duration_label: int = 5, group: str = "dev",
                  occluder: str = "full_solid") -> Path:
    scen = root / group / name
    (scen / "images").mkdir(parents=True)
    for k in range(N):
        (scen / "images" / f"{k + 1:07d}.jpg").write_bytes(b"")  # chi can ten file
    lines = []
    gt_rows = []
    for k in range(N):
        b = gt_box(k)
        lines.append(f"{k + 1},{GT_ID},{int(b.x1)},{int(b.y1)},{int(b.width)},{int(b.height)},1,4,0,0")
        occluded = OCC_LO <= k <= OCC_HI
        gt_rows.append({"frame_idx": k, "source_frame": k + 1, "bbox": list(b.as_tuple()),
                        "occluded": occluded, "occluder_bbox": list(b.as_tuple()) if occluded else None})
    (scen / "annotations.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "scenario_id": f"{group}/{name}", "name": name, "group": group,
        "source": {"split_dir": "VAL", "sequence": "seqA", "first_source_frame": 1, "last_source_frame": N},
        "target": {"gt_target_id": GT_ID, "class": 3, "category": 4},
        "frames": N, "first_frame": 1,
        "occlusion": {"start_frame_idx": OCC_LO, "end_frame_idx": OCC_HI,
                      "duration_frames": duration_label, "occluder": occluder,
                      "coverage": "full", "texture": "solid"},
        "ground_truth": gt_rows, "seed": 1, "config_version": 1, "config_sha256": "abc",
    }
    (scen / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return scen


def tracks_for(case: str, scen: Path = Path("s")):
    """BoT-SORT gia: ID 1 thay o frame 0..5, mat 6..10, sau do tuy case."""
    def trk(tid, k):
        b = gt_box(k)
        return {"track_id": tid, "bbox": list(b.as_tuple()), "conf": 0.8, "cls": 3}
    records = []
    for k in range(N):
        tracks = [{"track_id": 9, "bbox": [500, 400, 540, 420], "conf": 0.9, "cls": 3}]
        if k <= 5:
            tracks.append(trk(1, k))
        elif k > OCC_HI:
            if case == "new_id":
                tracks.append(trk(2, k))
            elif case == "same_id":
                tracks.append(trk(1, k))
        records.append({"frame_idx": k, "path": str(scen / "images" / f"{k + 1:07d}.jpg"),
                        "tracks": tracks})
    return records


def run_scenario(monkeypatch, scen: Path, exp_dir: Path, case: str, *extra: str) -> Path:
    monkeypatch.setattr(rp, "iter_tracks", lambda w, s, **k: iter(tracks_for(case, scen)))
    name = scen.name
    argv = ["--weights", "w.pt", "--source", str(scen / "images"), "--experiment-id", name,
            "--gt-target-id", str(GT_ID), "--gt-annotation", str(scen / "annotations.txt"),
            "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
            "--output-root", str(exp_dir / "scenarios"), "--imgsz", "960", *extra]
    assert rp.main(argv) == 0
    return exp_dir / "scenarios" / name


@pytest.mark.parametrize("case", ["new_id", "never", "same_id"])
def test_run_level_cases(tmp_path, monkeypatch, case):
    scen = make_scenario(tmp_path / "scn", f"s_{case}")
    base = evaluate_occlusion_run(scen, run_scenario(monkeypatch, scen, tmp_path / "base", case), METRICS_CFG)
    mm = evaluate_occlusion_run(
        scen, run_scenario(monkeypatch, scen, tmp_path / "mm", case, "--motion-memory"), METRICS_CFG
    )
    bm, mmm = base["metrics"], mm["metrics"]
    assert base["run"]["motion_memory"] is False and mm["run"]["motion_memory"] is True
    # baseline khong co output khi bi che; MM ngoai suy dung vi tri (van toc deu)
    assert bm["correct_prediction_ratio"] == 0.0
    assert mmm["correct_prediction_ratio"] == 1.0
    assert bm["target_acquired_before_occlusion"] and mmm["target_acquired_before_occlusion"]
    if case == "never":
        assert not bm["reidentified"] and not mmm["reidentified"]
    elif case == "same_id":
        assert bm["reidentified"] and mmm["reidentified"]
        assert not bm["reidentified_with_new_track_id"] and not mmm["reidentified_with_new_track_id"]
        assert mmm["recover_reasons"] == ["same_id"]
    else:  # new_id: baseline chi theo ID 1 -> that bai; MM khop ID 2 bang gate
        assert not bm["reidentified"]
        assert mmm["reidentified"] and mmm["time_to_reidentify_frames"] == 1
        assert mmm["reidentified_with_new_track_id"] is True
        assert mmm["track_id_before_occlusion"] == 1 and mmm["track_id_at_reidentify"] == 2
        assert mmm["recover_reasons"] == ["motion_gate"]


def test_run_rejects_wrong_source(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    other = make_scenario(tmp_path / "scn", "s2")
    run_dir = run_scenario(monkeypatch, other, tmp_path / "e", "same_id")
    with pytest.raises(EvaluationError):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


def test_run_rejects_truncated_log(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    log = run_dir / "logs" / "pipeline.jsonl"
    log.write_text("\n".join(log.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(EvaluationError):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


def test_run_rejects_appended_log(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")  # JsonlLogger ghi noi
    with pytest.raises(EvaluationError):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


def test_run_rejects_other_gt_target(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    sel = json.loads((run_dir / "target_selection.json").read_text())
    sel["gt_target_id"] = 7
    (run_dir / "target_selection.json").write_text(json.dumps(sel))
    with pytest.raises(EvaluationError):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


# ---------------- gop ----------------


def fake_item(name, duration, occ, correct, reid, ttr=None, group="dev", mm=False, conf=0.1):
    return {
        "scenario_id": f"{group}/{name}",
        "scenario": {"name": name, "group": group, "duration_frames": duration, "occluder": "full_solid"},
        "run": {"weights": "w", "tracker": "t", "conf": conf, "imgsz": 960, "classes": None,
                "motion_memory": mm, "motion_memory_config": None},
        "metrics_config": METRICS_CFG,
        "metrics": {
            "occlusion_frames": occ, "correct_occluded_frames": correct,
            "occluded_frames_with_output": correct, "position_error_sum_px": 2.0 * correct,
            "reidentified": reid, "time_to_reidentify_frames": ttr,
            "reidentified_with_new_track_id": False, "longest_tracking_gap": occ - correct,
            "target_acquired_before_occlusion": True,
        },
    }


def test_aggregate_pools_counts_not_ratio_means():
    items = [fake_item("a", 5, 5, 5, True, 1), fake_item("b", 60, 60, 0, False)]
    agg = aggregate_occlusion_metrics(items)
    o = agg["overall"]
    assert o["correct_prediction_ratio"] == pytest.approx(5 / 65)  # khong phai (1.0 + 0.0) / 2
    assert o["reidentified"] == 1 and o["reidentification_failed"] == 1
    assert o["mean_time_to_reidentify_frames"] == 1.0
    assert o["mean_position_error_px"] == pytest.approx(2.0)
    assert set(agg["by_duration_frames"]) == {"5", "60"}
    assert agg["by_duration_frames"]["60"]["correct_prediction_ratio"] == 0.0


def test_aggregate_rejects_mixed_groups():
    with pytest.raises(EvaluationError):
        aggregate_occlusion_metrics([fake_item("a", 5, 5, 5, True, 1),
                                     fake_item("b", 5, 5, 5, True, 1, group="heldout")])


def test_aggregate_rejects_mixed_run_settings():
    with pytest.raises(EvaluationError):
        aggregate_occlusion_metrics([fake_item("a", 5, 5, 5, True, 1),
                                     fake_item("b", 5, 5, 5, True, 1, conf=0.25)])


# ---------------- CLI + compare ----------------


def _build_experiments(tmp_path, monkeypatch, cases=("new_id", "never", "same_id"), mm_extra=()):
    scen_root = tmp_path / "scn"
    names = []
    for i, case in enumerate(cases):
        scen = make_scenario(scen_root, f"s{i}_{case}", duration_label=5 if i % 2 == 0 else 15)
        names.append(scen.name)
        for exp, extra in (("base", ()), ("mm", ("--motion-memory", *mm_extra))):
            run_dir = run_scenario(monkeypatch, scen, tmp_path / exp, case, *extra)
            assert ev.main(["occlusion", "--scenario-dir", str(scen), "--run-dir", str(run_dir)]) == 0
    index = scen_root / "dev" / "index.json"
    index.write_text(json.dumps({"scenarios": names, "partial": False}))
    return index


def test_cli_aggregate_and_compare(tmp_path, monkeypatch):
    index = _build_experiments(tmp_path, monkeypatch)
    for exp in ("base", "mm"):
        assert ev.main(["aggregate", "--exp-dir", str(tmp_path / exp), "--index", str(index)]) == 0
    agg_mm = json.loads((tmp_path / "mm" / "metrics.json").read_text())
    agg_base = json.loads((tmp_path / "base" / "metrics.json").read_text())
    assert agg_mm["completeness"]["complete"] is True
    assert agg_mm["overall"]["scenarios"] == 3
    assert agg_base["overall"]["reidentified"] == 1   # chi same_id
    assert agg_mm["overall"]["reidentified"] == 2     # same_id + new_id
    assert agg_mm["overall"]["reidentified_with_new_track_id"] == 1
    assert set(agg_mm["by_duration_frames"]) == {"5", "15"}

    out = tmp_path / "cmp.json"
    assert ev.main(["compare", "--baseline", str(tmp_path / "base"), "--memory", str(tmp_path / "mm"),
                    "--output", str(out)]) == 0
    cmp_ = json.loads(out.read_text())
    assert cmp_["scenarios"] == 3 and cmp_["warnings"] == []


def test_aggregate_refuses_missing_scenarios(tmp_path, monkeypatch):
    index = _build_experiments(tmp_path, monkeypatch, cases=("same_id",))
    data = json.loads(index.read_text())
    data["scenarios"].append("s9_missing")
    index.write_text(json.dumps(data))
    with pytest.raises(EvaluationError):
        ev.main(["aggregate", "--exp-dir", str(tmp_path / "base"), "--index", str(index)])
    assert ev.main(["aggregate", "--exp-dir", str(tmp_path / "base"), "--index", str(index),
                    "--allow-incomplete"]) == 0
    agg = json.loads((tmp_path / "base" / "metrics.json").read_text())
    assert agg["completeness"]["complete"] is False
    assert agg["completeness"]["missing"] == ["s9_missing"]


def test_aggregate_requires_index_or_flag(tmp_path, monkeypatch):
    _build_experiments(tmp_path, monkeypatch, cases=("same_id",))
    with pytest.raises(EvaluationError):
        ev.main(["aggregate", "--exp-dir", str(tmp_path / "base")])


def test_compare_refuses_mismatched_settings(tmp_path, monkeypatch):
    _build_experiments(tmp_path, monkeypatch, cases=("same_id",), mm_extra=("--conf", "0.3"))
    with pytest.raises(EvaluationError, match="conf"):
        ev.main(["compare", "--baseline", str(tmp_path / "base"), "--memory", str(tmp_path / "mm"),
                 "--output", str(tmp_path / "c.json")])


def test_old_cli_still_parses():
    with pytest.raises(SystemExit):
        ev.parse_args([])  # van can --log/--split/... nhu cu


def test_run_rejects_wrong_image_order(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    log = run_dir / "logs" / "pipeline.jsonl"
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    rows[3]["path"], rows[4]["path"] = rows[4]["path"], rows[3]["path"]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(EvaluationError, match="Thu tu anh"):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


def test_run_rejects_missing_image(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    (scen / "images" / "0000005.jpg").unlink()
    with pytest.raises(EvaluationError, match="Thu tu anh"):
        evaluate_occlusion_run(scen, run_dir, METRICS_CFG)


def test_aggregate_keeps_provenance():
    """Metrics tung kich ban khong commit -> ban gop phai giu commit/version."""
    a, b = fake_item("a", 5, 5, 5, True, 1), fake_item("b", 5, 5, 0, False)
    a["run"].update(git_commit="c1", git_dirty=False, versions={"ultralytics": "8.4.174"},
                    tracks_sha256="s1")
    b["run"].update(git_commit="c2", git_dirty=True, versions={"ultralytics": "8.4.174"},
                    tracks_sha256="s2")
    prov = aggregate_occlusion_metrics([a, b])["provenance"]
    assert prov["git_commits"] == ["c1", "c2"] and prov["git_dirty_runs"] == 1
    assert prov["versions"] == [{"ultralytics": "8.4.174"}] and prov["tracks_logs"] == 2


def test_scenario_metrics_record_versions(tmp_path, monkeypatch):
    scen = make_scenario(tmp_path / "scn", "s1")
    run_dir = run_scenario(monkeypatch, scen, tmp_path / "e", "same_id")
    run = evaluate_occlusion_run(scen, run_dir, METRICS_CFG)["run"]
    assert "versions" in run and "git_dirty" in run and run["tracks_sha256"] is None
