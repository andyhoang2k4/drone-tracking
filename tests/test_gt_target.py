"""Test chon muc tieu theo ground truth (src/utils/gt_target.py + run_pipeline).

Annotation VisDrone va log BoT-SORT deu gia lap, khong chay YOLO.
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
from src.motion_memory import BBox  # noqa: E402
from src.utils.gt_target import (  # noqa: E402
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SELECTED,
    GtTargetSelector,
    load_target_boxes,
)

CATEGORY_MAP = {c: c - 1 for c in range(1, 11)}


def ann_line(frame, tid, left, top, w=40, h=20, score=1, cat=4):
    return f"{frame},{tid},{left},{top},{w},{h},{score},{cat},0,0"


def write_annotation(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "seq.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def trk(tid, x1, y1, w=40, h=20, cls=3):
    return {"track_id": tid, "bbox": [x1, y1, x1 + w, y1 + h], "conf": 0.8, "cls": cls}


# ---------------- load_target_boxes ----------------


def test_load_target_boxes_filters_target_and_ignore(tmp_path):
    path = write_annotation(tmp_path, [
        ann_line(1, 5, 100, 100),
        ann_line(2, 5, 103, 100),
        ann_line(1, 6, 300, 300),            # muc tieu khac
        ann_line(3, 5, 106, 100, score=0),   # vung bo qua -> loai
        ann_line(4, 5, 109, 100, cat=11),    # "others" -> loai
    ])
    boxes = load_target_boxes(path, 5, CATEGORY_MAP)
    assert set(boxes) == {1, 2}
    assert boxes[2] == BBox(103.0, 100.0, 143.0, 120.0)


def test_load_target_boxes_missing_target(tmp_path):
    path = write_annotation(tmp_path, [ann_line(1, 6, 0, 0)])
    with pytest.raises(ValueError):
        load_target_boxes(path, 5, CATEGORY_MAP)


def test_load_target_boxes_duplicate_frame(tmp_path):
    path = write_annotation(tmp_path, [ann_line(1, 5, 0, 0), ann_line(1, 5, 9, 9)])
    with pytest.raises(ValueError):
        load_target_boxes(path, 5, CATEGORY_MAP)


# ---------------- GtTargetSelector ----------------


def selector(gt: dict[int, BBox], **kw) -> GtTargetSelector:
    params = dict(match_iou=0.5, max_search_frames=3, first_frame=1, gt_target_id=5)
    params.update(kw)
    return GtTargetSelector(gt_boxes=gt, **params)


GT = {f: BBox(100.0 + 3 * (f - 3), 100.0, 140.0 + 3 * (f - 3), 120.0) for f in range(3, 20)}


def test_selects_best_iou_on_first_gt_frame():
    s = selector(GT)
    assert s.offer({"frame_idx": 0, "tracks": [trk(1, 100, 100)]}) is None  # frame 1: chua co GT
    assert s.status == STATUS_PENDING
    rec = {"frame_idx": 2, "tracks": [trk(8, 104, 100), trk(4, 100, 101), trk(9, 400, 400)]}
    assert s.offer(rec) == 4  # IoU cao nhat
    assert s.status == STATUS_SELECTED
    assert s.selected_frame_idx == 2 and s.selected_iou > 0.9
    # da chon -> giu nguyen ID du frame sau co track khop hon
    assert s.offer({"frame_idx": 3, "tracks": [trk(8, 103, 100)]}) == 4


def test_tie_breaks_on_smaller_track_id():
    s = selector(GT)
    rec = {"frame_idx": 2, "tracks": [trk(7, 100, 100), trk(3, 100, 100)]}
    assert s.offer(rec) == 3


def test_waits_for_track_within_window():
    s = selector(GT, max_search_frames=3)
    assert s.offer({"frame_idx": 2, "tracks": []}) is None          # BoT-SORT chua xuat track
    assert s.offer({"frame_idx": 3, "tracks": [trk(None, 103, 100)]}) is None  # chua co ID
    assert s.offer({"frame_idx": 4, "tracks": [trk(2, 106, 100)]}) == 2
    assert s.frames_searched == 3


def test_fails_after_window():
    s = selector(GT, max_search_frames=2)
    s.offer({"frame_idx": 2, "tracks": []})
    s.offer({"frame_idx": 3, "tracks": []})
    assert s.offer({"frame_idx": 4, "tracks": [trk(2, 106, 100)]}) is None
    assert s.status == STATUS_FAILED
    assert s.as_dict()["selected_track_id"] is None


def test_low_iou_not_selected():
    s = selector(GT, match_iou=0.5)
    assert s.offer({"frame_idx": 2, "tracks": [trk(1, 125, 100)]}) is None  # IoU ~0.23


def test_first_frame_offset():
    s = selector(GT, first_frame=3)  # frame_idx 0 <-> frame VisDrone 3
    assert s.offer({"frame_idx": 0, "tracks": [trk(6, 100, 100)]}) == 6


def test_finish_marks_pending_as_failed():
    s = selector(GT)
    s.offer({"frame_idx": 0, "tracks": []})
    s.finish()
    assert s.status == STATUS_FAILED


@pytest.mark.parametrize("kw", [{"match_iou": 0.0}, {"max_search_frames": 0}])
def test_invalid_params(kw):
    with pytest.raises(ValueError):
        selector(GT, **kw)


# ---------------- run_pipeline --gt-target-id ----------------


def switch_records():
    """GT target 5 tai frame VisDrone 3.. (frame_idx 2..). BoT-SORT: frame_idx
    2 chua xuat, ID 11 tu frame_idx 3..9, mat 10, ID 12 tu 11.. cung vi tri GT.
    Vat khac ID 1 o xa moi frame."""
    records = []
    for f in range(16):
        x = 100.0 + 3 * (f - 2)
        tracks = [trk(1, 500, 400)]
        if 3 <= f <= 9:
            tracks.append(trk(11, x, 100))
        elif f >= 11:
            tracks.append(trk(12, x, 100))
        records.append({"frame_idx": f, "tracks": tracks})
    return records


def gt_annotation(tmp_path):
    lines = [ann_line(f + 1, 5, 100 + 3 * (f - 2), 100) for f in range(2, 16)]
    lines += [ann_line(f + 1, 1, 500, 400) for f in range(16)]
    return write_annotation(tmp_path, lines)


def run(tmp_path, monkeypatch, *extra, exp="exp"):
    monkeypatch.setattr(rp, "iter_tracks", lambda w, s, **k: iter(switch_records()))
    argv = ["--weights", "w.pt", "--source", "seq", "--experiment-id", exp,
            "--tracker", str(REPO_ROOT / "configs" / "botsort.yaml"),
            "--output-root", str(tmp_path), "--imgsz", "960", *extra]
    assert rp.main(argv) == 0
    d = tmp_path / exp
    rows = [json.loads(x) for x in (d / "logs" / "pipeline.jsonl").read_text().splitlines()]
    selection = json.loads((d / "target_selection.json").read_text())
    return rows, selection


def test_pipeline_gt_target_baseline(tmp_path, monkeypatch):
    ann = gt_annotation(tmp_path)
    rows, sel = run(tmp_path, monkeypatch, "--gt-target-id", "5", "--gt-annotation", str(ann))
    assert sel["status"] == STATUS_SELECTED and sel["selected_track_id"] == 11
    assert sel["selected_frame_idx"] == 3
    states = [r["state"] for r in rows]
    assert states[:3] == ["LOST"] * 3              # truoc khi chon
    assert states[3:10] == ["TRACKING"] * 7
    assert states[10:] == ["LOST"] * 6             # baseline khong theo ID 12
    assert set(rows[0]) == {"frame_idx", "detected", "tracker_bbox", "detector_confidence", "state"}


def test_pipeline_gt_target_with_memory_follows_new_id(tmp_path, monkeypatch):
    ann = gt_annotation(tmp_path)
    rows, sel = run(tmp_path, monkeypatch, "--gt-target-id", "5", "--gt-annotation", str(ann),
                    "--motion-memory")
    assert sel["selected_track_id"] == 11
    assert [r["state"] for r in rows[:3]] == [None] * 3
    assert rows[3]["state"] == "TRACKING" and rows[3]["track_id"] == 11
    assert rows[10]["state"] == "LOST"
    assert rows[11]["track_id"] == 12 and rows[11]["recover_reason"] == "motion_gate"


def test_pipeline_cli_target_writes_selection(tmp_path, monkeypatch):
    _, sel = run(tmp_path, monkeypatch, "--target-track-id", "11")
    assert sel == {"mode": "cli", "target_track_id": 11}


def test_pipeline_gt_selection_failure(tmp_path, monkeypatch):
    ann = write_annotation(tmp_path, [ann_line(f + 1, 5, 900, 900) for f in range(16)])
    rows, sel = run(tmp_path, monkeypatch, "--gt-target-id", "5", "--gt-annotation", str(ann),
                    "--motion-memory")
    assert sel["status"] == STATUS_FAILED
    assert all(r["state"] is None for r in rows)


def test_target_args_mutually_exclusive():
    base = ["--weights", "w", "--source", "s", "--experiment-id", "e"]
    with pytest.raises(SystemExit):
        rp.parse_args(base)  # thieu ca hai
    with pytest.raises(SystemExit):
        rp.parse_args(base + ["--target-track-id", "1", "--gt-target-id", "2",
                              "--gt-annotation", "a.txt"])
    with pytest.raises(SystemExit):
        rp.parse_args(base + ["--gt-target-id", "2"])  # thieu --gt-annotation
    args = rp.parse_args(base + ["--target-track-id", "1"])
    assert args.target_track_id == 1 and args.gt_target_id is None
