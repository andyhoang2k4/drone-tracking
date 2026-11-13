"""Test ground truth VisDrone-MOT + ghep cap + precision/recall + ID switch
(src/evaluate.py, src/utils/visdrone.py). Du lieu gia, khong chay model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import (  # noqa: E402
    EvalBox,
    compute_detection_metrics,
    compute_id_switches,
    intersection_over_area,
    load_ground_truth,
    match_frame,
    match_sequence,
)
from src.motion_memory import BBox  # noqa: E402
from src.utils.visdrone import (  # noqa: E402
    frame_index_from_path,
    is_ignore_region,
    parse_annotation_line,
    read_annotation_file,
)

CATEGORY_MAP = {c: c - 1 for c in range(1, 11)}


def box(x1, y1, x2, y2, cls=0, obj_id=None) -> EvalBox:
    return EvalBox(BBox(x1, y1, x2, y2), cls, obj_id)


# ---------------- ground truth ----------------


def test_load_ground_truth_splits_valid_and_ignored(tmp_path: Path):
    ann = tmp_path / "seq.txt"
    ann.write_text(
        "1,7,10,20,30,40,1,4,0,0\n"      # car -> class 3
        "1,0,100,100,50,50,0,0,0,0\n"    # ignored region (score 0, cat 0)
        "1,9,200,200,10,10,0,11,0,0\n"   # others, score 0 -> bo qua
        "2,7,12,20,30,40,1,4,0,1\n"
        "3,0,0,0,5,5,0,0,0,0\n",         # frame chi co vung bo qua
        encoding="utf-8",
    )
    frames = load_ground_truth(ann, CATEGORY_MAP)
    assert sorted(frames) == [1, 2, 3]
    assert frames[1].boxes == [EvalBox(BBox(10, 20, 40, 60), 3, 7)]
    assert frames[1].ignore_regions == [BBox(100, 100, 150, 150), BBox(200, 200, 210, 210)]
    assert frames[3].boxes == [] and len(frames[3].ignore_regions) == 1


def test_is_ignore_region_rule():
    def b(score, cat):
        return parse_annotation_line(f"1,1,0,0,5,5,{score},{cat},0,0")

    assert not is_ignore_region(b(1, 4), CATEGORY_MAP)
    assert is_ignore_region(b(0, 4), CATEGORY_MAP)      # score 0
    assert is_ignore_region(b(1, 0), CATEGORY_MAP)      # ignored region
    assert is_ignore_region(b(1, 11), CATEGORY_MAP)     # ngoai category_map


def test_read_annotation_file_reports_line(tmp_path: Path):
    ann = tmp_path / "bad.txt"
    ann.write_text("1,7,10,20,30,40,1,4,0,0\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.txt:2"):
        read_annotation_file(ann)


def test_frame_index_from_path():
    assert frame_index_from_path("a/uav0000086_00000_v/0000007.jpg") == 7
    with pytest.raises(ValueError):
        frame_index_from_path("a/clip.mp4")


def test_prepare_visdrone_reexports_parser():
    import scripts.prepare_visdrone as prep
    import src.utils.visdrone as vd

    assert prep.parse_annotation_line is vd.parse_annotation_line
    assert prep.VisDroneBox is vd.VisDroneBox


# ---------------- ghep cap ----------------


def test_intersection_over_area():
    assert intersection_over_area(BBox(0, 0, 10, 10), BBox(5, 0, 20, 10)) == 0.5
    assert intersection_over_area(BBox(0, 0, 0, 10), BBox(0, 0, 10, 10)) == 0.0


def test_greedy_match_highest_iou_first():
    gts = [box(0, 0, 10, 10, obj_id=1)]
    preds = [box(1, 0, 11, 10), box(0, 0, 10, 10)]   # pred 1 khop hoan hao
    m = match_frame(preds, gts)
    assert m.matches == [(1, 0)] and m.false_positives == [0] and m.false_negatives == []


def test_iou_threshold_and_class_aware():
    gts = [box(0, 0, 10, 10, cls=3)]
    # IoU = 50/150 = 0.333 < 0.5
    assert match_frame([box(5, 0, 15, 10, cls=3)], gts).matches == []
    # Khac class: class_aware -> khong ghep; agnostic -> ghep
    assert match_frame([box(0, 0, 10, 10, cls=4)], gts).matches == []
    assert match_frame([box(0, 0, 10, 10, cls=4)], gts, class_aware=False).matches == [(0, 0)]


def test_prediction_in_ignore_region_is_dropped_not_fp():
    region = [BBox(100, 100, 200, 200)]
    preds = [box(110, 110, 150, 150),   # IoA 1.0 -> loai
             box(180, 100, 220, 140),   # IoA 0.5 -> loai (>= nguong)
             box(190, 100, 230, 140)]   # IoA 0.25 -> van tinh FP
    m = match_frame(preds, [], region)
    assert m.ignored_predictions == [0, 1]
    assert m.false_positives == [2]


def test_ground_truth_in_ignore_region_is_dropped_not_fn():
    gts = [box(110, 110, 150, 150, obj_id=1), box(0, 0, 10, 10, obj_id=2)]
    m = match_frame([], gts, [BBox(100, 100, 200, 200)])
    assert m.ignored_ground_truth == [0] and m.false_negatives == [1]


def test_detection_metrics_precision_recall():
    gt = [[box(0, 0, 10, 10, obj_id=1), box(50, 50, 60, 60, obj_id=2)], [box(0, 0, 10, 10, obj_id=1)]]
    pred = [[box(0, 0, 10, 10, obj_id=5), box(80, 80, 90, 90, obj_id=6)], [box(0, 0, 10, 10, obj_id=5)]]
    regions = [[BBox(85, 85, 200, 200)], []]   # pred (80,80,90,90): IoA 0.25 -> van FP
    m = compute_detection_metrics(pred, gt, regions)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (2, 1, 1)
    assert m.precision == pytest.approx(2 / 3) and m.recall == pytest.approx(2 / 3)
    assert m.ignored_predictions == 0


def test_detection_metrics_empty():
    m = compute_detection_metrics([[]], [[]])
    assert m.precision == 0.0 and m.recall == 0.0


def test_match_sequence_length_mismatch():
    with pytest.raises(ValueError):
        match_sequence([[]], [[], []])


# ---------------- ID switch ----------------


def _seq(track_ids):
    """Mot GT (target 1) dung yen; track ID cua prediction theo tung frame (None = khong co pred)."""
    gt = [[box(0, 0, 10, 10, obj_id=1)] for _ in track_ids]
    pred = [[] if t == "miss" else [box(0, 0, 10, 10, obj_id=t)] for t in track_ids]
    return pred, gt


@pytest.mark.parametrize("track_ids,expected", [
    ([3, 3, 3], 0),
    ([3, 4, 4], 1),
    ([3, "miss", "miss", 4], 1),   # mat roi bat lai sai ID
    ([3, 4, 3], 2),                # doi qua lai: moi lan doi +1
    ([None, 3, None, 3], 0),       # pred khong co track ID: bo qua
])
def test_id_switches(track_ids, expected):
    pred, gt = _seq(track_ids)
    assert compute_id_switches(pred, gt) == expected


def test_id_switches_independent_per_target():
    gt = [[box(0, 0, 10, 10, obj_id=1), box(50, 50, 60, 60, obj_id=2)]] * 2
    pred = [[box(0, 0, 10, 10, obj_id=7), box(50, 50, 60, 60, obj_id=8)],
            [box(0, 0, 10, 10, obj_id=7), box(50, 50, 60, 60, obj_id=9)]]
    assert compute_id_switches(pred, gt) == 1


def test_id_switch_ignored_when_target_inside_ignore_region():
    pred, gt = _seq([3, 4])
    regions = [[], [BBox(-5, -5, 20, 20)]]   # frame 2: GT + pred deu trong vung bo qua
    assert compute_id_switches(pred, gt, regions) == 0
