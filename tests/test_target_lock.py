"""Test dinh nghia "khoa duoc truoc khi che" (src/utils/gt_target.evaluate_target_lock).

Log track va GT deu gia lap, khong chay YOLO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.motion_memory import BBox  # noqa: E402
from src.utils.gt_target import (  # noqa: E402
    LOCK_LOCKED,
    LOCK_LOST,
    LOCK_NOT_SELECTED,
    LockRule,
    evaluate_target_lock,
)

A = 10  # frame_idx dau tien bi che
RULE = LockRule(match_iou=0.5, max_search_frames=4, lock_confirm_frames=3, lock_iou=0.5)


def gt(k: int) -> BBox:
    return BBox(100.0 + 2 * k, 50.0, 140.0 + 2 * k, 70.0)


GT = {k + 1: gt(k) for k in range(A + 5)}  # frame VisDrone = frame_idx + 1


def trk(tid, b: BBox, cls=3, dx=0.0):
    return {"track_id": tid, "bbox": [b.x1 + dx, b.y1, b.x2 + dx, b.y2], "conf": 0.8, "cls": cls}


def records(visible, tid=1, cls=3, dx=0.0, n=A + 5):
    """Track `tid` khop GT o cac frame trong `visible`; them mot track nhieu (ID 9)."""
    out = []
    for k in range(n):
        tracks = [{"track_id": 9, "bbox": [500, 400, 540, 420], "conf": 0.9, "cls": 0}]
        if k in visible:
            tracks.append(trk(tid, gt(k), cls=cls, dx=dx))
        out.append({"frame_idx": k, "tracks": tracks})
    return out


def test_locked_when_selected_and_tracked_until_occlusion():
    res = evaluate_target_lock(records(set(range(A))), GT, RULE, A, gt_target_id=5)
    assert res["status"] == LOCK_LOCKED and res["locked"] is True
    assert res["selection"]["selected_track_id"] == 1 and res["selection"]["status"] == "selected"
    assert res["lock_ious"] == [pytest.approx(1.0)] * 3
    assert res["frames_with_match"] == A and res["best_iou"] == pytest.approx(1.0)
    assert res["selected_class"] == 3


def test_not_selected_when_target_never_tracked():
    res = evaluate_target_lock(records(set()), GT, RULE, A)
    assert res["status"] == LOCK_NOT_SELECTED and not res["locked"]
    assert res["selection"]["status"] == "failed" and res["lock_ious"] == []
    assert res["best_iou"] == 0.0 and res["frames_with_match"] == 0


def test_not_selected_when_first_seen_after_search_window():
    # thay tu frame 5 tro di, cua so tim chi 4 frame dau (0..3)
    res = evaluate_target_lock(records(set(range(5, A))), GT, RULE, A)
    assert res["status"] == LOCK_NOT_SELECTED
    assert res["frames_with_match"] == A - 5  # van ghi lai de biet ly do loai


def test_lost_before_occlusion():
    # chon duoc o frame 0, mat o frame 8 (trong 3 frame cuoi 7, 8, 9)
    res = evaluate_target_lock(records(set(range(A)) - {8}), GT, RULE, A)
    assert res["status"] == LOCK_LOST and not res["locked"]
    assert res["lock_ious"][1] is None


def test_lost_when_iou_below_lock_iou_in_last_frames():
    recs = records(set(range(A)))
    for k in range(A - 3, A):  # lech 30 px o 3 frame cuoi -> IoU < 0.5
        recs[k]["tracks"][1] = trk(1, gt(k), dx=30.0)
    res = evaluate_target_lock(recs, GT, RULE, A)
    assert res["status"] == LOCK_LOST
    assert all(v < 0.5 for v in res["lock_ious"])


def test_class_agnostic_but_class_recorded():
    # track co class khac class GT van duoc chon (khong xet class), class duoc ghi lai
    res = evaluate_target_lock(records(set(range(A)), cls=7), GT, RULE, A)
    assert res["locked"] and res["selected_class"] == 7


def test_frames_after_occlusion_start_are_ignored():
    # muc tieu chi xuat hien SAU khi che bat dau -> khong khoa duoc
    res = evaluate_target_lock(records(set(range(A, A + 5))), GT, RULE, A)
    assert res["status"] == LOCK_NOT_SELECTED and res["frames_with_match"] == 0


def test_missing_pre_occlusion_frames_rejected():
    with pytest.raises(ValueError):
        evaluate_target_lock(records(set(range(A)), n=A - 1), GT, RULE, A)


def test_rule_validation_and_mapping():
    with pytest.raises(ValueError):
        LockRule(match_iou=0.0, max_search_frames=1, lock_confirm_frames=1, lock_iou=0.5)
    with pytest.raises(ValueError):
        LockRule(match_iou=0.5, max_search_frames=1, lock_confirm_frames=0, lock_iou=0.5)
    with pytest.raises(ValueError):
        LockRule.from_mapping({"match_iou": 0.5})
    rule = LockRule.from_mapping({**RULE.as_dict(), "required_detectors": ["a"]})
    assert rule == RULE
