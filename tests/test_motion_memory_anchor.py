"""Unit test anchor (ghi nho ID goc) cua Motion Memory - config.anchor_recovery.

Kich ban chung (giong test_motion_memory_rematch.py): muc tieu ID 1, class 0,
bbox 40x20, di sang phai 3 px/frame o frame 0..9 (tam cuoi (127, 200)),
velocity_decay = 1. Frame 10 mat -> LOST; frame 11 ung vien ID 7 dung vi tri
du doan -> motion_gate sang ID 7 (anchor = 1). Moi lan goi co frame_idx.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.motion_memory import (  # noqa: E402
    RECOVER_ANCHOR_ID,
    RECOVER_MOTION_GATE,
    BBox,
    MotionMemory,
    MotionMemoryConfig,
    Observation,
    TrackState,
)

W, H = 40.0, 20.0
FAR_X = 600.0   # ID goc quay lai o vi tri bat ky (khong co dieu kien khoang cach)


def det(frame_idx: int, cx: float, cy: float, track_id: int, cls: Optional[int] = 0) -> Observation:
    return Observation(bbox=BBox(cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2), frame_idx=frame_idx,
                       track_id=track_id, detector_confidence=0.7, cls=cls)


def hijacked(anchor: bool = True, **cfg) -> MotionMemory:
    """Memory dang TRACKING tren ID 7 sau khi motion_gate o frame 11 (anchor = 1 neu bat)."""
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0, anchor_recovery=anchor, **cfg))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1), frame_idx=i)
    assert memory.update(None, frame_idx=10).state is TrackState.LOST
    pred = memory.update(None, [det(11, 133.0, 200.0, track_id=7)], frame_idx=11)
    assert pred.track_id == 7 and pred.recover_reason == RECOVER_MOTION_GATE
    return memory


def test_default_is_off_and_validated():
    assert MotionMemoryConfig().anchor_recovery is False
    memory = hijacked(anchor=False)
    assert memory.anchor_id is None


def test_anchor_set_at_motion_gate_switch():
    assert hijacked().anchor_id == 1


def test_anchor_returns_while_tracking_other_id_switches_back():
    memory = hijacked()
    back = det(12, FAR_X, 50.0, track_id=1)
    pred = memory.update(det(12, 136.0, 200.0, track_id=7), [back], frame_idx=12)   # B van duoc thay
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 1 and memory.track_id == 1
    assert pred.recover_reason == RECOVER_ANCHOR_ID
    assert pred.bbox == back.bbox and pred.frames_since_confirmed == 0
    assert memory.anchor_id is None                                                  # xoa khi quay ve


def test_flag_off_keeps_old_behaviour_on_new_id():
    memory = hijacked(anchor=False)
    pred = memory.update(det(12, 136.0, 200.0, track_id=7), [det(12, FAR_X, 50.0, track_id=1)], frame_idx=12)
    assert pred.track_id == 7 and pred.recover_reason is None


def test_anchor_returns_in_reidentifying():
    memory = hijacked(lost_timeout_frames=3, reid_timeout_frames=5)
    for f in range(12, 16):                                     # k = 1..4 (tu frame 11) -> REIDENTIFYING
        pred = memory.update(None, frame_idx=f)
    assert pred.state is TrackState.REIDENTIFYING
    near = det(16, 148.0, 200.0, track_id=9)                     # ID khac gan du doan: REID khong dung gate
    pred = memory.update(None, [near], frame_idx=16)
    assert pred.state is TrackState.REIDENTIFYING and pred.track_id == 7
    pred = memory.update(None, [near, det(17, FAR_X, 50.0, track_id=1)], frame_idx=17)
    assert pred.state is TrackState.TRACKING and pred.track_id == 1
    assert pred.recover_reason == RECOVER_ANCHOR_ID


def test_current_id_still_recovers_in_reidentifying_and_keeps_anchor():
    memory = hijacked(lost_timeout_frames=3, reid_timeout_frames=5)
    for f in range(12, 16):
        memory.update(None, frame_idx=f)
    pred = memory.update(det(16, 160.0, 200.0, track_id=7), frame_idx=16)
    assert pred.track_id == 7 and pred.recover_reason == "same_id"
    assert memory.anchor_id == 1


def test_anchor_returning_after_failed_is_ignored():
    memory = hijacked(lost_timeout_frames=3, reid_timeout_frames=2)
    for f in range(12, 18):                                     # k = 6 > 3 + 2 -> FAILED
        pred = memory.update(None, frame_idx=f)
    assert pred.state is TrackState.FAILED and memory.anchor_id is None
    pred = memory.update(None, [det(18, FAR_X, 50.0, track_id=1)], frame_idx=18)
    assert pred.state is TrackState.FAILED and pred.bbox is None


def test_chain_b_to_c_keeps_original_anchor():
    memory = hijacked()
    assert memory.update(None, frame_idx=12).state is TrackState.LOST
    pred = memory.update(None, [det(13, 139.0, 200.0, track_id=8)], frame_idx=13)   # gate 7 -> 8
    assert pred.track_id == 8 and pred.recover_reason == RECOVER_MOTION_GATE
    assert memory.anchor_id == 1
    pred = memory.update(det(14, 142.0, 200.0, track_id=8), [det(14, FAR_X, 50.0, track_id=1)], frame_idx=14)
    assert pred.track_id == 1 and pred.recover_reason == RECOVER_ANCHOR_ID


def test_anchor_returns_while_lost_on_new_id():
    memory = hijacked()
    assert memory.update(None, frame_idx=12).state is TrackState.LOST
    pred = memory.update(None, [det(13, FAR_X, 50.0, track_id=1)], frame_idx=13)
    assert pred.track_id == 1 and pred.recover_reason == RECOVER_ANCHOR_ID


def test_original_id_returning_without_hijack_is_plain_same_id():
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0, anchor_recovery=True))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1), frame_idx=i)
    memory.update(None, frame_idx=10)
    assert memory.anchor_id is None                              # LOST khong dat anchor
    pred = memory.update(det(11, 133.0, 200.0, track_id=1), frame_idx=11)
    assert pred.recover_reason == "same_id" and memory.anchor_id is None


def test_reset_clears_anchor():
    memory = hijacked()
    memory.reset()
    assert memory.anchor_id is None and memory.track_id is None


def test_settings_file_enables_flag(tmp_path):
    from src.utils.motion_memory_settings import resolve_motion_memory_config

    path = tmp_path / "mm.yaml"
    path.write_text("motion_memory:\n  anchor_recovery: true\n", encoding="utf-8")
    assert resolve_motion_memory_config(None, path).config.anchor_recovery is True
    path.write_text("motion_memory:\n  anchor_recovery: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="true/false"):
        resolve_motion_memory_config(None, path)
    default = REPO_ROOT / "configs" / "motion_memory.yaml"     # file mac dinh khong co khoa -> tat
    assert resolve_motion_memory_config(None, default).config.anchor_recovery is False
