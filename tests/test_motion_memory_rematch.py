"""Unit test cho khop lai bang gating chuyen dong trong LOST (roadmap buoc 5).

Du lieu gia lap, khong chay YOLO. Kich ban chung: muc tieu ID 1, class 0,
bbox 40x20 (scale = 30 px), di sang phai 3 px/frame o frame 0..9 (tam cuoi
(127, 200)); velocity_decay = 1 de so de tinh:
    predicted center tai gap g = (127 + 3g, 200)
    ban kinh gate = 30 * 0.5 * (1 + 0.1 g)
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
    BBox,
    MotionMemory,
    MotionMemoryConfig,
    Observation,
    TrackState,
)

W, H = 40.0, 20.0


def det(frame_idx: int, cx: float, cy: float, track_id: int, cls: Optional[int] = 0,
        w: float = W, h: float = H) -> Observation:
    return Observation(
        bbox=BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        frame_idx=frame_idx,
        track_id=track_id,
        detector_confidence=0.7,
        cls=cls,
    )


def predicted_x(gap: int) -> float:
    return 127.0 + 3.0 * gap


@pytest.fixture
def lost_memory() -> MotionMemory:
    """Memory dang LOST sau khi mat muc tieu o frame 10."""
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1))
    assert memory.update(None).state is TrackState.LOST  # frame 10
    return memory


def test_matches_new_id_at_predicted_position(lost_memory: MotionMemory):
    cand = det(11, predicted_x(2), 200.0, track_id=7)
    pred = lost_memory.update(None, candidates=[cand])
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 7
    assert lost_memory.track_id == 7
    assert pred.bbox == cand.bbox
    assert pred.frames_since_confirmed == 0
    assert pred.confidence == 1.0


def test_after_match_follows_new_id(lost_memory: MotionMemory):
    lost_memory.update(None, candidates=[det(11, predicted_x(2), 200.0, track_id=7)])
    # frame sau: ID 7 la muc tieu; ID 1 (neu xuat hien lai) chi la ung vien
    pred = lost_memory.update(None, candidates=[det(12, predicted_x(3), 200.0, track_id=7)])
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 7


def test_candidate_outside_gate_is_rejected(lost_memory: MotionMemory):
    cand = det(11, predicted_x(2) + 60.0, 200.0, track_id=7)
    pred = lost_memory.update(None, candidates=[cand])
    assert pred.state is TrackState.LOST
    assert pred.track_id == 1


def test_two_candidates_in_gate_is_ambiguous(lost_memory: MotionMemory):
    a = det(11, predicted_x(2) - 3.0, 200.0, track_id=7)
    b = det(11, predicted_x(2) + 3.0, 200.0, track_id=8)
    pred = lost_memory.update(None, candidates=[a, b])
    assert pred.state is TrackState.LOST
    assert pred.track_id == 1


def test_one_in_gate_one_outside_matches(lost_memory: MotionMemory):
    inside = det(11, predicted_x(2), 200.0, track_id=7)
    outside = det(11, predicted_x(2) + 80.0, 200.0, track_id=8)
    pred = lost_memory.update(None, candidates=[outside, inside])
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 7


@pytest.mark.parametrize("cls", [1, None])
def test_different_or_unknown_class_is_rejected(lost_memory: MotionMemory, cls):
    cand = det(11, predicted_x(2), 200.0, track_id=7, cls=cls)
    assert lost_memory.update(None, candidates=[cand]).state is TrackState.LOST


def test_class_ignored_when_target_class_unknown():
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1, cls=None))
    memory.update(None)
    cand = det(11, predicted_x(2), 200.0, track_id=7, cls=3)
    assert memory.update(None, candidates=[cand]).state is TrackState.TRACKING


@pytest.mark.parametrize(
    "scale_factor, expected",
    [
        (1.5, TrackState.LOST),       # dien tich x2.25 > 2 -> loai
        (1.4, TrackState.TRACKING),   # dien tich x1.96 -> nhan
        (0.7, TrackState.LOST),       # dien tich x0.49 < 0.5 -> loai
        (0.72, TrackState.TRACKING),  # dien tich x0.518 -> nhan
    ],
)
def test_scale_change_gate(lost_memory: MotionMemory, scale_factor, expected):
    # cung tam voi du doan: chi gate kich thuoc quyet dinh
    cand = det(11, predicted_x(2), 200.0, track_id=7, w=W * scale_factor, h=H * scale_factor)
    assert lost_memory.update(None, candidates=[cand]).state is expected


def test_iou_path_accepts_beyond_center_radius(lost_memory: MotionMemory):
    # gap 2: ban kinh 18 px; lech 19 px -> ngoai ban kinh nhung IoU ~0.36 >= 0.3
    cand = det(11, predicted_x(2) + 19.0, 200.0, track_id=7)
    assert lost_memory.update(None, candidates=[cand]).state is TrackState.TRACKING


def test_gate_grows_with_frame_gap(lost_memory: MotionMemory):
    # lech doc 20 px (IoU = 0). gap 2: ban kinh 18 -> loai.
    assert lost_memory.update(
        None, candidates=[det(11, predicted_x(2), 220.0, track_id=7)]
    ).state is TrackState.LOST
    for frame in (12, 13, 14):  # gap 3, 4, 5: ban kinh 19.5, 21 (>=20 tu gap 4)
        state = lost_memory.update(
            None, candidates=[det(frame, predicted_x(frame - 9), 220.0, track_id=7)]
        ).state
        if frame - 9 >= 4:
            assert state is TrackState.TRACKING
            break
        assert state is TrackState.LOST


def test_gate_uses_frame_idx_not_call_count(lost_memory: MotionMemory):
    """Chi 1 lan goi update() nhung ung vien o frame 15 (gap 6, ban kinh 24)."""
    cand = det(15, predicted_x(6), 220.0, track_id=7)
    assert lost_memory.update(None, candidates=[cand]).state is TrackState.TRACKING


def test_same_id_in_candidates_is_target_observation(lost_memory: MotionMemory):
    # ID 1 quay lai o xa vi tri du doan van duoc nhan (khop track, khong qua gate)
    far = det(11, 600.0, 50.0, track_id=1)
    pred = lost_memory.update(None, candidates=[det(11, predicted_x(2), 200.0, track_id=7), far])
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 1
    assert pred.bbox == far.bbox


def test_same_id_observation_recovers(lost_memory: MotionMemory):
    pred = lost_memory.update(det(11, 600.0, 50.0, track_id=1))
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 1


def test_observation_with_other_id_must_pass_gate(lost_memory: MotionMemory):
    far = det(11, 600.0, 50.0, track_id=7)
    assert lost_memory.update(far).state is TrackState.LOST
    near = det(12, predicted_x(3), 200.0, track_id=7)
    assert lost_memory.update(near).state is TrackState.TRACKING


def test_no_gating_on_frame_target_disappears():
    """TRACKING -> LOST theo bang muc 6; gating bat dau tu frame LOST ke tiep."""
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1))
    cand10 = det(10, predicted_x(1), 200.0, track_id=7)
    assert memory.update(None, candidates=[cand10]).state is TrackState.LOST
    cand11 = det(11, predicted_x(2), 200.0, track_id=7)
    assert memory.update(None, candidates=[cand11]).state is TrackState.TRACKING


def test_no_candidates_keeps_backward_compatible_behaviour(lost_memory: MotionMemory):
    pred = lost_memory.update(None)
    assert pred.state is TrackState.LOST
    assert pred.bbox is not None
    assert pred.bbox.center == pytest.approx((predicted_x(2), 200.0))


@pytest.mark.parametrize(
    "overrides",
    [
        {"match_iou": 0.0},
        {"match_iou": 1.1},
        {"match_center_ratio": 0.0},
        {"match_gate_growth": -0.1},
        {"max_scale_change": 0.9},
    ],
)
def test_invalid_gate_config_rejected(overrides):
    with pytest.raises(ValueError):
        MotionMemoryConfig(**overrides)


# ---------------- gate_candidates_passed ----------------


def test_gate_count_none_when_gate_not_evaluated(lost_memory: MotionMemory):
    assert lost_memory.update(None).gate_candidates_passed is None  # khong co ung vien


def test_gate_count_zero_when_all_rejected(lost_memory: MotionMemory):
    far = det(11, predicted_x(2) + 80.0, 200.0, track_id=7)
    pred = lost_memory.update(None, candidates=[far])
    assert pred.state is TrackState.LOST
    assert pred.gate_candidates_passed == 0


def test_gate_count_ambiguous(lost_memory: MotionMemory):
    a = det(11, predicted_x(2) - 3.0, 200.0, track_id=7)
    b = det(11, predicted_x(2) + 3.0, 200.0, track_id=8)
    assert lost_memory.update(None, candidates=[a, b]).gate_candidates_passed == 2


def test_gate_count_one_on_match(lost_memory: MotionMemory):
    pred = lost_memory.update(None, candidates=[det(11, predicted_x(2), 200.0, track_id=7)])
    assert pred.gate_candidates_passed == 1


def test_gate_count_none_when_same_id_returns(lost_memory: MotionMemory):
    pred = lost_memory.update(None, candidates=[det(11, 600.0, 50.0, track_id=1),
                                               det(11, predicted_x(2), 200.0, track_id=7)])
    assert pred.recover_reason == "same_id"
    assert pred.gate_candidates_passed is None


def test_gate_count_none_on_transition_frame():
    memory = MotionMemory(MotionMemoryConfig(velocity_decay=1.0))
    for i in range(10):
        memory.update(det(i, 100.0 + 3.0 * i, 200.0, track_id=1))
    pred = memory.update(None, candidates=[det(10, predicted_x(1), 200.0, track_id=7)])
    assert pred.state is TrackState.LOST
    assert pred.gate_candidates_passed is None
