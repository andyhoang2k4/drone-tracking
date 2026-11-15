"""Unit test cho ngoai suy chuyen dong cua Motion Memory (roadmap buoc 5).

Chi dung quy dao gia lap, khong chay YOLO. Toa do pixel, bbox xyxy.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Iterable, Optional

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

W, H = 40.0, 20.0  # kich thuoc bbox gia lap -> scale (w+h)/2 = 30 px


def box_at(cx: float, cy: float, w: float = W, h: float = H) -> BBox:
    return BBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def obs(frame_idx: int, cx: float, cy: float, w: float = W, h: float = H) -> Observation:
    return Observation(bbox=box_at(cx, cy, w, h), frame_idx=frame_idx, track_id=1,
                       detector_confidence=0.8)


def feed(memory: MotionMemory, observations: Iterable[Observation]) -> None:
    for o in observations:
        memory.update(o)


def linear(n: int, vx: float, vy: float, start: tuple[float, float] = (100.0, 200.0),
           first_frame: int = 0) -> list[Observation]:
    x0, y0 = start
    return [obs(first_frame + i, x0 + vx * i, y0 + vy * i) for i in range(n)]


class FixedConfidence(MotionMemory):
    """Co lap phan ngoai suy: confidence chua phai doi tuong test o day."""

    def _prediction_confidence(self) -> float:
        return 0.5


def make(**overrides) -> FixedConfidence:
    params = dict(lost_timeout_frames=15, reid_timeout_frames=45, velocity_decay=1.0)
    params.update(overrides)
    return FixedConfidence(MotionMemoryConfig(**params))


def center(b: Optional[BBox]) -> tuple[float, float]:
    assert b is not None
    return b.center


# ---------------- history rong ----------------


def test_extrapolate_empty_history_returns_none():
    assert make()._extrapolate(1) is None


def test_update_lost_without_any_observation_gives_no_bbox():
    memory = make()
    pred = memory.update(None)
    assert pred.state is TrackState.LOST
    assert pred.bbox is None
    assert pred.confidence == 0.0


# ---------------- quy dao thang deu ----------------


def test_one_missed_frame_on_linear_motion():
    memory = make()
    feed(memory, linear(10, vx=3.0, vy=-2.0))  # frame 0..9, tam cuoi (127, 182)
    pred = memory.update(None)
    assert pred.state is TrackState.LOST
    cx, cy = center(pred.bbox)
    assert cx == pytest.approx(130.0)
    assert cy == pytest.approx(180.0)


def test_extrapolates_until_end_of_horizon_without_decay():
    memory = make(lost_timeout_frames=10)
    feed(memory, linear(10, vx=3.0, vy=0.0))
    for k in range(1, 11):
        pred = memory.update(None)
        assert pred.state is TrackState.LOST
        assert center(pred.bbox)[0] == pytest.approx(127.0 + 3.0 * k)
    # qua horizon -> REIDENTIFYING, khong con bbox ngoai suy
    pred = memory.update(None)
    assert pred.state is TrackState.REIDENTIFYING
    assert pred.bbox is None


def test_bbox_size_is_kept():
    memory = make()
    feed(memory, linear(10, vx=3.0, vy=1.0))
    pred = memory.update(None)
    assert pred.bbox is not None
    assert pred.bbox.width == pytest.approx(W)
    assert pred.bbox.height == pytest.approx(H)


# ---------------- suy giam ----------------


def test_decay_formula_and_bounded_displacement():
    decay = 0.9
    memory = make(velocity_decay=decay)
    feed(memory, linear(10, vx=3.0, vy=0.0))
    x_last = 127.0
    for k in (1, 5, 10):
        expected = x_last + 3.0 * (1 - decay**k) / (1 - decay)
        assert center(memory._extrapolate(k))[0] == pytest.approx(expected)
    # do doi bi chan boi v / (1 - decay) = 30 px
    assert center(memory._extrapolate(1000))[0] <= x_last + 3.0 / (1 - decay) + 1e-6


def test_decay_never_overtakes_constant_velocity():
    with_decay = make(velocity_decay=0.95)
    without = make(velocity_decay=1.0)
    for m in (with_decay, without):
        feed(m, linear(10, vx=4.0, vy=0.0))
    for k in range(1, 16):
        assert center(with_decay._extrapolate(k))[0] <= center(without._extrapolate(k))[0]


# ---------------- dung yen ----------------


def test_stationary_target_with_jitter_does_not_drift():
    rng = random.Random(0)
    memory = make()
    # jitter +-0.3 px, nho hon nhieu so voi 2% * 30 px = 0.6 px/frame
    feed(memory, [obs(i, 300.0 + rng.uniform(-0.3, 0.3), 150.0 + rng.uniform(-0.3, 0.3))
                  for i in range(12)])
    last_center = memory.last_confirmed.bbox.center
    for k in range(1, 16):
        assert center(memory._extrapolate(k)) == pytest.approx(last_center)


def test_min_speed_threshold_scales_with_bbox_size():
    # cung toc do 1 px/frame: bbox nho (scale 10) -> chuyen dong; bbox lon (scale 100) -> dung yen
    small = make(min_speed_ratio=0.02)
    large = make(min_speed_ratio=0.02)
    feed(small, [obs(i, 100.0 + i, 100.0, w=10, h=10) for i in range(10)])
    feed(large, [obs(i, 100.0 + i, 100.0, w=100, h=100) for i in range(10)])
    assert center(small._extrapolate(1))[0] == pytest.approx(110.0)
    assert center(large._extrapolate(1))[0] == pytest.approx(109.0)


def test_too_few_observations_assumes_stationary():
    memory = make(min_history=3)
    feed(memory, linear(2, vx=10.0, vy=0.0))
    assert center(memory._extrapolate(3)) == pytest.approx((110.0, 200.0))


# ---------------- doi huong ----------------


def test_direction_change_follows_recent_motion():
    memory = make(velocity_window_frames=4)
    # 10 frame sang phai, roi 6 frame di xuong; cua so 4 frame chi thay doan di xuong
    path = linear(10, vx=5.0, vy=0.0)
    x_turn = 145.0
    path += [obs(10 + i, x_turn, 200.0 + 4.0 * (i + 1)) for i in range(6)]
    feed(memory, path)
    (cx, cy), (lx, ly) = center(memory._extrapolate(1)), memory.last_confirmed.bbox.center
    assert cx == pytest.approx(lx)
    assert cy == pytest.approx(ly + 4.0)


def test_direction_change_during_occlusion_is_known_limitation():
    """Doi huong TRONG luc bi che: mo hinh chi ngoai suy huong cu (gioi han da biet)."""
    memory = make()
    feed(memory, linear(10, vx=5.0, vy=0.0))
    pred_x, pred_y = center(memory._extrapolate(5))
    true_x, true_y = 145.0, 200.0 + 4.0 * 5  # thuc te da re xuong tu frame 10
    assert math.hypot(pred_x - true_x, pred_y - true_y) > 20.0


# ---------------- frame_idx khong lien tuc ----------------


def test_velocity_uses_frame_idx_not_sample_count():
    memory = make()
    # chi co observation o frame chan, tam di 6 px moi 2 frame -> 3 px/frame
    feed(memory, [obs(2 * i, 100.0 + 6.0 * i, 100.0) for i in range(4)])
    assert center(memory._extrapolate(1))[0] == pytest.approx(118.0 + 3.0)


def test_old_motion_outside_window_is_ignored_after_long_gap():
    memory = make(velocity_window_frames=8, min_history=3)
    feed(memory, linear(10, vx=5.0, vy=0.0))  # frame 0..9 di sang phai
    # mat 30 frame roi bat lai tai frame 40, dung yen o vi tri moi
    feed(memory, [obs(40, 400.0, 300.0), obs(41, 400.0, 300.0)])
    # chi 2 observation trong cua so (< min_history) -> gia dinh dung yen
    assert center(memory._extrapolate(3)) == pytest.approx((400.0, 300.0))


def test_short_gap_keeps_velocity():
    memory = make()
    path = linear(6, vx=2.0, vy=0.0)       # frame 0..5
    path += [obs(7, 114.0, 200.0), obs(8, 116.0, 200.0)]  # mat frame 6
    feed(memory, path)
    assert center(memory._extrapolate(1))[0] == pytest.approx(118.0)


def test_duplicate_frame_idx_does_not_crash():
    memory = make(min_history=2)
    feed(memory, [obs(5, 100.0, 100.0), obs(5, 110.0, 100.0)])
    assert center(memory._extrapolate(1)) == pytest.approx((110.0, 100.0))


# ---------------- config ----------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"velocity_decay": 0.0},
        {"velocity_decay": 1.5},
        {"velocity_window_frames": 1},
        {"min_history": 1},
        {"min_history": 100, "history_size": 30},
        {"min_speed_ratio": -0.1},
        {"lost_timeout_frames": 0},
    ],
)
def test_invalid_config_rejected(overrides):
    with pytest.raises(ValueError):
        MotionMemoryConfig(**overrides)


def test_default_config_is_valid():
    MotionMemoryConfig()
