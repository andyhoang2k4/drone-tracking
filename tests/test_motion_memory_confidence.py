"""Unit test cho prediction confidence cua Motion Memory (roadmap buoc 5).

Dung lop MotionMemory that (khong stub), quy dao gia lap, khong chay YOLO.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

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


def obs(frame_idx: int, cx: float, cy: float, det_conf: float = 0.8) -> Observation:
    return Observation(
        bbox=BBox(cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2),
        frame_idx=frame_idx,
        track_id=1,
        detector_confidence=det_conf,
    )


def linear_memory(n: int = 10, lost_timeout: int = 15, det_conf: float = 0.8,
                  **cfg) -> MotionMemory:
    memory = MotionMemory(MotionMemoryConfig(lost_timeout_frames=lost_timeout, **cfg))
    for i in range(n):
        memory.update(obs(i, 100.0 + 3.0 * i, 200.0, det_conf))
    return memory


def lost_confidences(memory: MotionMemory) -> list[float]:
    values = []
    while True:
        pred = memory.update(None)
        if pred.state is not TrackState.LOST:
            return values
        values.append(pred.confidence)


def test_tracking_confidence_is_one():
    memory = MotionMemory()
    assert memory.update(obs(0, 10.0, 10.0)).confidence == 1.0


def test_perfect_linear_motion_first_lost_frame():
    memory = linear_memory(lost_timeout=15)
    pred = memory.update(None)
    assert pred.state is TrackState.LOST
    assert pred.confidence == pytest.approx(1.0 - 1.0 / 16.0)


def test_confidence_bounded_monotonic_and_positive_over_horizon():
    values = lost_confidences(linear_memory(lost_timeout=10))
    assert len(values) == 10
    assert all(0.0 < v <= 1.0 for v in values)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(1.0 / 11.0)


def test_confidence_zero_after_horizon():
    memory = linear_memory(lost_timeout=3)
    lost_confidences(memory)  # tieu het LOST
    assert memory.state is TrackState.REIDENTIFYING
    assert memory.update(None).confidence == 0.0


def test_full_cycle_runs_without_stub():
    memory = linear_memory(lost_timeout=3, reid_timeout_frames=4)
    states = [memory.update(None).state for _ in range(10)]
    assert states[:3] == [TrackState.LOST] * 3
    assert states[3:7] == [TrackState.REIDENTIFYING] * 4
    assert states[7:] == [TrackState.FAILED] * 3


def test_independent_of_detector_confidence():
    """CLAUDE.md muc 6: confidence du doan khong phu thuoc confidence detector."""
    low = linear_memory(det_conf=0.11)
    high = linear_memory(det_conf=0.99)
    assert lost_confidences(low) == lost_confidences(high)


def test_little_history_lowers_confidence():
    full = linear_memory(n=10, min_history=3)
    sparse = linear_memory(n=1, min_history=3)
    c_full = full.update(None).confidence
    c_sparse = sparse.update(None).confidence
    assert c_sparse == pytest.approx(c_full / 3.0)


def test_erratic_motion_lowers_confidence():
    rng = random.Random(1)
    clean = linear_memory()
    noisy = MotionMemory(MotionMemoryConfig())
    for i in range(10):
        noisy.update(obs(i, 100.0 + 3.0 * i + rng.uniform(-8, 8), 200.0 + rng.uniform(-8, 8)))
    c_clean = clean.update(None).confidence
    c_noisy = noisy.update(None).confidence
    assert 0.0 < c_noisy < c_clean


def test_recovery_resets_confidence_to_one():
    memory = linear_memory()
    memory.update(None)
    memory.update(None)
    pred = memory.update(obs(12, 136.0, 200.0))
    assert pred.state is TrackState.TRACKING
    assert pred.confidence == 1.0
    # mat lai: bat dau lai tu k = 1
    assert memory.update(None).confidence == pytest.approx(1.0 - 1.0 / 16.0)
