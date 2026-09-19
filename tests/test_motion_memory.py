"""Unit test cho state machine cua Motion Memory (CLAUDE.md muc 6).

Chi test phan DA implement: bang transition. Phan ngoai suy chuyen dong
(`_extrapolate`) chua viet - o day duoc stub lai de co lap state machine.
Khi roadmap buoc 5 xong, them test rieng cho phan ngoai suy.
"""

from __future__ import annotations

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


class StubMotionMemory(MotionMemory):
    """Thay ngoai suy that bang gia tri co dinh de test state machine."""

    def _extrapolate(self, frames_ahead: int) -> BBox:
        return BBox(0.0, 0.0, 10.0, 10.0)

    def _prediction_confidence(self) -> float:
        return 0.5


def make_observation(frame_idx: int) -> Observation:
    return Observation(
        bbox=BBox(0.0, 0.0, 10.0, 10.0),
        frame_idx=frame_idx,
        track_id=1,
        detector_confidence=0.9,
    )


@pytest.fixture
def memory() -> StubMotionMemory:
    return StubMotionMemory(
        MotionMemoryConfig(lost_timeout_frames=3, reid_timeout_frames=5)
    )


def test_starts_tracking(memory: StubMotionMemory) -> None:
    assert memory.state is TrackState.TRACKING
    assert memory.frames_since_confirmed == 0


def test_confirmed_detection_stays_tracking(memory: StubMotionMemory) -> None:
    prediction = memory.update(make_observation(0))
    assert prediction.state is TrackState.TRACKING
    assert prediction.frames_since_confirmed == 0
    assert prediction.confidence == 1.0
    assert prediction.bbox is not None


def test_missing_detection_goes_lost(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    prediction = memory.update(None)
    assert prediction.state is TrackState.LOST
    assert prediction.frames_since_confirmed == 1
    assert prediction.bbox is not None  # van du doan duoc trong motion horizon


def test_lost_recovers_to_tracking(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    memory.update(None)
    assert memory.state is TrackState.LOST

    prediction = memory.update(make_observation(2))
    assert prediction.state is TrackState.TRACKING
    assert prediction.frames_since_confirmed == 0


def test_lost_timeout_goes_reidentifying(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    # lost_timeout_frames=3 -> dung 3 frame o LOST
    for _ in range(3):
        assert memory.update(None).state is TrackState.LOST

    prediction = memory.update(None)
    assert prediction.state is TrackState.REIDENTIFYING
    # Ngoai suy chuyen dong khong con dang tin -> khong tra bbox nua
    assert prediction.bbox is None
    assert prediction.confidence == 0.0


def test_reidentifying_needs_reid_match(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    for _ in range(4):
        memory.update(None)
    assert memory.state is TrackState.REIDENTIFYING

    # Detection don thuan khong du de thoat REIDENTIFYING
    prediction = memory.update(make_observation(10))
    assert prediction.state is TrackState.REIDENTIFYING

    prediction = memory.update(make_observation(11), reid_matched=True)
    assert prediction.state is TrackState.TRACKING
    assert prediction.frames_since_confirmed == 0


def test_reid_timeout_goes_failed(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    for _ in range(4):
        memory.update(None)
    assert memory.state is TrackState.REIDENTIFYING

    # reid_timeout_frames=5
    for _ in range(5):
        assert memory.update(None).state is TrackState.REIDENTIFYING

    assert memory.update(None).state is TrackState.FAILED


def test_failed_is_terminal_until_reset(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    for _ in range(20):
        memory.update(None)
    assert memory.state is TrackState.FAILED

    prediction = memory.update(make_observation(30), reid_matched=True)
    assert prediction.state is TrackState.FAILED

    memory.reset()
    assert memory.state is TrackState.TRACKING
    assert memory.update(make_observation(31)).state is TrackState.TRACKING


def test_detector_and_motion_confidence_stay_separate(
    memory: StubMotionMemory,
) -> None:
    """CLAUDE.md muc 6: khong gop hai loai confidence thanh mot con so."""
    observation = make_observation(0)
    memory.update(observation)
    prediction = memory.update(None)

    assert memory.last_confirmed is not None
    assert memory.last_confirmed.detector_confidence == 0.9
    assert prediction.confidence == 0.5  # confidence cua DU DOAN, doc lap


def test_history_keeps_only_confirmed(memory: StubMotionMemory) -> None:
    memory.update(make_observation(0))
    memory.update(None)
    memory.update(make_observation(2))

    assert [obs.frame_idx for obs in memory.history] == [0, 2]
