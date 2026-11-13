"""Unit test cho update(..., frame_idx=...) cua Motion Memory.

Du lieu gia lap, khong chay YOLO. Kiem tra: hoi quy hanh vi cu khi frame lien
tuc, nguon bo frame, phia goi bo qua loi goi, va kiem tra dau vao.
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

L, R = 4, 3


def det(frame_idx: int, cx: Optional[float] = None, track_id: int = 1) -> Observation:
    x = 100.0 + 3.0 * frame_idx if cx is None else cx
    return Observation(bbox=BBox(x - 20, 190, x + 20, 210), frame_idx=frame_idx,
                       track_id=track_id, cls=0)


def make() -> MotionMemory:
    return MotionMemory(MotionMemoryConfig(lost_timeout_frames=L, reid_timeout_frames=R,
                                           velocity_decay=1.0))


def tracked(last_frame: int = 9) -> MotionMemory:
    memory = make()
    for f in range(last_frame + 1):
        memory.update(det(f), frame_idx=f)
    return memory


# Kich ban: thay 0..9, mat 10..12, thay lai 13..15, mat tu 16 den het (FAILED)
SCENARIO: list[Optional[int]] = (
    list(range(10)) + [None] * 3 + [13, 14, 15] + [None] * (L + R + 4)
)


def test_contiguous_frames_match_old_behaviour():
    old, new = make(), make()
    for f, item in enumerate(SCENARIO):
        o = det(item) if item is not None else None
        p_old = old.update(o)
        p_new = new.update(o, frame_idx=f)
        assert p_new == p_old, f"lech o frame {f}"


def test_source_drops_frames():
    memory = tracked()  # confirmed cuoi o frame 9
    pred = memory.update(None, frame_idx=13)  # frame 10..12 khong co trong nguon
    assert pred.frames_since_confirmed == 4
    assert pred.state is TrackState.LOST
    assert pred.bbox is not None
    assert pred.bbox.center[0] == pytest.approx(127.0 + 3.0 * 4)
    assert pred.confidence == pytest.approx(1.0 - 4 / (L + 1))


def test_skipped_calls_jump_to_reidentifying():
    memory = tracked()
    pred = memory.update(None, frame_idx=9 + L + 1)
    assert pred.state is TrackState.REIDENTIFYING
    assert pred.frames_since_confirmed == L + 1
    assert pred.bbox is None


def test_skipped_calls_jump_straight_to_failed():
    memory = tracked()
    pred = memory.update(None, frame_idx=9 + L + R + 2)
    assert pred.state is TrackState.FAILED


def test_boundaries_measured_on_frame_idx():
    memory = tracked()
    assert memory.update(None, frame_idx=9 + L).state is TrackState.LOST
    assert memory.update(None, frame_idx=9 + L + R).state is TrackState.REIDENTIFYING
    assert memory.update(None, frame_idx=9 + L + R + 1).state is TrackState.FAILED


def test_recovery_after_jump_resets_counter():
    memory = tracked()
    memory.update(None, frame_idx=12)
    pred = memory.update(det(20), frame_idx=20)
    assert pred.state is TrackState.TRACKING
    assert pred.frames_since_confirmed == 0
    assert memory.update(None, frame_idx=22).frames_since_confirmed == 2


def test_without_frame_idx_counts_calls():
    memory = tracked()
    memory.update(None, frame_idx=12)  # k = 3
    assert memory.update(None).frames_since_confirmed == 4  # +1


def test_frame_idx_without_history_counts_calls():
    memory = make()
    assert memory.update(None, frame_idx=50).frames_since_confirmed == 1


def test_frame_idx_must_increase():
    memory = tracked()
    with pytest.raises(ValueError):
        memory.update(None, frame_idx=9)


def test_frame_idx_must_match_observation():
    memory = tracked()
    with pytest.raises(ValueError):
        memory.update(det(11), frame_idx=10)


def test_reset_clears_last_frame_idx():
    memory = tracked()
    memory.reset()
    assert memory.update(det(0), frame_idx=0).state is TrackState.TRACKING
