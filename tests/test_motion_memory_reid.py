"""Unit test cho chinh sach REIDENTIFYING (quyet dinh B), recover_reason va
cac transition o frame bien (roadmap buoc 5). Du lieu gia lap, khong chay YOLO.

Moc thoi gian voi lost_timeout_frames=L, reid_timeout_frames=R (k = so frame
tu lan confirmed cuoi, frame_idx lien tuc):
    k = 1..L            LOST
    k = L+1..L+R        REIDENTIFYING (dung R frame)
    k >= L+R+1          FAILED
Detection o frame k duoc xet neu output o frame k-1 chua phai FAILED.
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
    RECOVER_MOTION_GATE,
    RECOVER_REID_MATCHED,
    RECOVER_SAME_ID,
    BBox,
    MotionMemory,
    MotionMemoryConfig,
    Observation,
    TrackState,
)

L, R = 4, 3
W, H = 40.0, 20.0
LAST_FRAME = 9  # frame confirmed cuoi; dung yen tai (200, 200)


def det(frame_idx: int, track_id: Optional[int] = 1, cx: float = 200.0, cy: float = 200.0,
        cls: Optional[int] = 0) -> Observation:
    return Observation(
        bbox=BBox(cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2),
        frame_idx=frame_idx,
        track_id=track_id,
        detector_confidence=0.6,
        cls=cls,
    )


def memory_missing(k: int) -> MotionMemory:
    """Memory da mat muc tieu k frame (frame LAST_FRAME+1 .. LAST_FRAME+k)."""
    memory = MotionMemory(MotionMemoryConfig(lost_timeout_frames=L, reid_timeout_frames=R))
    for i in range(LAST_FRAME + 1):
        memory.update(det(i))
    for _ in range(k):
        memory.update(None)
    return memory


# ---------------- moc chuyen state ----------------


def test_state_timeline_boundaries():
    memory = memory_missing(0)
    states = [memory.update(None).state for _ in range(L + R + 3)]
    assert states[:L] == [TrackState.LOST] * L
    assert states[L:L + R] == [TrackState.REIDENTIFYING] * R
    assert states[L + R:] == [TrackState.FAILED] * 3


# ---------------- recover_reason ----------------


def test_no_reason_while_tracking_or_lost():
    memory = memory_missing(0)
    assert memory.update(det(LAST_FRAME + 1)).recover_reason is None
    assert memory.update(None).recover_reason is None


def test_same_id_recovers_from_lost_with_reason():
    memory = memory_missing(2)
    pred = memory.update(det(LAST_FRAME + 3))
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_SAME_ID
    # frame tiep theo: TRACKING binh thuong, reason ve None
    assert memory.update(det(LAST_FRAME + 4)).recover_reason is None


def test_motion_gate_reason():
    memory = memory_missing(1)
    pred = memory.update(None, candidates=[det(LAST_FRAME + 2, track_id=5)])
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 5
    assert pred.recover_reason == RECOVER_MOTION_GATE


# ---------------- frame bien LOST <-> REIDENTIFYING ----------------


def test_same_id_on_last_lost_frame():
    memory = memory_missing(L - 1)
    assert memory.state is TrackState.LOST
    pred = memory.update(det(LAST_FRAME + L))
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_SAME_ID


def test_gate_accepts_at_horizon_and_rejects_beyond():
    at_horizon = memory_missing(L - 1)
    pred = at_horizon.update(None, candidates=[det(LAST_FRAME + L, track_id=5)])
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_MOTION_GATE

    beyond = memory_missing(L)  # van LOST, frame tiep theo co gap L+1
    assert beyond.state is TrackState.LOST
    pred = beyond.update(None, candidates=[det(LAST_FRAME + L + 1, track_id=5)])
    assert pred.state is TrackState.REIDENTIFYING
    assert pred.track_id == 1


# ---------------- REIDENTIFYING (quyet dinh B) ----------------


def test_reidentifying_same_id_recovers():
    memory = memory_missing(L + 1)
    assert memory.state is TrackState.REIDENTIFYING
    pred = memory.update(det(LAST_FRAME + L + 2, cx=400.0))  # o bat ky dau
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 1
    assert pred.recover_reason == RECOVER_SAME_ID
    assert pred.frames_since_confirmed == 0


def test_reidentifying_same_id_via_candidates():
    memory = memory_missing(L + 1)
    pred = memory.update(None, candidates=[det(LAST_FRAME + L + 2, track_id=1)])
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_SAME_ID


def test_reidentifying_ignores_other_id_even_at_predicted_position():
    memory = memory_missing(L + 1)
    pred = memory.update(None, candidates=[det(LAST_FRAME + L + 2, track_id=5)])
    assert pred.state is TrackState.REIDENTIFYING
    assert pred.track_id == 1
    assert pred.bbox is None


def test_reidentifying_observation_with_other_id_is_ignored():
    memory = memory_missing(L + 1)
    pred = memory.update(det(LAST_FRAME + L + 2, track_id=5))
    assert pred.state is TrackState.REIDENTIFYING


def test_reidentifying_unknown_id_needs_reid_matched():
    memory = memory_missing(L + 1)
    assert memory.update(det(LAST_FRAME + L + 2, track_id=None)).state is TrackState.REIDENTIFYING
    pred = memory.update(det(LAST_FRAME + L + 3, track_id=None), reid_matched=True)
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_REID_MATCHED


def test_reid_matched_with_new_id_switches_target():
    memory = memory_missing(L + 1)
    pred = memory.update(det(LAST_FRAME + L + 2, track_id=9, cx=500.0), reid_matched=True)
    assert pred.state is TrackState.TRACKING
    assert pred.track_id == 9
    assert pred.recover_reason == RECOVER_REID_MATCHED


def test_reid_matched_flag_ignored_outside_reidentifying():
    memory = memory_missing(1)  # LOST
    pred = memory.update(det(LAST_FRAME + 2), reid_matched=True)
    assert pred.recover_reason == RECOVER_SAME_ID


def test_same_id_on_last_reidentifying_frame():
    memory = memory_missing(L + R - 1)  # frame REID thu R-1
    assert memory.state is TrackState.REIDENTIFYING
    pred = memory.update(det(LAST_FRAME + L + R))  # frame REID thu R (cuoi)
    assert pred.state is TrackState.TRACKING
    assert pred.recover_reason == RECOVER_SAME_ID


def test_same_id_one_frame_too_late_stays_failed():
    memory = memory_missing(L + R)  # frame REID cuoi cung
    assert memory.state is TrackState.REIDENTIFYING
    assert memory.update(None).state is TrackState.FAILED
    assert memory.update(det(LAST_FRAME + L + R + 2)).state is TrackState.FAILED


# ---------------- FAILED la trang thai cuoi ----------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observation": det(100)},
        {"observation": det(100, track_id=5), "reid_matched": True},
        {"observation": None, "candidates": [det(100, track_id=1), det(100, track_id=5)]},
    ],
)
def test_failed_is_terminal(kwargs):
    memory = memory_missing(L + R + 2)
    assert memory.state is TrackState.FAILED
    pred = memory.update(**kwargs)
    assert pred.state is TrackState.FAILED
    assert pred.bbox is None
    assert pred.confidence == 0.0
    assert pred.recover_reason is None


def test_reset_clears_target_id():
    memory = memory_missing(L + R + 2)
    memory.reset()
    assert memory.track_id is None
    assert memory.state is TrackState.TRACKING
    pred = memory.update(det(200, track_id=3))
    assert pred.track_id == 3
    assert pred.recover_reason is None
