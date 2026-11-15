"""Adapter MotionMemory cho lop sim (src/sim/mm_target.py). Du lieu gia, khong PX4/Gazebo/GUI."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest

from src.motion_memory import MotionMemoryConfig
from src.sim.config import SimConfig, TargetConfig
from src.sim.mm_target import MotionMemoryTargetManager, build_target_manager
from src.sim.target_manager import TargetManager
from src.sim.types import Detection, TargetState

RATE = 10.0  # frame/giay cho test: 1 frame = 0.1 s


def det(i, cx, cy=200.0, h=60.0, label="pedestrian", conf=0.4):
    w = h / 2
    return Detection(i, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), label, conf)


def locked(lost=3, reid=5, **kw):
    cfg = TargetConfig(estimator="motion_memory", mm_frame_rate_hz=RATE)
    m = MotionMemoryTargetManager(cfg, MotionMemoryConfig(lost_timeout_frames=lost, reid_timeout_frames=reid, **kw))
    m.update([det(1, 100.0), det(2, 400.0, label="car")], 0.0)
    assert m.select_at(100, 200) == 1 and m.lock_selected()
    return m


def walk(m, start_t, n, x0, dx, tid=1):
    """Muc tieu di sang phai dx px moi frame, n frame."""
    t, x = start_t, x0
    for _ in range(n):
        t, x = t + 1 / RATE, x + dx
        m.update([det(tid, x)], t)
    return t, x


def test_factory_selects_estimator(tmp_path):
    assert type(build_target_manager(TargetConfig())) is TargetManager
    cfgfile = tmp_path / "mm.yaml"
    cfgfile.write_text("motion_memory:\n  lost_timeout_frames: 7\n", encoding="utf-8")
    m = build_target_manager(TargetConfig(estimator="motion_memory", motion_memory_config=str(cfgfile)))
    assert isinstance(m, MotionMemoryTargetManager) and m._mm_config.lost_timeout_frames == 7
    with pytest.raises(ValueError):
        build_target_manager(TargetConfig(estimator="khac"))


def test_default_sim_config_keeps_heuristic():
    assert SimConfig().target.estimator == "heuristic"


def test_tracking_keeps_detector_conf_separate_from_prediction_conf():
    m = locked()
    t, x = walk(m, 0.0, 3, 100.0, 5.0)
    e = m.estimate
    assert e.state is TargetState.TRACKING and e.track_id == 1
    assert e.bbox == det(1, x).bbox
    assert e.prediction_confidence == 1.0  # khong phai conf 0.4 cua detector


def test_lost_extrapolates_forward_then_reid_then_failed():
    m = locked(lost=3, reid=5)
    t, x = walk(m, 0.0, 6, 100.0, 5.0)
    t += 1 / RATE
    e = m.update([], t)
    assert e.state is TargetState.LOST and e.bbox is not None
    cx = (e.bbox[0] + e.bbox[2]) / 2
    assert cx > x  # ngoai suy theo huong di
    assert 0.0 < e.prediction_confidence < 1.0
    assert e.seconds_since_confirmed == pytest.approx(0.1)
    states = []
    for _ in range(10):
        t += 1 / RATE
        states.append(m.update([], t).state)
    assert TargetState.REIDENTIFYING in states and states[-1] is TargetState.FAILED
    assert m.estimate.bbox is None


def test_motion_gate_switches_to_new_track_id_in_lost():
    m = locked(lost=5, reid=10)
    t, x = walk(m, 0.0, 6, 100.0, 5.0)
    t += 1 / RATE
    assert m.update([], t).state is TargetState.LOST
    t += 1 / RATE
    e = m.update([det(7, x + 10.0), det(2, 400.0, label="car")], t)  # BoT-SORT cap ID moi gan vi tri du doan
    assert e.state is TargetState.TRACKING and m.locked_id == 7 and e.track_id == 7


def test_same_timestamp_twice_does_not_break_frame_order():
    m = locked()
    m.update([det(1, 105.0)], 0.1)
    e = m.update([det(1, 106.0)], 0.1)  # cung t -> frame_idx day len +1, khong loi
    assert e.state is TargetState.TRACKING


def test_unlock_clears_and_relock_works():
    m = locked()
    walk(m, 0.0, 2, 100.0, 5.0)
    m.unlock()
    assert m.locked_id is None and m.update([det(1, 120.0)], 0.5) is None
    assert m.select_at(120, 200) == 1 and m.lock_selected()
    assert m.update([det(1, 121.0)], 0.6).state is TargetState.TRACKING
