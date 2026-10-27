"""Chay ca vong lap SimApp tren the gioi gia lap (khong GUI, khong PX4). Kiem tra luong dieu khien."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math

from src.sim.app import SimApp
from src.sim.config import SimConfig
from src.sim.demo import DemoCamera, DemoDetector, DemoLink, DemoWorld
from src.sim.types import FlightMode, TargetState
from src.sim.virtual_radio import VirtualRadio


class Harness:
    def __init__(self):
        self.cfg = SimConfig()
        self.w = DemoWorld(self.cfg)
        self.radio = VirtualRadio(self.cfg.radio)
        self.link = DemoLink(self.w)
        self.app = SimApp(self.cfg, DemoCamera(self.w), DemoDetector(self.w), self.link, self.radio)
        self.t = 0.0
        self.dt = 1.0 / self.cfg.loop_hz
        self.states = []

    def run(self, sec):
        for _ in range(int(sec / self.dt)):
            self.app.tick(self.t)
            self.t += self.dt
            e = self.app.manager.estimate
            self.states.append((self.t, e.state if e else None, self.app.manager.locked_id))

    def arm_and_hover(self):
        self.run(0.3)
        self.radio.handle_key(ord(" "))
        for _ in range(10):
            self.radio.handle_key(ord("w"))
        self.run(1.0)

    def lock_person(self, pid=1):
        d = next(d for d in self.app.manager.candidates if d.track_id == pid)
        cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
        self.app.view.on_mouse(1, int(cx), int(cy))
        self.app.view.handle_key(ord("l"))

    def msgs(self):
        return [m.text for m in self.app.board.active(self.t)]


def test_arm_blocked_until_throttle_zero_then_arms_in_sim():
    h = Harness()
    h.arm_and_hover()
    assert h.w.armed and ("arm", True) in h.link.log


def test_manual_lock_gimbal_tracks_then_auto_follows_loses_searches_and_reacquires():
    h = Harness()
    h.arm_and_hover()
    h.lock_person(1)
    assert h.app.manager.locked_id == 1
    h.run(1.0)
    assert h.app.last_output.desired_mode == FlightMode.STABILIZED
    az_deg = abs(h.w.gimbal_yaw)
    assert 5 < az_deg < 45  # gimbal da xoay ve phia muc tieu du dang bay thu cong
    d0 = math.hypot(*(h.w.people[1] - h.w.pos[:2]))

    h.radio.handle_key(ord("3"))
    h.run(1.0)
    assert h.app.last_output.auto_engaged and h.app.last_output.desired_mode == FlightMode.AUTO_FOLLOW
    assert h.link.log[-1] == ("mode", FlightMode.AUTO_FOLLOW)

    h.run(20.0)
    seen = [s for _, s, _ in h.states]
    assert TargetState.LOST in seen and TargetState.REIDENTIFYING in seen
    # sau khi bi che, bat lai bang ID moi va tiep tuc theo
    assert h.app.manager.locked_id >= 100
    assert h.app.manager.estimate.state in (TargetState.TRACKING, TargetState.LOST)
    assert h.w.pos[0] > -5  # drone da bay ve phia muc tieu


def test_lost_message_and_search_message_are_posted():
    h = Harness()
    h.arm_and_hover()
    h.lock_person(1)
    h.radio.handle_key(ord("3"))
    texts = set()
    for _ in range(int(25.0 / h.dt)):
        h.app.tick(h.t)
        h.t += h.dt
        texts.update(h.msgs())
    assert any("MUC TIEU BIEN MAT" in x for x in texts)
    assert any("DANG TIM MUC TIEU" in x for x in texts)
    assert any("DA BAT LAI MUC TIEU" in x for x in texts)


def test_auto_without_target_prompts_then_lock_engages_then_unlock_returns_manual():
    h = Harness()
    h.arm_and_hover()
    h.radio.handle_key(ord("3"))
    h.run(0.5)
    assert not h.app.last_output.auto_engaged
    assert h.app.last_output.desired_mode == FlightMode.STABILIZED
    txt = " ".join(h.msgs())
    assert "CHUA CO MUC TIEU" in txt and "person#1" in txt and "car#3" in txt
    h.lock_person(1)
    h.run(0.5)
    assert h.app.last_output.auto_engaged
    assert not any("CHUA CO MUC TIEU" in m for m in h.msgs())
    h.app.view.handle_key(ord("u"))
    h.run(0.5)
    assert not h.app.last_output.auto_engaged and h.app.last_output.desired_mode == FlightMode.STABILIZED
    assert any("CHUA CO MUC TIEU" in m for m in h.msgs())


def test_detector_receives_pristine_frame_without_osd():
    import numpy as np

    class SpyCam(DemoCamera):
        last = None
        def read(self):
            SpyCam.last = super().read()
            return SpyCam.last

    class SpyDet(DemoDetector):
        got = None
        def detect(self, frame, t):
            SpyDet.got = frame.copy()
            return super().detect(frame, t)

    h = Harness()
    h.app.camera, h.app.detector = SpyCam(h.w), SpyDet(h.w)
    h.run(0.3)
    assert np.array_equal(SpyDet.got, SpyCam.last)  # AI nhan dung anh goc
    assert (h.app.last_view != SpyCam.last).any()  # ban hien thi co OSD


def test_event_log_written_on_changes_only(tmp_path):
    from src.utils.logging_utils import JsonlLogger
    import json

    h = Harness()
    path = tmp_path / "sim.jsonl"
    with JsonlLogger(path) as lg:
        h.app.set_logger(lg)
        h.arm_and_hover()
        h.lock_person(1)
        h.radio.handle_key(ord("3"))
        h.run(10.0)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert 3 <= len(rows) < 60  # su kien, khong phai moi frame
    assert any(r["locked_id"] == 1 and r["status"].startswith("AUTO") for r in rows)


def test_target_state_is_shared_enum_from_motion_memory():
    from src.motion_memory import TrackState
    from src.sim.types import TargetState

    assert TargetState is TrackState
