import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.sim.camera_view import CameraView
from src.sim.config import RadioConfig, TargetConfig
from src.sim.messages import Message
from src.sim.target_manager import TargetManager
from src.sim.types import Detection, FlightMode, RCInput, TargetState, TargetEstimate, Telemetry
from src.sim import virtual_radio as vr


class Clock:
    t = 0.0
    def __call__(self):
        return self.t


def drag(radio, start, end):
    radio.on_mouse(cv2.EVENT_LBUTTONDOWN, *start)
    radio.on_mouse(cv2.EVENT_MOUSEMOVE, *end)


def test_right_stick_drag_and_spring_back():
    r = vr.VirtualRadio(RadioConfig())
    c = vr.RIGHT_C
    drag(r, c, (c[0] + 50, c[1] - 50))
    s = r.update()
    assert abs(s.roll - 0.5) < 1e-6 and abs(s.pitch - 0.5) < 1e-6
    r.on_mouse(cv2.EVENT_LBUTTONUP, c[0], c[1])
    s = r.update()
    assert s.roll == 0 and s.pitch == 0


def test_left_stick_throttle_holds_and_yaw_springs():
    r = vr.VirtualRadio(RadioConfig())
    c = vr.LEFT_C
    drag(r, c, (c[0] + 100, c[1] - 50))  # ga 0.75, yaw +1
    s = r.update()
    assert abs(s.throttle - 0.75) < 1e-6 and s.yaw == 1.0
    r.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
    s = r.update()
    assert abs(s.throttle - 0.75) < 1e-6 and s.yaw == 0.0


def test_cannot_arm_with_high_throttle_and_can_after_zero():
    r = vr.VirtualRadio(RadioConfig())
    c = vr.LEFT_C
    drag(r, c, (c[0], c[1] - 50))
    r.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
    r.handle_key(ord(" "))
    assert not r.state.arm_switch and r.notice
    r.handle_key(ord("x"))
    r.handle_key(ord(" "))
    assert r.state.arm_switch
    r.handle_key(ord(" "))
    assert not r.state.arm_switch


def test_mode_switch_by_click_and_key():
    r = vr.VirtualRadio(RadioConfig())
    y = vr.MODE_Y0 + 2 * vr.MODE_DY + 5
    r.on_mouse(cv2.EVENT_LBUTTONDOWN, vr.MODE_X[0] + 5, y)
    r.on_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
    assert r.state.mode_switch == FlightMode.AUTO_FOLLOW
    r.handle_key(ord("2"))
    assert r.state.mode_switch == FlightMode.ACRO


def test_key_pulse_expires():
    clk = Clock()
    r = vr.VirtualRadio(RadioConfig(), clock=clk)
    r.handle_key(ord("l"))
    assert r.update().roll > 0
    clk.t = 1.0
    assert r.update().roll == 0


def test_radio_renders_expected_size():
    img = vr.VirtualRadio(RadioConfig()).render()
    assert img.shape == (vr.H, vr.W, 3)


def test_camera_view_select_lock_unlock_next():
    m = TargetManager(TargetConfig())
    m.update([Detection(1, (100, 100, 200, 300), "person", .9), Detection(2, (300, 100, 400, 300), "car", .8)], 0.0)
    v = CameraView(m)
    v.render(np.zeros((480, 640, 3), np.uint8), detections=m.candidates, estimate=None, telem=Telemetry(),
             rc=RCInput(), desired_mode=FlightMode.STABILIZED, status="MANUAL", messages=[], gimbal=None)
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, 150, 200)
    assert m.selected_id == 1
    v.handle_key(ord("n"))
    assert m.selected_id == 2
    v.handle_key(ord("l"))
    assert m.locked_id == 2
    v.on_mouse(cv2.EVENT_LBUTTONDOWN, 140, 470 - 4)  # nut UNLOCK
    assert m.locked_id is None


def test_osd_draws_all_states_without_error():
    base = np.zeros((480, 640, 3), np.uint8)
    d = [Detection(1, (100, 100, 200, 300), "person", .9)]
    for st, bb in ((TargetState.TRACKING, (100, 100, 200, 300)), (TargetState.LOST, (110, 100, 210, 300)),
                   (TargetState.REIDENTIFYING, None), (TargetState.FAILED, None)):
        est = TargetEstimate(1, st, bb, 0.5, 0.5)
        out = CameraView(TargetManager(TargetConfig())).render(
            base, detections=d, estimate=est, telem=Telemetry(connected=True, armed=True, battery_pct=80),
            rc=RCInput(), desired_mode=FlightMode.AUTO_FOLLOW, status="AUTO", gimbal=(-10, 5),
            messages=[Message("MUC TIEU BIEN MAT", "alert", None)])
        assert out.shape == base.shape and out.any()
