"""Tay dieu khien ao ve bang OpenCV: 2 stick, cong tac Arm, cong tac 3 nac chon mode.

Chuot chi keo duoc 1 stick mot luc, nen co them phim de dieu khien song song:
  W/S: tang/giam ga   A/D: yaw trai/phai   I/K: chuc/ngua (pitch)   J/L: nghieng trai/phai (roll)
  Space: doi Arm   1/2/3: Stabilized/Acro/Auto   X: ga ve 0
"""
from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np

from .config import RadioConfig
from .geometry import clamp
from .types import FlightMode, RCInput

W, H = 720, 400
R = 100  # nua canh o stick (px)
LEFT_C = (170, 250)
RIGHT_C = (550, 250)
ARM_RECT = (310, 90, 410, 130)
MODE_X = (310, 410)
MODE_Y0, MODE_DY = 170, 36
MODES = (FlightMode.STABILIZED, FlightMode.ACRO, FlightMode.AUTO_FOLLOW)
MODE_LABEL = {
    FlightMode.STABILIZED: "ANGLE (Stabilized)",
    FlightMode.ACRO: "ACRO",
    FlightMode.AUTO_FOLLOW: "AUTO (bam muc tieu)",
}
KEY_PULSE_S = 0.35
KEY_AXIS_VALUE = 0.5
THROTTLE_STEP = 0.05


class VirtualRadio:
    def __init__(self, cfg: RadioConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self._cfg = cfg
        self._clock = clock
        self.state = RCInput()
        self.notice = ""
        self._drag: str | None = None
        self._mouse = {"yaw": 0.0, "throttle": 0.0, "roll": 0.0, "pitch": 0.0}
        self._pulse: dict[str, tuple[float, float]] = {}  # truc -> (gia tri, het han)

    # ---- chuot ----
    def on_mouse(self, event: int, x: int, y: int, flags: int = 0, param: object = None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if _in_box(x, y, LEFT_C):
                self._drag = "L"
            elif _in_box(x, y, RIGHT_C):
                self._drag = "R"
            elif ARM_RECT[0] <= x <= ARM_RECT[2] and ARM_RECT[1] <= y <= ARM_RECT[3]:
                self.toggle_arm()
            else:
                for i, m in enumerate(MODES):
                    y0 = MODE_Y0 + i * MODE_DY
                    if MODE_X[0] <= x <= MODE_X[1] and y0 <= y <= y0 + MODE_DY - 6:
                        self.set_mode(m)
        if self._drag and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE):
            c = LEFT_C if self._drag == "L" else RIGHT_C
            nx = clamp((x - c[0]) / R, -1.0, 1.0)
            ny = clamp(-(y - c[1]) / R, -1.0, 1.0)
            if self._drag == "L":
                self._mouse["yaw"] = nx
                self._mouse["throttle"] = (ny + 1.0) / 2.0
            else:
                self._mouse["roll"], self._mouse["pitch"] = nx, ny
        if event == cv2.EVENT_LBUTTONUP:
            if self._drag == "L":
                self._mouse["yaw"] = 0.0  # yaw tu ve giua, ga giu nguyen
            elif self._drag == "R":
                self._mouse["roll"] = self._mouse["pitch"] = 0.0
            self._drag = None

    # ---- ban phim ----
    def handle_key(self, key: int) -> None:
        if key < 0:
            return
        c = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
        now = self._clock()
        pulses = {"a": ("yaw", -1), "d": ("yaw", 1), "i": ("pitch", 1), "k": ("pitch", -1),
                  "j": ("roll", -1), "l": ("roll", 1)}
        if c in pulses:
            axis, sign = pulses[c]
            self._pulse[axis] = (sign * KEY_AXIS_VALUE, now + KEY_PULSE_S)
        elif c == "w":
            self._mouse["throttle"] = clamp(self._mouse["throttle"] + THROTTLE_STEP, 0.0, 1.0)
        elif c == "s":
            self._mouse["throttle"] = clamp(self._mouse["throttle"] - THROTTLE_STEP, 0.0, 1.0)
        elif c == "x":
            self._mouse["throttle"] = 0.0
        elif c == " ":
            self.toggle_arm()
        elif c in "123":
            self.set_mode(MODES[int(c) - 1])

    # ---- cong tac ----
    def toggle_arm(self) -> None:
        if self.state.arm_switch:
            self.state.arm_switch = False
            self.notice = ""
        elif self._current("throttle") > self._cfg.arm_throttle_max:
            self.notice = "KHONG ARM DUOC: keo ga ve 0 truoc"
        else:
            self.state.arm_switch = True
            self.notice = ""

    def set_mode(self, mode: FlightMode) -> None:
        self.state.mode_switch = mode

    # ---- cap nhat/ve ----
    def _current(self, axis: str) -> float:
        if axis == "throttle":
            return self._mouse["throttle"]
        v = self._mouse[axis]
        p = self._pulse.get(axis)
        if abs(v) < 1e-6 and p is not None and p[1] > self._clock():
            v = p[0]
        return v

    def update(self) -> RCInput:
        dz = self._cfg.stick_deadzone

        def dead(v: float) -> float:
            return 0.0 if abs(v) < dz else v

        s = self.state
        s.roll = dead(self._current("roll"))
        s.pitch = dead(self._current("pitch"))
        s.yaw = dead(self._current("yaw"))
        s.throttle = self._current("throttle")
        return s

    def render(self) -> np.ndarray:
        s = self.update()
        img = np.full((H, W, 3), 38, np.uint8)
        _text(img, "VIRTUAL RADIO", (20, 30), 0.7, (230, 230, 230))
        for c, label in ((LEFT_C, "THROTTLE / YAW"), (RIGHT_C, "PITCH / ROLL")):
            cv2.rectangle(img, (c[0] - R, c[1] - R), (c[0] + R, c[1] + R), (90, 90, 90), 1)
            cv2.line(img, (c[0] - R, c[1]), (c[0] + R, c[1]), (60, 60, 60), 1)
            cv2.line(img, (c[0], c[1] - R), (c[0], c[1] + R), (60, 60, 60), 1)
            _text(img, label, (c[0] - 70, c[1] - R - 10), 0.5, (180, 180, 180))
        lx = int(LEFT_C[0] + s.yaw * R)
        ly = int(LEFT_C[1] - (s.throttle * 2 - 1) * R)
        rx = int(RIGHT_C[0] + s.roll * R)
        ry = int(RIGHT_C[1] - s.pitch * R)
        cv2.circle(img, (lx, ly), 14, (80, 200, 255), -1)
        cv2.circle(img, (rx, ry), 14, (80, 200, 255), -1)

        color = (60, 60, 220) if s.arm_switch else (70, 70, 70)
        cv2.rectangle(img, (ARM_RECT[0], ARM_RECT[1]), (ARM_RECT[2], ARM_RECT[3]), color, -1)
        _text(img, "ARMED" if s.arm_switch else "DISARMED", (ARM_RECT[0] + 10, ARM_RECT[1] + 27), 0.55, (255, 255, 255))
        for i, m in enumerate(MODES):
            y0 = MODE_Y0 + i * MODE_DY
            on = m == s.mode_switch
            cv2.rectangle(img, (MODE_X[0], y0), (MODE_X[1] + 60, y0 + MODE_DY - 6),
                          (60, 160, 60) if on else (70, 70, 70), -1)
            _text(img, f"{i + 1} {MODE_LABEL[m]}", (MODE_X[0] + 6, y0 + 22), 0.45, (255, 255, 255))
        _text(img, f"thr {s.throttle:4.2f} yaw {s.yaw:+.2f} pit {s.pitch:+.2f} rol {s.roll:+.2f}",
              (20, H - 40), 0.5, (200, 200, 200))
        _text(img, "W/S ga  A/D yaw  I/K pitch  J/L roll  Space arm  1/2/3 mode  X ga=0",
              (20, H - 15), 0.45, (150, 150, 150))
        if self.notice:
            _text(img, self.notice, (250, 60), 0.55, (60, 60, 255))
        return img


def _in_box(x: int, y: int, c: tuple[int, int]) -> bool:
    return c[0] - R <= x <= c[0] + R and c[1] - R <= y <= c[1] + R


def _text(img: np.ndarray, s: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]) -> None:
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
