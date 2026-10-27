"""Moi truong gia lap nhe (khong can PX4/Gazebo) de chay thu giao dien va logic bay.

Co: drone dong hoc don gian, nguoi di bo (muc tieu), nguoi nhieu, xe dung yen, cay che khuat.
Khi muc tieu di sau cay roi ra lai, `DemoDetector` cap ID MOI (gia lap BoT-SORT doi ID sau che khuat).
Day KHONG phai vat ly PX4 - chi de kiem tra luong dieu khien.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import SimConfig
from .follow_controller import BodyVelocity
from .geometry import clamp
from .types import BBox, Detection, FlightMode, Telemetry

PERSON_H, PERSON_W = 1.7, 0.5


class DemoWorld:
    """He toa do: x bac, y dong, z len. yaw 0 = huong bac, tang theo chieu kim dong ho."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.W, self.H = cfg.camera.width, cfg.camera.height
        self.f_px = (self.W / 2) / math.tan(math.radians(cfg.camera.hfov_deg) / 2)
        self.t = 0.0
        self.pos = np.array([-10.0, 0.0, 5.0])
        self.yaw = 0.0
        self.armed = False
        self.mode = FlightMode.STABILIZED
        self.gimbal_pitch, self.gimbal_yaw = cfg.gimbal.default_pitch_deg, 0.0
        self.trees = [(6.0, 3.3, 2.2), (16.0, -2.0, 0.8)]  # (x, y, ban kinh); vat can dau rong de che lau
        self._cmd = dict(pitch=0.0, roll=0.0, throttle=0.0, yaw=0.0)
        self._vel = BodyVelocity()
        # muc tieu di vong thang bac->dong, dung ve phia sau cay thu nhat
        self.people: dict[int, np.ndarray] = {1: np.array([0.0, 6.0]), 2: np.array([-2.0, -6.0])}
        self.car = np.array([4.0, -8.0])
        self.id_shift = 0  # tang moi lan muc tieu 1 bi che -> ra lai

    # ---- dong hoc ----
    def step(self, dt: float) -> None:
        self.t += dt
        self.people[1] += np.array([0.0, 0.0]) + dt * np.array([1.2, -0.0])
        self.people[1][1] = 6.0 - 0.45 * self.people[1][0]  # di cheo, qua sau cay (6,3)
        self.people[2] += dt * np.array([0.3, 0.2])
        if not self.armed:
            return
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        if self.mode == FlightMode.AUTO_FOLLOW:
            fwd, right, down, yr = self._vel.forward, self._vel.right, self._vel.down, self._vel.yaw_rate
        else:
            fwd, right = self._cmd["pitch"] * 6.0, self._cmd["roll"] * 6.0
            down = -(self._cmd["throttle"] - 0.5) * 4.0
            yr = self._cmd["yaw"] * math.radians(90)
        self.pos[0] += (fwd * c - right * s) * dt
        self.pos[1] += (fwd * s + right * c) * dt
        self.pos[2] = max(0.0, self.pos[2] - down * dt)
        self.yaw = (self.yaw + yr * dt) % (2 * math.pi)

    # ---- camera ----
    def _axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        yaw = self.yaw + math.radians(self.gimbal_yaw)
        p = math.radians(self.gimbal_pitch)
        f = np.array([math.cos(p) * math.cos(yaw), math.cos(p) * math.sin(yaw), math.sin(p)])
        r = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return f, r, np.cross(f, r)

    def project(self, p: np.ndarray) -> tuple[float, float, float] | None:
        f, r, u = self._axes()
        d = p - self.pos
        zc = float(d @ f)
        if zc < 0.5:
            return None
        return (self.W / 2 + self.f_px * float(d @ r) / zc, self.H / 2 - self.f_px * float(d @ u) / zc, zc)

    def _occluded(self, xy: np.ndarray) -> bool:
        a, b = self.pos[:2], xy
        ab = b - a
        for tx, ty, rad in self.trees:
            t = clamp(float((np.array([tx, ty]) - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
            if np.linalg.norm(a + t * ab - np.array([tx, ty])) < rad and t < 0.98:
                return True
        return False

    def person_bbox(self, xy: np.ndarray) -> BBox | None:
        base = np.array([xy[0], xy[1], 0.0])
        top = base + np.array([0, 0, PERSON_H])
        pb, pt = self.project(base), self.project(top)
        if pb is None or pt is None:
            return None
        w = PERSON_W * self.f_px / pb[2]
        box = (pb[0] - w / 2, pt[1], pb[0] + w / 2, pb[1])
        if box[2] < 0 or box[0] > self.W or box[3] < 0 or box[1] > self.H or (box[3] - box[1]) < 6:
            return None
        return box

    def visible_people(self) -> dict[int, BBox]:
        out = {}
        for pid, xy in self.people.items():
            box = self.person_bbox(xy)
            if box is not None and not self._occluded(xy):
                out[pid] = box
        return out

    def render(self) -> np.ndarray:
        img = np.empty((self.H, self.W, 3), np.uint8)
        f, r, u = self._axes()
        horizon = int(clamp(self.H / 2 + self.f_px * math.tan(math.radians(self.gimbal_pitch)), 0, self.H))
        img[:horizon] = (235, 200, 150)
        img[horizon:] = (90, 120, 100)
        items: list[tuple[float, str, object]] = []
        for tx, ty, rad in self.trees:
            pb, pt = self.project(np.array([tx, ty, 0.0])), self.project(np.array([tx, ty, 7.0]))
            if pb and pt:
                items.append((pb[2], "tree", (pb, pt, rad)))
        for pid, xy in self.people.items():
            box = self.person_bbox(xy)
            if box:
                items.append((self.project(np.array([xy[0], xy[1], 0.0]))[2], "person", (pid, box)))
        cb = self.project(np.array([self.car[0], self.car[1], 0.0]))
        if cb:
            items.append((cb[2], "car", cb))
        for _, kind, data in sorted(items, key=lambda i: -i[0]):
            if kind == "tree":
                pb, pt, rad = data
                w = int(rad * 2 * self.f_px / pb[2])
                cv2.rectangle(img, (int(pb[0] - w / 2), int(pt[1])), (int(pb[0] + w / 2), int(pb[1])), (30, 90, 40), -1)
            elif kind == "person":
                pid, b = data
                col = (40, 40, 200) if pid == 1 else (200, 120, 40)
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, -1)
                cv2.circle(img, (int((b[0] + b[2]) / 2), int(b[1])), max(2, int((b[2] - b[0]) / 3)), (170, 200, 230), -1)
            else:
                w = 4.0 * self.f_px / data[2]
                cv2.rectangle(img, (int(data[0] - w / 2), int(data[1] - w / 3)), (int(data[0] + w / 2), int(data[1])), (150, 150, 150), -1)
        return img

    def car_bbox(self) -> BBox | None:
        cb = self.project(np.array([self.car[0], self.car[1], 0.0]))
        if cb is None:
            return None
        w = 4.0 * self.f_px / cb[2]
        b = (cb[0] - w / 2, cb[1] - w / 3, cb[0] + w / 2, cb[1])
        return b if 0 <= b[2] and b[0] <= self.W and 0 <= b[3] and b[1] <= self.H else None


class DemoCamera:
    def __init__(self, world: DemoWorld) -> None:
        self._w = world

    def read(self) -> np.ndarray:
        return self._w.render()

    def close(self) -> None:
        pass


class DemoDetector:
    """Thay cho YOLO+BoT-SORT: tra bbox that cua the gioi gia lap. Muc tieu 1 doi ID sau moi lan bi che."""

    def __init__(self, world: DemoWorld) -> None:
        self._w = world
        self._was_visible = True
        self._shift = 0

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        vis = self._w.visible_people()
        now_vis = 1 in vis
        if now_vis and not self._was_visible:
            self._shift += 100  # BoT-SORT cap ID moi sau che khuat
        self._was_visible = now_vis
        out = [Detection(pid + (self._shift if pid == 1 else 0), b, "person", 0.9) for pid, b in vis.items()]
        cb = self._w.car_bbox()
        if cb:
            out.append(Detection(3, cb, "car", 0.85))
        return out


class DemoLink:
    """FlightLink gia: dieu khien DemoWorld. Dung de chay demo va test, khong noi PX4."""

    def __init__(self, world: DemoWorld) -> None:
        self.w = world
        self._last_now: float | None = None
        self._desired = FlightMode.STABILIZED
        self.log: list[tuple[str, object]] = []

    def connect(self) -> None: ...
    def close(self) -> None: ...

    def tick(self, now: float) -> None:
        if self._last_now is not None:
            self.w.step(max(0.0, now - self._last_now))
        self._last_now = now
        self.w.mode = self._desired

    def telemetry(self) -> Telemetry:
        w = self.w
        return Telemetry(True, w.armed, w.mode.value, float(w.pos[2]), 0.0, w.yaw, 100.0, 3, 12)

    def arm(self, on: bool) -> None:
        self.w.armed = on
        self.log.append(("arm", on))

    def set_mode(self, mode: FlightMode) -> None:
        if mode != self._desired:
            self.log.append(("mode", mode))
        self._desired = mode

    def send_manual(self, pitch: float, roll: float, throttle: float, yaw: float) -> None:
        self.w._cmd = dict(pitch=pitch, roll=roll, throttle=throttle, yaw=yaw)

    def send_velocity(self, v: BodyVelocity) -> None:
        self.w._vel = v

    def send_gimbal(self, pitch_deg: float, yaw_deg: float) -> None:
        self.w.gimbal_pitch, self.w.gimbal_yaw = pitch_deg, yaw_deg
