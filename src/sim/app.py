"""Vong lap chinh: camera -> detect/track -> chon/khoa muc tieu -> supervisor -> lenh toi PX4."""
from __future__ import annotations

import time

import cv2
import numpy as np

from .adapters import DetectionSource
from .camera_view import CameraView
from .config import SimConfig
from .link import FlightLink
from .messages import MessageBoard
from .sources import FrameSource
from .supervisor import FlightOutput, FlightSupervisor
from .target_manager import TargetManager
from .types import FrameSize, RCInput
from .virtual_radio import VirtualRadio


class SimApp:
    def __init__(self, cfg: SimConfig, camera: FrameSource, detector: DetectionSource,
                 link: FlightLink, radio: VirtualRadio) -> None:
        self.cfg, self.camera, self.detector, self.link, self.radio = cfg, camera, detector, link, radio
        self.board = MessageBoard()
        self.manager = TargetManager(cfg.target)
        self.supervisor = FlightSupervisor(cfg, self.board)
        self.view = CameraView(self.manager)
        self.last_output: FlightOutput | None = None
        self.last_view: np.ndarray | None = None
        self._frame: np.ndarray | None = None
        self._logger = None
        self._last_key: tuple | None = None

    def set_logger(self, logger) -> None:
        """Gan JsonlLogger (src.utils.logging_utils). Chi ghi khi co SU KIEN doi (mode, trang thai, ID khoa)."""
        self._logger = logger

    def _log_event(self, now: float, out: FlightOutput, estimate, telem) -> None:
        if self._logger is None:
            return
        state = estimate.state.value if estimate else None
        key = (out.desired_mode.value, out.status, self.manager.locked_id, state, telem.armed)
        if key == self._last_key:
            return
        self._last_key = key
        self._logger.write({
            "t": round(now, 3), "mode": out.desired_mode.value, "status": out.status,
            "locked_id": self.manager.locked_id, "target_state": state, "armed": telem.armed,
            "messages": [m.text for m in self.board.active(now)],
        })

    def tick(self, now: float) -> FlightOutput | None:
        frame = self.camera.read()
        if frame is not None:
            self._frame = frame
        if self._frame is None:
            return None
        frame = self._frame
        h, w = frame.shape[:2]
        self.link.tick(now)
        telem = self.link.telemetry()
        # Luong anh GOC di vao AI; OSD chi ve len ban sao de hien thi.
        detections = self.detector.detect(frame, now)
        estimate = self.manager.update(detections, now)
        rc: RCInput = self.radio.update()
        out = self.supervisor.step(now, rc, estimate, self.manager.candidates, FrameSize(w, h), telem)

        if out.arm is not None:
            self.link.arm(out.arm)
        self.link.set_mode(out.desired_mode)
        self.link.send_manual(*out.manual)
        self.link.send_velocity(out.velocity)
        if out.gimbal is not None:
            self.link.send_gimbal(*out.gimbal)

        self.last_output = out
        self._log_event(now, out, estimate, telem)
        self.last_view = self.view.render(
            frame, detections=detections, estimate=estimate, telem=telem, rc=rc,
            desired_mode=out.desired_mode, status=out.status,
            messages=self.board.active(now), gimbal=out.gimbal,
        )
        return out

    def run(self) -> None:  # pragma: no cover - can man hinh
        cam_win, radio_win = "Camera + OSD", self.cfg.radio.window_name
        cv2.namedWindow(cam_win)
        cv2.namedWindow(radio_win)
        cv2.setMouseCallback(cam_win, self.view.on_mouse)
        cv2.setMouseCallback(radio_win, self.radio.on_mouse)
        period = 1.0 / self.cfg.loop_hz
        try:
            while True:
                t0 = time.monotonic()
                self.tick(t0)
                if self.last_view is not None:
                    cv2.imshow(cam_win, self.last_view)
                cv2.imshow(radio_win, self.radio.render())
                key = cv2.waitKey(1)
                if (key & 0xFF) in (27, ord("q")):
                    break
                self.radio.handle_key(key)
                self.view.handle_key(key)
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        finally:
            cv2.destroyAllWindows()
            self.camera.close()
            self.link.close()
