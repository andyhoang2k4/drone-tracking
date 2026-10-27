"""Cua so hien anh camera + OSD, bat chuot/phim de chon va khoa muc tieu."""
from __future__ import annotations

import cv2
import numpy as np

from .osd import button_rects, draw_osd
from .target_manager import TargetManager


class CameraView:
    def __init__(self, manager: TargetManager, window_name: str = "Camera") -> None:
        self.manager = manager
        self.window_name = window_name
        self._frame_h = 480

    def on_mouse(self, event: int, x: int, y: int, flags: int = 0, param: object = None) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (x1, y1, x2, y2) in button_rects(self._frame_h).items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._press(name)
                return
        self.manager.select_at(x, y)

    def handle_key(self, key: int) -> None:
        if key < 0 or (key & 0xFF) >= 128:
            return
        c = chr(key & 0xFF).lower()
        if c == "l":
            self._press("lock")
        elif c == "u":
            self._press("unlock")
        elif c == "n":
            self._press("next")

    def _press(self, name: str) -> None:
        m = self.manager
        if name == "lock":
            m.lock_selected()
        elif name == "unlock":
            m.unlock()
        elif name == "next" and m.candidates:
            ids = [d.track_id for d in m.candidates]
            i = ids.index(m.selected_id) + 1 if m.selected_id in ids else 0
            m.selected_id = ids[i % len(ids)]

    def render(self, frame: np.ndarray, **osd_kwargs) -> np.ndarray:
        self._frame_h = frame.shape[0]
        return draw_osd(frame, selected_id=self.manager.selected_id, **osd_kwargs)
