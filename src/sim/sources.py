"""Nguon anh camera: Gazebo (gz-transport), file video. Moi nguon co `read()` tra ve anh BGR hoac None."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...
    def close(self) -> None: ...


class VideoFileSource:
    """Phat lai file video (de thu pipeline khi chua co Gazebo). Het video thi lap lai."""

    def __init__(self, path: str | Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Khong mo duoc video: {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        self._cap.release()


class GzCameraSource:
    """Nhan anh tu topic camera cua Gazebo qua gz-transport (python binding).

    Can goi apt: python3-gz-transport13 python3-gz-msgs10 (Gazebo Harmonic) hoac ban tuong ung.
    Chua kiem chung voi Gazebo that - kiem tra ten topic bang `gz topic -l` va dinh dang pixel.
    """

    def __init__(self, topic: str) -> None:
        try:
            from gz.msgs10.image_pb2 import Image
            from gz.transport13 import Node
        except ImportError as exc:  # pragma: no cover - phu thuoc moi truong Gazebo
            raise ImportError(
                "Thieu gz-transport python binding (python3-gz-transport13, python3-gz-msgs10). "
                "Dung ban khop voi Gazebo dang cai."
            ) from exc
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._node = Node()
        if not self._node.subscribe(Image, topic, self._on_image):
            raise RuntimeError(f"Khong subscribe duoc topic camera: {topic}")

    def _on_image(self, msg) -> None:  # pragma: no cover
        # pixel_format_type 3 = RGB_INT8 trong gz.msgs; nhan 3 kenh 8-bit
        arr = np.frombuffer(msg.data, np.uint8)
        if arr.size != msg.height * msg.width * 3:
            return
        rgb = arr.reshape(msg.height, msg.width, 3)
        with self._lock:
            self._frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def read(self) -> np.ndarray | None:  # pragma: no cover
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self) -> None:  # pragma: no cover
        pass
