"""Nguon "oracle detection" tu boundingbox_camera cua Gazebo (CLAUDE.md muc 2, quyet dinh 2026-11-15, muc 2).

Dung thay `UltralyticsBotSortSource` qua cung interface `DetectionSource` de lop sim chay duoc ma khong phu
thuoc detector. Khong sua detect/track/motion memory, va KHONG dung de danh gia chung (CLAUDE.md muc 12).

Khac voi pipeline that, can nho khi doc ket qua sim:
- `track_id` = label cua plugin Label trong world (moi vat the nen co label rieng). ID khong bao gio doi sau
  che khuat, nen khong co ID switch nhu BoT-SORT.
- `conf` = 1.0 co dinh. Day khong phai confidence cua detector that.
- Box lay tu message bbox moi nhat, khong dong bo theo timestamp voi anh camera. Hai sensor cung pose va
  cung update_rate thi lech toi da khoang mot chu ky.
- boundingbox_camera kieu 2d chi tra vat the nhin thay, nhung khi chi bi che mot phan thi box van co the
  bao ca phan bi che (tuy phien ban Gazebo, chua kiem).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping

import numpy as np

from .types import Detection

# (label, x1, y1, x2, y2) theo pixel
RawBox = tuple[int, float, float, float, float]


def detections_from_boxes(boxes: Iterable[RawBox], label_classes: Mapping[int, str] | None = None,
                          min_box_px: float = 0.0) -> list[Detection]:
    """Doi box tho thanh Detection. Label khong co trong `label_classes` thi ten lop la "label<n>".

    Box co canh nho hon `min_box_px` bi bo (vat the qua xa, chi vai pixel).
    """
    classes = label_classes or {}
    out: list[Detection] = []
    for label, x1, y1, x2, y2 in boxes:
        if min(x2 - x1, y2 - y1) < min_box_px:
            continue
        name = classes.get(int(label), f"label{int(label)}")
        out.append(Detection(int(label), (float(x1), float(y1), float(x2), float(y2)), name, 1.0))
    return out


def boxes_from_msg(msg) -> list[RawBox]:
    """Doc gz.msgs.AnnotatedAxisAligned2DBox_V (truong annotated_box[].{label, box.min_corner, box.max_corner})."""
    return [(int(a.label), float(a.box.min_corner.x), float(a.box.min_corner.y),
             float(a.box.max_corner.x), float(a.box.max_corner.y)) for a in msg.annotated_box]


class GzOracleDetectionSource:
    """Subscribe topic boundingbox_camera (kieu 2d) qua gz-transport, tra Detection theo `DetectionSource`.

    `max_age_s`: neu khong nhan message moi trong khoang nay (tinh theo dong ho may) thi tra danh sach rong,
    de lop sim thay "mat muc tieu" khi Gazebo dung, thay vi giu box cu mai.
    """

    def __init__(self, topic: str, label_classes: Mapping[int, str] | None = None, *,
                 min_box_px: float = 0.0, max_age_s: float = 1.0) -> None:
        try:
            from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
            from gz.transport13 import Node
        except ImportError as exc:  # pragma: no cover - phu thuoc moi truong Gazebo
            raise ImportError(
                "Thieu gz-transport python binding (python3-gz-transport13, python3-gz-msgs10). "
                "Dung ban khop voi Gazebo dang cai."
            ) from exc
        self._classes = dict(label_classes or {})
        self._min_box_px = min_box_px
        self._max_age_s = max_age_s
        self._lock = threading.Lock()
        self._boxes: list[RawBox] = []
        self._stamp: float | None = None
        self._node = Node()
        if not self._node.subscribe(AnnotatedAxisAligned2DBox_V, topic, self._on_boxes):
            raise RuntimeError(f"Khong subscribe duoc topic bbox: {topic}")

    def _on_boxes(self, msg) -> None:  # pragma: no cover - goi tu thread gz-transport
        self._store(boxes_from_msg(msg), time.monotonic())

    def _store(self, boxes: list[RawBox], stamp: float) -> None:
        with self._lock:
            self._boxes, self._stamp = boxes, stamp

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        with self._lock:
            boxes, stamp = list(self._boxes), self._stamp
        if stamp is None or time.monotonic() - stamp > self._max_age_s:
            return []
        return detections_from_boxes(boxes, self._classes, self._min_box_px)
