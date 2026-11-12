"""Noi pipeline detect/track voi lop mo phong qua `DetectionSource`.

Neu repo da co src/track.py bao ham YOLO-seg + BoT-SORT thi nen viet adapter mong goi no thay vi
dung class duoi day (tranh nhan doi logic). Class o day chi la ban toi thieu de chay duoc.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import Detection


class DetectionSource(Protocol):
    def detect(self, frame: np.ndarray, t: float) -> list[Detection]: ...


class UltralyticsBotSortSource:
    """YOLO-seg + BoT-SORT cua Ultralytics (chi dung bbox, dung voi Motion Memory contract).
    Chua kiem chung trong moi truong nay (khong cai ultralytics) - kiem tra API theo version da khoa."""

    def __init__(self, weights: str, tracker_cfg: str, conf: float = 0.25, *, imgsz: int,
                 device: str | None = None) -> None:
        """imgsz bat buoc (khong mac dinh): nguoi goi chon bang src.utils.inference_settings.resolve_imgsz."""
        from ultralytics import YOLO  # import tre

        self._model = YOLO(weights)
        self._tracker_cfg = tracker_cfg
        self.imgsz = imgsz
        self._kw = dict(conf=conf, imgsz=imgsz, verbose=False)
        if device is not None:
            self._kw["device"] = device

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        res = self._model.track(frame, persist=True, tracker=self._tracker_cfg, **self._kw)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        out: list[Detection] = []
        for xyxy, tid, cls, conf in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().tolist(),
                                        boxes.cls.int().cpu().tolist(), boxes.conf.cpu().tolist()):
            out.append(Detection(int(tid), tuple(float(v) for v in xyxy), self._model.names[int(cls)], float(conf)))
        return out
