"""Chon va khoa muc tieu tren man hinh camera, theo doi trang thai TRACKING/LOST/REIDENTIFYING/FAILED.

Day la lop tieu thu o phia bay: nhan detection da co track ID tu pipeline (BoT-SORT) va tao
`TargetEstimate`. Khi motion_memory.py cua repo san sang, co the thay bang adapter dung chung
interface `update(detections, t) -> TargetEstimate | None` (xem CLAUDE.md muc 6).
Phan du doan o day chi la ngoai suy van toc tam anh don gian de dieu khien bay, KHONG phai
motion memory.
"""
from __future__ import annotations

from collections import deque

from .config import TargetConfig
from .geometry import bbox_center, bbox_size
from .types import BBox, Detection, TargetEstimate, TargetState


class TargetManager:
    def __init__(self, cfg: TargetConfig) -> None:
        self._cfg = cfg
        self.candidates: list[Detection] = []
        self.selected_id: int | None = None
        self.locked_id: int | None = None
        self.locked_label: str | None = None
        self.estimate: TargetEstimate | None = None
        self._hist: deque[tuple[float, float, float]] = deque(maxlen=cfg.history_len)
        self._last_bbox: BBox | None = None
        self._last_t: float | None = None

    # ---- thao tac nguoi dung ----
    def select_at(self, x: float, y: float) -> int | None:
        """Chon vat the chua diem (x, y); neu nhieu vat the long nhau thi lay cai nho nhat."""
        best: Detection | None = None
        best_area = float("inf")
        for d in self.candidates:
            x1, y1, x2, y2 = d.bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = d, area
        self.selected_id = best.track_id if best else None
        return self.selected_id

    def lock_selected(self) -> bool:
        det = next((d for d in self.candidates if d.track_id == self.selected_id), None)
        if det is None:
            return False
        self.locked_id = det.track_id
        self.locked_label = det.label
        self._hist.clear()
        self._confirm(det, self._last_t if self._last_t is not None else 0.0)
        return True

    def unlock(self) -> None:
        self.locked_id = None
        self.locked_label = None
        self.estimate = None
        self._hist.clear()
        self._last_bbox = None

    # ---- cap nhat moi frame ----
    def update(self, detections: list[Detection], t: float) -> TargetEstimate | None:
        self.candidates = list(detections)
        if self.selected_id is not None and all(d.track_id != self.selected_id for d in detections):
            if self.selected_id != self.locked_id:
                self.selected_id = None
        if self.locked_id is None:
            self.estimate = None
            self._last_t = t
            return None

        det = next((d for d in detections if d.track_id == self.locked_id), None)
        if det is not None:
            self._confirm(det, t)
            return self.estimate

        assert self._last_t is not None and self._last_bbox is not None
        dt = t - self._last_t
        cx, cy = self._predict_center(dt)
        cfg = self._cfg
        failed = dt >= cfg.lost_timeout_s + cfg.reid_timeout_s  # FAILED la trang thai cuoi: khoa lai bang tay
        if not failed:
            cand = self._reacquire(cx, cy, dt, strict=dt >= cfg.lost_timeout_s)
            if cand is not None:  # thay ID (vi du BoT-SORT cap ID moi sau che khuat): theo ID moi
                self.locked_id = cand.track_id
                self._confirm(cand, t)
                return self.estimate

        if dt < cfg.lost_timeout_s:
            state = TargetState.LOST
            conf = max(0.0, 1.0 - dt / cfg.lost_timeout_s)
            w, h = bbox_size(self._last_bbox)
            bbox: BBox | None = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        elif dt < cfg.lost_timeout_s + cfg.reid_timeout_s:
            state, conf, bbox = TargetState.REIDENTIFYING, 0.0, None
        else:
            state, conf, bbox = TargetState.FAILED, 0.0, None
        self.estimate = TargetEstimate(self.locked_id, state, bbox, dt, conf)
        return self.estimate

    # ---- noi bo ----
    def _confirm(self, det: Detection, t: float) -> None:
        cx, cy = bbox_center(det.bbox)
        self._hist.append((t, cx, cy))
        self._last_bbox = det.bbox
        self._last_t = t
        self.estimate = TargetEstimate(det.track_id, TargetState.TRACKING, det.bbox, 0.0, 1.0)

    def _velocity(self) -> tuple[float, float]:
        if len(self._hist) < 2:
            return 0.0, 0.0
        t0, x0, y0 = self._hist[0]
        t1, x1, y1 = self._hist[-1]
        if t1 - t0 <= 1e-6:
            return 0.0, 0.0
        return (x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)

    def _predict_center(self, dt: float) -> tuple[float, float]:
        assert self._last_bbox is not None
        cx, cy = bbox_center(self._last_bbox)
        vx, vy = self._velocity()
        return cx + vx * dt, cy + vy * dt

    def _reacquire(self, px: float, py: float, dt: float, strict: bool) -> Detection | None:
        """Bat lai bang gating tren vi tri du doan va nhan lop. Day la heuristic tam thoi,
        thay cho Re-ID appearance that cua pipeline. Khi `strict` (REIDENTIFYING) chi bat khi
        trong khung hinh co DUNG MOT vat the cung nhan lop, tranh bat nham nguoi khac."""
        cfg = self._cfg
        radius = min(cfg.reacquire_base_px + cfg.reacquire_speed_px_s * dt, cfg.reacquire_max_px)
        same = [d for d in self.candidates if d.label == self.locked_label]
        if strict and len(same) != 1:
            return None
        best, best_d = None, float("inf")
        for d in same:
            cx, cy = bbox_center(d.bbox)
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist <= radius and dist < best_d:
                best, best_d = d, dist
        return best
