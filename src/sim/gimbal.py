"""Gimbal ao: xoay camera de giu muc tieu o giua hinh (dung ca khi bay thu cong)."""
from __future__ import annotations

import math

from .config import GimbalConfig
from .geometry import bbox_center, clamp, pixel_angles
from .types import BBox, FrameSize


class GimbalTracker:
    """Tich phan goc gimbal theo sai so tren anh. Goc la goc LENH (open-loop); neu muon dung goc
    that thi doc GIMBAL_DEVICE_ATTITUDE_STATUS thay cho bien noi bo."""

    def __init__(self, cfg: GimbalConfig) -> None:
        self._cfg = cfg
        self.pitch_deg = cfg.default_pitch_deg
        self.yaw_deg = 0.0

    def reset(self) -> None:
        self.pitch_deg = self._cfg.default_pitch_deg
        self.yaw_deg = 0.0

    def update(self, bbox: BBox, frame: FrameSize, hfov_deg: float) -> tuple[float, float]:
        cfg = self._cfg
        cx, cy = bbox_center(bbox)
        az, el = pixel_angles(cx, cy, frame, hfov_deg)
        dy = clamp(cfg.k * math.degrees(az), -cfg.max_step_deg, cfg.max_step_deg)
        dp = clamp(cfg.k * math.degrees(el), -cfg.max_step_deg, cfg.max_step_deg)
        self.yaw_deg = clamp(self.yaw_deg + dy, -cfg.yaw_limit_deg, cfg.yaw_limit_deg)
        self.pitch_deg = clamp(self.pitch_deg + dp, cfg.pitch_min_deg, cfg.pitch_max_deg)
        return self.pitch_deg, self.yaw_deg
