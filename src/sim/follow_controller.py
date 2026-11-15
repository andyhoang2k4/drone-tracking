"""Tinh van toc ham bam muc tieu tu bbox tren anh (khong phu thuoc PX4)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import FollowConfig
from .geometry import bbox_center, bbox_size, clamp, pixel_angles
from .types import BBox, FrameSize


@dataclass(frozen=True)
class BodyVelocity:
    """He than may: forward (+ ra truoc), right (+ sang phai), down (+ xuong), yaw_rate (+ quay phai)."""

    forward: float = 0.0
    right: float = 0.0
    down: float = 0.0
    yaw_rate: float = 0.0  # rad/s


ZERO_VELOCITY = BodyVelocity()


def bearing_error_rad(bbox: BBox, frame: FrameSize, hfov_deg: float, gimbal_yaw_deg: float = 0.0) -> float:
    """Sai so huong cua muc tieu so voi MUI may bay (rad, + la ben phai).
    Gimbal xoay yaw thi cong them goc gimbal."""
    cx, cy = bbox_center(bbox)
    az, _ = pixel_angles(cx, cy, frame, hfov_deg)
    return az + math.radians(gimbal_yaw_deg)


def follow_command(
    bbox: BBox, frame: FrameSize, hfov_deg: float, cfg: FollowConfig, gimbal_yaw_deg: float = 0.0
) -> BodyVelocity:
    err = bearing_error_rad(bbox, frame, hfov_deg, gimbal_yaw_deg)
    err_deg = math.degrees(err)
    max_yaw = math.radians(cfg.max_yaw_rate_deg)
    yaw_rate = 0.0 if abs(err_deg) < cfg.yaw_deadband_deg else clamp(cfg.k_yaw * err, -max_yaw, max_yaw)

    _, h = bbox_size(bbox)
    h_frac = h / frame.height
    size_err = (cfg.desired_height_frac - h_frac) / cfg.desired_height_frac
    fwd = clamp(cfg.k_fwd * size_err, -cfg.max_back_ms, cfg.max_fwd_ms)
    if bbox[3] >= frame.height * (1.0 - cfg.bottom_edge_frac):  # box bi cat o mep duoi -> khong tien
        fwd = min(fwd, 0.0)
    if fwd > 0:  # chua huong dung thi chua lao toi
        fwd *= clamp(1.0 - abs(err_deg) / cfg.slow_when_yaw_err_deg, 0.0, 1.0)
    return BodyVelocity(forward=fwd, yaw_rate=yaw_rate)
