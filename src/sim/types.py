"""Kieu du lieu dung chung cua lop mo phong."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..motion_memory import TrackState

# (x1, y1, x2, y2) theo pixel cua khung hinh camera
BBox = tuple[float, float, float, float]


class FlightMode(str, Enum):
    STABILIZED = "STABILIZED"  # "Angle" trong Betaflight/EdgeTX; PX4 goi la Stabilized
    ACRO = "ACRO"
    AUTO_FOLLOW = "AUTO_FOLLOW"  # Offboard, bam muc tieu


# Dung lai enum cua src/motion_memory.py (CLAUDE.md muc 6) thay vi dinh nghia trung lap.
TargetState = TrackState


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox: BBox
    label: str
    conf: float  # confidence cua DETECTOR, khac voi confidence du doan cua estimator


@dataclass(frozen=True)
class TargetEstimate:
    """Uoc luong muc tieu dang khoa. `prediction_confidence` la do tin cay cua DU DOAN,
    khong gop voi confidence cua detector."""

    track_id: int
    state: TargetState
    bbox: BBox | None
    seconds_since_confirmed: float
    prediction_confidence: float


@dataclass(frozen=True)
class FrameSize:
    width: int
    height: int


@dataclass
class RCInput:
    """Dau vao tu tay dieu khien ao. Stick -1..1, throttle 0..1."""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    throttle: float = 0.0
    arm_switch: bool = False
    mode_switch: FlightMode = FlightMode.STABILIZED


@dataclass
class Telemetry:
    connected: bool = False
    armed: bool = False
    mode_name: str = "?"
    rel_alt_m: float = 0.0
    ground_speed_ms: float = 0.0
    yaw_rad: float = 0.0
    battery_pct: float = -1.0
    gps_fix: int = 0
    gps_sats: int = 0
