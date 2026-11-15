"""Doc cau hinh mo phong tu configs/sim.yaml. Moi gia tri co default, file yaml chi can ghi de."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LinkConfig:
    url: str = "udpin:0.0.0.0:14540"  # cong offboard/onboard cua PX4 SITL
    heartbeat_hz: float = 1.0
    connect_timeout_s: float = 10.0
    allow_force_disarm: bool = False


@dataclass
class CameraConfig:
    gz_topic: str = "/camera"
    hfov_deg: float = 100.0  # x500_mono_cam ~1.74 rad; doi theo model dang dung
    width: int = 640
    height: int = 480


@dataclass
class RadioConfig:
    stick_deadzone: float = 0.03
    arm_throttle_max: float = 0.05  # chi cho arm khi throttle thap
    window_name: str = "Virtual radio"


@dataclass
class TargetConfig:
    lost_timeout_s: float = 1.5  # TRACKING mat -> LOST, qua nguong nay -> REIDENTIFYING
    reid_timeout_s: float = 8.0  # trong REIDENTIFYING, het nguong -> FAILED
    reacquire_base_px: float = 60.0
    reacquire_speed_px_s: float = 200.0
    reacquire_max_px: float = 200.0  # tran ban kinh gating, tranh bat nham nguoi khac
    history_len: int = 12
    # Bo uoc luong muc tieu: "heuristic" (TargetManager, ngoai suy tam anh) hoac "motion_memory"
    # (src/sim/mm_target.py dung src/motion_memory.py). Cac nguong *_timeout_s/reacquire_* chi dung cho heuristic.
    estimator: str = "heuristic"
    motion_memory_config: str = "configs/motion_memory.yaml"  # tuong doi theo thu muc chay (goc repo)
    mm_frame_rate_hz: float = 30.0  # doi giay -> frame_idx cho MotionMemory (nguong cua no tinh theo frame)


@dataclass
class FollowConfig:
    desired_height_frac: float = 0.15  # chieu cao bbox / chieu cao anh o khoang cach mong muon; chinh theo do cao bay va FOV
    k_yaw: float = 1.2  # rad/s tren moi rad sai so
    k_fwd: float = 4.0  # m/s tren moi don vi sai so kich thuoc tuong doi
    max_fwd_ms: float = 4.0
    max_back_ms: float = 1.5
    max_yaw_rate_deg: float = 60.0
    yaw_deadband_deg: float = 1.5
    slow_when_yaw_err_deg: float = 35.0  # sai so yaw lon thi giam toc do tien


@dataclass
class SearchConfig:
    search_speed_ms: float = 1.0
    k_yaw: float = 1.0
    max_lead_deg: float = 60.0
    fit_window_s: float = 1.0


@dataclass
class GimbalConfig:
    enabled: bool = True
    k: float = 0.6  # ti le sua goc moi buoc
    max_step_deg: float = 4.0
    pitch_min_deg: float = -90.0
    pitch_max_deg: float = 30.0
    yaw_limit_deg: float = 170.0
    default_pitch_deg: float = -15.0


@dataclass
class OracleConfig:
    """Nguon oracle detection tu boundingbox_camera cua Gazebo (src/sim/oracle.py). Khong dung de danh gia."""

    bbox_topic: str = "/camera/bboxes"  # topic cua sensor boundingbox_camera (box_type 2d); kiem tra bang `gz topic -l`
    label_classes: dict[int, str] = field(default_factory=dict)  # label (plugin Label) -> ten lop; thieu thi "label<n>"
    min_box_px: float = 0.0
    max_age_s: float = 1.0


@dataclass
class SimConfig:
    loop_hz: float = 30.0
    link: LinkConfig = field(default_factory=LinkConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    follow: FollowConfig = field(default_factory=FollowConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    gimbal: GimbalConfig = field(default_factory=GimbalConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    auto_engage_on_lock: bool = True  # cong tac dang o Auto, vua khoa muc tieu -> vao Auto luon


def _merge(obj: Any, data: dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"Khoa cau hinh khong hop le: {key!r} (hop le: {sorted(names)})")
        cur = getattr(obj, key)
        if is_dataclass(cur) and isinstance(value, dict):
            _merge(cur, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path | None) -> SimConfig:
    cfg = SimConfig()
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(cfg, data)
    return cfg
