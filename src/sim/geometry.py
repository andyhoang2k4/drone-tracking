"""Ham hinh hoc nho dung chung."""
from __future__ import annotations

import math

from .types import BBox, FrameSize


def wrap_pi(a: float) -> float:
    """Dua goc ve (-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bbox_center(b: BBox) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def bbox_size(b: BBox) -> tuple[float, float]:
    return b[2] - b[0], b[3] - b[1]


def focal_px(frame: FrameSize, hfov_deg: float) -> float:
    return (frame.width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


def pixel_angles(cx: float, cy: float, frame: FrameSize, hfov_deg: float) -> tuple[float, float]:
    """Goc lech (rad) cua diem anh so voi truc camera: (ngang, doc).
    Ngang duong = ben phai. Doc duong = phia tren tam anh."""
    f = focal_px(frame, hfov_deg)
    az = math.atan((cx - frame.width / 2.0) / f)
    el = -math.atan((cy - frame.height / 2.0) / f)
    return az, el
