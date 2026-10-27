"""Bay tim muc tieu theo huong ma muc tieu da di truoc khi bien mat."""
from __future__ import annotations

import math
from collections import deque

from .config import SearchConfig
from .follow_controller import BodyVelocity
from .geometry import clamp, wrap_pi


class BearingMemory:
    """Nho huong (so voi bac, rad) cua muc tieu trong cua so thoi gian gan nhat, de uoc luong toc do doi huong."""

    def __init__(self, window_s: float) -> None:
        self._window = window_s
        self._buf: deque[tuple[float, float]] = deque()

    def clear(self) -> None:
        self._buf.clear()

    def add(self, t: float, bearing: float) -> None:
        if self._buf:  # giu lien tuc, tranh nhay 2*pi
            bearing = self._buf[-1][1] + wrap_pi(bearing - self._buf[-1][1])
        self._buf.append((t, bearing))
        while self._buf and t - self._buf[0][0] > self._window:
            self._buf.popleft()

    @property
    def empty(self) -> bool:
        return not self._buf

    def last_and_rate(self) -> tuple[float, float]:
        """(huong cuoi, toc do doi huong rad/s) bang hoi quy tuyen tinh."""
        if not self._buf:
            return 0.0, 0.0
        last = self._buf[-1][1]
        n = len(self._buf)
        if n < 2:
            return last, 0.0
        mt = sum(t for t, _ in self._buf) / n
        mb = sum(b for _, b in self._buf) / n
        den = sum((t - mt) ** 2 for t, _ in self._buf)
        if den < 1e-9:
            return last, 0.0
        return last, sum((t - mt) * (b - mb) for t, b in self._buf) / den


def search_command(
    last_bearing: float,
    bearing_rate: float,
    since_lost_s: float,
    yaw_now: float,
    last_forward_ms: float,
    cfg: SearchConfig,
) -> BodyVelocity:
    """Quay ve huong cuoi cung cua muc tieu (co cong them huong dang di), vua quay vua tien."""
    lead = clamp(bearing_rate * since_lost_s, -math.radians(cfg.max_lead_deg), math.radians(cfg.max_lead_deg))
    err = wrap_pi(last_bearing + lead - yaw_now)
    yaw_rate = clamp(cfg.k_yaw * err, -math.radians(60.0), math.radians(60.0))
    speed = max(cfg.search_speed_ms, 0.5 * max(last_forward_ms, 0.0))
    if abs(math.degrees(err)) > 45.0:
        speed *= 0.3
    return BodyVelocity(forward=speed, yaw_rate=yaw_rate)
