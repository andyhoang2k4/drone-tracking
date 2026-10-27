"""Lien ket MAVLink toi PX4 (SITL hoac that). Dung pymavlink, import tre de phan logic test duoc khong can cai.

Khop dung vai tro cua Jetson trong drone that: nhan ANH + gui MAVLink (offboard, manual control,
gimbal) cho flight controller.
"""
from __future__ import annotations

import math
import time
from typing import Protocol

from .config import LinkConfig
from .follow_controller import BodyVelocity
from .types import FlightMode, Telemetry

# PX4 custom main mode (px4_custom_mode.h)
PX4_MAIN_MODE = {FlightMode.ACRO: 5, FlightMode.AUTO_FOLLOW: 6, FlightMode.STABILIZED: 7}
PX4_MAIN_NAME = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO", 6: "OFFBOARD", 7: "STABILIZED", 8: "RATTITUDE"}

# SET_POSITION_TARGET_LOCAL_NED type_mask: bit = 1 nghia la BO QUA truong do
_IGN_POS, _IGN_VEL, _IGN_ACC, _IGN_YAW, _IGN_YAW_RATE = 0b111, 0b111 << 3, 0b111 << 6, 1 << 10, 1 << 11
VELOCITY_YAWRATE_MASK = _IGN_POS | _IGN_ACC | _IGN_YAW  # dung van toc + toc do yaw

CMD_DO_GIMBAL_MANAGER_PITCHYAW = 1000
CMD_DO_GIMBAL_MANAGER_CONFIGURE = 1001
FORCE_DISARM_MAGIC = 21196.0


def decode_main_mode(custom_mode: int) -> str:
    return PX4_MAIN_NAME.get((int(custom_mode) >> 16) & 0xFF, "?")


def body_to_ned(v: BodyVelocity, yaw_rad: float) -> tuple[float, float, float]:
    """Van toc he than (forward/right/down) -> NED (north/east/down) theo yaw hien tai."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return v.forward * c - v.right * s, v.forward * s + v.right * c, v.down


class FlightLink(Protocol):
    def connect(self) -> None: ...
    def close(self) -> None: ...
    def tick(self, now: float) -> None: ...
    def telemetry(self) -> Telemetry: ...
    def arm(self, on: bool) -> None: ...
    def set_mode(self, mode: FlightMode) -> None: ...
    def send_manual(self, pitch: float, roll: float, throttle: float, yaw: float) -> None: ...
    def send_velocity(self, v: BodyVelocity) -> None: ...
    def send_gimbal(self, pitch_deg: float, yaw_deg: float) -> None: ...


class PymavlinkLink:
    SRC_SYSTEM = 255
    SRC_COMPONENT = 191  # MAV_COMP_ID_ONBOARD_COMPUTER

    def __init__(self, cfg: LinkConfig, gimbal_enabled: bool = True) -> None:
        self._cfg = cfg
        self._gimbal_enabled = gimbal_enabled
        self._m = None
        self._telem = Telemetry()
        self._desired: FlightMode | None = None
        self._last_mode_cmd = 0.0
        self._last_hb = 0.0
        self._gimbal_configured = False
        self._last_gimbal: tuple[float, float] | None = None

    # ---- vong doi ----
    def connect(self) -> None:
        from pymavlink import mavutil

        self._mav = mavutil.mavlink
        self._m = mavutil.mavlink_connection(
            self._cfg.url, source_system=self.SRC_SYSTEM, source_component=self.SRC_COMPONENT
        )
        hb = self._m.wait_heartbeat(timeout=self._cfg.connect_timeout_s)
        if hb is None:
            raise TimeoutError(f"Khong nhan duoc heartbeat tu PX4 tai {self._cfg.url}")
        self._telem.connected = True

    def close(self) -> None:
        if self._m is not None:
            self._m.close()

    def tick(self, now: float) -> None:
        """Gui heartbeat, doc het tin den, gui lai lenh doi mode neu PX4 chua vao dung mode."""
        if self._m is None:
            return
        if now - self._last_hb >= 1.0 / self._cfg.heartbeat_hz:
            self._m.mav.heartbeat_send(
                self._mav.MAV_TYPE_ONBOARD_CONTROLLER, self._mav.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )
            self._last_hb = now
        while True:
            msg = self._m.recv_match(blocking=False)
            if msg is None:
                break
            self._on_message(msg)
        if self._desired is not None and now - self._last_mode_cmd > 0.5:
            want = PX4_MAIN_NAME[PX4_MAIN_MODE[self._desired]]
            if self._telem.mode_name != want:
                self._send_mode(self._desired)
            self._last_mode_cmd = now

    def telemetry(self) -> Telemetry:
        return self._telem

    # ---- lenh ----
    def arm(self, on: bool) -> None:
        force = FORCE_DISARM_MAGIC if (not on and self._cfg.allow_force_disarm) else 0.0
        self._cmd(self._mav.MAV_CMD_COMPONENT_ARM_DISARM, 1.0 if on else 0.0, force)

    def set_mode(self, mode: FlightMode) -> None:
        if mode != self._desired:
            self._desired = mode
            if self._m is not None:
                self._send_mode(mode)
                self._last_mode_cmd = time.monotonic()

    def send_manual(self, pitch: float, roll: float, throttle: float, yaw: float) -> None:
        if self._m is None:
            return
        i = lambda v, lo, hi: int(max(lo, min(hi, round(v * 1000))))  # noqa: E731
        self._m.mav.manual_control_send(
            self._m.target_system, i(pitch, -1000, 1000), i(roll, -1000, 1000),
            i(throttle, 0, 1000), i(yaw, -1000, 1000), 0,
        )

    def send_velocity(self, v: BodyVelocity) -> None:
        if self._m is None:
            return
        vn, ve, vd = body_to_ned(v, self._telem.yaw_rad)
        self._m.mav.set_position_target_local_ned_send(
            0, self._m.target_system, self._m.target_component, self._mav.MAV_FRAME_LOCAL_NED,
            VELOCITY_YAWRATE_MASK, 0, 0, 0, vn, ve, vd, 0, 0, 0, 0, v.yaw_rate,
        )

    def send_gimbal(self, pitch_deg: float, yaw_deg: float) -> None:
        if self._m is None or not self._gimbal_enabled:
            return
        if not self._gimbal_configured:  # lay quyen dieu khien gimbal (Gimbal Protocol v2)
            self._cmd(CMD_DO_GIMBAL_MANAGER_CONFIGURE, self.SRC_SYSTEM, self.SRC_COMPONENT, -1, -1, 0, 0, 0)
            self._gimbal_configured = True
        cur = (round(pitch_deg, 1), round(yaw_deg, 1))
        if cur == self._last_gimbal:
            return  # chi gui khi goc doi
        self._last_gimbal = cur
        nan = float("nan")
        self._cmd(CMD_DO_GIMBAL_MANAGER_PITCHYAW, pitch_deg, yaw_deg, nan, nan, 0, 0, 0)

    # ---- noi bo ----
    def _cmd(self, command: int, *params: float) -> None:
        if self._m is None:
            return
        p = list(params) + [0.0] * (7 - len(params))
        self._m.mav.command_long_send(self._m.target_system, self._m.target_component, command, 0, *p)

    def _send_mode(self, mode: FlightMode) -> None:
        self._cmd(self._mav.MAV_CMD_DO_SET_MODE, 1.0, float(PX4_MAIN_MODE[mode]), 0.0)

    def _on_message(self, msg) -> None:
        t, k = self._telem, msg.get_type()
        if k == "HEARTBEAT" and msg.get_srcSystem() == self._m.target_system and msg.autopilot != self._mav.MAV_AUTOPILOT_INVALID:
            t.armed = bool(msg.base_mode & self._mav.MAV_MODE_FLAG_SAFETY_ARMED)
            t.mode_name = decode_main_mode(msg.custom_mode)
            t.connected = True
        elif k == "ATTITUDE":
            t.yaw_rad = msg.yaw
        elif k == "GLOBAL_POSITION_INT":
            t.rel_alt_m = msg.relative_alt / 1000.0
            t.ground_speed_ms = math.hypot(msg.vx, msg.vy) / 100.0
        elif k == "SYS_STATUS":
            t.battery_pct = float(msg.battery_remaining)
        elif k == "GPS_RAW_INT":
            t.gps_fix, t.gps_sats = msg.fix_type, msg.satellites_visible
