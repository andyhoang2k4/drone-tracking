"""Kiem tra PymavlinkLink bang mot 'autopilot gia' qua UDP loopback (khong phai PX4 that)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
import platform
import socket
import threading
import time

import pytest

pytest.importorskip("pymavlink")
from pymavlink import mavutil  # noqa: E402

from src.sim.config import LinkConfig  # noqa: E402
from src.sim.follow_controller import BodyVelocity  # noqa: E402
from src.sim.link import (PymavlinkLink, VELOCITY_YAWRATE_MASK, body_to_ned, decode_main_mode)  # noqa: E402
from src.sim.types import FlightMode  # noqa: E402


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_helpers():
    assert decode_main_mode((7 << 16) | 0) == "STABILIZED"
    assert decode_main_mode(6 << 16) == "OFFBOARD"
    vn, ve, vd = body_to_ned(BodyVelocity(forward=1.0), math.pi / 2)  # huong dong
    assert abs(vn) < 1e-9 and abs(ve - 1.0) < 1e-9 and vd == 0
    vn, ve, _ = body_to_ned(BodyVelocity(right=1.0), 0.0)  # sang phai khi huong bac = dong
    assert abs(vn) < 1e-9 and abs(ve - 1.0) < 1e-9
    assert not (VELOCITY_YAWRATE_MASK & (0b111 << 3))  # van toc KHONG bi bo qua
    assert not (VELOCITY_YAWRATE_MASK & (1 << 11))  # yaw_rate KHONG bi bo qua
    assert VELOCITY_YAWRATE_MASK & 0b111 and VELOCITY_YAWRATE_MASK & (1 << 10)


def test_roundtrip_against_fake_autopilot():
    port = free_port()
    link = PymavlinkLink(LinkConfig(url=f"udpin:127.0.0.1:{port}", connect_timeout_s=5.0))
    ap = mavutil.mavlink_connection(f"udpout:127.0.0.1:{port}", source_system=1, source_component=1)
    if platform.system() == "Windows":
        # pymavlink udpout tren Windows bind 0.0.0.0 CUNG cong dich -> trung cong voi link udpin,
        # link gui ve chinh no. Thay bang socket cong tu cap.
        ap.port.close()
        ap.port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ap.port.bind(("127.0.0.1", 0))
        ap.port.setblocking(False)
        ap.fd = ap.port.fileno()
    mav = mavutil.mavlink

    def ap_send_state(yaw=0.5):
        custom = (7 << 16)
        ap.mav.heartbeat_send(mav.MAV_TYPE_QUADROTOR, mav.MAV_AUTOPILOT_PX4,
                              mav.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mav.MAV_MODE_FLAG_SAFETY_ARMED, custom, 4)
        ap.mav.attitude_send(0, 0, 0, yaw, 0, 0, 0)
        ap.mav.global_position_int_send(0, 0, 0, 0, 12345, 300, 400, 0, 0)
        ap.mav.sys_status_send(0, 0, 0, 0, 0, 0, 77, 0, 0, 0, 0, 0, 0)
        ap.mav.gps_raw_int_send(0, 3, 0, 0, 0, 0, 0, 0, 0, 9)

    stop = threading.Event()

    def pump():  # autopilot gia gui trang thai lien tuc, giong PX4
        while not stop.is_set():
            ap_send_state()
            time.sleep(0.05)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    link.connect()
    time.sleep(0.2)
    link.tick(time.monotonic())
    t = link.telemetry()
    assert t.connected and t.armed and t.mode_name == "STABILIZED"
    assert abs(t.rel_alt_m - 12.345) < 1e-6 and abs(t.ground_speed_ms - 5.0) < 1e-6
    assert abs(t.yaw_rad - 0.5) < 1e-4 and t.battery_pct == 77 and t.gps_fix == 3 and t.gps_sats == 9

    link.send_manual(0.5, -0.25, 0.3, 1.0)
    link.arm(True)
    link.set_mode(FlightMode.AUTO_FOLLOW)
    link.send_velocity(BodyVelocity(forward=2.0, yaw_rate=0.3))
    link.send_gimbal(-20.0, 10.0)
    time.sleep(0.2)

    got = {}
    while True:
        m = ap.recv_match(blocking=False)
        if m is None:
            break
        got.setdefault(m.get_type(), []).append(m)
    mc = got["MANUAL_CONTROL"][0]
    assert (mc.x, mc.y, mc.z, mc.r) == (500, -250, 300, 1000)
    cmds = {c.command: c for c in got["COMMAND_LONG"]}
    assert cmds[mav.MAV_CMD_COMPONENT_ARM_DISARM].param1 == 1.0
    assert cmds[mav.MAV_CMD_DO_SET_MODE].param2 == 6.0  # OFFBOARD
    assert cmds[1000].param1 == -20.0 and cmds[1000].param2 == 10.0
    sp = got["SET_POSITION_TARGET_LOCAL_NED"][0]
    assert sp.type_mask == VELOCITY_YAWRATE_MASK and sp.coordinate_frame == mav.MAV_FRAME_LOCAL_NED
    assert abs(sp.vx - 2.0 * math.cos(0.5)) < 1e-4 and abs(sp.vy - 2.0 * math.sin(0.5)) < 1e-4
    assert abs(sp.yaw_rate - 0.3) < 1e-6
    stop.set()
    th.join(timeout=1)
    link.close()
