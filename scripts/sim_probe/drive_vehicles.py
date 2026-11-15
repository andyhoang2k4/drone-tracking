"""Cho xe trong world follow_v2 chay (VelocityControl): gui cmd_vel mot lan, dua xe ve dau duong khi |y| > y_limit.

Chay rieng mot process (set_pose cham, khong duoc chan vong lap dieu khien). Dung khi bi kill hoac het --duration.
Chi la cong cu kiem tra vong kin, khong dung de danh gia (CLAUDE.md muc 12).

    python scripts/sim_probe/drive_vehicles.py --scenario ~/sim_probe/follow_v2/scenario.json --duration 200
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist
from gz.transport13 import Node


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=300.0)
    a = ap.parse_args()
    sc = json.loads(a.scenario.read_text(encoding="utf-8"))
    world, ylim = sc["world"], sc["y_limit"]
    cars = {f"car{v['label']}": v for v in sc["vehicles"]}
    node = Node()
    lock, poses = threading.Lock(), {}

    def on_pose(msg):
        with lock:
            for p in msg.pose:
                if p.name in cars:
                    poses[p.name] = (p.position.x, p.position.y)

    node.subscribe(Pose_V, f"/world/{world}/pose/info", on_pose)
    pubs = {n: node.advertise(f"/model/{n}/cmd_vel", Twist) for n in cars}

    def drive(name):
        tw = Twist()
        tw.linear.x = cars[name]["speed_ms"]
        pubs[name].publish(tw)

    def reset(name):
        v = cars[name]
        req = Pose()
        req.name = name
        req.position.x, req.position.y, req.position.z = v["start"][0], v["start"][1], 0.05
        req.orientation.w, req.orientation.z = math.cos(v["yaw"] / 2), math.sin(v["yaw"] / 2)
        node.request(f"/world/{world}/set_pose", req, Pose, Boolean, 500)

    time.sleep(1.0)  # cho publisher duoc nhan
    t_end = time.monotonic() + a.duration
    last_drive = 0.0
    while time.monotonic() < t_end:
        now = time.monotonic()
        if now - last_drive > 2.0:  # gui lai dinh ky (an toan neu lan dau bi mat)
            for n in cars:
                drive(n)
            last_drive = now
        with lock:
            snap = dict(poses)
        for n, (x, y) in snap.items():
            if abs(y) > ylim:
                reset(n)
                drive(n)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
