"""Sinh world "follow_v2" cho kiem tra vong kin bam muc tieu (CLAUDE.md muc 2, lop mo phong).

World = world default cua PX4 + duong cho xe + vat can (nha, cay, cot den) + muc tieu nguoi di bo (actor co
hoat anh, label 1) + 3 nguoi gay nham cung skin + 3 xe chay (VelocityControl, xem drive_vehicles.py) + gio.
Drone x500_oracle: x500 cua PX4 (ban co enable_wind, sinh tu model PX4 da cai) + camera RGB va
boundingbox_camera cung pose (box GT 2D de doi chieu, xem src/sim/oracle.py).

Chi la cong cu kiem tra hanh vi vong kin: KHONG dung de danh gia motion memory hay benchmark (CLAUDE.md muc 12).
Output (ngoai repo): <out>/follow_v2.sdf, <out>/models/..., <out>/scenario.json.

Vi du:
    python scripts/sim_probe/make_follow_world.py --out ~/sim_probe/follow_v2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

FUEL = "https://fuel.gazebosim.org/1.0/OpenRobotics/models/"
WALK = "https://fuel.gazebosim.org/1.0/Mingfei/models/actor/tip/files/meshes/walk.dae"
ACTOR_Z = 1.0  # skin walk.dae co goc o hong (theo vi du actor.sdf cua Gazebo)

# (label, ten, toc do m/s, waypoint (x, y) khep kin, cac diem dung {index: giay})
WALKERS = [
    (1, "target", 1.3, [(22, -6), (22, 10), (22, 26), (40, 26), (40, -6), (22, -6)], {2: 3.0}),
    (2, "distractor_a", 1.2, [(25, 32), (25, -8), (25, 32)], {}),
    (3, "distractor_b", 1.4, [(8, 8), (36, 8), (8, 8)], {}),
    (4, "distractor_c", 1.0, [(26, 21), (30, 21), (30, 29), (26, 29), (26, 21)], {}),
]
# (label, model Fuel, x, y, yaw, toc do m/s) - xe chay theo truc y tren duong x ~ 30.5 / 33.5
VEHICLES = [
    (5, "Hatchback", 30.5, -40.0, math.pi / 2, 7.0),
    (6, "SUV", 33.5, 40.0, -math.pi / 2, 6.0),
    (7, "Pickup", 30.5, 0.0, math.pi / 2, 5.0),
]
# vat can tinh (model Fuel, x, y, yaw)
OCCLUDERS = [
    ("House 1", 15.5, 16.0, 0.0),
    ("Oak tree", 18.5, 3.0, 0.0),
    ("Pine Tree", 33.0, 17.0, 0.0),
    ("Lamp Post", 28.5, 10.0, 0.0),
    ("Lamp Post", 28.5, -10.0, 0.0),
    ("House 2", 46.0, 12.0, 1.57),
]
ROAD_X, ROAD_W, ROAD_LEN = 32.0, 7.0, 120.0
Y_LIMIT = 45.0  # drive_vehicles.py dua xe ve dau duong khi |y| vuot nguong


def actor_xml(label, name, speed, pts, stops):
    wps, t = [], 0.0
    for i, (x, y) in enumerate(pts):
        if i > 0:
            px, py = pts[i - 1]
            t += math.hypot(x - px, y - py) / speed
        nx, ny = pts[i + 1] if i + 1 < len(pts) else pts[1]
        yaw = math.atan2(ny - y, nx - x)
        prev_yaw = math.atan2(y - pts[i - 1][1], x - pts[i - 1][0]) if i > 0 else yaw
        if i > 0:  # den diem nay voi huong cu
            wps.append((t, x, y, prev_yaw))
        if i in stops:
            t += stops[i]
            wps.append((t, x, y, prev_yaw))
        if i + 1 < len(pts):  # quay sang huong moi trong 0.5 s
            t += 0.5 if i > 0 else 0.0
            wps.append((t, x, y, yaw))
    w = "".join(f"<waypoint><time>{tt:.2f}</time><pose>{x} {y} {ACTOR_Z} 0 0 {yy:.4f}</pose></waypoint>" for tt, x, y, yy in wps)
    return f"""
    <actor name="{name}">
      <skin><filename>{WALK}</filename></skin>
      <animation name="walk"><filename>{WALK}</filename><interpolate_x>true</interpolate_x></animation>
      <script><loop>true</loop><auto_start>true</auto_start>
        <trajectory id="0" type="walk" tension="0.6">{w}</trajectory></script>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin>
    </actor>"""


def camera_model(pitch, hfov, w, h, rate):
    def sensor(kind, name, topic):
        box = "<box_type>2d</box_type>" if kind == "boundingbox_camera" else ""
        return (f'<sensor name="{name}" type="{kind}"><pose>0 0 0 0 {pitch:.4f} 0</pose><always_on>1</always_on>'
                f'<update_rate>{rate}</update_rate><topic>{topic}</topic><camera>{box}<horizontal_fov>{hfov}</horizontal_fov>'
                f'<image><width>{w}</width><height>{h}</height></image><clip><near>0.1</near><far>500</far></clip></camera></sensor>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- Sinh boi scripts/sim_probe/make_follow_world.py. Camera RGB + boundingbox_camera cung pose (theo mono_cam cua PX4). -->
<sdf version="1.9"><model name="oracle_cam"><link name="camera_link">
  <inertial><pose>0.03 0.03 0.03 0 0 0</pose><mass>0.050</mass>
    <inertia><ixx>0.00004</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.00004</iyy><iyz>0</iyz><izz>0.00004</izz></inertia></inertial>
  <visual name="housing"><geometry><box><size>0.02 0.04 0.04</size></box></geometry></visual>
  {sensor("camera", "rgb", "/oracle/camera")}
  {sensor("boundingbox_camera", "bbox", "/oracle/bbox")}
</link></model></sdf>
"""


def world_plugins(server_config: Path, wind_scale: float) -> str:
    """Plugin cap world lay tu server.config cua PX4 (Gazebo chi dung server.config khi world khong khai bao plugin
    nao). WindEffects duoc dat he so luc + gio giat: mac dinh he so 1 lam x500 nghieng ~23 do o gio 3.9 m/s."""
    root = ET.parse(server_config).getroot()
    out = []
    for pl in root.iter("plugin"):
        if pl.get("entity_type") != "world":
            continue
        name, fn = pl.get("name"), pl.get("filename")
        if name == "gz::sim::systems::WindEffects":
            body = (f"<force_approximation_scaling_factor>{wind_scale}</force_approximation_scaling_factor>"
                    "<horizontal><magnitude><time_for_rise>10</time_for_rise>"
                    "<sin><amplitude_percent>0.2</amplitude_percent><period>15</period></sin>"
                    '<noise type="gaussian"><mean>0</mean><stddev>0.05</stddev></noise></magnitude>'
                    "<direction><time_for_rise>30</time_for_rise><sin><amplitude>10</amplitude><period>25</period></sin>"
                    '<noise type="gaussian"><mean>0</mean><stddev>0.03</stddev></noise></direction></horizontal>'
                    '<vertical><noise type="gaussian"><mean>0</mean><stddev>0.03</stddev></noise></vertical>')
        else:
            body = "".join(ET.tostring(c, encoding="unicode") for c in pl)
        out.append(f'\n    <plugin filename="{fn}" name="{name}">{body}</plugin>')
    return "".join(out)


def write_model(root: Path, name: str, sdf: str) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.sdf").write_text(sdf, encoding="utf-8")
    (d / "model.config").write_text(
        f'<?xml version="1.0"?>\n<model><name>{name}</name><version>1.0</version>'
        f'<sdf version="1.9">model.sdf</sdf><description>sinh boi make_follow_world.py</description></model>\n',
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--px4-dir", type=Path, default=Path(os.environ.get("PX4_DIR", Path.home() / "PX4-Autopilot")))
    ap.add_argument("--wind", type=float, nargs=3, default=[3.0, 2.5, 0.0], metavar=("VX", "VY", "VZ"),
                    help="van toc gio (m/s, ENU); 0 0 0 = khong gio")
    ap.add_argument("--wind-force-scale", type=float, default=0.2,
                    help="force_approximation_scaling_factor cua WindEffects (chon de goc nghieng hop ly, khong phai so do)")
    ap.add_argument("--cam-pitch-deg", type=float, default=35.0)
    ap.add_argument("--hfov", type=float, default=1.5708)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=756)
    ap.add_argument("--cam-rate", type=float, default=15.0)
    a = ap.parse_args()

    gz_models = a.px4_dir / "Tools/simulation/gz/models"
    base = (gz_models / "x500_base/model.sdf").read_text(encoding="utf-8")
    x500 = (gz_models / "x500/model.sdf").read_text(encoding="utf-8")
    world = (a.px4_dir / "Tools/simulation/gz/worlds/default.sdf").read_text(encoding="utf-8-sig")

    models = a.out / "models"
    # x500_base_wind: x500_base + enable_wind tren base_link (de WindEffects tac dong); x500_wind dung no.
    marker = '<link name="base_link">'
    assert base.count(marker) == 1, "khong tim thay base_link trong x500_base"
    base_w = base.replace("<model name='x500_base'>", "<model name='x500_base_wind'>").replace(
        "<model name=\"x500_base\">", "<model name=\"x500_base_wind\">")
    base_w = base_w.replace(marker, marker + "\n      <enable_wind>true</enable_wind>", 1)
    write_model(models, "x500_base_wind", base_w)
    assert "<uri>model://x500_base</uri>" in x500, "cau truc x500 cua PX4 khac du kien"
    write_model(models, "x500_wind", x500.replace("<uri>model://x500_base</uri>", "<uri>model://x500_base_wind</uri>")
                .replace("<model name='x500'>", "<model name='x500_wind'>"))
    write_model(models, "oracle_cam", camera_model(math.radians(a.cam_pitch_deg), a.hfov, a.width, a.height, a.cam_rate))
    write_model(models, "x500_oracle", """<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9"><model name="x500_oracle">
  <include merge="true"><uri>model://x500_wind</uri></include>
  <include merge="true"><uri>model://oracle_cam</uri><pose>.12 .03 .242 0 0 0</pose></include>
  <joint name="CameraJoint" type="fixed"><parent>base_link</parent><child>camera_link</child>
    <pose relative_to="base_link">.12 .03 .242 0 0 0</pose></joint>
</model></sdf>
""")

    parts = []
    parts.append(f"""
    <model name="road"><static>true</static><pose>{ROAD_X} 0 0.01 0 0 0</pose><link name="l"><visual name="v">
      <geometry><box><size>{ROAD_W} {ROAD_LEN} 0.02</size></box></geometry>
      <material><ambient>0.25 0.25 0.27 1</ambient><diffuse>0.25 0.25 0.27 1</diffuse></material></visual></link></model>""")
    for i, (m, x, y, yaw) in enumerate(OCCLUDERS):
        parts.append(f'\n    <include><name>occ{i}_{m.replace(" ", "_").lower()}</name><uri>{FUEL}{m.replace(" ", "%20")}</uri>'
                     f'<static>true</static><pose>{x} {y} 0 0 0 {yaw}</pose></include>')
    for label, name, speed, pts, stops in WALKERS:
        parts.append(actor_xml(label, name, speed, pts, stops))
    for label, m, x, y, yaw, _ in VEHICLES:
        parts.append(f"""
    <include><name>car{label}</name><uri>{FUEL}{m}</uri><static>false</static><pose>{x} {y} 0.05 0 0 {yaw}</pose>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{label}</label></plugin></include>""")
    parts.append('\n    <include><name>x500_oracle_0</name><uri>model://x500_oracle</uri><pose>0 0 0.25 0 0 0</pose></include>')
    vx, vy, vz = a.wind
    parts.append(f"\n    <wind><linear_velocity>{vx} {vy} {vz}</linear_velocity></wind>\n")

    assert world.count('<world name="default">') == 1
    plugins = world_plugins(a.px4_dir / "src/modules/simulation/gz_bridge/server.config", a.wind_force_scale)
    world = world.replace('<world name="default">', '<world name="follow_v2">' + plugins, 1).replace("  </world>", "".join(parts) + "  </world>", 1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "follow_v2.sdf").write_text(world, encoding="utf-8")

    scenario = {
        "world": "follow_v2", "drone_model": "x500_oracle_0", "target_label": 1,
        "label_classes": {**{w[0]: "pedestrian" for w in WALKERS}, **{v[0]: "car" for v in VEHICLES}},
        "walkers": [{"label": w[0], "name": w[1], "speed_ms": w[2], "waypoints": w[3], "stops": w[4]} for w in WALKERS],
        "vehicles": [{"label": v[0], "model": v[1], "start": [v[2], v[3]], "yaw": v[4], "speed_ms": v[5]} for v in VEHICLES],
        "y_limit": Y_LIMIT, "wind_ms": a.wind, "wind_force_scale": a.wind_force_scale,
        "camera": {"topic": "/oracle/camera", "bbox_topic": "/oracle/bbox", "pitch_deg": a.cam_pitch_deg,
                   "hfov_rad": a.hfov, "width": a.width, "height": a.height, "rate_hz": a.cam_rate},
    }
    (a.out / "scenario.json").write_text(json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"world: {a.out / 'follow_v2.sdf'}\nscenario: {a.out / 'scenario.json'}")


if __name__ == "__main__":
    main()
