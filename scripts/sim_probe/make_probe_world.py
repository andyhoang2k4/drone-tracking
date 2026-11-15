"""Sinh world SDF cho probe domain-gap (CLAUDE.md muc 2, quyet dinh 2026-11-15).

World gom mot cum vat the (nguoi + xe tu Fuel) va mot luoi "rig" camera tinh,
moi rig = 1 camera RGB + 1 boundingbox_camera cung pose (ground truth 2D).
Moi vat the co label rieng (plugin Label), file JSON di kem anh xa label -> loai.

Chi la cong cu probe: khong dung de danh gia motion memory, khong dung cho benchmark.

Model tham chieu bang `model://<ten>` (person_walking, hatchback, suv, pickup), nen khi chay Gazebo phai dat
GZ_SIM_RESOURCE_PATH toi thu muc chua cac model do. Tren cloud, thu muc nay la ~/sim_probe/models (ban tai zip tu
Fuel, moi model mot thu muc <ten>/model.config); cache cua `gz fuel download` co them thu muc phien ban
(<ten>/<so>/) nen khong dung truc tiep duoc.

Vi du:
    python scripts/sim_probe/make_probe_world.py --out ~/sim_probe/runs/r1/world.sdf \
        --heights 10 20 30 50 80 --pitches 30 45 60 90
    GZ_SIM_RESOURCE_PATH=~/sim_probe/models gz sim -s -r --headless-rendering ~/sim_probe/runs/r1/world.sdf
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

# loai vat the -> (model uri, nhom). Nhom (person/vehicle) ghi vao layout JSON de so khop voi lop VisDrone
# (pedestrian/people <-> person, car/van/truck/bus <-> vehicle); script danh gia nam ngoai repo
MODEL_KINDS = {
    "person": ("model://person_walking", "person"),
    "hatchback": ("model://hatchback", "vehicle"),
    "suv": ("model://suv", "vehicle"),
    "pickup": ("model://pickup", "vehicle"),
}


def build_layout(n_person: int, n_each_vehicle: int, spread: float, min_gap: float, seed: int) -> list[dict]:
    """Dat vat the ngau nhien (co seed) trong o vuong canh `spread` m quanh goc toa do, khong chong len nhau."""
    rng = random.Random(seed)
    kinds = ["person"] * n_person + [k for k in ("hatchback", "suv", "pickup") for _ in range(n_each_vehicle)]
    placed: list[dict] = []
    for i, kind in enumerate(kinds):
        gap = min_gap if kind == "person" else 2 * min_gap
        for _ in range(1000):
            x, y = rng.uniform(-spread / 2, spread / 2), rng.uniform(-spread / 2, spread / 2)
            if all(math.hypot(x - p["x"], y - p["y"]) >= max(gap, p["gap"]) for p in placed):
                break
        else:
            raise RuntimeError(f"khong dat duoc vat the {kind}; tang --spread hoac giam so luong")
        placed.append({"label": i + 1, "kind": kind, "group": MODEL_KINDS[kind][1],
                       "x": round(x, 2), "y": round(y, 2), "yaw": round(rng.uniform(-math.pi, math.pi), 3),
                       "gap": gap})
    return placed


def camera_rigs(heights: list[float], pitches_deg: list[float]) -> list[dict]:
    """Moi (do cao, goc cui) mot rig nhin vao tam cum; doi phuong vi giua cac rig cho da dang goc nhin."""
    rigs = []
    n = len(heights) * len(pitches_deg)
    k = 0
    for h in heights:
        for p in pitches_deg:
            az = 2 * math.pi * k / n
            pr = math.radians(p)
            d = 0.0 if p >= 89.9 else h / math.tan(pr)
            rigs.append({"name": f"h{int(h)}_p{int(p)}", "height_m": h, "pitch_deg": p,
                         "ground_dist_m": round(d, 2), "azimuth_rad": round(az, 3),
                         "x": round(-d * math.cos(az), 3), "y": round(-d * math.sin(az), 3), "z": h,
                         "roll": 0.0, "pitch": round(pr, 5), "yaw": round(az, 5)})
            k += 1
    return rigs


def sensor_xml(kind: str, name: str, topic: str, save_dir: str, a: argparse.Namespace) -> str:
    box = "<box_type>2d</box_type>" if kind == "boundingbox_camera" else ""
    return f"""
      <sensor name="{name}" type="{kind}"><always_on>1</always_on><update_rate>{a.rate}</update_rate><topic>{topic}</topic>
        <camera>{box}<horizontal_fov>{a.hfov}</horizontal_fov>
          <image><width>{a.width}</width><height>{a.height}</height></image>
          <clip><near>0.1</near><far>{a.far}</far></clip>
          <save enabled="true"><path>{save_dir}</path></save></camera></sensor>"""


def world_xml(layout: list[dict], rigs: list[dict], a: argparse.Namespace) -> str:
    objs = []
    for o in layout:
        uri = MODEL_KINDS[o["kind"]][0]
        objs.append(f"""
    <include><name>{o['kind']}_{o['label']}</name><uri>{uri}</uri><static>true</static>
      <pose>{o['x']} {o['y']} 0 0 0 {o['yaw']}</pose>
      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{o['label']}</label></plugin></include>""")
    cams = []
    for r in rigs:
        n = r["name"]
        cams.append(f"""
    <model name="rig_{n}"><static>true</static><pose>{r['x']} {r['y']} {r['z']} {r['roll']} {r['pitch']} {r['yaw']}</pose><link name="link">"""
                    + sensor_xml("camera", "rgb", f"probe/{n}/rgb", f"frames/{n}/rgb", a)
                    + sensor_xml("boundingbox_camera", "bbox", f"probe/{n}/bbox", f"frames/{n}/bbox", a)
                    + "\n    </link></model>")
    g = a.ground_rgb
    return f"""<?xml version="1.0" ?>
<!-- Sinh boi scripts/sim_probe/make_probe_world.py - khong sua tay -->
<sdf version="1.9">
  <world name="probe">
    <physics name="p" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <scene><ambient>{a.ambient} {a.ambient} {a.ambient} 1</ambient><background>0.6 0.75 0.9 1</background></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.3 0.2 -0.9</direction></light>
    <model name="ground"><static>true</static><link name="l"><visual name="v">
      <geometry><plane><normal>0 0 1</normal><size>{a.ground_size} {a.ground_size}</size></plane></geometry>
      <material><ambient>{g} {g} {g} 1</ambient><diffuse>{g} {g} {g} 1</diffuse></material></visual></link></model>
{''.join(objs)}
{''.join(cams)}
  </world>
</sdf>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="file .sdf dau ra (layout JSON ghi canh do)")
    ap.add_argument("--heights", type=float, nargs="+", default=[10, 20, 30, 50, 80])
    ap.add_argument("--pitches", type=float, nargs="+", default=[30, 45, 60, 90], help="goc cui (do), 90 = nhin thang xuong")
    ap.add_argument("--n-person", type=int, default=6)
    ap.add_argument("--n-each-vehicle", type=int, default=2)
    ap.add_argument("--spread", type=float, default=30.0, help="canh o vuong dat vat the (m)")
    ap.add_argument("--min-gap", type=float, default=2.5, help="khoang cach toi thieu giua nguoi (m); xe gap doi")
    ap.add_argument("--seed", type=int, default=20261115)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=756)
    ap.add_argument("--hfov", type=float, default=1.4, help="rad")
    ap.add_argument("--rate", type=float, default=1.0, help="Hz moi sensor")
    ap.add_argument("--far", type=float, default=600.0)
    ap.add_argument("--ambient", type=float, default=0.6)
    ap.add_argument("--ground-rgb", type=float, default=0.45, help="do sang mat dat xam (0-1)")
    ap.add_argument("--ground-size", type=float, default=1000.0)
    a = ap.parse_args()

    layout = build_layout(a.n_person, a.n_each_vehicle, a.spread, a.min_gap, a.seed)
    rigs = camera_rigs(a.heights, a.pitches)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(world_xml(layout, rigs, a), encoding="utf-8")
    meta = {"generator": "scripts/sim_probe/make_probe_world.py", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "objects": [{k: v for k, v in o.items() if k != "gap"} for o in layout], "rigs": rigs}
    meta_path = a.out.with_suffix(".layout.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"world: {a.out}\nlayout: {meta_path}\nso vat the: {len(layout)}, so rig: {len(rigs)}")


if __name__ == "__main__":
    main()
