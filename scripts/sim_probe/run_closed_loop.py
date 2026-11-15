"""Chay vong kin headless tren world follow_v2: SimApp + PymavlinkLink (PX4 SITL) + camera Gazebo.

Nguon detection: --detector oracle (box GT cua boundingbox_camera) hoac yolo (YOLO-seg + BoT-SORT, --weights).
Bo uoc luong: --estimator heuristic (TargetManager) hoac motion_memory (src/sim/mm_target.py).
Oracle LUON chay song song chi de ghi log doi chieu (track dang khoa co nam tren muc tieu that khong) va de chon
track ban dau; khi --detector yolo no KHONG tham gia dieu khien.

Can truoc: scripts/sim_probe/sim_up.sh va drone dang bay (commander takeoff) - xem run_scenario.sh.
Radio gia: cong tac Auto tu dau, khoa muc tieu truoc tick dau -> supervisor vao OFFBOARD ngay.

KHONG phai danh gia Motion Memory hay benchmark (CLAUDE.md muc 2, 12): mot world, mot lan chay, kich ban tu dung.
So lieu danh gia van do tren video co ground truth qua src/evaluate.py.

    python scripts/sim_probe/run_closed_loop.py --scenario ~/sim_probe/follow_v2/scenario.json \\
        --detector yolo --weights W.pt --estimator motion_memory --out ~/sim_probe/follow_v2/runs/r1
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402

from src.sim.app import SimApp  # noqa: E402
from src.sim.config import load_config  # noqa: E402
from src.sim.follow_controller import ZERO_VELOCITY  # noqa: E402
from src.sim.link import PymavlinkLink  # noqa: E402
from src.sim.oracle import GzOracleDetectionSource  # noqa: E402
from src.sim.sources import GzCameraSource  # noqa: E402
from src.sim.types import FlightMode, RCInput  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


class ScriptedRadio:
    def __init__(self) -> None:
        self.rc = RCInput(throttle=0.5, arm_switch=False, mode_switch=FlightMode.AUTO_FOLLOW)

    def update(self) -> RCInput:
        return self.rc


class LoggingLink(PymavlinkLink):
    """Giu them ATTITUDE roll/pitch va STATUSTEXT de log (khong doi hanh vi link)."""

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        self.roll = self.pitch = 0.0
        self.texts: list[str] = []

    def _on_message(self, msg) -> None:
        k = msg.get_type()
        if k == "ATTITUDE":
            self.roll, self.pitch = msg.roll, msg.pitch
        elif k == "STATUSTEXT":
            self.texts.append(msg.text)
        super()._on_message(msg)


class Poses:
    def __init__(self, node, world: str, names: set[str]) -> None:
        from gz.msgs10.pose_v_pb2 import Pose_V

        self._lock, self._names, self.p = threading.Lock(), names, {}
        node.subscribe(Pose_V, f"/world/{world}/pose/info", self._cb)

    def _cb(self, msg) -> None:
        d = {q.name: (q.position.x, q.position.y, q.position.z) for q in msg.pose if q.name in self._names}
        with self._lock:
            self.p.update(d)

    def get(self, name: str):
        with self._lock:
            return self.p.get(name)


def summarize(recs: list[dict], target: int) -> dict:
    """Tom tat tu log tick (so dem that, khong uoc luong)."""
    n = len(recs)
    vis = [r for r in recs if r["gt_target_visible"]]
    on = [r for r in vis if r["on_target"]]
    wrong = [r for r in recs if r["on_other"] is not None]
    states: dict[str, int] = {}
    for r in recs:
        states[str(r["state"])] = states.get(str(r["state"]), 0) + 1
    ids = [r["locked"] for r in recs if r["locked"] is not None]
    id_changes = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
    # cac doan muc tieu khong nhin thay (bi che / ra khoi khung) >= 0.5 s
    episodes, start = [], None
    for i, r in enumerate(recs):
        if not r["gt_target_visible"] and start is None:
            start = i
        elif r["gt_target_visible"] and start is not None:
            episodes.append((start, i))
            start = None
    if start is not None:
        episodes.append((start, n))
    occl = []
    for s_i, e_i in episodes:
        dur = (recs[e_i - 1]["t"] - recs[s_i]["t"]) if e_i > s_i else 0.0
        if dur < 0.5:
            continue
        item = {"t_start": recs[s_i]["t"], "duration_s": round(dur, 2), "reappeared": e_i < n}
        if e_i < n:
            t_back = recs[e_i]["t"]
            hit = next((r for r in recs[e_i:] if r["on_target"]), None)
            item["reacquired_on_target"] = hit is not None and hit["t"] - t_back <= 5.0
            item["time_to_reacquire_s"] = round(hit["t"] - t_back, 2) if hit else None
            item["locked_other_after"] = any(r["on_other"] is not None for r in recs[e_i:e_i + 100])
        item["states_during"] = sorted({str(r["state"]) for r in recs[s_i:e_i]})
        occl.append(item)
    cx = [abs(r["cx_err_px"]) for r in on if r["cx_err_px"] is not None]
    tilt = [math.degrees(math.acos(max(-1.0, min(1.0, math.cos(r["roll"]) * math.cos(r["pitch"]))))) for r in recs]
    first_failed = next((r["t"] for r in recs if r["state"] == "FAILED"), None)
    return {
        "ticks": n, "duration_s": recs[-1]["t"] if recs else 0, "loop_hz": round(n / recs[-1]["t"], 1) if recs else 0,
        "modes": sorted({r["mode"] for r in recs}), "target_label": target,
        "target_visible_ticks": len(vis), "on_target_ticks": len(on),
        "on_target_ratio_when_visible": round(len(on) / len(vis), 3) if vis else None,
        "locked_on_other_object_ticks": len(wrong),
        "locked_on_other_labels": sorted({r["on_other"] for r in wrong}),
        "state_ticks": states, "first_failed_t": first_failed, "locked_id_changes": id_changes,
        "occlusions": occl,
        "abs_cx_err_px_on_target": {"median": st.median(cx), "max": max(cx)} if cx else None,
        "tilt_deg": {"median": round(st.median(tilt), 1), "max": round(max(tilt), 1)} if tilt else None,
        "recover_reasons": {k: sum(1 for r in recs if r.get("recover_reason") == k)
                            for k in sorted({r.get("recover_reason") for r in recs if r.get("recover_reason")})},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=REPO / "configs/sim.yaml")
    ap.add_argument("--detector", choices=["oracle", "yolo"], default="oracle")
    ap.add_argument("--weights")
    ap.add_argument("--imgsz", type=int, default=960, help="mac dinh theo configs/train_config.yaml")
    ap.add_argument("--conf", type=float, default=0.1, help="conf truoc tracker (configs/train_config.yaml)")
    ap.add_argument("--estimator", choices=["heuristic", "motion_memory"], default="heuristic")
    ap.add_argument("--mm-config", type=Path, default=REPO / "configs/motion_memory.yaml",
                    help="config MotionMemory khi --estimator motion_memory")
    ap.add_argument("--desired-height-frac", type=float, default=0.06,
                    help="nguoi 1.7 m o tam anh khi bay 12 m, camera chuc 35 do ~ 45 px / 756")
    ap.add_argument("--duration", type=float, default=150.0)
    ap.add_argument("--osd-every", type=float, default=1.0, help="giay; 0 = khong luu anh OSD")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.detector == "yolo" and not a.weights:
        ap.error("--detector yolo can --weights")

    sc = json.loads(a.scenario.read_text(encoding="utf-8"))
    target = int(sc["target_label"])
    cfg = load_config(a.config)
    cam_sc = sc["camera"]
    cfg.camera.gz_topic, cfg.camera.width, cfg.camera.height = cam_sc["topic"], cam_sc["width"], cam_sc["height"]
    cfg.camera.hfov_deg = math.degrees(cam_sc["hfov_rad"])
    cfg.gimbal.enabled = False  # camera gan cung tren x500_oracle
    cfg.follow.desired_height_frac = a.desired_height_frac
    cfg.loop_hz = 20.0
    cfg.target.estimator = a.estimator
    cfg.target.motion_memory_config = str(a.mm_config)
    cfg.oracle.bbox_topic = cam_sc["bbox_topic"]
    cfg.oracle.label_classes = {int(k): v for k, v in sc["label_classes"].items()}

    a.out.mkdir(parents=True, exist_ok=True)
    osd_dir = a.out / "osd"
    if a.osd_every > 0:
        osd_dir.mkdir(exist_ok=True)

    from gz.transport13 import Node

    node = Node()
    poses = Poses(node, sc["world"], {sc["drone_model"], "target"})
    cam = GzCameraSource(cfg.camera.gz_topic)
    oracle = GzOracleDetectionSource(cfg.oracle.bbox_topic, cfg.oracle.label_classes, max_age_s=1.0)
    if a.detector == "yolo":
        from src.sim.adapters import UltralyticsBotSortSource

        det = UltralyticsBotSortSource(a.weights, str(REPO / "configs/botsort.yaml"), a.conf, imgsz=a.imgsz, device="0")
    else:
        det = oracle
    link = LoggingLink(cfg.link, gimbal_enabled=False)
    link.connect()
    app = SimApp(cfg, cam, det, link, ScriptedRadio())

    drv = subprocess.Popen([sys.executable, str(REPO / "scripts/sim_probe/drive_vehicles.py"), "--scenario",
                            str(a.scenario), "--duration", str(a.duration + 30)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # cho muc tieu hien ra, khoa detection khop box GT cua muc tieu (IoU >= 0.3)
        t0, lock_id = time.monotonic(), None
        while lock_id is None:
            if time.monotonic() - t0 > 60:
                raise SystemExit("khong khoa duoc muc tieu trong 60 s")
            link.tick(time.monotonic())
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            dets = det.detect(frame, time.monotonic())
            gt = next((g.bbox for g in oracle.detect(frame, 0.0) if g.track_id == target), None)
            if gt is not None:
                best = max(dets, key=lambda d: iou(d.bbox, gt), default=None)
                if best is not None and iou(best.bbox, gt) >= 0.3:
                    lock_id = best.track_id
                    app.manager.update(dets, time.monotonic())
                    x1, y1, x2, y2 = best.bbox
                    assert app.manager.select_at((x1 + x2) / 2, (y1 + y2) / 2) == lock_id
                    assert app.manager.lock_selected()
            time.sleep(0.05)

        recs, period, start, last_osd = [], 1.0 / cfg.loop_hz, time.monotonic(), -1e9
        while True:
            t_loop = time.monotonic()
            t = t_loop - start
            if t > a.duration:
                break
            out = app.tick(t_loop)
            m, telem = app.manager, link.telemetry()
            est = m.estimate
            bbox = next((d.bbox for d in m.candidates if d.track_id == m.locked_id), None)
            gts = oracle.detect(None, 0.0)
            gt_t = next((g.bbox for g in gts if g.track_id == target), None)
            on_target = bool(bbox and gt_t and iou(bbox, gt_t) >= 0.3)
            on_other = None
            if bbox and not on_target:
                o = max((g for g in gts if g.track_id != target), key=lambda g: iou(bbox, g.bbox), default=None)
                if o is not None and iou(bbox, o.bbox) >= 0.3:
                    on_other = o.track_id
            dp = poses.get(sc["drone_model"])
            pred = getattr(m, "last_prediction", None)
            rec = {
                "t": round(t, 3), "mode": telem.mode_name, "status": out.status if out else None,
                "state": est.state.value if est else None, "locked": m.locked_id,
                "locked_label": m.locked_label, "n_dets": len(m.candidates),
                "est_bbox": [round(v, 1) for v in est.bbox] if (est and est.bbox) else None,
                "pred_conf": round(est.prediction_confidence, 3) if est else None,
                "recover_reason": getattr(pred, "recover_reason", None),
                "gt_target_visible": gt_t is not None, "on_target": on_target, "on_other": on_other,
                "cx_err_px": round((bbox[0] + bbox[2]) / 2 - cfg.camera.width / 2, 1) if bbox else None,
                "h_frac": round((bbox[3] - bbox[1]) / cfg.camera.height, 4) if bbox else None,
                "v_fwd": round(out.velocity.forward, 2) if out else None,
                "v_yaw_deg": round(math.degrees(out.velocity.yaw_rate), 1) if out else None,
                "alt": round(telem.rel_alt_m, 2), "roll": round(link.roll, 4), "pitch": round(link.pitch, 4),
                "drone_xyz": [round(v, 2) for v in dp] if dp else None,
            }
            if link.texts:
                rec["statustext"], link.texts = link.texts, []
            recs.append(rec)
            if a.osd_every > 0 and t - last_osd >= a.osd_every and app.last_view is not None:
                cv2.imwrite(str(osd_dir / f"osd_{t:06.1f}.jpg"), app.last_view, [cv2.IMWRITE_JPEG_QUALITY, 75])
                last_osd = t
            time.sleep(max(0.0, period - (time.monotonic() - t_loop)))
        for _ in range(10):
            link.tick(time.monotonic())
            link.send_velocity(ZERO_VELOCITY)
            time.sleep(0.05)
    finally:
        drv.terminate()

    with (a.out / "ticks.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = {"detector": a.detector, "weights": a.weights, "estimator": a.estimator,
               "mm_config": str(a.mm_config) if a.estimator == "motion_memory" else None, "imgsz": a.imgsz,
               "conf": a.conf, "scenario": str(a.scenario), "note": "kiem tra vong kin, khong phai danh gia (CLAUDE.md muc 12)",
               **summarize(recs, target)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("detector", "estimator", "loop_hz", "on_target_ratio_when_visible",
                                              "locked_on_other_object_ticks", "locked_id_changes", "first_failed_t")}))


if __name__ == "__main__":
    main()
