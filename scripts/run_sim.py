"""Chay mo phong bay co tay dieu khien ao + man hinh camera co OSD.

  python scripts/run_sim.py --demo                     # chay thu khong can PX4/Gazebo
  python scripts/run_sim.py --weights W.pt --tracker-cfg configs/botsort.yaml --log experiments/sim_runs/run01.jsonl   # PX4 SITL + Gazebo
  python scripts/run_sim.py --video clip.mp4 --weights W.pt ...                 # anh tu file video
  python scripts/run_sim.py --oracle --gz-topic /camera --oracle-topic /camera/bboxes   # box tu boundingbox_camera Gazebo
                                                                                 # (khong can weights; khong dung de danh gia)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.app import SimApp  # noqa: E402
from src.sim.config import load_config  # noqa: E402
from src.sim.virtual_radio import VirtualRadio  # noqa: E402
from src.track import DEFAULT_TRACKER_CONFIG  # noqa: E402
from src.utils.inference_settings import DEFAULT_TRAIN_CONFIG, ImgszChoice, resolve_imgsz  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sim.yaml"))
    ap.add_argument("--demo", action="store_true", help="the gioi gia lap nhe, khong can PX4/Gazebo")
    ap.add_argument("--video", type=Path, help="dung file video thay cho camera Gazebo")
    ap.add_argument("--gz-topic", help="ghi de camera.gz_topic")
    ap.add_argument("--mavlink-url", help="ghi de link.url, vi du udpin:0.0.0.0:14540")
    ap.add_argument("--weights", help="trong so YOLO-seg (che do khong --demo, khong --oracle)")
    ap.add_argument("--oracle", action="store_true",
                    help="lay detection tu boundingbox_camera cua Gazebo (src/sim/oracle.py, muc oracle trong "
                         "configs/sim.yaml) thay cho YOLO + BoT-SORT; khong dung de danh gia")
    ap.add_argument("--oracle-topic", help="ghi de oracle.bbox_topic")
    ap.add_argument("--tracker-cfg", type=Path, default=DEFAULT_TRACKER_CONFIG)
    ap.add_argument("--log", type=Path, default=None, help="ghi su kien (khoa/mat/doi mode) ra file .jsonl")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="bo trong = theo experiment (config.yaml / run_context.json canh weights), "
                         "roi configs/train_config.yaml")
    ap.add_argument("--experiment-dir", type=Path, default=None,
                    help="thu muc experiment co config.yaml (detector.imgsz)")
    ap.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    args = ap.parse_args(argv)
    if args.demo and args.oracle:
        ap.error("--demo va --oracle khong dung cung nhau")
    if args.oracle and args.video:
        ap.error("--oracle lay box tu Gazebo, khong dung voi --video")
    if not args.demo and not args.oracle and not args.weights:
        ap.error("can --weights (hoac dung --demo / --oracle)")
    return args


def resolve_detector_imgsz(args: argparse.Namespace) -> Optional[ImgszChoice]:
    """imgsz cho YOLO theo resolve_imgsz; None o che do --demo/--oracle (khong dung YOLO)."""
    if args.demo or args.oracle:
        return None
    return resolve_imgsz(args.imgsz, experiment_dir=args.experiment_dir,
                         weights=args.weights, train_config=args.train_config)


def detector_kind(args: argparse.Namespace) -> str:
    """Nguon detection dung trong phien sim (ghi vao log de khong nham ket qua oracle voi YOLO)."""
    return "demo" if args.demo else "gz_oracle" if args.oracle else "yolo_botsort"


def session_record(args: argparse.Namespace, choice: Optional[ImgszChoice]) -> dict[str, Any]:
    """Ban ghi dau tien cua log sim: tham so detector da dung (gom imgsz)."""
    record: dict[str, Any] = {
        "event": "session_start",
        "demo": bool(args.demo),
        "detector": detector_kind(args),
        "video": str(args.video) if args.video else None,
        "weights": args.weights,
        "tracker_cfg": str(args.tracker_cfg),
        "conf": args.conf,
    }
    if choice is None:
        why = "oracle: khong dung YOLO" if args.oracle else "demo: khong dung YOLO"
        record.update(imgsz=None, imgsz_source=why, imgsz_warnings=[])
    else:
        record.update(choice.as_dict())
    return record


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    choice = resolve_detector_imgsz(args)
    if choice is not None:
        print(f"imgsz = {choice.imgsz} (nguon: {choice.source})")
        for warning in choice.warnings:
            print(f"CANH BAO: {warning}")

    cfg = load_config(args.config if args.config.exists() else None)
    if args.gz_topic:
        cfg.camera.gz_topic = args.gz_topic
    if args.mavlink_url:
        cfg.link.url = args.mavlink_url
    if args.oracle_topic:
        cfg.oracle.bbox_topic = args.oracle_topic

    if args.demo:
        from src.sim.demo import DemoCamera, DemoDetector, DemoLink, DemoWorld

        world = DemoWorld(cfg)
        camera, detector, link = DemoCamera(world), DemoDetector(world), DemoLink(world)
    else:
        from src.sim.adapters import UltralyticsBotSortSource
        from src.sim.link import PymavlinkLink
        from src.sim.sources import GzCameraSource, VideoFileSource

        camera = VideoFileSource(args.video) if args.video else GzCameraSource(cfg.camera.gz_topic)
        if args.oracle:
            from src.sim.oracle import GzOracleDetectionSource

            detector = GzOracleDetectionSource(cfg.oracle.bbox_topic, cfg.oracle.label_classes,
                                               min_box_px=cfg.oracle.min_box_px, max_age_s=cfg.oracle.max_age_s)
        else:
            detector = UltralyticsBotSortSource(args.weights, str(args.tracker_cfg), args.conf,
                                                imgsz=choice.imgsz, device=args.device)
        link = PymavlinkLink(cfg.link, gimbal_enabled=cfg.gimbal.enabled)
    link.connect()
    app = SimApp(cfg, camera, detector, link, VirtualRadio(cfg.radio))
    if args.log:
        from src.utils.logging_utils import JsonlLogger

        with JsonlLogger(args.log) as logger:
            logger.write(session_record(args, choice))
            app.set_logger(logger)
            app.run()
    else:
        app.run()


if __name__ == "__main__":
    main()
