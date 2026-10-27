"""Chay mo phong bay co tay dieu khien ao + man hinh camera co OSD.

  python scripts/run_sim.py --demo                     # chay thu khong can PX4/Gazebo
  python scripts/run_sim.py --weights W.pt --tracker-cfg configs/botsort.yaml --log experiments/sim_runs/run01.jsonl   # PX4 SITL + Gazebo
  python scripts/run_sim.py --video clip.mp4 --weights W.pt ...                 # anh tu file video
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sim.app import SimApp  # noqa: E402
from src.sim.config import load_config  # noqa: E402
from src.sim.virtual_radio import VirtualRadio  # noqa: E402
from src.track import DEFAULT_TRACKER_CONFIG  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path("configs/sim.yaml"))
    ap.add_argument("--demo", action="store_true", help="the gioi gia lap nhe, khong can PX4/Gazebo")
    ap.add_argument("--video", type=Path, help="dung file video thay cho camera Gazebo")
    ap.add_argument("--gz-topic", help="ghi de camera.gz_topic")
    ap.add_argument("--mavlink-url", help="ghi de link.url, vi du udpin:0.0.0.0:14540")
    ap.add_argument("--weights", help="trong so YOLO-seg (che do khong --demo)")
    ap.add_argument("--tracker-cfg", type=Path, default=DEFAULT_TRACKER_CONFIG)
    ap.add_argument("--log", type=Path, default=None, help="ghi su kien (khoa/mat/doi mode) ra file .jsonl")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config if args.config.exists() else None)
    if args.gz_topic:
        cfg.camera.gz_topic = args.gz_topic
    if args.mavlink_url:
        cfg.link.url = args.mavlink_url

    if args.demo:
        from src.sim.demo import DemoCamera, DemoDetector, DemoLink, DemoWorld

        world = DemoWorld(cfg)
        camera, detector, link = DemoCamera(world), DemoDetector(world), DemoLink(world)
    else:
        if not args.weights:
            ap.error("can --weights (hoac dung --demo)")
        from src.sim.adapters import UltralyticsBotSortSource
        from src.sim.link import PymavlinkLink
        from src.sim.sources import GzCameraSource, VideoFileSource

        camera = VideoFileSource(args.video) if args.video else GzCameraSource(cfg.camera.gz_topic)
        detector = UltralyticsBotSortSource(args.weights, str(args.tracker_cfg), args.conf, device=args.device)
        link = PymavlinkLink(cfg.link, gimbal_enabled=cfg.gimbal.enabled)
    link.connect()
    app = SimApp(cfg, camera, detector, link, VirtualRadio(cfg.radio))
    if args.log:
        from src.utils.logging_utils import JsonlLogger

        with JsonlLogger(args.log) as logger:
            app.set_logger(logger)
            app.run()
    else:
        app.run()


if __name__ == "__main__":
    main()
