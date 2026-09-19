"""Chay pipeline day du: YOLO-seg -> BoT-SORT -> (Motion Memory) -> log.

Entry point cho ablation o CLAUDE.md muc 12: cung detector + tracker, chay
hai lan voi va khong voi `--motion-memory` roi so sanh.

Vi du::

    # Baseline: YOLO11 + BoT-SORT
    python scripts/run_pipeline.py --weights runs/best.pt \
        --source data/occlusion_test_scenarios/scn01.mp4 \
        --experiment-id yolo11_botsort_001 --target-track-id 1

    # Ablation: + Motion Memory
    python scripts/run_pipeline.py ... --motion-memory

LUU Y: `--motion-memory` hien se loi NotImplementedError ngay khi muc tieu
bi mat, vi phan ngoai suy chuyen dong chua duoc viet (roadmap buoc 5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.motion_memory import MotionMemory, MotionMemoryConfig, TrackState  # noqa: E402
from src.track import (  # noqa: E402
    DEFAULT_TRACKER_CONFIG,
    iter_tracks,
    track_record_to_observation,
)
from src.utils.logging_utils import JsonlLogger, write_json  # noqa: E402
from src.utils.run_context import build_run_context  # noqa: E402


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Duong dan file .pt")
    parser.add_argument("--source", required=True, help="Video/thu muc anh/camera")
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Ten thu muc con trong experiments/ (CLAUDE.md muc 14)",
    )
    parser.add_argument(
        "--target-track-id",
        type=int,
        required=True,
        help="Track ID cua muc tieu can bam theo",
    )
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help='vd "0" hoac "cpu"')
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--motion-memory",
        action="store_true",
        help="Bat Motion Memory (mac dinh TAT - de chay nhanh baseline)",
    )
    parser.add_argument("--lost-timeout-frames", type=int, default=15)
    parser.add_argument("--reid-timeout-frames", type=int, default=45)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Thu muc goc chua cac experiment",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.tracker.exists():
        raise FileNotFoundError(f"Khong tim thay tracker config: {args.tracker}")

    exp_dir = Path(args.output_root) / args.experiment_id
    log_path = exp_dir / "logs" / "pipeline.jsonl"

    # Metadata tai hien experiment (CLAUDE.md muc 14)
    write_json(
        exp_dir / "run_context.json",
        build_run_context(
            experiment_id=args.experiment_id,
            config=vars(args),
            dataset=args.source,
            repo_root=REPO_ROOT,
        ),
    )

    memory: Optional[MotionMemory] = None
    if args.motion_memory:
        memory = MotionMemory(
            MotionMemoryConfig(
                lost_timeout_frames=args.lost_timeout_frames,
                reid_timeout_frames=args.reid_timeout_frames,
            )
        )

    frames = 0
    with JsonlLogger(log_path) as log:
        for record in iter_tracks(
            args.weights,
            args.source,
            tracker=args.tracker,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
        ):
            frames += 1
            observation = track_record_to_observation(record, args.target_track_id)

            entry: dict = {
                "frame_idx": record["frame_idx"],
                "detected": observation is not None,
                "tracker_bbox": (
                    list(observation.bbox.as_tuple()) if observation else None
                ),
                # Giu RIENG - khong gop voi motion confidence (muc 6)
                "detector_confidence": (
                    observation.detector_confidence if observation else None
                ),
            }

            if memory is not None:
                prediction = memory.update(observation)
                entry.update(
                    {
                        "state": prediction.state.value,
                        "predicted_bbox": (
                            list(prediction.bbox.as_tuple())
                            if prediction.bbox
                            else None
                        ),
                        "motion_confidence": prediction.confidence,
                        "frames_since_confirmed": prediction.frames_since_confirmed,
                    }
                )
            else:
                entry["state"] = (
                    TrackState.TRACKING.value if observation else TrackState.LOST.value
                )

            log.write(entry)

    print(f"Da chay {frames} frame. Log: {log_path}")
    print(f"Metadata: {exp_dir / 'run_context.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
