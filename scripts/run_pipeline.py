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

Tham so Motion Memory: CLI (--lost-timeout-frames, --reid-timeout-frames) >
--motion-memory-config (mac dinh configs/motion_memory.yaml) > mac dinh
MotionMemoryConfig; gia tri da resolve ghi vao run_context.json.

Khi KHONG bat --motion-memory, log giu nhu truoc (baseline), chi them truong
`path` (anh nguon cua frame, de kiem tra thu tu anh). BoT-SORT
duoc goi giong het o hai che do - Motion Memory chi la post-processing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.motion_memory import (  # noqa: E402
    BBox,
    MotionMemory,
    MotionPrediction,
    Observation,
    TrackState,
)
from src.track import (  # noqa: E402
    DEFAULT_TRACKER_CONFIG,
    iter_tracks,
    track_record_to_observation,
)
from src.utils.inference_settings import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG,
    resolve_conf,
    resolve_imgsz,
)
from src.utils.logging_utils import JsonlLogger, write_json  # noqa: E402
from src.utils.motion_memory_settings import (  # noqa: E402
    DEFAULT_MOTION_MEMORY_CONFIG,
    resolve_motion_memory_config,
)
from src.utils.gt_target import GtTargetSelector, load_target_boxes  # noqa: E402
from src.utils.run_context import build_run_context  # noqa: E402

DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval_visdrone.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Duong dan file .pt")
    parser.add_argument("--source", required=True, help="Video/thu muc anh/camera")
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Ten thu muc con trong experiments/ (CLAUDE.md muc 14)",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-track-id",
        type=int,
        help="Track ID cua muc tieu can bam theo",
    )
    target.add_argument(
        "--gt-target-id",
        type=int,
        help="target_id trong ground truth VisDrone; track ID duoc chon tu dong bang "
        "IoU (src/utils/gt_target.py). Can --gt-annotation",
    )
    parser.add_argument(
        "--gt-annotation",
        type=Path,
        default=None,
        help="File annotation VisDrone-MOT (.txt) chua --gt-target-id",
    )
    parser.add_argument(
        "--gt-first-frame",
        type=int,
        default=1,
        help="So frame VisDrone ung voi frame_idx 0 cua log (anh 0000001.jpg -> 1)",
    )
    parser.add_argument(
        "--gt-match-iou",
        type=float,
        default=None,
        help="Bo trong = gt_target_selection.match_iou trong --eval-config",
    )
    parser.add_argument(
        "--gt-max-search-frames",
        type=int,
        default=None,
        help="Bo trong = gt_target_selection.max_search_frames trong --eval-config",
    )
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Bo trong = inference.conf trong configs/train_config.yaml (dung chung)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Bo trong = detector.imgsz trong experiments/<id>/config.yaml, roi "
        "run_context.json canh weights, roi configs/train_config.yaml",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--device", default=None, help='vd "0" hoac "cpu"')
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument(
        "--motion-memory",
        action="store_true",
        help="Bat Motion Memory (mac dinh TAT - de chay nhanh baseline)",
    )
    parser.add_argument(
        "--motion-memory-config",
        type=Path,
        default=DEFAULT_MOTION_MEMORY_CONFIG,
        help="File tham so Motion Memory (khoa motion_memory)",
    )
    parser.add_argument(
        "--lost-timeout-frames",
        type=int,
        default=None,
        help="Bo trong = lay tu --motion-memory-config",
    )
    parser.add_argument(
        "--reid-timeout-frames",
        type=int,
        default=None,
        help="Bo trong = lay tu --motion-memory-config",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="Thu muc goc chua cac experiment",
    )
    args = parser.parse_args(argv)
    if args.gt_target_id is not None and args.gt_annotation is None:
        parser.error("--gt-target-id can --gt-annotation")
    return args


def build_target_selector(args: argparse.Namespace) -> Optional[GtTargetSelector]:
    """GtTargetSelector khi dung --gt-target-id; None khi dung --target-track-id."""
    if args.gt_target_id is None:
        return None
    import yaml

    eval_config = yaml.safe_load(Path(args.eval_config).read_text(encoding="utf-8")) or {}
    selection = eval_config.get("gt_target_selection") or {}
    dataset_config = Path(eval_config["dataset_config"])
    if not dataset_config.is_absolute():
        dataset_config = REPO_ROOT / dataset_config
    category_map = {
        int(k): int(v)
        for k, v in (
            yaml.safe_load(dataset_config.read_text(encoding="utf-8")).get("category_map") or {}
        ).items()
    }
    match_iou = args.gt_match_iou if args.gt_match_iou is not None else selection.get("match_iou")
    max_search = (
        args.gt_max_search_frames
        if args.gt_max_search_frames is not None
        else selection.get("max_search_frames")
    )
    if match_iou is None or max_search is None:
        raise ValueError(f"{args.eval_config}: thieu gt_target_selection.match_iou/max_search_frames")
    return GtTargetSelector(
        gt_boxes=load_target_boxes(args.gt_annotation, args.gt_target_id, category_map),
        match_iou=float(match_iou),
        max_search_frames=int(max_search),
        first_frame=args.gt_first_frame,
        gt_target_id=args.gt_target_id,
    )


def record_to_candidates(record: dict[str, Any]) -> list[Observation]:
    """Doi cac track BoT-SORT xuat ra trong mot frame sang Observation.

    Chi lay track da co track_id (track chua duoc gan ID bi bo). Giu class
    (cls) cho gating cung class va detector conf RIENG (khong gop, muc 6).
    """
    candidates: list[Observation] = []
    for track in record["tracks"]:
        if track.get("track_id") is None:
            continue
        x1, y1, x2, y2 = track["bbox"]
        cls = track.get("cls")
        candidates.append(
            Observation(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                frame_idx=record["frame_idx"],
                track_id=int(track["track_id"]),
                detector_confidence=track.get("conf"),
                cls=int(cls) if cls is not None else None,
            )
        )
    return candidates


class MotionMemoryRunner:
    """Noi output BoT-SORT cua tung frame vao MotionMemory (post-processing).

    - Chua goi memory cho den frame dau tien thay `initial_target_id`; truoc
      do log state = null.
    - Sau do muc tieu = MotionPrediction.track_id (doi khi khop lai bang ID moi).
    - Truyen frame_idx that va moi track khac lam ung vien.
    """

    def __init__(self, memory: MotionMemory, initial_target_id: Optional[int]) -> None:
        self.memory = memory
        self.initial_target_id = initial_target_id
        self.started = False

    @property
    def target_id(self) -> Optional[int]:
        if not self.started:
            return self.initial_target_id
        return self.memory.track_id

    def step(self, record: dict[str, Any]) -> dict[str, Any]:
        candidates = record_to_candidates(record)
        target_id = self.target_id
        observation = next((c for c in candidates if c.track_id == target_id), None)

        if not self.started and observation is None:
            return self._entry(record, candidates, None)

        self.started = True
        others = [c for c in candidates if c is not observation]
        prediction = self.memory.update(observation, others, frame_idx=record["frame_idx"])
        return self._entry(record, candidates, prediction)

    @staticmethod
    def _entry(
        record: dict[str, Any],
        candidates: list[Observation],
        prediction: Optional[MotionPrediction],
    ) -> dict[str, Any]:
        # detected/tracker_bbox: track BoT-SORT cua ID muc tieu HIEN TAI (sau update).
        target_id = prediction.track_id if prediction is not None else None
        target = next((c for c in candidates if c.track_id == target_id), None)
        entry: dict[str, Any] = {
            "frame_idx": record["frame_idx"],
            # Anh nguon cua frame (kiem tra thu tu anh khi danh gia kich ban).
            "path": record.get("path"),
            "detected": target is not None,
            "tracker_bbox": list(target.bbox.as_tuple()) if target else None,
            # Giu RIENG - khong gop voi motion confidence (muc 6)
            "detector_confidence": target.detector_confidence if target else None,
        }
        if prediction is None:  # chua thay muc tieu lan nao
            entry.update(
                {
                    "state": None,
                    "predicted_bbox": None,
                    "motion_confidence": None,
                    "frames_since_confirmed": None,
                    "track_id": None,
                    "recover_reason": None,
                    "gate_candidates_passed": None,
                }
            )
            return entry
        entry.update(
            {
                "state": prediction.state.value,
                "predicted_bbox": (
                    list(prediction.bbox.as_tuple()) if prediction.bbox else None
                ),
                "motion_confidence": prediction.confidence,
                "frames_since_confirmed": prediction.frames_since_confirmed,
                "track_id": prediction.track_id,
                "recover_reason": prediction.recover_reason,
                "gate_candidates_passed": prediction.gate_candidates_passed,
            }
        )
        return entry


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not args.tracker.exists():
        raise FileNotFoundError(f"Khong tim thay tracker config: {args.tracker}")

    exp_dir = Path(args.output_root) / args.experiment_id
    log_path = exp_dir / "logs" / "pipeline.jsonl"

    # imgsz: CLI > experiment > train_config (src/utils/inference_settings.py).
    # Ghi vao run_context de so sanh co/khong Motion Memory cung imgsz.
    choice = resolve_imgsz(
        args.imgsz,
        experiment_dir=exp_dir,
        weights=args.weights,
        train_config=args.train_config,
    )
    args.imgsz = choice.imgsz
    print(f"imgsz = {choice.imgsz} (nguon: {choice.source})")
    for warning in choice.warnings:
        print(f"CANH BAO: {warning}")
    conf_choice = resolve_conf(args.conf, train_config=args.train_config)
    args.conf = conf_choice.conf
    print(f"conf = {conf_choice.conf} (nguon: {conf_choice.source})")
    if conf_choice.conf != conf_choice.config_conf:
        print(f"CANH BAO: conf khac inference.conf={conf_choice.config_conf} trong config")

    selector = build_target_selector(args)

    runner: Optional[MotionMemoryRunner] = None
    mm_context: Optional[dict] = None
    if args.motion_memory:
        mm_choice = resolve_motion_memory_config(
            {
                "lost_timeout_frames": args.lost_timeout_frames,
                "reid_timeout_frames": args.reid_timeout_frames,
            },
            args.motion_memory_config,
        )
        mm_context = mm_choice.as_dict()
        runner = MotionMemoryRunner(MotionMemory(mm_choice.config), args.target_track_id)

    # Metadata tai hien experiment (CLAUDE.md muc 14)
    write_json(
        exp_dir / "run_context.json",
        build_run_context(
            experiment_id=args.experiment_id,
            config=vars(args),
            dataset=args.source,
            repo_root=REPO_ROOT,
            extra={
                "inference": {**choice.as_dict(), **conf_choice.as_dict()},
                # Gia tri da resolve (CLI > file > dataclass); None khi tat.
                "motion_memory": mm_context,
            },
        ),
    )

    frames = 0
    with JsonlLogger(log_path) as log:
        for record in iter_tracks(
            args.weights,
            args.source,
            tracker=args.tracker,
            conf=args.conf,
            imgsz=choice.imgsz,
            device=args.device,
            classes=args.classes,
        ):
            frames += 1
            target_id = (
                args.target_track_id if selector is None else selector.offer(record)
            )
            if runner is not None:
                if not runner.started:
                    runner.initial_target_id = target_id
                log.write(runner.step(record))
                continue

            # Baseline (khong Motion Memory): giu nguyen hanh vi cu.
            observation = (
                track_record_to_observation(record, target_id)
                if target_id is not None
                else None
            )

            entry: dict = {
                "frame_idx": record["frame_idx"],
                # Anh nguon cua frame (kiem tra thu tu anh khi danh gia kich ban).
                "path": record.get("path"),
                "detected": observation is not None,
                "tracker_bbox": (
                    list(observation.bbox.as_tuple()) if observation else None
                ),
                # Giu RIENG - khong gop voi motion confidence (muc 6)
                "detector_confidence": (
                    observation.detector_confidence if observation else None
                ),
            }

            entry["state"] = (
                TrackState.TRACKING.value if observation else TrackState.LOST.value
            )

            log.write(entry)

    # Cach chon muc tieu (doc boi danh gia kich ban occlusion).
    if selector is not None:
        selector.finish()
        selection = selector.as_dict()
        print(f"Chon muc tieu theo GT: {selection['status']}, track {selection['selected_track_id']}")
    else:
        selection = {"mode": "cli", "target_track_id": args.target_track_id}
    write_json(exp_dir / "target_selection.json", selection)

    print(f"Da chay {frames} frame. Log: {log_path}")
    print(f"Metadata: {exp_dir / 'run_context.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
