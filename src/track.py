"""Chay YOLO-seg + BoT-SORT, log track ID/bbox theo tung frame.

Buoc 2 cua pipeline (CLAUDE.md muc 5). Tracker duoc chi dinh qua file config
rieng cua project (`configs/botsort.yaml`) - khong sua config goc trong
package ultralytics.

Vi du::

    python -m src.track --weights runs/best.pt --source data/raw/clip01.mp4 \
        --tracker configs/botsort.yaml \
        --output experiments/yolo11_baseline_001/logs/track.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator, Optional

from src.motion_memory import BBox, Observation
from src.utils.inference_settings import (
    DEFAULT_TRAIN_CONFIG,
    resolve_conf,
    resolve_imgsz,
    write_inference_meta,
)
from src.utils.logging_utils import JsonlLogger

DEFAULT_TRACKER_CONFIG = Path("configs/botsort.yaml")


def iter_tracks(
    weights: str,
    source: str,
    *,
    tracker: Path = DEFAULT_TRACKER_CONFIG,
    conf: float,
    imgsz: int,
    device: Optional[str] = None,
    classes: Optional[list[int]] = None,
    persist: bool = True,
) -> Iterator[dict[str, Any]]:
    """Duyet tung frame, yield ban ghi track cua frame do.

    Yield dict dang::

        {
            "frame_idx": 0,
            "tracks": [
                {"track_id": 1, "bbox": [x1, y1, x2, y2], "conf": 0.9, "cls": 0}
            ],
        }

    Track nao chua duoc tracker gan ID (track_id None) van duoc giu lai de
    buoc sau tu quyet dinh co dung hay khong.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    stream = model.track(
        source=source,
        tracker=str(tracker),
        conf=conf,
        imgsz=imgsz,
        device=device,
        classes=classes,
        persist=persist,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        tracks: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                track_id = box.id
                tracks.append(
                    {
                        "track_id": int(track_id[0]) if track_id is not None else None,
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        "conf": float(box.conf[0]),
                        "cls": int(box.cls[0]),
                    }
                )
        yield {"frame_idx": frame_idx, "tracks": tracks}


def track_record_to_observation(
    record: dict[str, Any],
    target_track_id: int,
) -> Optional[Observation]:
    """Lay track cua muc tieu trong mot frame, doi sang Observation.

    Tra ve None neu frame do khong co track ID mong muon - chinh la truong hop
    motion memory can xu ly (CLAUDE.md muc 6).
    """
    for track in record["tracks"]:
        if track["track_id"] == target_track_id:
            x1, y1, x2, y2 = track["bbox"]
            return Observation(
                bbox=BBox(x1, y1, x2, y2),
                frame_idx=record["frame_idx"],
                track_id=target_track_id,
                detector_confidence=track["conf"],
            )
    return None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Duong dan file .pt")
    parser.add_argument("--source", required=True, help="Video/thu muc anh/camera")
    parser.add_argument("--output", type=Path, required=True, help="File .jsonl")
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
        help="Bo trong = lay theo experiment (config.yaml / run_context.json canh "
        "weights), roi configs/train_config.yaml",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Thu muc experiment co config.yaml (detector.imgsz) - bac uu tien 2",
    )
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--device", default=None, help='vd "0" hoac "cpu"')
    parser.add_argument("--classes", type=int, nargs="*", default=None)
    parser.add_argument("--echo", action="store_true", help="In ra console")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.tracker.exists():
        raise FileNotFoundError(f"Khong tim thay tracker config: {args.tracker}")

    choice = resolve_imgsz(
        args.imgsz,
        experiment_dir=args.experiment_dir,
        weights=args.weights,
        train_config=args.train_config,
    )
    print(f"imgsz = {choice.imgsz} (nguon: {choice.source})")
    for warning in choice.warnings:
        print(f"CANH BAO: {warning}")
    conf_choice = resolve_conf(args.conf, train_config=args.train_config)
    args.conf = conf_choice.conf
    print(f"conf = {conf_choice.conf} (nguon: {conf_choice.source})")
    if conf_choice.conf != conf_choice.config_conf:
        print(f"CANH BAO: conf khac inference.conf={conf_choice.config_conf} trong config")
    meta_path = write_inference_meta(args.output, vars(args), choice, conf_choice)

    with JsonlLogger(args.output, echo=args.echo) as log:
        for record in iter_tracks(
            args.weights,
            args.source,
            tracker=args.tracker,
            conf=args.conf,
            imgsz=choice.imgsz,
            device=args.device,
            classes=args.classes,
        ):
            log.write(record)
    print(f"Da ghi track log: {args.output}")
    print(f"Tham so inference: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
