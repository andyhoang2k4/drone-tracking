"""Chay YOLO-seg detection tren video/anh, log ket qua ra file JSONL.

Buoc dau cua pipeline (CLAUDE.md muc 5). Chi detect, KHONG track.
Dung `track.py` khi can track ID.

Vi du::

    python -m src.detect --weights runs/best.pt --source data/raw/clip01.mp4 \
        --output experiments/yolo11_baseline_001/logs/detect.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator, Optional

from src.utils.logging_utils import JsonlLogger


def load_model(weights: str):
    """Nap model Ultralytics.

    Import ben trong ham de module nay van import duoc tren may chua cai
    ultralytics (vd may local Windows chi dung de sua code).
    """
    from ultralytics import YOLO

    return YOLO(weights)


def iter_detections(
    weights: str,
    source: str,
    *,
    conf: float = 0.25,
    imgsz: int = 640,
    device: Optional[str] = None,
    classes: Optional[list[int]] = None,
) -> Iterator[dict[str, Any]]:
    """Duyet tung frame, yield ban ghi detection cua frame do.

    Yield dict dang::

        {"frame_idx": 0, "detections": [{"bbox": [...], "conf": ..., "cls": ...}]}

    Luu y: bbox o dinh dang xyxy pixel. Mask cua model seg KHONG duoc dua vao
    ban ghi nay - motion memory chi lam viec tren bbox (CLAUDE.md muc 6).
    """
    model = load_model(weights)
    stream = model.predict(
        source=source,
        conf=conf,
        imgsz=imgsz,
        device=device,
        classes=classes,
        stream=True,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                detections.append(
                    {
                        "bbox": [float(v) for v in box.xyxy[0].tolist()],
                        "conf": float(box.conf[0]),
                        "cls": int(box.cls[0]),
                    }
                )
        yield {"frame_idx": frame_idx, "detections": detections}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Duong dan file .pt")
    parser.add_argument("--source", required=True, help="Video/thu muc anh/camera")
    parser.add_argument("--output", type=Path, required=True, help="File .jsonl")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help='vd "0" hoac "cpu"')
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Loc theo class id; bo trong = lay tat ca",
    )
    parser.add_argument("--echo", action="store_true", help="In ra console")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    with JsonlLogger(args.output, echo=args.echo) as log:
        for record in iter_detections(
            args.weights,
            args.source,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
        ):
            log.write(record)
    print(f"Da ghi detection log: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
