"""Sinh nhan segmentation (polygon) tu nhan bbox bang SAM.

Chay SAU scripts/prepare_visdrone.py, tren cloud (SAM tren CPU rat cham).

Vi sao khong dung thang `ultralytics.data.converter.yolo_bbox2segment`:
    - YOLO-seg tinh box train TU MASK (ultralytics data/utils.py), nen mask
      lech lam box train lech. Script nay do IoU(bbox cua mask, box goc) va
      thay mask lech (< fallback_iou) bang hinh chu nhat dung bang box goc.
    - Giu lai nhan box goc o labels_bbox/ de DANH GIA (CLAUDE.md muc 12) -
      danh gia luon so voi box goc, khong so voi box suy ra tu mask.

Bo cuc thu muc (dataset_dir = output_dir trong configs/dataset_visdrone.yaml)::

    images/<split>/*.jpg
    labels_bbox/<split>/*.txt   box goc (lan chay dau: doi ten tu labels/)
    labels/<split>/*.txt        polygon cho YOLO-seg (sinh boi script nay)

Chay lai an toan: luon doc box tu labels_bbox/, ghi de labels/.

Vi du::

    # Smoke test tren CPU
    python scripts/generate_masks.py --splits val --max-images 5 --device cpu

    # Day du tren cloud (baseline: chi train/val, test giu nhan bbox)
    python scripts/generate_masks.py --splits train val --device 0
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import write_json  # noqa: E402

BOX_LABEL_DIR = "labels_bbox"
SEG_LABEL_DIR = "labels"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# segment_fn(xyxy_px (N, 4)) -> N polygon, moi polygon (K, 2) toa do chuan hoa
SegmentFn = Callable[[np.ndarray], Sequence[np.ndarray]]


# ---------------- ham thuan (co unit test) ----------------


def parse_box_line(line: str) -> tuple[int, np.ndarray]:
    """'cls cx cy w h' -> (cls, [cx, cy, w, h]) chuan hoa."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Dong label khong phai bbox (can 5 cot): {line!r}")
    return int(parts[0]), np.array([float(v) for v in parts[1:]], dtype=np.float64)


def xywhn_to_xyxyn(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = box
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU hai box xyxy."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def rect_polygon(xyxy: np.ndarray) -> np.ndarray:
    """Hinh chu nhat 4 dinh tu box xyxy (chuan hoa)."""
    x1, y1, x2, y2 = np.clip(xyxy, 0.0, 1.0)
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def choose_polygon(
    box_xywhn: np.ndarray, polygon: Optional[np.ndarray], fallback_iou: float
) -> tuple[np.ndarray, bool, float]:
    """Chon polygon de ghi: mask cua SAM, hoac hinh chu nhat neu mask lech.

    Returns:
        (polygon, used_fallback, iou). iou = 0 neu polygon khong hop le.
    """
    box_xyxy = xywhn_to_xyxyn(box_xywhn)
    if polygon is None or len(polygon) < 3:
        return rect_polygon(box_xyxy), True, 0.0
    poly_box = np.array(
        [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()]
    )
    iou = box_iou(box_xyxy, poly_box)
    if iou < fallback_iou:
        return rect_polygon(box_xyxy), True, iou
    return np.clip(polygon, 0.0, 1.0), False, iou


def format_polygon_line(cls: int, polygon: np.ndarray) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in polygon.reshape(-1))


def convert_label_lines(
    box_lines: Sequence[str],
    segment_fn: SegmentFn,
    img_w: int,
    img_h: int,
    fallback_iou: float,
    stats: Counter,
    ious: list[float],
) -> list[str]:
    """Doi cac dong bbox cua mot anh thanh cac dong polygon."""
    parsed = [parse_box_line(line) for line in box_lines if line.strip()]
    if not parsed:
        return []
    boxes = np.stack([b for _, b in parsed])
    xyxy_px = np.stack([xywhn_to_xyxyn(b) for b in boxes]) * [img_w, img_h, img_w, img_h]
    polygons = list(segment_fn(xyxy_px))
    if len(polygons) != len(parsed):
        raise RuntimeError(
            f"SAM tra ve {len(polygons)} mask cho {len(parsed)} box - khong khop thu tu"
        )

    out = []
    for (cls, box), poly in zip(parsed, polygons):
        chosen, fallback, iou = choose_polygon(box, poly, fallback_iou)
        stats[f"class_{cls}"] += 1
        stats[f"class_{cls}_fallback"] += int(fallback)
        ious.append(iou)
        out.append(format_polygon_line(cls, chosen))
    return out


def summarize_ious(ious: Sequence[float], fallback_iou: float) -> dict:
    if not ious:
        return {"count": 0}
    arr = np.asarray(ious)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "below_fallback": int((arr < fallback_iou).sum()),
        "below_fallback_ratio": round(float((arr < fallback_iou).mean()), 4),
    }


# ---------------- SAM ----------------


def load_sam(model_name: str, weights_dir: Path, device: Optional[str]) -> SegmentFn:
    """Nap SAM; tai weight vao weights_dir (gitignore) neu chua co."""
    from ultralytics import SAM

    weights_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    os.chdir(weights_dir)
    try:
        model = SAM(model_name)
    finally:
        os.chdir(previous_cwd)

    def segment(image: np.ndarray, xyxy_px: np.ndarray) -> list[np.ndarray]:
        result = model(image, bboxes=xyxy_px, verbose=False, save=False, device=device)[0]
        if result.masks is None:
            return [None] * len(xyxy_px)
        return list(result.masks.xyn)

    return segment


# ---------------- I/O ----------------


def _ensure_box_labels(dataset_dir: Path, split: str) -> Path:
    """Lan dau: doi ten labels/<split> -> labels_bbox/<split> de giu box goc."""
    box_dir = dataset_dir / BOX_LABEL_DIR / split
    seg_dir = dataset_dir / SEG_LABEL_DIR / split
    if not box_dir.exists():
        if not seg_dir.exists():
            raise FileNotFoundError(f"Khong thay nhan box: {seg_dir}")
        box_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seg_dir), str(box_dir))
    return box_dir


def process_split(
    dataset_dir: Path,
    split: str,
    sam: Callable[[np.ndarray, np.ndarray], list],
    fallback_iou: float,
    max_images: Optional[int],
) -> dict:
    import cv2
    from tqdm import tqdm

    box_dir = _ensure_box_labels(dataset_dir, split)
    seg_dir = dataset_dir / SEG_LABEL_DIR / split
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Cache cu cua Ultralytics tro toi nhan box -> xoa de nap lai.
    for cache in (dataset_dir / SEG_LABEL_DIR).glob(f"{split}*.cache"):
        cache.unlink()

    images = sorted(
        p for p in (dataset_dir / "images" / split).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images is not None:
        images = images[:max_images]

    stats: Counter = Counter()
    ious: list[float] = []
    for img_path in tqdm(images, desc=f"SAM {split}"):
        box_lines = (box_dir / f"{img_path.stem}.txt").read_text(encoding="utf-8").splitlines()
        lines: list[str] = []
        if any(line.strip() for line in box_lines):
            image = cv2.imread(str(img_path))
            if image is None:
                raise RuntimeError(f"Khong doc duoc anh: {img_path}")
            h, w = image.shape[:2]
            lines = convert_label_lines(
                box_lines,
                lambda xyxy: sam(image, xyxy),
                w,
                h,
                fallback_iou,
                stats,
                ious,
            )
        (seg_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["images"] += 1

    return {
        "images": stats.pop("images", 0),
        "max_images": max_images,
        "per_class": dict(stats),
        "iou": summarize_ious(ious, fallback_iou),
    }


# ---------------- CLI ----------------


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "dataset_visdrone.yaml"
    )
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Ghi de output_dir")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--sam-model", default=None, help="Ghi de masks.sam_model")
    parser.add_argument("--fallback-iou", type=float, default=None)
    parser.add_argument("--device", default=None, help='vd "0" hoac "cpu"')
    parser.add_argument("--weights-dir", type=Path, default=REPO_ROOT / "weights")
    parser.add_argument(
        "--max-images", type=int, default=None, help="Chi xu ly N anh dau moi split (smoke test)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    mask_cfg = cfg.get("masks", {})
    dataset_dir = args.dataset_dir or Path(cfg["output_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = REPO_ROOT / dataset_dir
    sam_model = args.sam_model or mask_cfg.get("sam_model", "sam_b.pt")
    fallback_iou = (
        args.fallback_iou if args.fallback_iou is not None else float(mask_cfg.get("fallback_iou", 0.5))
    )

    for split in args.splits:
        if not (dataset_dir / "images" / split).is_dir():
            raise FileNotFoundError(f"Khong thay {dataset_dir / 'images' / split}")

    sam = load_sam(sam_model, args.weights_dir, args.device)

    report = {"sam_model": sam_model, "fallback_iou": fallback_iou, "splits": {}}
    for split in args.splits:
        report["splits"][split] = process_split(
            dataset_dir, split, sam, fallback_iou, args.max_images
        )
        print(f"[{split}] {report['splits'][split]['iou']}")

    # Ghi them vao manifest cua prepare_visdrone.py (neu co).
    manifest_path = dataset_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["masks"] = report
    write_json(manifest_path, manifest)
    print(f"Da ghi thong ke vao {manifest_path}")
    if args.max_images is not None:
        print(
            "CANH BAO: --max-images -> anh chua xu ly KHONG co file trong labels/ "
            "(Ultralytics coi la anh nen). Khong train tren dataset nay; chay lai "
            "khong co --max-images de sinh du."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
