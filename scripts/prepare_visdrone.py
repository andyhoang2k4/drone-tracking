"""Chuyen VisDrone2019-MOT sang dinh dang Ultralytics (detection labels).

Giai doan A cua dataset (xem configs/dataset_visdrone.yaml): 3 class, split
theo sequence dung split chinh thuc cua VisDrone (CLAUDE.md muc 7).

Script nay chi tao nhan BOUNDING BOX. Mask cho YOLO-seg duoc sinh o buoc sau
bang SAM (`ultralytics.data.converter.yolo_bbox2segment`) - chay tren cloud.

Dinh dang annotation VisDrone-MOT (moi dong 1 box)::

    frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
    score,object_category,truncation,occlusion

    score = 0 -> box bi bo qua khi danh gia (ignored), script cung bo qua.

Output::

    <output_dir>/
        images/{train,val,test}/<seq>_<frame:07d>.jpg   (hardlink/copy/symlink)
        labels/{train,val,test}/<seq>_<frame:07d>.txt   (cls cx cy w h, chuan hoa)
        manifest.json                                   (thong ke + config da dung)
    data/dataset.yaml                                   (yaml cho Ultralytics)

Vi du::

    # Smoke test: moi split 1 sequence
    python scripts/prepare_visdrone.py --max-sequences 1

    # Day du
    python scripts/prepare_visdrone.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.logging_utils import write_json  # noqa: E402
from src.utils.run_context import git_info  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class VisDroneBox:
    frame_idx: int
    target_id: int
    left: float
    top: float
    width: float
    height: float
    score: int
    category: int
    truncation: int
    occlusion: int


# ---------------- ham thuan (co unit test) ----------------


def parse_annotation_line(line: str) -> Optional[VisDroneBox]:
    """Doc mot dong annotation. Tra ve None voi dong trong."""
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
    if len(parts) < 10:
        raise ValueError(f"Dong annotation thieu cot (can 10): {line!r}")
    values = [int(float(p)) for p in parts[:10]]
    return VisDroneBox(*values)


def to_yolo_bbox(
    box: VisDroneBox, img_w: int, img_h: int, min_size: float = 0.0
) -> Optional[tuple[float, float, float, float]]:
    """Doi box pixel (left, top, w, h) sang (cx, cy, w, h) chuan hoa [0, 1].

    Cat box theo bien anh; tra ve None neu box con lai nho hon `min_size`.
    """
    x1 = max(0.0, float(box.left))
    y1 = max(0.0, float(box.top))
    x2 = min(float(img_w), float(box.left + box.width))
    y2 = min(float(img_h), float(box.top + box.height))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or w < min_size or h < min_size:
        return None
    return ((x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h, w / img_w, h / img_h)


def build_frame_labels(
    boxes: Iterable[VisDroneBox],
    category_map: Mapping[int, int],
    img_w: int,
    img_h: int,
    *,
    min_size: float = 0.0,
    stats: Optional[Counter] = None,
) -> dict[int, list[str]]:
    """Gom box theo frame, tra ve {frame_idx: [dong label YOLO]}."""
    stats = stats if stats is not None else Counter()
    labels: dict[int, list[str]] = {}
    for box in boxes:
        if box.score == 0:
            stats["skipped_ignored"] += 1
            continue
        cls = category_map.get(box.category)
        if cls is None:
            stats["skipped_unmapped_category"] += 1
            continue
        yolo = to_yolo_bbox(box, img_w, img_h, min_size)
        if yolo is None:
            stats["skipped_too_small"] += 1
            continue
        stats[f"class_{cls}"] += 1
        labels.setdefault(box.frame_idx, []).append(
            f"{cls} " + " ".join(f"{v:.6f}" for v in yolo)
        )
    return labels


def keep_frame(frame_idx: int, stride: int) -> bool:
    """Frame VisDrone danh so tu 1; giu frame 1, 1+stride, 1+2*stride..."""
    return (frame_idx - 1) % stride == 0


# ---------------- I/O ----------------


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Khong doc duoc anh: {path}")
    h, w = img.shape[:2]
    return w, h


def _place_image(src: Path, dst: Path, mode: str) -> str:
    """Dat anh vao output. hardlink loi (khac o dia...) -> fallback copy."""
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlink"
    shutil.copy2(src, dst)
    return "copy"


def process_sequence(
    seq_dir: Path,
    annotation_path: Path,
    *,
    split: str,
    output_dir: Path,
    category_map: Mapping[int, int],
    stride: int,
    min_size: float,
    link_mode: str,
) -> dict:
    frames = sorted(
        p for p in seq_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise RuntimeError(f"Sequence khong co anh: {seq_dir}")
    img_w, img_h = _image_size(frames[0])

    boxes = []
    with annotation_path.open(encoding="utf-8") as f:
        for line in f:
            box = parse_annotation_line(line)
            if box is not None:
                boxes.append(box)

    stats: Counter = Counter()
    labels = build_frame_labels(
        boxes, category_map, img_w, img_h, min_size=min_size, stats=stats
    )

    img_out = output_dir / "images" / split
    lbl_out = output_dir / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    kept = 0
    placed: Counter = Counter()
    for frame_path in frames:
        frame_idx = int(frame_path.stem)
        if not keep_frame(frame_idx, stride):
            continue
        name = f"{seq_dir.name}_{frame_idx:07d}"
        placed[_place_image(frame_path, img_out / f"{name}{frame_path.suffix}", link_mode)] += 1
        # Frame khong co box van ghi file rong (anh nen).
        (lbl_out / f"{name}.txt").write_text(
            "\n".join(labels.get(frame_idx, [])) + ("\n" if labels.get(frame_idx) else ""),
            encoding="utf-8",
        )
        kept += 1

    return {
        "sequence": seq_dir.name,
        "image_size": [img_w, img_h],
        "frames_total": len(frames),
        "frames_kept": kept,
        "image_placement": dict(placed),
        "box_stats": dict(stats),
    }


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} da co du lieu. Dung --overwrite de tao lai."
            )
        # Chi xoa thu muc do chinh script nay tao ra (co manifest).
        if not (output_dir / MANIFEST_NAME).exists():
            raise FileExistsError(
                f"{output_dir} khong co {MANIFEST_NAME} - khong phai output cua "
                "script nay, tu choi xoa."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(
    path: Path, output_dir: Path, classes: Mapping[int, str], overwrite: bool
) -> None:
    import yaml

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} da ton tai. Dung --overwrite de ghi de.")
    content = {
        # Duong dan tuyet doi: file nay bi gitignore, sinh lai tren moi may.
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in classes.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# SINH TU DONG boi scripts/prepare_visdrone.py - khong sua tay.\n"
    path.write_text(
        header + yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


# ---------------- CLI ----------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "dataset_visdrone.yaml"
    )
    parser.add_argument("--source-root", type=Path, default=None, help="Ghi de source_root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Ghi de output_dir")
    parser.add_argument(
        "--dataset-yaml", type=Path, default=REPO_ROOT / "data" / "dataset.yaml"
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="Cach dat anh vao output (hardlink: khong ton them dung luong)",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Chi xu ly N sequence dau moi split (smoke test)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)

    source_root = _resolve(args.source_root or Path(cfg["source_root"]))
    output_dir = _resolve(args.output_dir or Path(cfg["output_dir"]))
    category_map = {int(k): int(v) for k, v in cfg["category_map"].items()}
    classes = {int(k): str(v) for k, v in cfg["classes"].items()}
    unknown = set(category_map.values()) - set(classes)
    if unknown:
        raise ValueError(f"category_map tro toi class id khong co trong classes: {unknown}")

    # Kiem tra du input truoc khi xoa/ghi bat ky thu gi.
    split_dirs: dict[str, Path] = {}
    for split, folder in cfg["splits"].items():
        split_dir = source_root / folder
        for sub in ("sequences", "annotations"):
            if not (split_dir / sub).is_dir():
                raise FileNotFoundError(f"Khong thay {split_dir / sub}")
        split_dirs[split] = split_dir

    _prepare_output_dir(output_dir, args.overwrite)

    manifest: dict = {
        "dataset": {"name": cfg["name"], "version": cfg["version"]},
        "config_file": str(args.config),
        "config": cfg,
        "source_root": str(source_root),
        "git": git_info(REPO_ROOT),
        "max_sequences": args.max_sequences,
        "splits": {},
    }

    for split, split_dir in split_dirs.items():
        stride = int(cfg["frame_stride"][split])
        seq_dirs = sorted(p for p in (split_dir / "sequences").iterdir() if p.is_dir())
        if args.max_sequences is not None:
            seq_dirs = seq_dirs[: args.max_sequences]

        seq_reports = []
        for seq_dir in seq_dirs:
            ann = split_dir / "annotations" / f"{seq_dir.name}.txt"
            if not ann.exists():
                raise FileNotFoundError(f"Thieu annotation cho {seq_dir.name}: {ann}")
            report = process_sequence(
                seq_dir,
                ann,
                split=split,
                output_dir=output_dir,
                category_map=category_map,
                stride=stride,
                min_size=float(cfg.get("min_box_size", 0)),
                link_mode=args.link_mode,
            )
            seq_reports.append(report)
            print(
                f"[{split}] {seq_dir.name}: {report['frames_kept']}/"
                f"{report['frames_total']} frame, {report['box_stats']}"
            )

        totals: Counter = Counter()
        for r in seq_reports:
            totals.update(r["box_stats"])
        manifest["splits"][split] = {
            "source": str(split_dir),
            "frame_stride": stride,
            "num_sequences": len(seq_reports),
            "frames_kept": sum(r["frames_kept"] for r in seq_reports),
            "box_stats": dict(totals),
            "sequences": seq_reports,
        }

    write_json(output_dir / MANIFEST_NAME, manifest)
    write_dataset_yaml(args.dataset_yaml, output_dir, classes, args.overwrite)

    print("\nTong ket:")
    for split, info in manifest["splits"].items():
        print(
            f"  {split}: {info['num_sequences']} sequence, "
            f"{info['frames_kept']} frame, {info['box_stats']}"
        )
    print(f"Manifest: {output_dir / MANIFEST_NAME}")
    print(f"Dataset yaml: {args.dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
