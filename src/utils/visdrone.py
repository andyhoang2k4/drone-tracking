"""Doc annotation VisDrone2019-MOT (dung chung cho prepare_visdrone va evaluate).

Dinh dang (moi dong 1 box, frame danh so tu 1, khop ten anh 0000001.jpg)::

    frame_index,target_id,bbox_left,bbox_top,bbox_width,bbox_height,
    score,object_category,truncation,occlusion

score = 0 -> box bi bo qua khi danh gia. Tren VisDrone2019-MOT val/test-dev,
moi dong score=0 la category 0 (ignored region) hoac 11 (others); moi dong
category 1..10 deu co score=1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Union


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


def read_annotation_file(path: Union[str, Path]) -> list[VisDroneBox]:
    """Doc ca file annotation cua mot sequence."""
    boxes: list[VisDroneBox] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        try:
            box = parse_annotation_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if box is not None:
            boxes.append(box)
    return boxes


def is_ignore_region(box: VisDroneBox, category_map: Mapping[int, int]) -> bool:
    """Box la VUNG BI BO QUA khi danh gia (khong phai ground truth hop le).

    Quy tac: score == 0, HOAC category == 0 (ignored region), HOAC category
    khong co trong category_map (vd 11 "others" - model khong hoc class nay).
    """
    return box.score == 0 or box.category == 0 or box.category not in category_map


def frame_index_from_path(path: Union[str, Path]) -> int:
    """So frame VisDrone tu ten anh: .../0000007.jpg -> 7."""
    stem = Path(path).stem
    if not stem.isdigit():
        raise ValueError(f"Ten anh khong phai so frame VisDrone: {path}")
    return int(stem)
