"""Chon muc tieu cho run_pipeline theo ground truth (thay cho --target-track-id).

Dung trong danh gia kich ban occlusion: biet target_id cua GT (VisDrone-MOT)
nhung KHONG biet track ID ma BoT-SORT se cap. Bo chon chay ONLINE, tung frame:

    1. Doc bbox GT cua `gt_target_id` tu file annotation VisDrone (dung lai
       src/utils/visdrone.py). Chi lay dong hop le (khong phai vung bi bo qua).
    2. Frame log `frame_idx` (dem tu 0) <-> frame VisDrone
       `frame_idx + first_frame` (mac dinh first_frame = 1, anh 0000001.jpg).
    3. Tu frame dau tien muc tieu co trong GT, o moi frame co GT: ghep bbox GT
       voi cac track BoT-SORT da co ID bang IoU; chon track IoU cao nhat neu
       IoU >= match_iou (hoa -> track ID nho hon). Khong xet class.
    4. Chi tim trong `max_search_frames` frame tinh tu frame GT dau tien
       (BoT-SORT chi xuat track moi tu frame thu 2 tro di, tru frame 1 cua
       video). Het cua so ma chua ghep duoc -> status "failed": khong co muc
       tieu cho ca lan chay (danh gia se tinh la that bai).
    5. Da chon thi giu nguyen ID do (run_pipeline/MotionMemory lo phan sau).

Chi dung GT o buoc khoi tao (tuong duong nguoi dung bam chon muc tieu), khong
dung GT o cac frame sau.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.motion_memory import BBox
from src.utils.visdrone import is_ignore_region, read_annotation_file

PathLike = Union[str, Path]

STATUS_PENDING = "pending"
STATUS_SELECTED = "selected"
STATUS_FAILED = "failed"


def _iou(a: BBox, b: BBox) -> float:
    iw = min(a.x2, b.x2) - max(a.x1, b.x1)
    ih = min(a.y2, b.y2) - max(a.y1, b.y1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0.0 else 0.0


def load_target_boxes(
    annotation_path: PathLike,
    gt_target_id: int,
    category_map: Mapping[int, int],
) -> dict[int, BBox]:
    """{frame VisDrone: bbox} cua mot target_id. Rong -> ValueError."""
    boxes: dict[int, BBox] = {}
    for row in read_annotation_file(annotation_path):
        if row.target_id != gt_target_id or is_ignore_region(row, category_map):
            continue
        if row.frame_idx in boxes:
            raise ValueError(
                f"{annotation_path}: target {gt_target_id} co 2 box o frame {row.frame_idx}"
            )
        boxes[row.frame_idx] = BBox(
            float(row.left), float(row.top),
            float(row.left + row.width), float(row.top + row.height),
        )
    if not boxes:
        raise ValueError(f"{annotation_path}: khong co box hop le cho target_id {gt_target_id}")
    return boxes


@dataclass
class GtTargetSelector:
    gt_boxes: Mapping[int, BBox]
    match_iou: float
    max_search_frames: int
    first_frame: int = 1
    gt_target_id: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.match_iou <= 1.0:
            raise ValueError("match_iou phai trong (0, 1]")
        if self.max_search_frames < 1:
            raise ValueError("max_search_frames phai >= 1")
        if not self.gt_boxes:
            raise ValueError("gt_boxes rong")
        self.first_gt_frame = min(self.gt_boxes)
        self.status = STATUS_PENDING
        self.selected_track_id: Optional[int] = None
        self.selected_frame_idx: Optional[int] = None
        self.selected_iou: Optional[float] = None
        self.frames_searched = 0

    def offer(self, record: Mapping[str, Any]) -> Optional[int]:
        """Nhan ban ghi track cua mot frame; tra ve track ID muc tieu (hoac None)."""
        if self.status != STATUS_PENDING:
            return self.selected_track_id
        frame = record["frame_idx"] + self.first_frame
        if frame < self.first_gt_frame:
            return None
        if frame >= self.first_gt_frame + self.max_search_frames:
            self.status = STATUS_FAILED
            return None
        gt = self.gt_boxes.get(frame)
        if gt is None:
            return None
        self.frames_searched += 1
        best: Optional[tuple[float, int]] = None
        for track in record["tracks"]:
            track_id = track.get("track_id")
            if track_id is None:
                continue
            overlap = _iou(gt, BBox(*[float(v) for v in track["bbox"]]))
            if overlap < self.match_iou:
                continue
            key = (-overlap, int(track_id))
            if best is None or key < (-best[0], best[1]):
                best = (overlap, int(track_id))
        if best is not None:
            self.status = STATUS_SELECTED
            self.selected_iou, self.selected_track_id = best
            self.selected_frame_idx = int(record["frame_idx"])
        return self.selected_track_id

    def finish(self) -> None:
        """Goi khi het log: con pending (log ngan hon cua so) -> failed."""
        if self.status == STATUS_PENDING:
            self.status = STATUS_FAILED

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "gt",
            "gt_target_id": self.gt_target_id,
            "status": self.status,
            "selected_track_id": self.selected_track_id,
            "selected_frame_idx": self.selected_frame_idx,
            "selected_iou": self.selected_iou,
            "first_gt_frame": self.first_gt_frame,
            "first_frame": self.first_frame,
            "match_iou": self.match_iou,
            "max_search_frames": self.max_search_frames,
            "frames_searched": self.frames_searched,
        }
