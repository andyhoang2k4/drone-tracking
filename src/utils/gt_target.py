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
from typing import Any, Mapping, Optional, Sequence, Union

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


# ---------------- "khoa duoc truoc khi che" (screening + danh gia v2) ----------------
#
# Dinh nghia DUNG CHUNG cho buoc screening muc tieu (src/occlusion_screening.py)
# va danh gia kich ban occlusion v2 (src/evaluate.py), de hai noi khong lech:
#
#   khoa duoc  <=>  (1) GtTargetSelector (match_iou, max_search_frames) chon
#                       duoc track tren cac frame TRUOC khi che;
#                   (2) track da chon co mat va IoU(bbox track, GT) >= lock_iou
#                       o MOI frame trong lock_confirm_frames frame lien truoc
#                       khi che [a - L, a - 1].
# Khong xet class (nguoi van hanh chon vat the, khong quan tam nhan); class cua
# track da chon chi duoc GHI LAI.

LOCK_LOCKED = "locked"
LOCK_NOT_SELECTED = "not_selected"
LOCK_LOST = "lost_before_occlusion"


@dataclass(frozen=True)
class LockRule:
    match_iou: float
    max_search_frames: int
    lock_confirm_frames: int
    lock_iou: float

    def __post_init__(self) -> None:
        if not 0.0 < self.match_iou <= 1.0 or not 0.0 < self.lock_iou <= 1.0:
            raise ValueError("match_iou/lock_iou phai trong (0, 1]")
        if self.max_search_frames < 1 or self.lock_confirm_frames < 1:
            raise ValueError("max_search_frames/lock_confirm_frames phai >= 1")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LockRule":
        try:
            return cls(
                match_iou=float(data["match_iou"]),
                max_search_frames=int(data["max_search_frames"]),
                lock_confirm_frames=int(data["lock_confirm_frames"]),
                lock_iou=float(data["lock_iou"]),
            )
        except KeyError as exc:
            raise ValueError(f"target_lock thieu khoa {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "match_iou": self.match_iou,
            "max_search_frames": self.max_search_frames,
            "lock_confirm_frames": self.lock_confirm_frames,
            "lock_iou": self.lock_iou,
        }


def _track_box(track: Mapping[str, Any]) -> BBox:
    return BBox(*[float(v) for v in track["bbox"]])


def evaluate_target_lock(
    records: Sequence[Mapping[str, Any]],
    gt_boxes: Mapping[int, BBox],
    rule: LockRule,
    occlusion_start_frame_idx: int,
    *,
    first_frame: int = 1,
    gt_target_id: Optional[int] = None,
) -> dict[str, Any]:
    """Xet muc tieu co "khoa duoc truoc khi che" tren log track (src.track).

    Args:
        records: ban ghi track theo frame_idx 0.. (chi dung frame < a).
        gt_boxes: {frame VisDrone: bbox GT muc tieu} (frame = frame_idx + first_frame).
        occlusion_start_frame_idx: a = frame_idx dau tien bi che.

    Tra ve dict: status (locked | not_selected | lost_before_occlusion), locked,
    selection (GtTargetSelector.as_dict()), selected_class, lock_ious (IoU track
    da chon o [a-L, a-1], None = track khong co mat), best_iou (IoU lon nhat cua
    bat ky track co ID voi GT tren frame truoc che), frames_with_match (so frame
    truoc che co track IoU >= match_iou), frames_with_gt.
    """
    a = int(occlusion_start_frame_idx)
    if a < 1:
        raise ValueError("occlusion_start_frame_idx phai >= 1")
    by_idx = {int(r["frame_idx"]): r for r in records}
    pre = [by_idx[k] for k in range(a) if k in by_idx]
    if len(pre) != a:
        raise ValueError(f"log track thieu frame truoc khi che (co {len(pre)}/{a})")

    selector = GtTargetSelector(
        gt_boxes=gt_boxes,
        match_iou=rule.match_iou,
        max_search_frames=rule.max_search_frames,
        first_frame=first_frame,
        gt_target_id=gt_target_id,
    )
    best_iou = 0.0
    frames_with_match = 0
    frames_with_gt = 0
    selected_class: Optional[int] = None
    for record in pre:
        selector.offer(record)
        gt = gt_boxes.get(int(record["frame_idx"]) + first_frame)
        if gt is None:
            continue
        frames_with_gt += 1
        ious = [_iou(gt, _track_box(t)) for t in record["tracks"] if t.get("track_id") is not None]
        top = max(ious, default=0.0)
        best_iou = max(best_iou, top)
        frames_with_match += int(top >= rule.match_iou)
    selector.finish()
    selection = selector.as_dict()

    lock_ious: list[Optional[float]] = []
    status = LOCK_NOT_SELECTED
    if selector.status == STATUS_SELECTED:
        sid = selector.selected_track_id
        for t in by_idx[selector.selected_frame_idx]["tracks"]:
            if t.get("track_id") == sid:
                cls = t.get("cls")
                selected_class = int(cls) if cls is not None else None
                break
        for k in range(a - rule.lock_confirm_frames, a):
            gt = gt_boxes.get(k + first_frame) if k >= 0 else None
            track = (
                next((t for t in by_idx[k]["tracks"] if t.get("track_id") == sid), None)
                if k >= 0 else None
            )
            lock_ious.append(_iou(gt, _track_box(track)) if gt is not None and track is not None else None)
        ok = all(v is not None and v >= rule.lock_iou for v in lock_ious)
        status = LOCK_LOCKED if ok else LOCK_LOST
    return {
        "status": status,
        "locked": status == LOCK_LOCKED,
        "rule": rule.as_dict(),
        "occlusion_start_frame_idx": a,
        "selection": selection,
        "selected_class": selected_class,
        "lock_ious": lock_ious,
        "best_iou": best_iou,
        "frames_with_match": frames_with_match,
        "frames_with_gt": frames_with_gt,
    }
