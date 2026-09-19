"""Tinh metric danh gia (CLAUDE.md muc 12).

Trang thai file: KHUNG (scaffold).
    - `iou()` va cac dataclass metric: DA implement.
    - Cac ham tinh metric: CHUA implement - can chot dinh dang ground truth
      truoc (roadmap buoc 4 sinh kich ban occlusion co ground truth).

Nguyen tac bat buoc khi bao cao:
    - Recall va precision bao cao SONG SONG, khong dung recall don doc.
    - Moi kich ban occlusion phai co ground truth bbox cho TOAN BO sequence,
      ke ca frame bi che khuat - neu khong se khong tinh duoc "% frame du doan
      dung" va "sai so vi tri".
    - Danh gia Motion Memory phai co ablation co/khong Motion Memory tren
      CUNG detector + tracker.
    - IDF1/MOTA/HOTA KHONG bat buoc o giai doan nay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.motion_memory import BBox


def iou(a: BBox, b: BBox) -> float:
    """Intersection over Union cua hai bbox xyxy."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0.0 else 0.0


def center_error(a: BBox, b: BBox) -> float:
    """Khoang cach Euclid giua tam hai bbox (pixel)."""
    ax, ay = a.center
    bx, by = b.center
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass
class DetectionMetrics:
    """Precision/recall bao cao SONG SONG (CLAUDE.md muc 12)."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    iou_threshold: float = 0.5

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0


@dataclass
class TrackingMetrics:
    """Metric muc tracking cho ca sequence."""

    id_switches: int = 0
    frames: int = 0
    mean_fps: Optional[float] = None
    mean_latency_ms: Optional[float] = None


@dataclass
class OcclusionMetrics:
    """Metric bat buoc cho moi kich ban occlusion (CLAUDE.md muc 12)."""

    scenario_id: str = ""
    occlusion_frames: int = 0
    correct_prediction_ratio: float = 0.0
    longest_tracking_gap: int = 0
    reidentified: bool = False
    time_to_reidentify_frames: Optional[int] = None
    mean_position_error_px: Optional[float] = None
    per_frame_errors: list[float] = field(default_factory=list)


def compute_detection_metrics(
    predictions: Sequence[Sequence[BBox]],
    ground_truth: Sequence[Sequence[BBox]],
    *,
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Ghep prediction voi ground truth theo frame, dem TP/FP/FN.

    CHUA IMPLEMENT - can chot dinh dang ground truth truoc (roadmap buoc 4).
    """
    raise NotImplementedError("Chua chot dinh dang ground truth - roadmap buoc 4.")


def compute_id_switches(
    predicted_ids: Sequence[Optional[int]],
    ground_truth_ids: Sequence[Optional[int]],
) -> int:
    """Dem so lan track ID gan cho cung mot muc tieu bi doi.

    CHUA IMPLEMENT - can chot dinh dang ground truth truoc (roadmap buoc 4).
    """
    raise NotImplementedError("Chua chot dinh dang ground truth - roadmap buoc 4.")


def compute_occlusion_metrics(
    predictions: Sequence[Optional[BBox]],
    ground_truth: Sequence[Optional[BBox]],
    occluded_frames: Sequence[bool],
    *,
    iou_threshold: float = 0.5,
    scenario_id: str = "",
) -> OcclusionMetrics:
    """Tinh metric cho mot kich ban occlusion.

    `ground_truth` phai co gia tri o CA cac frame bi che khuat.

    CHUA IMPLEMENT - can chot dinh dang ground truth truoc (roadmap buoc 4).
    """
    raise NotImplementedError("Chua chot dinh dang ground truth - roadmap buoc 4.")
