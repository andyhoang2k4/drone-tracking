"""Tinh metric danh gia (CLAUDE.md muc 12).

Trang thai file:
    - `iou()`, `center_error()`, cac dataclass metric: DA implement.
    - Ground truth VisDrone2019-MOT + `match_frame()` + `compute_detection_metrics()`
      + `compute_id_switches()`: DA implement (roadmap buoc 3, chua occlusion).
    - `load_pipeline_settings()` / `ablation_mismatches()`: DA implement - kiem
      tra hai lan chay co/khong Motion Memory dung cung weights/source/tracker/
      conf/imgsz/classes (doc tu run_context.json cua scripts/run_pipeline.py).
    - `compute_occlusion_metrics()`: CHUA implement (roadmap buoc 4).

Quy tac danh gia tren VisDrone2019-MOT (ghi ca trong README):
    1. Ground truth hop le = dong co score=1 va category nam trong category_map
       (1..10 -> class 0..9). Con lai (score=0, category 0 "ignored region",
       category ngoai map vd 11 "others") la VUNG BI BO QUA.
    2. Truoc khi ghep cap, moi frame:
       - prediction co IoA = dien tich giao / dien tich PREDICTION >= ignore_ioa
         voi bat ky vung bi bo qua nao -> LOAI khoi tinh toan (khong TP, khong FP);
       - ground truth hop le co IoA (theo dien tich GT) >= ignore_ioa voi vung bi
         bo qua cung bi loai (khong FN) - doi xung, de detection dung nam trong
         vung do khong bi tinh FN.
    3. Ghep cap tham lam theo IoU giam dan (hoa -> chi so nho truoc), IoU >=
       iou_threshold, cung class neu class_aware. Moi prediction/GT ghep toi da 1.
    4. ID switch: voi moi target_id cua GT, nho track ID cua lan ghep gan nhat;
       lan ghep sau co track ID khac -> +1 (ke ca khi giua hai lan co khoang
       khong ghep duoc - dung truong hop mat roi bat lai sai ID). Prediction
       khong co track ID (None) khong tinh vao ID switch.
    Don gian hoa so voi MOTA/CLEAR: ghep moi frame doc lap (khong uu tien giu
    cap cua frame truoc). IDF1/MOTA/HOTA khong bat buoc o giai doan nay.

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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from src.motion_memory import BBox
from src.utils.visdrone import is_ignore_region, read_annotation_file


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
    # Bi loai vi chong lan vung bi bo qua (quy tac 2 o docstring module)
    ignored_predictions: int = 0
    ignored_ground_truth: int = 0

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


# ---------------- ground truth + ghep cap (quy tac o docstring module) ----------------


@dataclass(frozen=True)
class EvalBox:
    """Mot box de danh gia: GT (obj_id = target_id) hoac prediction (obj_id = track ID)."""

    bbox: BBox
    cls: int
    obj_id: Optional[int] = None
    conf: Optional[float] = None


@dataclass
class FrameGroundTruth:
    boxes: list[EvalBox] = field(default_factory=list)
    ignore_regions: list[BBox] = field(default_factory=list)


@dataclass
class FrameMatch:
    """Ket qua ghep cap mot frame - chi so tro vao list prediction/GT cua frame."""

    matches: list[tuple[int, int]] = field(default_factory=list)  # (pred_idx, gt_idx)
    false_positives: list[int] = field(default_factory=list)
    false_negatives: list[int] = field(default_factory=list)
    ignored_predictions: list[int] = field(default_factory=list)
    ignored_ground_truth: list[int] = field(default_factory=list)


def load_ground_truth(
    annotation_path: Union[str, Path], category_map: Mapping[int, int]
) -> dict[int, FrameGroundTruth]:
    """Doc annotation VisDrone-MOT cua mot sequence -> {so frame (tu 1): FrameGroundTruth}.

    Frame co trong file nhung chi co vung bi bo qua van co mat (boxes rong).
    """
    frames: dict[int, FrameGroundTruth] = {}
    for box in read_annotation_file(annotation_path):
        frame = frames.setdefault(box.frame_idx, FrameGroundTruth())
        bbox = BBox(float(box.left), float(box.top), float(box.left + box.width), float(box.top + box.height))
        if is_ignore_region(box, category_map):
            frame.ignore_regions.append(bbox)
        else:
            frame.boxes.append(EvalBox(bbox=bbox, cls=category_map[box.category], obj_id=box.target_id))
    return frames


def intersection_over_area(box: BBox, region: BBox) -> float:
    """Dien tich giao / dien tich `box` (IoA). Box dien tich 0 -> 0."""
    inter_w = max(0.0, min(box.x2, region.x2) - max(box.x1, region.x1))
    inter_h = max(0.0, min(box.y2, region.y2) - max(box.y1, region.y1))
    area = box.width * box.height
    return inter_w * inter_h / area if area > 0.0 else 0.0


def _in_ignore_region(box: BBox, regions: Sequence[BBox], ignore_ioa: float) -> bool:
    return any(intersection_over_area(box, region) >= ignore_ioa for region in regions)


def match_frame(
    predictions: Sequence[EvalBox],
    ground_truth: Sequence[EvalBox],
    ignore_regions: Sequence[BBox] = (),
    *,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
    class_aware: bool = True,
) -> FrameMatch:
    """Ghep prediction voi GT trong mot frame (quy tac 2-3 o docstring module)."""
    result = FrameMatch()
    preds = []
    for i, pred in enumerate(predictions):
        if _in_ignore_region(pred.bbox, ignore_regions, ignore_ioa):
            result.ignored_predictions.append(i)
        else:
            preds.append(i)
    gts = []
    for j, gt in enumerate(ground_truth):
        if _in_ignore_region(gt.bbox, ignore_regions, ignore_ioa):
            result.ignored_ground_truth.append(j)
        else:
            gts.append(j)

    candidates = []
    for i in preds:
        for j in gts:
            if class_aware and predictions[i].cls != ground_truth[j].cls:
                continue
            overlap = iou(predictions[i].bbox, ground_truth[j].bbox)
            if overlap >= iou_threshold:
                candidates.append((-overlap, i, j))
    candidates.sort()

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for _, i, j in candidates:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        result.matches.append((i, j))
    result.matches.sort()
    result.false_positives = [i for i in preds if i not in used_pred]
    result.false_negatives = [j for j in gts if j not in used_gt]
    return result


def match_sequence(
    predictions: Sequence[Sequence[EvalBox]],
    ground_truth: Sequence[Sequence[EvalBox]],
    ignore_regions: Optional[Sequence[Sequence[BBox]]] = None,
    *,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
    class_aware: bool = True,
) -> list[FrameMatch]:
    """match_frame cho tung frame; ba danh sach phai cung so frame, cung thu tu."""
    if len(predictions) != len(ground_truth):
        raise ValueError(f"So frame khac nhau: prediction {len(predictions)}, GT {len(ground_truth)}")
    if ignore_regions is None:
        ignore_regions = [()] * len(ground_truth)
    if len(ignore_regions) != len(ground_truth):
        raise ValueError("ignore_regions phai co cung so frame voi ground truth")
    return [
        match_frame(p, g, r, iou_threshold=iou_threshold, ignore_ioa=ignore_ioa, class_aware=class_aware)
        for p, g, r in zip(predictions, ground_truth, ignore_regions)
    ]


def detection_metrics_from_matches(
    frame_matches: Sequence[FrameMatch], *, iou_threshold: float = 0.5
) -> DetectionMetrics:
    metrics = DetectionMetrics(iou_threshold=iou_threshold)
    for frame in frame_matches:
        metrics.true_positives += len(frame.matches)
        metrics.false_positives += len(frame.false_positives)
        metrics.false_negatives += len(frame.false_negatives)
        metrics.ignored_predictions += len(frame.ignored_predictions)
        metrics.ignored_ground_truth += len(frame.ignored_ground_truth)
    return metrics


def id_switches_from_matches(
    frame_matches: Sequence[FrameMatch],
    predictions: Sequence[Sequence[EvalBox]],
    ground_truth: Sequence[Sequence[EvalBox]],
) -> int:
    """Dem ID switch (quy tac 4 o docstring module)."""
    last_track: dict[int, int] = {}
    switches = 0
    for frame, preds, gts in zip(frame_matches, predictions, ground_truth):
        for i, j in frame.matches:
            gt_id, track_id = gts[j].obj_id, preds[i].obj_id
            if gt_id is None or track_id is None:
                continue
            previous = last_track.get(gt_id)
            if previous is not None and previous != track_id:
                switches += 1
            last_track[gt_id] = track_id
    return switches


def compute_detection_metrics(
    predictions: Sequence[Sequence[EvalBox]],
    ground_truth: Sequence[Sequence[EvalBox]],
    ignore_regions: Optional[Sequence[Sequence[BBox]]] = None,
    *,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
    class_aware: bool = True,
) -> DetectionMetrics:
    """TP/FP/FN theo frame (precision + recall bao cao song song)."""
    matches = match_sequence(
        predictions, ground_truth, ignore_regions,
        iou_threshold=iou_threshold, ignore_ioa=ignore_ioa, class_aware=class_aware,
    )
    return detection_metrics_from_matches(matches, iou_threshold=iou_threshold)


def compute_id_switches(
    predictions: Sequence[Sequence[EvalBox]],
    ground_truth: Sequence[Sequence[EvalBox]],
    ignore_regions: Optional[Sequence[Sequence[BBox]]] = None,
    *,
    iou_threshold: float = 0.5,
    ignore_ioa: float = 0.5,
    class_aware: bool = True,
) -> int:
    """So lan track ID gan cho cung mot target_id GT bi doi."""
    matches = match_sequence(
        predictions, ground_truth, ignore_regions,
        iou_threshold=iou_threshold, ignore_ioa=ignore_ioa, class_aware=class_aware,
    )
    return id_switches_from_matches(matches, predictions, ground_truth)


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


# ---------------- dieu kien so sanh cong bang (ablation, CLAUDE.md muc 12) ----------------

# Tham so phai GIONG NHAU giua lan chay co va khong Motion Memory.
ABLATION_MUST_MATCH = ("weights", "source", "tracker", "conf", "imgsz", "classes")


def load_pipeline_settings(exp_dir: Path) -> dict[str, Any]:
    """Doc tham so inference cua mot lan chay run_pipeline.py tu run_context.json.

    imgsz uu tien `inference.imgsz` (gia tri thuc te da dung); run cu khong co
    truong nay thi lay `config.imgsz`.
    """
    context = json.loads((Path(exp_dir) / "run_context.json").read_text(encoding="utf-8"))
    config = context.get("config") or {}
    settings: dict[str, Any] = {key: config.get(key) for key in ABLATION_MUST_MATCH}
    inference = context.get("inference") or {}
    if inference.get("imgsz") is not None:
        settings["imgsz"] = inference["imgsz"]
    settings["motion_memory"] = config.get("motion_memory")
    settings["experiment_id"] = context.get("experiment_id")
    return settings


def ablation_mismatches(
    baseline: Mapping[str, Any], with_memory: Mapping[str, Any]
) -> list[str]:
    """Danh sach ly do hai lan chay KHONG so sanh cong bang duoc; rong = OK."""
    problems: list[str] = []
    for key in ABLATION_MUST_MATCH:
        if baseline.get(key) != with_memory.get(key):
            problems.append(f"{key} khac nhau: {baseline.get(key)!r} vs {with_memory.get(key)!r}")
    if baseline.get("imgsz") is None or with_memory.get("imgsz") is None:
        problems.append("thieu imgsz trong run_context (lan chay cu?) - khong xac nhan duoc")
    if baseline.get("motion_memory") or not with_memory.get("motion_memory"):
        problems.append(
            "can baseline motion_memory=False va lan chay kia motion_memory=True, dang la "
            f"{baseline.get('motion_memory')!r} / {with_memory.get('motion_memory')!r}"
        )
    return problems
