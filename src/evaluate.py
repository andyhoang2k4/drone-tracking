"""Tinh metric danh gia (CLAUDE.md muc 12).

Trang thai file:
    - `iou()`, `center_error()`, cac dataclass metric: DA implement.
    - Ground truth VisDrone2019-MOT + `match_frame()` + `compute_detection_metrics()`
      + `compute_id_switches()`: DA implement (roadmap buoc 3, chua occlusion).
    - `load_pipeline_settings()` / `ablation_mismatches()`: DA implement - kiem
      tra hai lan chay co/khong Motion Memory dung cung weights/source/tracker/
      conf/imgsz/classes (doc tu run_context.json cua scripts/run_pipeline.py).
    - CLI (`python -m src.evaluate`): DA implement - danh gia tung sequence
      VisDrone (val hoac test-dev) tu log cua src/track.py, ghi metrics JSON.
    - `compute_occlusion_metrics()` + danh gia kich ban occlusion tu log
      run_pipeline va manifest (`python -m src.evaluate occlusion|aggregate|
      compare`): DA implement (roadmap buoc 4). Dinh nghia do do:
      configs/occlusion_metrics.yaml va docstring compute_occlusion_metrics.

Vi du (moi sequence mot lan, log tao boi src/track.py voi --source la thu muc
anh cua dung sequence do)::

    python -m src.evaluate --split val --sequence uav0000086_00000_v \
        --log experiments/<id>/logs/val/uav0000086_00000_v.jsonl \
        --output experiments/<id>/metrics/val/uav0000086_00000_v.json

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

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from src.motion_memory import BBox
from src.utils.visdrone import frame_index_from_path, is_ignore_region, read_annotation_file

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval_visdrone.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    """Metric bat buoc cho moi kich ban occlusion (CLAUDE.md muc 12).

    Dinh nghia chinh xac: docstring compute_occlusion_metrics va
    configs/occlusion_metrics.yaml. Cac truong *_frames / *_sum la tu so,
    mau so de gop nhieu kich ban bang cong gop.
    """

    scenario_id: str = ""
    occlusion_frames: int = 0
    correct_prediction_ratio: float = 0.0
    longest_tracking_gap: int = 0
    reidentified: bool = False
    time_to_reidentify_frames: Optional[int] = None
    mean_position_error_px: Optional[float] = None
    # sai so tam trong khoang che (None = frame do khong co output)
    per_frame_errors: list[Optional[float]] = field(default_factory=list)
    total_frames: int = 0
    occlusion_start_frame: int = 0
    occlusion_end_frame: int = 0
    correct_occluded_frames: int = 0
    occluded_frames_with_output: int = 0
    position_error_sum_px: float = 0.0
    target_acquired_before_occlusion: bool = False
    first_correct_frame: Optional[int] = None
    iou_threshold: float = 0.5
    reid_min_consecutive_frames: int = 3


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
    reid_min_consecutive_frames: int = 3,
    confirmed: Optional[Sequence[bool]] = None,
) -> OcclusionMetrics:
    """Tinh metric cho mot kich ban occlusion (dinh nghia giong
    configs/occlusion_metrics.yaml).

    Args:
        predictions: output cua pipeline moi frame (None = khong co output).
        ground_truth: bbox GT cua muc tieu o MOI frame (ke ca frame bi che);
            None o bat ky frame nao -> ValueError.
        occluded_frames: True o frame bi che; phai la MOT doan lien tuc [a, b].
        confirmed: True o frame ma output den tu detection cua tracker (khong
            phai ngoai suy). None = moi output deu la detection (baseline).

    Dinh nghia (frame k, M = reid_min_consecutive_frames):
        dung(k)          = predictions[k] co va IoU(predictions[k], GT_k) >= iou_threshold
        occlusion_frames = b - a + 1
        correct_prediction_ratio = #{k in [a,b]: dung(k)} / occlusion_frames
        longest_tracking_gap = doan dai nhat cac frame lien tiep khong dung,
            tu frame dung dau tien den het; chua tung dung -> so frame
        reidentified     = ton tai k > b voi dung(k..k+M-1) VA confirmed(k..k+M-1)
                           (trong kich ban) - ngoai suy khong tinh la bat lai
        time_to_reidentify_frames = k_min - b (>= 1), None neu khong
        mean_position_error_px = trung binh center_error tren frame trong [a,b]
            co output; None neu khong co frame nao
        target_acquired_before_occlusion = ton tai k < a voi dung(k)
    """
    n = len(ground_truth)
    if len(predictions) != n or len(occluded_frames) != n:
        raise ValueError(
            f"{scenario_id}: so frame khac nhau (pred {len(predictions)}, GT {n}, "
            f"occluded {len(occluded_frames)})"
        )
    if any(g is None for g in ground_truth):
        raise ValueError(f"{scenario_id}: ground truth thieu o mot so frame (bat buoc du moi frame)")
    occ = [k for k, flag in enumerate(occluded_frames) if flag]
    if not occ:
        raise ValueError(f"{scenario_id}: khong co frame bi che")
    a, b = occ[0], occ[-1]
    if occ != list(range(a, b + 1)):
        raise ValueError(f"{scenario_id}: frame bi che phai la mot doan lien tuc")
    if reid_min_consecutive_frames < 1:
        raise ValueError("reid_min_consecutive_frames phai >= 1")
    if confirmed is None:
        confirmed = [p is not None for p in predictions]
    if len(confirmed) != n:
        raise ValueError(f"{scenario_id}: confirmed khac so frame")

    correct = [
        p is not None and iou(p, g) >= iou_threshold for p, g in zip(predictions, ground_truth)
    ]
    metrics = OcclusionMetrics(
        scenario_id=scenario_id,
        occlusion_frames=b - a + 1,
        total_frames=n,
        occlusion_start_frame=a,
        occlusion_end_frame=b,
        iou_threshold=iou_threshold,
        reid_min_consecutive_frames=reid_min_consecutive_frames,
    )
    metrics.correct_occluded_frames = sum(correct[a:b + 1])
    metrics.correct_prediction_ratio = metrics.correct_occluded_frames / metrics.occlusion_frames
    metrics.target_acquired_before_occlusion = any(correct[:a])

    first = next((k for k, ok in enumerate(correct) if ok), None)
    metrics.first_correct_frame = first
    if first is None:
        metrics.longest_tracking_gap = n
    else:
        run = longest = 0
        for ok in correct[first:]:
            run = 0 if ok else run + 1
            longest = max(longest, run)
        metrics.longest_tracking_gap = longest

    m = reid_min_consecutive_frames
    reacquired = [ok and bool(c) for ok, c in zip(correct, confirmed)]
    for k in range(b + 1, n - m + 1):
        if all(reacquired[k:k + m]):
            metrics.reidentified = True
            metrics.time_to_reidentify_frames = k - b
            break

    errors: list[Optional[float]] = []
    for k in range(a, b + 1):
        p = predictions[k]
        errors.append(center_error(p, ground_truth[k]) if p is not None else None)
    measured = [e for e in errors if e is not None]
    metrics.per_frame_errors = errors
    metrics.occluded_frames_with_output = len(measured)
    metrics.position_error_sum_px = float(sum(measured))
    metrics.mean_position_error_px = (
        metrics.position_error_sum_px / len(measured) if measured else None
    )
    return metrics


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


# ---------------- CLI: danh gia mot sequence VisDrone-MOT ----------------


class EvaluationError(ValueError):
    """Dau vao khong hop le de danh gia (log thieu frame, sai sequence, sai conf...)."""


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sequence_paths(eval_config: Mapping[str, Any], split: str, sequence: str) -> tuple[Path, Path]:
    """(file annotation, thu muc anh) cua mot sequence trong split val/test-dev."""
    splits = eval_config.get("splits") or {}
    if split not in splits:
        raise EvaluationError(f"split {split!r} khong co trong config (co: {sorted(splits)})")
    split_dir = _repo_path(eval_config["raw_root"]) / splits[split]
    annotation = split_dir / "annotations" / f"{sequence}.txt"
    if not annotation.is_file():
        raise EvaluationError(f"Khong thay annotation: {annotation}")
    return annotation, split_dir / "sequences" / sequence


def read_track_log(log_path: Path) -> dict[int, dict[str, Any]]:
    """Log JSONL cua src/track.py -> {so frame VisDrone: ban ghi}. Can truong `path`."""
    records: dict[int, dict[str, Any]] = {}
    for line_no, line in enumerate(Path(log_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not record.get("path"):
            raise EvaluationError(
                f"{log_path}:{line_no}: ban ghi khong co `path` - log cu? Chay lai src/track.py"
            )
        if "tracks" not in record:
            raise EvaluationError(f"{log_path}:{line_no}: khong co `tracks` - khong phai log cua src/track.py")
        frame = frame_index_from_path(record["path"])
        if frame in records:
            raise EvaluationError(f"{log_path}: frame {frame} xuat hien hai lan (log bi ghi noi?)")
        records[frame] = record
    if not records:
        raise EvaluationError(f"{log_path}: log rong")
    return records


def _image_frames(sequence_dir: Path) -> Optional[set[int]]:
    if not sequence_dir.is_dir():
        return None
    return {
        frame_index_from_path(p) for p in sequence_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Noi suy tuyen tinh (giong numpy mac dinh)."""
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def speed_summary(records: Sequence[Mapping[str, Any]], warmup_frames: int) -> dict[str, Any]:
    """FPS/latency thuc do tu frame_time_ms (bo `warmup_frames` frame dau)."""
    times = [r.get("frame_time_ms") for r in records]
    timed = [float(t) for t in times[warmup_frames:] if t is not None]
    summary: dict[str, Any] = {
        "warmup_frames_excluded": min(warmup_frames, len(records)),
        "timed_frames": len(timed),
        "mean_latency_ms": None,
        "median_latency_ms": None,
        "p95_latency_ms": None,
        "mean_fps": None,
        "ultralytics_speed_ms_mean": {},
        "note": "frame_time_ms gom doc anh + preprocess + inference + postprocess + tracker; "
        "ultralytics_speed_ms khong gom tracker",
    }
    if timed:
        ordered = sorted(timed)
        mean = sum(timed) / len(timed)
        summary.update(
            mean_latency_ms=mean,
            median_latency_ms=_percentile(ordered, 0.5),
            p95_latency_ms=_percentile(ordered, 0.95),
            mean_fps=1000.0 / mean if mean > 0 else None,
        )
    stages: dict[str, list[float]] = {}
    for record in records[warmup_frames:]:
        for key, value in (record.get("speed_ms") or {}).items():
            if value is not None:
                stages.setdefault(key, []).append(float(value))
    summary["ultralytics_speed_ms_mean"] = {k: sum(v) / len(v) for k, v in stages.items()}
    return summary


def _check_meta(meta: Mapping[str, Any], sequence: str, configured_conf: float) -> list[str]:
    """Kiem tra log dung sequence + dung conf config. Tra ve canh bao (khong chan)."""
    source = meta.get("source")
    if not source or Path(str(source)).name != sequence:
        raise EvaluationError(
            f"Log chay tren source={source!r}, khong phai sequence {sequence!r} "
            "(--source phai la thu muc anh cua dung sequence)"
        )
    if meta.get("conf") != configured_conf:
        raise EvaluationError(
            f"conf cua log = {meta.get('conf')!r} khac inference.conf = {configured_conf} trong "
            "train_config (conf co dinh, dung chung cho hai model)"
        )
    warnings = []
    if meta.get("classes"):
        warnings.append(
            f"log loc classes={meta['classes']} nhung GT khong loc -> recall bi tinh tren moi class"
        )
    return warnings


def evaluate_sequence(
    *,
    log_path: Path,
    split: str,
    sequence: str,
    eval_config: Mapping[str, Any],
    configured_conf: float,
) -> dict[str, Any]:
    """Danh gia mot sequence -> dict metrics (chua ghi file)."""
    annotation, sequence_dir = sequence_paths(eval_config, split, sequence)
    meta_path = Path(log_path).with_suffix(".meta.json")
    if not meta_path.is_file():
        raise EvaluationError(f"Khong thay {meta_path} (src/track.py ghi canh log)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    warnings = _check_meta(meta, sequence, configured_conf)

    category_map = {
        int(k): int(v)
        for k, v in (_load_yaml(_repo_path(eval_config["dataset_config"])).get("category_map") or {}).items()
    }
    if not category_map:
        raise EvaluationError(f"{eval_config['dataset_config']} khong co category_map")
    ground_truth = load_ground_truth(annotation, category_map)
    records = read_track_log(log_path)

    expected = _image_frames(sequence_dir)
    if expected is None:
        expected = {f for f, g in ground_truth.items() if g.boxes}
        warnings.append(f"khong thay thu muc anh {sequence_dir}; chi kiem tra du frame co GT")
    missing = sorted(expected - set(records))
    if missing:
        raise EvaluationError(
            f"Log thieu {len(missing)} frame cua sequence (vd {missing[:5]}) - log chua chay het?"
        )

    frames = sorted(records)
    preds = [
        [
            EvalBox(BBox(*t["bbox"]), int(t["cls"]), t.get("track_id"), t.get("conf"))
            for t in records[f]["tracks"]
        ]
        for f in frames
    ]
    gts = [ground_truth.get(f, FrameGroundTruth()).boxes for f in frames]
    regions = [ground_truth.get(f, FrameGroundTruth()).ignore_regions for f in frames]
    iou_threshold = float(eval_config["iou_threshold"])
    matches = match_sequence(
        preds, gts, regions,
        iou_threshold=iou_threshold,
        ignore_ioa=float(eval_config["ignore_ioa"]),
        class_aware=bool(eval_config["class_aware"]),
    )
    detection = detection_metrics_from_matches(matches, iou_threshold=iou_threshold)
    id_switches = id_switches_from_matches(matches, preds, gts)

    gt_targets = {b.obj_id for frame in gts for b in frame}
    matched_targets = {gts[k][j].obj_id for k, m in enumerate(matches) for _, j in m.matches}
    predicted_tracks = {b.obj_id for frame in preds for b in frame if b.obj_id is not None}

    from src.utils.run_context import git_info

    return {
        "split": split,
        "sequence": sequence,
        "log": str(log_path),
        "annotation": str(annotation),
        "frames_evaluated": len(frames),
        "inference": {
            key: meta.get(key)
            for key in ("weights", "tracker", "conf", "conf_source", "imgsz", "imgsz_source",
                        "device", "classes")
        },
        "eval_config": {
            key: eval_config.get(key)
            for key in ("iou_threshold", "ignore_ioa", "class_aware", "fps_warmup_frames")
        },
        "detection": {**asdict(detection), "precision": detection.precision, "recall": detection.recall},
        "tracking": {
            "id_switches": id_switches,
            "gt_targets": len(gt_targets),
            "gt_targets_matched": len(matched_targets),
            "predicted_tracks": len(predicted_tracks),
        },
        "speed": speed_summary([records[f] for f in frames], int(eval_config.get("fps_warmup_frames", 0))),
        "warnings": warnings,
        "git": git_info(REPO_ROOT),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Danh gia mot sequence VisDrone2019-MOT tu log cua src/track.py"
    )
    parser.add_argument("--log", type=Path, required=True, help="track.jsonl (canh no la .meta.json)")
    parser.add_argument("--split", required=True, help="val hoac test-dev (theo eval config)")
    parser.add_argument("--sequence", required=True, help="vd uav0000086_00000_v")
    parser.add_argument("--output", type=Path, required=True, help="File metrics JSON")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument(
        "--train-config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_config.yaml",
        help="Lay inference.conf (conf co dinh, dung chung)",
    )
    return parser.parse_args(argv)


# ---------------- kich ban occlusion: log run_pipeline + manifest ----------------

DEFAULT_OCCLUSION_METRICS_CONFIG = REPO_ROOT / "configs" / "occlusion_metrics.yaml"
PIPELINE_LOG = Path("logs") / "pipeline.jsonl"
SCENARIO_RUNS_DIR = "scenarios"   # experiments/<id>/scenarios/<ten kich ban>/
SCENARIO_METRICS_NAME = "metrics.json"
# Tham so phai giong nhau giua moi kich ban duoc gop vao mot bao cao.
RUN_KEYS = ("weights", "tracker", "conf", "imgsz", "classes", "motion_memory")


def load_occlusion_metrics_config(path: Union[str, Path]) -> dict[str, Any]:
    cfg = _load_yaml(Path(path))
    for key in ("version", "correct_iou", "reid_min_consecutive_frames"):
        if key not in cfg:
            raise EvaluationError(f"{path}: thieu khoa {key}")
    return {
        "version": int(cfg["version"]),
        "correct_iou": float(cfg["correct_iou"]),
        "reid_min_consecutive_frames": int(cfg["reid_min_consecutive_frames"]),
        "path": str(path),
    }


def read_pipeline_log(path: Path) -> list[dict[str, Any]]:
    """Log run_pipeline -> danh sach ban ghi theo frame_idx 0..N-1 (bat buoc lien tuc)."""
    if not Path(path).is_file():
        raise EvaluationError(f"Khong thay log {path}")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = [r.get("frame_idx") for r in rows]
    if indices != list(range(len(rows))):
        raise EvaluationError(
            f"{path}: frame_idx phai la 0..N-1 lien tuc, khong trung (log bi ghi noi hoac thieu frame?)"
        )
    return rows


def pipeline_outputs(
    rows: Sequence[Mapping[str, Any]],
    motion_memory: bool,
    selection: Mapping[str, Any],
) -> tuple[list[Optional[BBox]], list[Optional[int]], list[bool]]:
    """(output bbox, track ID muc tieu, output xac nhan) moi frame - dinh
    nghia "output" va "output xac nhan" o configs/occlusion_metrics.yaml."""
    boxes: list[Optional[BBox]] = []
    ids: list[Optional[int]] = []
    confirmed: list[bool] = []
    baseline_id = selection.get("selected_track_id", selection.get("target_track_id"))
    for row in rows:
        if motion_memory:
            raw = row.get("predicted_bbox")
            boxes.append(BBox(*raw) if raw else None)
            ids.append(row.get("track_id") if raw else None)
            confirmed.append(bool(raw) and row.get("state") == "TRACKING")
        else:
            raw = row.get("tracker_bbox")
            boxes.append(BBox(*raw) if raw else None)
            ids.append(baseline_id if raw else None)
            confirmed.append(bool(raw))
    return boxes, ids, confirmed


def evaluate_occlusion_run(
    scenario_dir: Union[str, Path],
    run_dir: Union[str, Path],
    metrics_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Danh gia mot lan chay run_pipeline tren mot kich ban -> dict metrics."""
    from src.occlusion_scenarios import load_manifest, manifest_ground_truth, occluded_flags

    scenario_dir, run_dir = Path(scenario_dir), Path(run_dir)
    manifest = load_manifest(scenario_dir)
    context_path = run_dir / "run_context.json"
    selection_path = run_dir / "target_selection.json"
    for p in (context_path, selection_path):
        if not p.is_file():
            raise EvaluationError(f"Khong thay {p} (do scripts/run_pipeline.py ghi)")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = context.get("config") or {}
    warnings: list[str] = []

    source_parts = Path(str(config.get("source", ""))).parts[-2:]
    if tuple(source_parts) != (manifest["name"], "images"):
        raise EvaluationError(
            f"Lan chay dung source={config.get('source')!r}, khong phai {manifest['name']}/images"
        )
    gt_id = manifest["target"]["gt_target_id"]
    if selection.get("mode") == "gt":
        if selection.get("gt_target_id") != gt_id:
            raise EvaluationError(
                f"target_selection.gt_target_id={selection.get('gt_target_id')} khac manifest {gt_id}"
            )
    else:
        warnings.append("muc tieu chon bang --target-track-id: khong kiem chung duoc la dung muc tieu GT")

    rows = read_pipeline_log(run_dir / PIPELINE_LOG)
    if len(rows) != int(manifest["frames"]):
        raise EvaluationError(
            f"Log co {len(rows)} frame, manifest co {manifest['frames']} - log chua chay het?"
        )
    motion_memory = bool(config.get("motion_memory"))
    boxes, ids, confirmed = pipeline_outputs(rows, motion_memory, selection)
    gt = manifest_ground_truth(manifest)
    metrics = compute_occlusion_metrics(
        boxes, gt, occluded_flags(manifest),
        iou_threshold=metrics_config["correct_iou"],
        scenario_id=manifest["scenario_id"],
        reid_min_consecutive_frames=metrics_config["reid_min_consecutive_frames"],
        confirmed=confirmed,
    )

    a = metrics.occlusion_start_frame
    correct_before = [
        k for k in range(a) if boxes[k] is not None and iou(boxes[k], gt[k]) >= metrics.iou_threshold
    ]
    id_before = ids[correct_before[-1]] if correct_before else None
    reid_frame = (
        metrics.occlusion_end_frame + metrics.time_to_reidentify_frames
        if metrics.reidentified else None
    )
    id_at_reid = ids[reid_frame] if reid_frame is not None else None
    recovery = {
        "track_id_before_occlusion": id_before,
        "reidentify_frame": reid_frame,
        "track_id_at_reidentify": id_at_reid,
        "reidentified_with_new_track_id": (
            bool(metrics.reidentified and id_before is not None and id_at_reid != id_before)
        ),
        "recover_reasons": sorted(
            {r["recover_reason"] for r in rows if r.get("recover_reason")}
        ) if motion_memory else [],
    }

    from src.utils.run_context import git_info

    occ = manifest["occlusion"]
    return {
        "scenario_id": manifest["scenario_id"],
        "scenario": {
            "name": manifest["name"],
            "group": manifest["group"],
            "sequence": manifest["source"]["sequence"],
            "gt_target_id": gt_id,
            "duration_frames": occ["duration_frames"],
            "occluder": occ["occluder"],
            "coverage": occ["coverage"],
            "texture": occ["texture"],
            "config_sha256": manifest.get("config_sha256"),
            "scenario_dir": str(scenario_dir),
        },
        "run": {
            "run_dir": str(run_dir),
            "experiment_id": context.get("experiment_id"),
            **{k: config.get(k) for k in RUN_KEYS},
            "motion_memory_config": (context.get("motion_memory") or {}).get("config"),
            "git_commit": (context.get("git") or {}).get("commit"),
        },
        "target_selection": selection,
        "metrics_config": dict(metrics_config),
        "metrics": {**asdict(metrics), **recovery},
        "warnings": warnings,
        "git": git_info(REPO_ROOT),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _pool(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gop metrics nhieu kich ban bang CONG GOP tu so / mau so."""
    ms = [it["metrics"] for it in items]
    n = len(ms)
    occ = sum(m["occlusion_frames"] for m in ms)
    correct = sum(m["correct_occluded_frames"] for m in ms)
    with_output = sum(m["occluded_frames_with_output"] for m in ms)
    err_sum = sum(m["position_error_sum_px"] for m in ms)
    reid = [m for m in ms if m["reidentified"]]
    ttr = sorted(m["time_to_reidentify_frames"] for m in reid)
    gaps = [m["longest_tracking_gap"] for m in ms]
    return {
        "scenarios": n,
        "occlusion_frames": occ,
        "correct_occluded_frames": correct,
        "correct_prediction_ratio": correct / occ if occ else None,
        "occluded_frames_with_output": with_output,
        "mean_position_error_px": err_sum / with_output if with_output else None,
        "reidentified": len(reid),
        "reidentification_failed": n - len(reid),
        "reidentification_rate": len(reid) / n if n else None,
        "reidentified_with_new_track_id": sum(1 for m in ms if m["reidentified_with_new_track_id"]),
        "mean_time_to_reidentify_frames": sum(ttr) / len(ttr) if ttr else None,
        "max_time_to_reidentify_frames": ttr[-1] if ttr else None,
        "longest_tracking_gap_max": max(gaps) if gaps else None,
        "longest_tracking_gap_mean": sum(gaps) / n if n else None,
        "target_not_acquired_before_occlusion": sum(
            1 for m in ms if not m["target_acquired_before_occlusion"]
        ),
    }


def aggregate_occlusion_metrics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Gop nhieu kich ban cua CUNG mot cau hinh chay, CUNG mot nhom."""
    if not items:
        raise EvaluationError("Khong co kich ban nao de gop")
    groups = {it["scenario"]["group"] for it in items}
    if len(groups) != 1:
        raise EvaluationError(f"Khong gop nhieu nhom vao mot bao cao: {sorted(groups)}")
    for key in RUN_KEYS:
        values = {json.dumps(it["run"].get(key), sort_keys=True) for it in items}
        if len(values) != 1:
            raise EvaluationError(f"Cac kich ban khac {key}: {sorted(values)} - khong gop duoc")
    mm_cfgs = {json.dumps(it["run"].get("motion_memory_config"), sort_keys=True) for it in items}
    if len(mm_cfgs) != 1:
        raise EvaluationError("Cac kich ban dung tham so Motion Memory khac nhau - khong gop duoc")
    metric_cfgs = {
        (it["metrics_config"]["version"], it["metrics_config"]["correct_iou"],
         it["metrics_config"]["reid_min_consecutive_frames"])
        for it in items
    }
    if len(metric_cfgs) != 1:
        raise EvaluationError(f"Cac kich ban dung dinh nghia do do khac nhau: {sorted(metric_cfgs)}")

    def by(key: str) -> dict[str, Any]:
        buckets: dict[Any, list] = {}
        for it in items:
            buckets.setdefault(it["scenario"][key], []).append(it)
        return {str(k): _pool(v) for k, v in sorted(buckets.items())}

    first = items[0]
    return {
        "group": groups.pop(),
        "run": {k: first["run"].get(k) for k in RUN_KEYS},
        "motion_memory_config": first["run"].get("motion_memory_config"),
        "metrics_config": first["metrics_config"],
        "overall": _pool(items),
        "by_duration_frames": by("duration_frames"),
        "by_occluder": by("occluder"),
        "scenario_ids": sorted(it["scenario_id"] for it in items),
    }


def load_scenario_metrics(exp_dir: Path) -> dict[str, dict[str, Any]]:
    """{ten kich ban: metrics} tu experiments/<id>/scenarios/*/metrics.json."""
    root = Path(exp_dir) / SCENARIO_RUNS_DIR
    if not root.is_dir():
        raise EvaluationError(f"Khong thay {root}")
    out = {}
    for path in sorted(root.glob(f"*/{SCENARIO_METRICS_NAME}")):
        out[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _check_complete(found: Mapping[str, Any], index_path: Optional[Path], allow_incomplete: bool) -> dict:
    if index_path is None:
        if not allow_incomplete:
            raise EvaluationError("Can --index (index.json cua nhom kich ban) hoac --allow-incomplete")
        return {"index": None, "missing": [], "complete": False}
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    expected = set(index["scenarios"])
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if extra:
        raise EvaluationError(f"Co metrics cua kich ban khong thuoc index: {extra[:5]}")
    if missing and not allow_incomplete:
        raise EvaluationError(
            f"Thieu metrics cua {len(missing)} kich ban (vd {missing[:3]}) - lan chay that bai cung "
            "phai co mat; dung --allow-incomplete de gop mot phan (danh dau incomplete)"
        )
    complete = not missing and not index.get("partial", False)
    return {"index": str(index_path), "missing": missing, "complete": complete,
            "index_partial": bool(index.get("partial", False))}


def compare_ablation(baseline_dir: Path, memory_dir: Path) -> dict[str, Any]:
    """So sanh co/khong Motion Memory tren cung kich ban. Tu choi neu bat ky cap
    kich ban nao khac weights/source/tracker/conf/imgsz/classes (ablation_mismatches)."""
    base = load_scenario_metrics(baseline_dir)
    mem = load_scenario_metrics(memory_dir)
    if set(base) != set(mem):
        raise EvaluationError(
            f"Hai experiment khong cung tap kich ban: chi baseline {sorted(set(base) - set(mem))[:3]}, "
            f"chi memory {sorted(set(mem) - set(base))[:3]}"
        )
    problems: list[str] = []
    warnings: list[str] = []
    for name in sorted(base):
        b_run = Path(baseline_dir) / SCENARIO_RUNS_DIR / name
        m_run = Path(memory_dir) / SCENARIO_RUNS_DIR / name
        for p in ablation_mismatches(load_pipeline_settings(b_run), load_pipeline_settings(m_run)):
            problems.append(f"{name}: {p}")
        b_sel = base[name]["target_selection"].get("selected_track_id")
        m_sel = mem[name]["target_selection"].get("selected_track_id")
        if b_sel != m_sel:
            warnings.append(f"{name}: track muc tieu duoc chon khac nhau ({b_sel} vs {m_sel})")
    if problems:
        raise EvaluationError("Khong so sanh cong bang duoc:\n  " + "\n  ".join(problems[:20]))
    b_agg = aggregate_occlusion_metrics(list(base.values()))
    m_agg = aggregate_occlusion_metrics(list(mem.values()))
    if b_agg["metrics_config"]["version"] != m_agg["metrics_config"]["version"]:
        raise EvaluationError("Hai experiment dung dinh nghia do do khac version")
    return {
        "baseline_experiment": str(baseline_dir),
        "memory_experiment": str(memory_dir),
        "scenarios": len(base),
        "baseline": b_agg,
        "memory": m_agg,
        "warnings": warnings,
    }


def _occlusion_cli(argv: list[str]) -> int:
    from src.utils.logging_utils import write_json

    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Do do kich ban occlusion (configs/occlusion_metrics.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("occlusion", help="Danh gia mot lan chay tren mot kich ban")
    one.add_argument("--scenario-dir", type=Path, required=True)
    one.add_argument("--run-dir", type=Path, required=True, help="Thu muc --output-root/<experiment-id> cua run_pipeline")
    one.add_argument("--output", type=Path, default=None, help="Mac dinh <run-dir>/metrics.json")
    one.add_argument("--metrics-config", type=Path, default=DEFAULT_OCCLUSION_METRICS_CONFIG)

    agg = sub.add_parser("aggregate", help="Gop metrics cac kich ban cua mot experiment")
    agg.add_argument("--exp-dir", type=Path, required=True)
    agg.add_argument("--index", type=Path, default=None, help="index.json cua nhom kich ban")
    agg.add_argument("--allow-incomplete", action="store_true")
    agg.add_argument("--output", type=Path, default=None, help="Mac dinh <exp-dir>/metrics.json")

    cmp_ = sub.add_parser("compare", help="So sanh co/khong Motion Memory (ablation)")
    cmp_.add_argument("--baseline", type=Path, required=True)
    cmp_.add_argument("--memory", type=Path, required=True)
    cmp_.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "occlusion":
        result = evaluate_occlusion_run(
            args.scenario_dir, args.run_dir, load_occlusion_metrics_config(args.metrics_config)
        )
        out = args.output or args.run_dir / SCENARIO_METRICS_NAME
        write_json(out, result)
        m = result["metrics"]
        print(f"{result['scenario_id']}: dung {m['correct_occluded_frames']}/{m['occlusion_frames']} "
              f"frame che, reid={m['reidentified']} (sau {m['time_to_reidentify_frames']} frame)")
        print(f"Da ghi: {out}")
    elif args.command == "aggregate":
        found = load_scenario_metrics(args.exp_dir)
        completeness = _check_complete(found, args.index, args.allow_incomplete)
        result = {**aggregate_occlusion_metrics(list(found.values())), "completeness": completeness}
        out = args.output or args.exp_dir / SCENARIO_METRICS_NAME
        write_json(out, result)
        print(f"Gop {result['overall']['scenarios']} kich ban, complete={completeness['complete']}")
        print(f"Da ghi: {out}")
    else:
        result = compare_ablation(args.baseline, args.memory)
        write_json(args.output, result)
        for w in result["warnings"]:
            print(f"CANH BAO: {w}")
        print(f"So sanh {result['scenarios']} kich ban. Da ghi: {args.output}")
    return 0


OCCLUSION_COMMANDS = ("occlusion", "aggregate", "compare")


def main(argv: Optional[list[str]] = None) -> int:
    from src.utils.inference_settings import config_conf
    from src.utils.logging_utils import write_json

    import sys

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in OCCLUSION_COMMANDS:
        return _occlusion_cli(raw_argv)

    args = parse_args(argv)
    metrics = evaluate_sequence(
        log_path=args.log,
        split=args.split,
        sequence=args.sequence,
        eval_config=_load_yaml(args.eval_config),
        configured_conf=config_conf(args.train_config),
    )
    metrics["eval_config"]["path"] = str(args.eval_config)
    write_json(args.output, metrics)
    det, trk, spd = metrics["detection"], metrics["tracking"], metrics["speed"]
    print(f"{args.split}/{args.sequence}: {metrics['frames_evaluated']} frame")
    print(f"  precision={det['precision']:.4f} recall={det['recall']:.4f} "
          f"(TP={det['true_positives']} FP={det['false_positives']} FN={det['false_negatives']})")
    print(f"  ID switch={trk['id_switches']}  GT targets={trk['gt_targets']}")
    fps = spd["mean_fps"]
    print(f"  FPS={fps:.2f}" if fps is not None else "  FPS=khong du frame de do")
    for warning in metrics["warnings"]:
        print(f"CANH BAO: {warning}")
    print(f"Da ghi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
