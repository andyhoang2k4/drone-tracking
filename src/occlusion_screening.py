"""Screening "khoa muc tieu truoc khi che" cho kich ban occlusion v2.

Doc configs/occlusion_scenarios_v2.yaml. Voi moi ung vien (plan_candidates):

    1. Clip SACH = dung pre_occlusion_frames frame nguon truoc khi che
       [start - pre, start - 1], ten anh danh so lai tu 0000001 (hardlink anh
       goc) -> GIONG HET frame 0..pre-1 cua moi kich ban cua muc tieu do
       (write_scenario cung hardlink cung anh, cung cach danh so).
    2. src.track (YOLO + BoT-SORT, cung weights/config nhu ablation) tren clip,
       moi detector mot log (chay o scripts/screen_occlusion_targets.py).
    3. evaluate_target_lock (src/utils/gt_target.py) voi tham so target_lock:
       luat chon muc tieu THAT cua pipeline (GtTargetSelector) + track da chon
       dung vi tri o lock_confirm_frames frame cuoi.
    4. Ung vien DAT <=> MOI detector trong required_detectors deu 'locked'.

Ket qua gon (commit duoc): <output_root>/<group>/screening.json - ly do loai
tung detector (trang thai, IoU tot nhat, so frame bat duoc, class track da
chon), ti le dat theo detector / sequence, phan bo lop + kich thuoc cua ung
vien so voi muc tieu dat, co `partial` khi chay --limit.

Chi lam viec tren bbox va clip sach: KHONG chay Motion Memory, KHONG sinh kich
ban che -> dung duoc cho heldout ma khong xem ket qua occlusion.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.occlusion_scenarios import (
    ANNOTATION_NAME,
    IMAGES_DIR,
    SCREENING_KIND,
    ScenarioConfig,
    ScreeningError,
    SequenceData,
    TargetPlan,
    _annotation_line,
    _link_or_copy,
    candidate_eligible,
    candidate_id,
    list_sequences,
    load_sequence,
    plan_candidates,
    select_screened_targets,
)
from src.utils.gt_target import LOCK_LOCKED, evaluate_target_lock, load_target_boxes

CLIP_INFO_NAME = "clip.json"
WEIGHTS_SIDECAR_NAME = "screening_weights.json"


def plan_group_candidates(cfg: ScenarioConfig, group: str) -> list[tuple[TargetPlan, SequenceData]]:
    """Moi ung vien cua nhom: sequence theo ten, trong sequence theo thu tu xep hang."""
    if not cfg.uses_screening:
        raise ScreeningError("config khong co target_lock (can configs/occlusion_scenarios_v2.yaml)")
    if group not in cfg.groups:
        raise ScreeningError(f"Nhom {group!r} khong co trong config (co: {sorted(cfg.groups)})")
    split_dir = cfg.raw_root / cfg.groups[group]
    out: list[tuple[TargetPlan, SequenceData]] = []
    for seq in list_sequences(split_dir):
        data = load_sequence(split_dir, seq, cfg.category_map)
        out.extend((p, data) for p in plan_candidates(data, cfg))
    return out


def clip_dir_for(cfg: ScenarioConfig, group: str, plan: TargetPlan) -> Path:
    assert cfg.screening_work_root is not None
    return cfg.screening_work_root / group / candidate_id(plan.sequence, plan.target_id)


def tracks_log_for(clip_dir: Path, detector: str) -> Path:
    return clip_dir / "logs" / detector / "tracks.jsonl"


def _clip_info(plan: TargetPlan, cfg: ScenarioConfig) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(plan.sequence, plan.target_id),
        "sequence": plan.sequence,
        "gt_target_id": plan.target_id,
        "occlusion_start": plan.occlusion_start,
        "first_source_frame": plan.occlusion_start - cfg.pre_occlusion_frames,
        "frames": cfg.pre_occlusion_frames,
    }


def ensure_screening_clip(data: SequenceData, plan: TargetPlan, cfg: ScenarioConfig, clip_dir: Path) -> None:
    """Ghi clip sach (neu chua co). Da co ma khac ung vien/vi tri -> loi (xu ly tay)."""
    info = _clip_info(plan, cfg)
    info_path = clip_dir / CLIP_INFO_NAME
    if clip_dir.exists():
        existing = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else None
        if existing != info:
            raise ScreeningError(f"{clip_dir} da ton tai voi noi dung khac (hoac thieu {CLIP_INFO_NAME}) - xu ly tay")
        return
    images_dir = clip_dir / IMAGES_DIR
    images_dir.mkdir(parents=True)
    first = info["first_source_frame"]
    rows_by_frame: dict[int, list] = {}
    for row in data.rows:
        rows_by_frame.setdefault(row.frame_idx, []).append(row)
    lines = []
    for k, src_frame in enumerate(range(first, plan.occlusion_start)):
        src = data.image_paths[src_frame]
        _link_or_copy(src, images_dir / f"{k + 1:07d}{src.suffix.lower()}")
        for row in sorted(rows_by_frame.get(src_frame, []), key=lambda r: (r.target_id, r.left, r.top)):
            lines.append(_annotation_line(row, k + 1))
    (clip_dir / ANNOTATION_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def screen_candidate(
    plan: TargetPlan,
    rank: int,
    cfg: ScenarioConfig,
    clip_dir: Path,
    detector_records: Mapping[str, Sequence[Mapping[str, Any]]],
    detector_tracks_sha256: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Ket qua screening cua mot ung vien tu log track cua tung detector."""
    assert cfg.target_lock is not None
    gt_boxes = load_target_boxes(clip_dir / ANNOTATION_NAME, plan.target_id, cfg.category_map)
    last = gt_boxes[max(gt_boxes)]  # bbox GT o frame cuoi truoc khi che
    entry = {
        **base_entry(plan, rank, cfg),
        "gt_box_wh": [last.width, last.height],
        "min_side_px": min(last.width, last.height),
        "status": "screened",
        "detectors": {},
    }
    for name in cfg.required_detectors:
        lock = evaluate_target_lock(
            detector_records[name], gt_boxes, cfg.target_lock, cfg.pre_occlusion_frames,
            first_frame=1, gt_target_id=plan.target_id,
        )
        sel = lock["selection"]
        entry["detectors"][name] = {
            "status": lock["status"],
            "selected_track_id": sel["selected_track_id"],
            "selected_frame_idx": sel["selected_frame_idx"],
            "selected_iou": sel["selected_iou"],
            "selected_class": lock["selected_class"],
            "class_matches_gt": (lock["selected_class"] == plan.cls) if lock["selected_class"] is not None else None,
            "lock_ious": lock["lock_ious"],
            "best_iou": lock["best_iou"],
            "frames_with_match": lock["frames_with_match"],
            "frames_with_gt": lock["frames_with_gt"],
            "tracks_sha256": (detector_tracks_sha256 or {}).get(name),
        }
    entry["eligible"] = candidate_eligible(entry, cfg.required_detectors)
    return entry


def base_entry(plan: TargetPlan, rank: int, cfg: ScenarioConfig) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id(plan.sequence, plan.target_id),
        "sequence": plan.sequence,
        "gt_target_id": plan.target_id,
        "rank": rank,
        "occlusion_start": plan.occlusion_start,
        "first_source_frame": plan.occlusion_start - cfg.pre_occlusion_frames,
        "class": plan.cls,
        "category": plan.category,
    }


# ---------------- tong hop ----------------


def size_bin(value: float, edges: Sequence[float]) -> str:
    if not edges or value < edges[0]:
        return f"<{edges[0]:g}" if edges else "all"
    for lo, hi in zip(edges, list(edges[1:]) + [None]):
        if hi is None or value < hi:
            return f"[{lo:g},{hi:g})" if hi is not None else f">={lo:g}"
    return "all"  # khong toi day


def _distribution(entries: Sequence[Mapping[str, Any]], edges: Sequence[float]) -> dict[str, Any]:
    classes: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for e in entries:
        classes[str(e["class"])] = classes.get(str(e["class"]), 0) + 1
        b = size_bin(float(e["min_side_px"]), edges)
        sizes[b] = sizes.get(b, 0) + 1
    sides = [float(e["min_side_px"]) for e in entries]
    return {
        "count": len(entries),
        "by_class": dict(sorted(classes.items(), key=lambda kv: int(kv[0]))),
        "by_min_side_px": sizes,
        "median_min_side_px": statistics.median(sides) if sides else None,
    }


def summarize(entries: Sequence[Mapping[str, Any]], cfg: ScenarioConfig) -> dict[str, Any]:
    """Ti le dat theo detector / sequence, phan bo lop + kich thuoc ung vien vs dat."""
    screened = [e for e in entries if e.get("status") == "screened"]
    eligible = [e for e in screened if e["eligible"]]
    per_detector: dict[str, Any] = {}
    for name in cfg.required_detectors:
        statuses: dict[str, int] = {}
        for e in screened:
            s = e["detectors"][name]["status"]
            statuses[s] = statuses.get(s, 0) + 1
        locked = statuses.get(LOCK_LOCKED, 0)
        per_detector[name] = {
            "screened": len(screened),
            "locked": locked,
            "pass_rate": locked / len(screened) if screened else None,
            "by_status": dict(sorted(statuses.items())),
            "locked_with_other_class": sum(
                1 for e in screened
                if e["detectors"][name]["status"] == LOCK_LOCKED and e["detectors"][name]["class_matches_gt"] is False
            ),
        }
    per_sequence: dict[str, Any] = {}
    for seq in sorted({e["sequence"] for e in entries}):
        seq_entries = [e for e in entries if e["sequence"] == seq]
        seq_screened = [e for e in seq_entries if e.get("status") == "screened"]
        row: dict[str, Any] = {
            "candidates": len(seq_entries),
            "screened": len(seq_screened),
            "eligible": sum(1 for e in seq_screened if e["eligible"]),
        }
        for name in cfg.required_detectors:
            n_locked = sum(1 for e in seq_screened if e["detectors"][name]["status"] == LOCK_LOCKED)
            row[f"{name}_locked"] = n_locked
            row[f"{name}_pass_rate"] = n_locked / len(seq_screened) if seq_screened else None
        row["selected"] = len(select_screened_targets(
            [_plan_from_entry(e) for e in seq_screened],
            {e["candidate_id"]: e for e in seq_screened}, cfg,
        ))
        per_sequence[seq] = row
    return {
        "candidates": len(entries),
        "screened": len(screened),
        "eligible": len(eligible),
        "eligible_rate": len(eligible) / len(screened) if screened else None,
        "selected_targets": sum(r["selected"] for r in per_sequence.values()),
        "per_detector": per_detector,
        "per_sequence": per_sequence,
        "candidate_distribution": _distribution(screened, cfg.size_bins_px),
        "eligible_distribution": _distribution(eligible, cfg.size_bins_px),
        "selection_bias_note": (
            "Muc tieu dat la muc tieu MOI detector khoa duoc truoc khi che: nghieng ve vat lon, "
            "tuong phan cao, lop pho bien. Ket qua occlusion chi co dieu kien tren tap nay."
        ),
    }


def _plan_from_entry(e: Mapping[str, Any]) -> TargetPlan:
    return TargetPlan(e["sequence"], int(e["gt_target_id"]), int(e["occlusion_start"]),
                      int(e["class"]), int(e["category"]))


def build_screening(
    cfg: ScenarioConfig,
    group: str,
    entries: Sequence[Mapping[str, Any]],
    detectors: Mapping[str, Mapping[str, Any]],
    *,
    limit: Optional[int],
    errors: Sequence[str],
    git: Optional[Mapping[str, Any]] = None,
    versions: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    assert cfg.target_lock is not None
    not_run = sum(1 for e in entries if e.get("status") != "screened")
    return {
        "kind": SCREENING_KIND,
        "group": group,
        "split_dir": cfg.groups[group],
        "config_version": cfg.version,
        "config_sha256": cfg.config_sha256,
        "seed": cfg.seed,
        "target_lock": {**cfg.target_lock.as_dict(), "required_detectors": list(cfg.required_detectors)},
        "pre_occlusion_frames": cfg.pre_occlusion_frames,
        "candidates_per_sequence": cfg.candidates_per_sequence,
        "targets_per_sequence": cfg.targets_per_sequence,
        "detectors": {k: dict(v) for k, v in sorted(detectors.items())},
        # partial: chay --limit hoac co ung vien chua screening xong -> generator tu choi.
        "partial": bool(limit is not None or not_run),
        "limit": limit,
        "errors": list(errors),
        "summary": summarize(entries, cfg),
        "candidates": list(entries),
        "git": dict(git or {}),
        "versions": dict(versions or {}),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


__all__ = [
    "plan_group_candidates", "clip_dir_for", "tracks_log_for", "ensure_screening_clip",
    "screen_candidate", "base_entry", "summarize", "build_screening", "size_bin",
    "CLIP_INFO_NAME", "WEIGHTS_SIDECAR_NAME",
]
