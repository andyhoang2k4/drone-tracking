"""Sinh kich ban occlusion nhan tao co ground truth (roadmap buoc 4).

Doc configs/occlusion_scenarios.yaml (quy tac + seed co dinh), thu muc anh va
annotation cua sequence VisDrone2019-MOT, ghi ra moi kich ban::

    <output_root>/<group>/<scenario>/
        images/0000001.jpg ...   anh cua so kich ban (frame bi che ghi lai,
                                 frame khac hardlink/copy tu anh goc)
        annotations.txt          annotation VisDrone cua cua so, danh so lai tu 1
        manifest.json            GT bbox muc tieu MOI frame, khoang che, kieu che,
                                 seed, sequence nguon
    <output_root>/<group>/index.json

Xac dinh (deterministic): cung config + seed -> cung muc tieu, cung vi tri,
cung anh. RNG duoc seed bang chuoi (seed, sequence, target, ...) qua SHA-256,
khong dung hash() cua Python (bi salt moi tien trinh).

Chi lam viec tren bbox; vat che la hinh chu nhat bam theo bbox GT.
Han che: vat che nhan tao KHONG giong che khuat that (cay, cong trinh).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from src.motion_memory import BBox
from src.utils.gt_target import LockRule
from src.utils.visdrone import (
    VisDroneBox,
    frame_index_from_path,
    is_ignore_region,
    read_annotation_file,
)

PathLike = Union[str, Path]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.json"
ANNOTATION_NAME = "annotations.txt"
IMAGES_DIR = "images"
COVERAGES = {"full", "partial"}
TEXTURES = {"solid", "noise"}


class ScenarioConfigError(ValueError):
    """Config kich ban khong hop le."""


# ---------------- config ----------------


@dataclass(frozen=True)
class OccluderSpec:
    name: str
    coverage: str
    texture: str


@dataclass(frozen=True)
class ScenarioConfig:
    version: int
    seed: int
    raw_root: Path
    output_root: Path
    category_map: Mapping[int, int]
    groups: Mapping[str, str]  # ten nhom -> split_dir
    classes: frozenset[int]
    min_box_side_px: float
    max_gt_truncation: int
    max_gt_occlusion: int
    max_ignore_ioa: float
    pre_occlusion_frames: int
    post_occlusion_frames: int
    targets_per_sequence: int
    durations_frames: tuple[int, ...]
    occluders: tuple[OccluderSpec, ...]
    margin_ratio: float
    partial_fraction: float
    solid_color_bgr: tuple[int, int, int]
    image_format: str
    jpeg_quality: int
    config_sha256: str = ""
    # --- v2 (configs/occlusion_scenarios_v2.yaml); None/rong = hanh vi v1 ---
    candidates_per_sequence: Optional[int] = None
    target_lock: Optional[LockRule] = None
    required_detectors: tuple[str, ...] = ()
    screening_work_root: Optional[Path] = None
    size_bins_px: tuple[float, ...] = ()

    @property
    def max_duration(self) -> int:
        return max(self.durations_frames)

    @property
    def uses_screening(self) -> bool:
        return self.target_lock is not None


def _resolve(repo_root: Path, value: PathLike) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def config_sha256(raw_bytes: bytes) -> str:
    """SHA-256 cua noi dung config sau khi bo ky tu CR: cung noi dung tren
    Windows (CRLF) va Linux (LF) cho cung hash."""
    return hashlib.sha256(raw_bytes.replace(b"\r", b"")).hexdigest()


def load_scenario_config(path: PathLike, repo_root: Optional[PathLike] = None) -> ScenarioConfig:
    import yaml

    path = Path(path)
    raw_bytes = path.read_bytes()
    data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    try:
        dataset = yaml.safe_load(
            _resolve(root, data["dataset_config"]).read_text(encoding="utf-8")
        )
        rules = data["target_rules"]
        params = data["occluder_params"]
        image = data["image"]
        groups = {name: str(g["split_dir"]) for name, g in data["groups"].items()}
        lock_data = data.get("target_lock")
        screening = data.get("screening") or {}
        v2: dict[str, Any] = {}
        if lock_data is not None:
            try:
                rule = LockRule.from_mapping(lock_data)
            except ValueError as exc:
                raise ScenarioConfigError(f"{path}: {exc}") from exc
            v2 = {
                "candidates_per_sequence": int(rules["candidates_per_sequence"]),
                "target_lock": rule,
                "required_detectors": tuple(str(d) for d in lock_data["required_detectors"]),
                "screening_work_root": _resolve(root, screening["work_root"]),
                "size_bins_px": tuple(float(b) for b in screening.get("size_bins_px", [])),
            }
        elif "candidates_per_sequence" in rules or screening:
            raise ScenarioConfigError(f"{path}: candidates_per_sequence/screening can khoi target_lock")
        cfg = ScenarioConfig(
            version=int(data["version"]),
            seed=int(data["seed"]),
            raw_root=_resolve(root, data["raw_root"]),
            output_root=_resolve(root, data["output_root"]),
            category_map={int(k): int(v) for k, v in dataset["category_map"].items()},
            groups=groups,
            classes=frozenset(int(c) for c in rules["classes"]),
            min_box_side_px=float(rules["min_box_side_px"]),
            max_gt_truncation=int(rules["max_gt_truncation"]),
            max_gt_occlusion=int(rules["max_gt_occlusion"]),
            max_ignore_ioa=float(rules["max_ignore_ioa"]),
            pre_occlusion_frames=int(rules["pre_occlusion_frames"]),
            post_occlusion_frames=int(rules["post_occlusion_frames"]),
            targets_per_sequence=int(rules["targets_per_sequence"]),
            durations_frames=tuple(int(d) for d in data["durations_frames"]),
            occluders=tuple(
                OccluderSpec(str(o["name"]), str(o["coverage"]), str(o["texture"]))
                for o in data["occluders"]
            ),
            margin_ratio=float(params["margin_ratio"]),
            partial_fraction=float(params["partial_fraction"]),
            solid_color_bgr=tuple(int(c) for c in params["solid_color_bgr"]),  # type: ignore[arg-type]
            image_format=str(image["format"]).lower(),
            jpeg_quality=int(image.get("jpeg_quality", 95)),
            config_sha256=config_sha256(raw_bytes),
            **v2,
        )
    except (KeyError, TypeError) as exc:
        raise ScenarioConfigError(f"{path}: thieu/sai khoa config: {exc}") from exc
    if data.get("start_policy") != "seeded_shared":
        raise ScenarioConfigError(f"{path}: start_policy chi ho tro 'seeded_shared'")
    _validate(cfg, path)
    return cfg


def _validate(cfg: ScenarioConfig, path: Path) -> None:
    def fail(msg: str) -> None:
        raise ScenarioConfigError(f"{path}: {msg}")

    if not cfg.groups:
        fail("groups rong")
    split_dirs = list(cfg.groups.values())
    if len(set(split_dirs)) != len(split_dirs):
        fail("hai nhom dung chung split_dir -> co the trung sequence")
    if not cfg.durations_frames or min(cfg.durations_frames) < 1:
        fail("durations_frames phai co gia tri >= 1")
    if len(set(cfg.durations_frames)) != len(cfg.durations_frames):
        fail("durations_frames trung lap")
    if cfg.pre_occlusion_frames < 1 or cfg.post_occlusion_frames < 1:
        fail("pre/post_occlusion_frames phai >= 1")
    if cfg.targets_per_sequence < 1:
        fail("targets_per_sequence phai >= 1")
    if not cfg.classes <= set(cfg.category_map.values()):
        fail("classes ngoai category_map")
    names = [o.name for o in cfg.occluders]
    if not names or len(set(names)) != len(names):
        fail("occluders rong hoac trung ten")
    for o in cfg.occluders:
        if o.coverage not in COVERAGES or o.texture not in TEXTURES:
            fail(f"occluder {o.name}: coverage {COVERAGES}, texture {TEXTURES}")
    if not 0.0 < cfg.partial_fraction < 1.0:
        fail("partial_fraction phai trong (0, 1)")
    if cfg.margin_ratio < 0.0:
        fail("margin_ratio phai >= 0")
    if len(cfg.solid_color_bgr) != 3 or not all(0 <= c <= 255 for c in cfg.solid_color_bgr):
        fail("solid_color_bgr phai la 3 so 0..255")
    if cfg.image_format not in {"jpg", "png"}:
        fail("image.format chi ho tro jpg|png")
    if not 0 <= cfg.max_ignore_ioa <= 1:
        fail("max_ignore_ioa phai trong [0, 1]")
    if cfg.uses_screening:
        if cfg.version < 2:
            fail("target_lock chi dung tu version 2 (giu nguyen hanh vi v1)")
        if cfg.candidates_per_sequence is None or cfg.candidates_per_sequence < cfg.targets_per_sequence:
            fail("candidates_per_sequence phai >= targets_per_sequence")
        dets = cfg.required_detectors
        if not dets or len(set(dets)) != len(dets):
            fail("target_lock.required_detectors rong hoac trung ten")
        assert cfg.target_lock is not None
        if cfg.target_lock.lock_confirm_frames > cfg.pre_occlusion_frames:
            fail("lock_confirm_frames phai <= pre_occlusion_frames")
        if list(cfg.size_bins_px) != sorted(set(cfg.size_bins_px)):
            fail("screening.size_bins_px phai tang dan, khong trung")
        if cfg.output_root == cfg.screening_work_root:
            fail("screening.work_root phai khac output_root")


# ---------------- RNG xac dinh ----------------


def derived_seed(*parts: Any) -> int:
    """So nguyen 64-bit xac dinh tu cac thanh phan (SHA-256, khong dung hash())."""
    text = ":".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def derived_rng(*parts: Any) -> random.Random:
    return random.Random(derived_seed(*parts))


# ---------------- chon muc tieu ----------------


def _box_of(row: VisDroneBox) -> BBox:
    return BBox(float(row.left), float(row.top), float(row.left + row.width),
                float(row.top + row.height))


def _ioa(box: BBox, region: BBox) -> float:
    iw = max(0.0, min(box.x2, region.x2) - max(box.x1, region.x1))
    ih = max(0.0, min(box.y2, region.y2) - max(box.y1, region.y1))
    area = box.width * box.height
    return iw * ih / area if area > 0.0 else 0.0


@dataclass
class SequenceData:
    name: str
    image_paths: dict[int, Path]                       # frame VisDrone -> anh
    rows: list[VisDroneBox]
    targets: dict[int, dict[int, VisDroneBox]] = field(default_factory=dict)  # tid -> frame -> row
    ignore_regions: dict[int, list[BBox]] = field(default_factory=dict)


def load_sequence(split_dir: Path, sequence: str, category_map: Mapping[int, int]) -> SequenceData:
    seq_dir = split_dir / "sequences" / sequence
    annotation = split_dir / "annotations" / f"{sequence}.txt"
    if not seq_dir.is_dir() or not annotation.is_file():
        raise FileNotFoundError(f"Thieu thu muc anh hoac annotation cua {sequence} trong {split_dir}")
    images = {
        frame_index_from_path(p): p
        for p in sorted(seq_dir.iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES
    }
    rows = read_annotation_file(annotation)
    data = SequenceData(name=sequence, image_paths=images, rows=rows)
    for row in rows:
        if is_ignore_region(row, category_map):
            data.ignore_regions.setdefault(row.frame_idx, []).append(_box_of(row))
            continue
        frames = data.targets.setdefault(row.target_id, {})
        if row.frame_idx in frames:
            raise ValueError(f"{annotation}: target {row.target_id} co 2 box o frame {row.frame_idx}")
        frames[row.frame_idx] = row
    return data


def valid_frames(data: SequenceData, target_id: int, cfg: ScenarioConfig) -> set[int]:
    """Cac frame ma muc tieu thoa moi quy tac target_rules (va co anh)."""
    out: set[int] = set()
    for frame, row in data.targets.get(target_id, {}).items():
        if frame not in data.image_paths:
            continue
        if cfg.category_map[row.category] not in cfg.classes:
            continue
        if min(row.width, row.height) < cfg.min_box_side_px:
            continue
        if row.truncation > cfg.max_gt_truncation or row.occlusion > cfg.max_gt_occlusion:
            continue
        box = _box_of(row)
        if any(_ioa(box, r) > cfg.max_ignore_ioa for r in data.ignore_regions.get(frame, [])):
            continue
        out.add(frame)
    return out


def valid_starts(frames: set[int], cfg: ScenarioConfig, duration: int) -> list[int]:
    """Frame bat dau che s sao cho moi frame [s - pre, s + duration - 1 + post] hop le."""
    starts = []
    for s in sorted(frames):
        lo = s - cfg.pre_occlusion_frames
        hi = s + duration - 1 + cfg.post_occlusion_frames
        if all(f in frames for f in range(lo, hi + 1)):
            starts.append(s)
    return starts


@dataclass(frozen=True)
class TargetPlan:
    sequence: str
    target_id: int
    occlusion_start: int  # frame VisDrone
    cls: int
    category: int


def plan_sequence(data: SequenceData, cfg: ScenarioConfig) -> list[TargetPlan]:
    """v1: chon muc tieu + vi tri bat dau che (xac dinh theo seed), chi theo GT."""
    eligible = _eligible_starts(data, cfg)
    if not eligible:
        return []
    ids = sorted(eligible)
    chosen = sorted(
        derived_rng(cfg.seed, data.name, "targets").sample(ids, min(cfg.targets_per_sequence, len(ids)))
    )
    return [_target_plan(data, cfg, tid, eligible[tid]) for tid in chosen]


def _eligible_starts(data: SequenceData, cfg: ScenarioConfig) -> dict[int, list[int]]:
    eligible: dict[int, list[int]] = {}
    for tid in sorted(data.targets):
        starts = valid_starts(valid_frames(data, tid, cfg), cfg, cfg.max_duration)
        if starts:
            eligible[tid] = starts
    return eligible


def _target_plan(data: SequenceData, cfg: ScenarioConfig, tid: int, starts: list[int]) -> TargetPlan:
    """Vi tri bat dau che: cung cong thuc seed voi v1 (seed, sequence, target, 'start')."""
    start = derived_rng(cfg.seed, data.name, tid, "start").choice(starts)
    row = data.targets[tid][start]
    return TargetPlan(data.name, tid, start, cfg.category_map[row.category], row.category)


# ---------------- v2: ung vien + screening ----------------

SCREENING_NAME = "screening.json"
SCREENING_KIND = "occlusion_target_screening"


def candidate_id(sequence: str, target_id: int) -> str:
    return f"{sequence}__t{target_id}"


def plan_candidates(data: SequenceData, cfg: ScenarioConfig) -> list[TargetPlan]:
    """v2: ung vien screening cua mot sequence, THEO THU TU XEP HANG co seed.

    Thu tu = tron (seed, sequence, 'candidates') tren ID muc tieu hop le da sap
    xep; lay candidates_per_sequence dau. Vi tri che giong cong thuc v1.
    """
    if not cfg.uses_screening or cfg.candidates_per_sequence is None:
        raise ScenarioConfigError("plan_candidates chi dung cho config co target_lock (v2)")
    eligible = _eligible_starts(data, cfg)
    order = sorted(eligible)
    derived_rng(cfg.seed, data.name, "candidates").shuffle(order)
    return [_target_plan(data, cfg, tid, eligible[tid]) for tid in order[: cfg.candidates_per_sequence]]


def file_sha256(path: PathLike, chunk: int = 1 << 20) -> str:
    """SHA-256 noi dung file (weights) - doc tung khoi, khong nap ca file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ScreeningError(ValueError):
    """screening.json khong dung duoc de sinh kich ban v2."""


def screening_path(cfg: ScenarioConfig, group: str) -> Path:
    return cfg.output_root / group / SCREENING_NAME


def candidate_eligible(entry: Mapping[str, Any], required: Sequence[str]) -> bool:
    """Dat <=> MOI detector bat buoc deu 'locked'."""
    dets = entry.get("detectors") or {}
    return all((dets.get(name) or {}).get("status") == "locked" for name in required)


def validate_screening(
    screening: Mapping[str, Any],
    cfg: ScenarioConfig,
    group: str,
    candidates: Mapping[str, Sequence[TargetPlan]],
    weights_sha256: Mapping[str, str],
) -> None:
    """Tu choi screening partial / co loi / khac config_sha256 / khac danh sach
    ung vien / khac weights sha256 (weights_sha256: tinh tu --detector luc sinh)."""
    problems: list[str] = []
    if screening.get("kind") != SCREENING_KIND:
        problems.append(f"kind={screening.get('kind')!r}, mong doi {SCREENING_KIND!r}")
    if screening.get("group") != group:
        problems.append(f"group={screening.get('group')!r} khac {group!r}")
    if screening.get("config_sha256") != cfg.config_sha256:
        problems.append("config_sha256 khac config hien tai (config da doi sau screening?)")
    if screening.get("partial"):
        problems.append("screening la partial (--limit) - chay screening day du truoc")
    if screening.get("errors"):
        problems.append(f"screening co {len(screening['errors'])} loi (vd {screening['errors'][0]})")
    dets = screening.get("detectors") or {}
    if sorted(dets) != sorted(cfg.required_detectors):
        problems.append(f"detector screening {sorted(dets)} khac required_detectors {sorted(cfg.required_detectors)}")
    if sorted(weights_sha256) != sorted(cfg.required_detectors):
        problems.append(f"can --detector cho dung {sorted(cfg.required_detectors)} de kiem tra weights sha256")
    for name, sha in weights_sha256.items():
        recorded = (dets.get(name) or {}).get("weights_sha256")
        if recorded != sha:
            problems.append(f"weights sha256 cua {name} khac screening ({str(sha)[:12]} vs {str(recorded)[:12]})")
    expected = [
        (candidate_id(p.sequence, p.target_id), p.occlusion_start)
        for seq in sorted(candidates) for p in candidates[seq]
    ]
    entries = screening.get("candidates") or []
    found = [(e.get("candidate_id"), e.get("occlusion_start")) for e in entries]
    if found != expected:
        problems.append(f"danh sach ung vien khac ke hoach hien tai ({len(found)} vs {len(expected)} ung vien)")
    for e in entries:
        if e.get("status") != "screened":
            problems.append(f"{e.get('candidate_id')}: status={e.get('status')!r} (chua screening xong)")
            break
        if bool(e.get("eligible")) != candidate_eligible(e, cfg.required_detectors):
            problems.append(f"{e.get('candidate_id')}: eligible khong khop trang thai detector")
            break
    if problems:
        raise ScreeningError("screening.json khong dung duoc:\n  " + "\n  ".join(problems))


def select_screened_targets(
    candidates: Sequence[TargetPlan],
    screening_by_id: Mapping[str, Mapping[str, Any]],
    cfg: ScenarioConfig,
) -> list[TargetPlan]:
    """Toi da targets_per_sequence ung vien DAT dau tien theo thu tu xep hang."""
    passed = [
        p for p in candidates
        if candidate_eligible(screening_by_id[candidate_id(p.sequence, p.target_id)], cfg.required_detectors)
    ]
    return passed[: cfg.targets_per_sequence]


# ---------------- vat che ----------------


def occluder_box(
    target: BBox, spec: OccluderSpec, cfg: ScenarioConfig, side: str, width: int, height: int
) -> BBox:
    """Hinh chu nhat vat che (da cat theo bien anh)."""
    mx = cfg.margin_ratio * target.width
    my = cfg.margin_ratio * target.height
    y1, y2 = target.y1 - my, target.y2 + my
    if spec.coverage == "full":
        x1, x2 = target.x1 - mx, target.x2 + mx
    elif side == "left":
        x1, x2 = target.x1 - mx, target.x1 + cfg.partial_fraction * target.width
    else:
        x1, x2 = target.x2 - cfg.partial_fraction * target.width, target.x2 + mx
    return BBox(max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2))


def _pixel_rect(box: BBox) -> tuple[int, int, int, int]:
    """Pixel phu boi box: [floor(x1), ceil(x2)) x [floor(y1), ceil(y2))."""
    import math

    return math.floor(box.x1), math.floor(box.y1), math.ceil(box.x2), math.ceil(box.y2)


def draw_occluder(image, box: BBox, spec: OccluderSpec, cfg: ScenarioConfig, rng) -> None:
    """Ve vat che len anh (numpy HxWx3 BGR) tai cho."""
    x1, y1, x2, y2 = _pixel_rect(box)
    if x2 <= x1 or y2 <= y1:
        return
    if spec.texture == "solid":
        image[y1:y2, x1:x2] = cfg.solid_color_bgr
    else:
        image[y1:y2, x1:x2] = rng.integers(0, 256, size=(y2 - y1, x2 - x1, 3), dtype="uint8")


# ---------------- ghi kich ban ----------------


def scenario_name(sequence: str, target_id: int, duration: int, occluder: str) -> str:
    return f"{sequence}__t{target_id}__d{duration:03d}__{occluder}"


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _annotation_line(row: VisDroneBox, frame: int) -> str:
    return (f"{frame},{row.target_id},{row.left},{row.top},{row.width},{row.height},"
            f"{row.score},{row.category},{row.truncation},{row.occlusion}")


def write_scenario(
    data: SequenceData,
    plan: TargetPlan,
    duration: int,
    spec: OccluderSpec,
    cfg: ScenarioConfig,
    group: str,
    out_dir: Path,
    extra_manifest: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Ghi mot kich ban ra out_dir (phai chua ton tai). Tra ve manifest.

    extra_manifest (v2): them target_lock + nguon goc screening vao manifest.
    """
    import cv2
    import numpy as np

    name = scenario_name(plan.sequence, plan.target_id, duration, spec.name)
    first_src = plan.occlusion_start - cfg.pre_occlusion_frames
    last_src = plan.occlusion_start + duration - 1 + cfg.post_occlusion_frames
    occ_lo, occ_hi = plan.occlusion_start, plan.occlusion_start + duration - 1
    side = derived_rng(cfg.seed, name, "side").choice(["left", "right"])
    noise_rng = np.random.default_rng(derived_seed(cfg.seed, name, "noise"))

    images_dir = out_dir / IMAGES_DIR
    images_dir.mkdir(parents=True)
    target_rows = data.targets[plan.target_id]
    rows_by_frame: dict[int, list[VisDroneBox]] = {}
    for row in data.rows:
        rows_by_frame.setdefault(row.frame_idx, []).append(row)

    ground_truth = []
    annotation_lines = []
    for k, src_frame in enumerate(range(first_src, last_src + 1)):
        new_frame = k + 1
        src_path = data.image_paths[src_frame]
        target = _box_of(target_rows[src_frame])
        occluded = occ_lo <= src_frame <= occ_hi
        occ_box: Optional[BBox] = None
        if occluded:
            image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Khong doc duoc anh {src_path}")
            h, w = image.shape[:2]
            occ_box = occluder_box(target, spec, cfg, side, w, h)
            draw_occluder(image, occ_box, spec, cfg, noise_rng)
            dst = images_dir / f"{new_frame:07d}.{cfg.image_format}"
            params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality] if cfg.image_format == "jpg" else []
            if not cv2.imwrite(str(dst), image, params):
                raise OSError(f"Khong ghi duoc anh {dst}")
        else:
            _link_or_copy(src_path, images_dir / f"{new_frame:07d}{src_path.suffix.lower()}")
        ground_truth.append({
            "frame_idx": k,
            "source_frame": src_frame,
            "bbox": list(target.as_tuple()),
            "occluded": occluded,
            "occluder_bbox": list(occ_box.as_tuple()) if occ_box else None,
        })
        for row in sorted(rows_by_frame.get(src_frame, []), key=lambda r: (r.target_id, r.left, r.top)):
            annotation_lines.append(_annotation_line(row, new_frame))

    (out_dir / ANNOTATION_NAME).write_text("\n".join(annotation_lines) + "\n", encoding="utf-8")
    manifest = {
        "scenario_id": f"{group}/{name}",
        "name": name,
        "group": group,
        "source": {
            "split_dir": cfg.groups[group],
            "sequence": plan.sequence,
            "first_source_frame": first_src,
            "last_source_frame": last_src,
        },
        "target": {"gt_target_id": plan.target_id, "class": plan.cls, "category": plan.category},
        "frames": len(ground_truth),
        "first_frame": 1,
        "occlusion": {
            "start_frame_idx": occ_lo - first_src,
            "end_frame_idx": occ_hi - first_src,
            "duration_frames": duration,
            "occluder": spec.name,
            "coverage": spec.coverage,
            "texture": spec.texture,
            "partial_side": side if spec.coverage == "partial" else None,
            "margin_ratio": cfg.margin_ratio,
            "partial_fraction": cfg.partial_fraction if spec.coverage == "partial" else None,
        },
        "ground_truth": ground_truth,
        "seed": cfg.seed,
        "config_version": cfg.version,
        "config_sha256": cfg.config_sha256,
        **dict(extra_manifest or {}),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def list_sequences(split_dir: Path) -> list[str]:
    seq_root = split_dir / "sequences"
    if not seq_root.is_dir():
        raise FileNotFoundError(f"Khong thay {seq_root}")
    return sorted(p.name for p in seq_root.iterdir() if p.is_dir())


def check_group_disjoint(cfg: ScenarioConfig) -> None:
    """Hai nhom khong duoc chung sequence (chi kiem tra nhom co thu muc tren dia)."""
    seen: dict[str, str] = {}
    for group, split_dir in cfg.groups.items():
        path = cfg.raw_root / split_dir / "sequences"
        if not path.is_dir():
            continue
        for seq in list_sequences(cfg.raw_root / split_dir):
            if seq in seen:
                raise ScenarioConfigError(f"sequence {seq} co o ca nhom {seen[seq]} va {group}")
            seen[seq] = group


def generate_group(
    cfg: ScenarioConfig,
    group: str,
    *,
    overwrite: bool = False,
    max_sequences: Optional[int] = None,
    dry_run: bool = False,
    screening_file: Optional[PathLike] = None,
    weights_sha256: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Sinh moi kich ban cua mot nhom. Tra ve index (va ghi index.json neu khong dry_run).

    Kich ban da ton tai voi cung config_sha256 -> bo qua (khong ghi lai).
    Khac config -> loi, tru khi overwrite=True (xoa thu muc kich ban do roi sinh lai).

    v2 (config co target_lock): bat buoc screening.json day du (mac dinh
    <output_root>/<group>/screening.json) khop config_sha256, danh sach ung vien
    va weights_sha256 ({detector: sha256} tinh tu weights luc sinh); chi sinh
    kich ban cho toi da targets_per_sequence ung vien DAT dau tien moi sequence.
    """
    if group not in cfg.groups:
        raise ScenarioConfigError(f"Nhom {group!r} khong co trong config (co: {sorted(cfg.groups)})")
    check_group_disjoint(cfg)
    split_dir = cfg.raw_root / cfg.groups[group]
    all_sequences = list_sequences(split_dir)
    sequences = all_sequences
    partial = max_sequences is not None and max_sequences < len(sequences)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
    group_dir = cfg.output_root / group

    seq_data: dict[str, SequenceData] = {}
    screening_info: Optional[dict[str, Any]] = None
    screening_by_id: dict[str, Mapping[str, Any]] = {}
    candidates: dict[str, list[TargetPlan]] = {}
    if cfg.uses_screening:
        # Kiem tra screening tren TOAN BO nhom (khong chi cac sequence --max-sequences).
        path = Path(screening_file) if screening_file else screening_path(cfg, group)
        if not path.is_file():
            raise ScreeningError(
                f"Khong thay {path} - chay scripts/screen_occlusion_targets.py --group {group} truoc"
            )
        raw = path.read_bytes()
        screening = json.loads(raw.decode("utf-8"))
        for seq in all_sequences:
            seq_data[seq] = load_sequence(split_dir, seq, cfg.category_map)
            candidates[seq] = plan_candidates(seq_data[seq], cfg)
        validate_screening(screening, cfg, group, candidates, weights_sha256 or {})
        screening_by_id = {e["candidate_id"]: e for e in screening["candidates"]}
        screening_info = {
            "file": SCREENING_NAME,
            "sha256": config_sha256(raw),
            "detectors": {
                name: {"weights_sha256": d.get("weights_sha256")}
                for name, d in sorted(screening["detectors"].items())
            },
        }

    index: dict[str, Any] = {
        "group": group,
        "split_dir": cfg.groups[group],
        "seed": cfg.seed,
        "config_version": cfg.version,
        "config_sha256": cfg.config_sha256,
        "partial": partial,
        "sequences": sequences,
        "sequences_without_target": [],
        "scenarios": [],
        "skipped_existing": [],
    }
    if cfg.uses_screening:
        assert cfg.target_lock is not None and screening_info is not None
        index.update({
            "target_lock": {**cfg.target_lock.as_dict(), "required_detectors": list(cfg.required_detectors)},
            "screening": screening_info,
            "candidates_per_sequence": {},
            "eligible_per_sequence": {},
            "selected_targets": {},
        })
    for seq in sequences:
        data = seq_data.get(seq) or load_sequence(split_dir, seq, cfg.category_map)
        extra: dict[int, dict[str, Any]] = {}
        if cfg.uses_screening:
            assert cfg.target_lock is not None and screening_info is not None
            cands = candidates[seq]
            plans = select_screened_targets(cands, screening_by_id, cfg)
            index["candidates_per_sequence"][seq] = len(cands)
            index["eligible_per_sequence"][seq] = sum(
                1 for p in cands
                if candidate_eligible(screening_by_id[candidate_id(seq, p.target_id)], cfg.required_detectors)
            )
            index["selected_targets"][seq] = [p.target_id for p in plans]
            rank = {p.target_id: k for k, p in enumerate(cands)}
            for p in plans:
                extra[p.target_id] = {
                    "target_lock": {
                        **cfg.target_lock.as_dict(),
                        "required_detectors": list(cfg.required_detectors),
                        "occlusion_start_frame_idx": cfg.pre_occlusion_frames,
                    },
                    "screening": {**screening_info, "candidate_id": candidate_id(seq, p.target_id),
                                  "candidate_rank": rank[p.target_id]},
                }
        else:
            plans = plan_sequence(data, cfg)
        if not plans:
            index["sequences_without_target"].append(seq)
            continue
        for plan in plans:
            for duration in sorted(cfg.durations_frames):
                for spec in cfg.occluders:
                    name = scenario_name(seq, plan.target_id, duration, spec.name)
                    index["scenarios"].append(name)
                    if dry_run:
                        continue
                    out_dir = group_dir / name
                    if out_dir.exists():
                        existing = out_dir / MANIFEST_NAME
                        same = (
                            existing.is_file()
                            and json.loads(existing.read_text(encoding="utf-8")).get("config_sha256")
                            == cfg.config_sha256
                        )
                        if same and not overwrite:
                            index["skipped_existing"].append(name)
                            continue
                        if not overwrite:
                            raise FileExistsError(
                                f"{out_dir} da ton tai voi config khac (hoac thieu manifest); "
                                "dung --overwrite de sinh lai"
                            )
                        shutil.rmtree(out_dir)
                    write_scenario(data, plan, duration, spec, cfg, group, out_dir,
                                   extra_manifest=extra.get(plan.target_id))
    if not dry_run:
        group_dir.mkdir(parents=True, exist_ok=True)
        # index.json duoc commit: chi ghi phan xac dinh boi config (bo skipped_existing,
        # von thay doi giua cac lan chay lai) de chay lai khong tao diff.
        stored = {k: v for k, v in index.items() if k != "skipped_existing"}
        (group_dir / INDEX_NAME).write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    return index


def load_manifest(scenario_dir: PathLike) -> dict[str, Any]:
    return json.loads((Path(scenario_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def manifest_ground_truth(manifest: Mapping[str, Any]) -> list[BBox]:
    """GT bbox muc tieu theo frame_idx (0..frames-1). Kiem tra du moi frame."""
    rows = sorted(manifest["ground_truth"], key=lambda r: r["frame_idx"])
    if [r["frame_idx"] for r in rows] != list(range(int(manifest["frames"]))):
        raise ValueError(f"{manifest.get('scenario_id')}: ground truth khong du moi frame")
    return [BBox(*r["bbox"]) for r in rows]


def occluded_flags(manifest: Mapping[str, Any]) -> list[bool]:
    rows = sorted(manifest["ground_truth"], key=lambda r: r["frame_idx"])
    return [bool(r["occluded"]) for r in rows]


def scenario_image_names(scenario_dir: PathLike) -> list[str]:
    """Ten file anh cua kich ban, sap xep theo ten (= thu tu frame vi ten co
    so 0 dem dau 7 chu so)."""
    images_dir = Path(scenario_dir) / IMAGES_DIR
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Khong thay {images_dir}")
    return sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def frame_order_problems(
    paths: Sequence[Optional[str]], scenario_dir: PathLike, max_report: int = 5
) -> list[str]:
    """So path anh theo thu tu frame_idx trong log voi danh sach ten anh da
    sap xep cua kich ban. Rong = khop het (dung thu tu, khong thieu, khong thua).

    Kiem tra ten file va thu muc cha (<kich ban>/images) cua tung path; khong
    gia dinh path tuyet doi giong nhau giua cac may.
    """
    scenario_dir = Path(scenario_dir)
    expected = scenario_image_names(scenario_dir)
    problems: list[str] = []
    if len(paths) != len(expected):
        problems.append(f"log co {len(paths)} frame, thu muc anh co {len(expected)} anh")
    want_parent = (scenario_dir.name, IMAGES_DIR)
    bad = 0
    for k, (path, name) in enumerate(zip(paths, expected)):
        if path is None:
            msg = f"frame_idx {k}: log khong co path (mong doi {name})"
        else:
            p = Path(str(path))
            if p.name != name:
                msg = f"frame_idx {k}: log la {p.name}, mong doi {name} (lech thu tu/thieu anh)"
            elif tuple(p.parts[-3:-1]) != want_parent:
                msg = f"frame_idx {k}: anh {path} khong thuoc {'/'.join(want_parent)}"
            else:
                continue
        bad += 1
        if bad <= max_report:
            problems.append(msg)
    if bad > max_report:
        problems.append(f"... tong cong {bad} frame lech")
    return problems


__all__: Sequence[str] = [
    "ScenarioConfig", "ScenarioConfigError", "OccluderSpec", "TargetPlan", "load_scenario_config",
    "load_sequence", "valid_frames", "valid_starts", "plan_sequence", "occluder_box",
    "plan_candidates", "candidate_id", "candidate_eligible", "validate_screening",
    "select_screened_targets", "screening_path", "file_sha256", "ScreeningError",
    "SCREENING_NAME", "SCREENING_KIND",
    "write_scenario", "generate_group", "scenario_name", "derived_seed", "load_manifest",
    "manifest_ground_truth", "occluded_flags", "config_sha256", "scenario_image_names", "frame_order_problems",
]
