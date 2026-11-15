"""Chan doan MO TA (chi dev, chi doc): phan cpr cua Motion Memory den tu box cua track bi "cuop".

Voi moi experiment Motion Memory (vd <det>_botsort_mm_occl_dev_tune_default,
<det>_botsort_mm_occl_dev_anchor_v1_anchor) va moi kich ban DA KHOA (metrics v2):
moi frame bi che co box cua MM duoc phan loai
    lost_extrapolated     state LOST (box ngoai suy)
    tracking_original_id  state TRACKING, track_id = ID goc (target_selection.selected_track_id)
    tracking_other_id     state TRACKING, track_id khac ID goc
    other                 state khac co box (khong mong doi; dem de khong mat frame)
roi dem so frame co box va so frame "dung" (IoU >= correct_iou voi GT, dinh nghia
metrics v2) cua tung loai, gop bang cong count, theo duration. Them: so kich ban ma
ID cua MM o frame a-1 da khac ID goc (bi cuop truoc khi che), so frame theo
recover_reason.

Dung dung ham hien co: src.evaluate (read_pipeline_log, pipeline_outputs, iou,
load_occlusion_metrics_config), src.occlusion_scenarios (load_manifest,
manifest_ground_truth, occluded_flags). GT lay tu manifest.json cua kich ban.
Kiem tra cheo: tong frame co box / frame dung moi kich ban PHAI bang
occluded_frames_with_output / correct_occluded_frames trong metrics.json cua kich
ban (cung dinh nghia) - lech -> loi.

Khong dung de chon cau hinh (chi mo ta). Doc file per-scenario (gitignore, tren cloud)::

    python scripts/diagnose_mm_hijack.py \\
        --experiment yolo11_botsort_mm_occl_dev_tune_default --experiment yolo26_botsort_mm_occl_dev_tune_default \\
        --experiment yolo11_botsort_mm_occl_dev_anchor_v1_anchor --experiment yolo26_botsort_mm_occl_dev_anchor_v1_anchor \\
        --output experiments/motion_memory_anchor_dev/diagnose_hijack.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import evaluate as ev  # noqa: E402
from src import occlusion_scenarios as osc  # noqa: E402

CATEGORIES = ("lost_extrapolated", "tracking_original_id", "tracking_other_id", "other")
KIND = "mm_hijack_diagnosis"


def classify_frame(row: Mapping[str, Any], original_id: Optional[int]) -> str:
    """Loai cua mot frame CO BOX cua MM (theo state + track_id cua log pipeline)."""
    state = row.get("state")
    if state == "LOST":
        return "lost_extrapolated"
    if state == "TRACKING":
        return "tracking_original_id" if row.get("track_id") == original_id else "tracking_other_id"
    return "other"


def _empty() -> dict[str, dict[str, int]]:
    return {c: {"frames_with_box": 0, "correct": 0} for c in CATEGORIES}


def diagnose_scenario(run_dir: Path, scenario_dir: Path, correct_iou: float) -> dict[str, Any]:
    """Phan loai frame bi che co box cua MOT kich ban (MM)."""
    manifest = osc.load_manifest(scenario_dir)
    gt = osc.manifest_ground_truth(manifest)
    occ = osc.occluded_flags(manifest)
    a = occ.index(True)
    b = len(occ) - 1 - occ[::-1].index(True)
    rows = ev.read_pipeline_log(run_dir / ev.PIPELINE_LOG)
    selection = json.loads((run_dir / "target_selection.json").read_text(encoding="utf-8"))
    boxes, _, _ = ev.pipeline_outputs(rows, True, selection)
    original = selection.get("selected_track_id")
    counts = _empty()
    for k in range(a, b + 1):
        if boxes[k] is None:
            continue
        cat = classify_frame(rows[k], original)
        counts[cat]["frames_with_box"] += 1
        counts[cat]["correct"] += int(ev.iou(boxes[k], gt[k]) >= correct_iou)
    before = rows[a - 1].get("track_id") if a > 0 else None
    return {
        "categories": counts,
        "original_id": original,
        "track_id_before_occlusion": before,
        "hijacked_before_occlusion": before is not None and before != original,
        "recover_reason_frames": dict(Counter(r["recover_reason"] for r in rows if r.get("recover_reason"))),
        "duration_frames": manifest["occlusion"]["duration_frames"],
    }


def _add(total: dict, part: Mapping[str, Mapping[str, int]]) -> None:
    for c in CATEGORIES:
        for key in ("frames_with_box", "correct"):
            total[c][key] += part[c][key]


def _finish(cats: dict) -> dict[str, Any]:
    out = {c: dict(v) for c, v in cats.items()}
    with_box = sum(v["frames_with_box"] for v in cats.values())
    correct = sum(v["correct"] for v in cats.values())
    for c in CATEGORIES:
        v = out[c]
        v["share_of_correct"] = v["correct"] / correct if correct else None
        v["precision"] = v["correct"] / v["frames_with_box"] if v["frames_with_box"] else None
    out["total"] = {"frames_with_box": with_box, "correct": correct}
    return out


def diagnose_experiment(exp_dir: Path, scenario_root: Path, correct_iou: float) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    metrics_by_scen = ev.load_scenario_metrics(exp_dir)
    if not metrics_by_scen:
        raise ev.EvaluationError(f"{exp_dir}: khong co metrics.json kich ban nao")
    overall, by_duration = _empty(), {}
    locked = skipped = hijacked = 0
    reasons: Counter = Counter()
    for name, m in sorted(metrics_by_scen.items()):
        run = m.get("run") or {}
        if not run.get("motion_memory"):
            raise ev.EvaluationError(f"{exp_dir.name}/{name}: khong phai lan chay Motion Memory")
        if "target_locked_at_occlusion" not in (m.get("metrics") or {}):
            raise ev.EvaluationError(f"{exp_dir.name}/{name}: metrics khong phai v2 (thieu target_locked_at_occlusion)")
        if not m["metrics"]["target_locked_at_occlusion"]:
            skipped += 1
            continue
        locked += 1
        if float(m["metrics_config"]["correct_iou"]) != correct_iou:
            raise ev.EvaluationError(f"{name}: correct_iou trong metrics kich ban khac metrics config")
        res = diagnose_scenario(exp_dir / ev.SCENARIO_RUNS_DIR / name,
                                Path(scenario_root) / m["scenario"]["group"] / name, correct_iou)
        with_box = sum(v["frames_with_box"] for v in res["categories"].values())
        correct = sum(v["correct"] for v in res["categories"].values())
        if (with_box, correct) != (m["metrics"]["occluded_frames_with_output"], m["metrics"]["correct_occluded_frames"]):
            raise ev.EvaluationError(
                f"{exp_dir.name}/{name}: kiem tra cheo lech (box {with_box} vs {m['metrics']['occluded_frames_with_output']}, "
                f"dung {correct} vs {m['metrics']['correct_occluded_frames']})")
        _add(overall, res["categories"])
        _add(by_duration.setdefault(str(res["duration_frames"]), _empty()), res["categories"])
        hijacked += res["hijacked_before_occlusion"]
        reasons.update(res["recover_reason_frames"])
    return {
        "scenarios_locked": locked,
        "scenarios_not_locked_skipped": skipped,
        "hijacked_before_occlusion_scenarios": hijacked,
        "recover_reason_frames": dict(sorted(reasons.items())),
        "overall": _finish(overall),
        "by_duration_frames": {d: _finish(c) for d, c in sorted(by_duration.items(), key=lambda kv: int(kv[0]))},
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", action="append", required=True,
                        help="Ten thu muc experiment Motion Memory trong --output-root (lap lai)")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--scenario-root", type=Path, default=REPO_ROOT / "data" / "occlusion_test_scenarios" / "v2")
    parser.add_argument("--metrics-config", type=Path, default=ev.DEFAULT_OCCLUSION_METRICS_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="File JSON ket qua")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    from src.utils.logging_utils import write_json
    from src.utils.run_context import git_info

    args = parse_args(argv)
    metrics_cfg = ev.load_occlusion_metrics_config(args.metrics_config)
    if int(metrics_cfg["version"]) < 2:
        raise SystemExit("Chan doan can metrics v2 (kich ban da khoa)")
    result = {
        "kind": KIND,
        "note": "Chi mo ta (dev); khong dung de chon cau hinh. Chi kich ban da khoa truoc khi che.",
        "metrics_config": {"path": str(args.metrics_config), "version": metrics_cfg["version"],
                           "config_sha256": metrics_cfg["config_sha256"], "correct_iou": metrics_cfg["correct_iou"]},
        "scenario_root": str(args.scenario_root),
        "experiments": {},
        "git": git_info(REPO_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for name in args.experiment:
        res = diagnose_experiment(args.output_root / name, args.scenario_root, float(metrics_cfg["correct_iou"]))
        result["experiments"][name] = res
        o = res["overall"]
        parts = ", ".join(f"{c} {o[c]['correct']}/{o[c]['frames_with_box']}" for c in CATEGORIES)
        print(f"{name}: {res['scenarios_locked']} kich ban da khoa; dung/co box: {parts}; "
              f"bi cuop truoc khi che: {res['hijacked_before_occlusion_scenarios']}")
    write_json(args.output, result)
    print(f"Da ghi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
