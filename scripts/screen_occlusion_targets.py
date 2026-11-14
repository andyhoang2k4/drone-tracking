"""Screening "khoa muc tieu truoc khi che" cho kich ban occlusion v2 (chay tren GPU).

Voi moi ung vien cua nhom (configs/occlusion_scenarios_v2.yaml) va moi detector
trong target_lock.required_detectors:
    1. ghi clip SACH pre_occlusion_frames frame truoc khi che (hardlink, CPU);
    2. python -m src.track (YOLO + BoT-SORT, cung lenh nhu run_occlusion_ablation)
       -> log track; kiem tra so frame + thu tu anh;
    3. evaluate_target_lock (luat chon muc tieu that + 3 frame cuoi dung vi tri).
Ghi <output_root>/<group>/screening.json (commit duoc). Ung vien DAT khi MOI
detector deu khoa duoc. Sau do: scripts/generate_occlusion_scenarios.py.

Resume: clip va log track hop le da co duoc dung lai (chay lai sau --limit
hoac sau khi bi ngat chi chay phan con thieu). Log hong / weights doi so voi
log cu -> bao loi, KHONG xoa (xu ly tay); screening co loi bi generator tu choi.

--dry-run: in lenh + so lan goi du kien, khong chay, khong ghi file.
--limit N: chi N ung vien dau (theo sequence, roi thu tu xep hang) -> screening
danh dau partial (generator tu choi) - de do thoi gian / smoke test.

Heldout: chi chay detector + tracker tren clip sach, KHONG Motion Memory, KHONG
kich ban che, khong tinh chinh gi.

Vi du (tren cloud)::

    python scripts/screen_occlusion_targets.py --group dev \\
        --detector yolo11=weights/yolo11_best.pt --detector yolo26=weights/yolo26_best.pt \\
        --device 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_occlusion_ablation import (  # noqa: E402
    default_runner,
    parse_detector,
    tracker_command,
    tracks_log_problems,
)
from src.occlusion_scenarios import (  # noqa: E402
    ScreeningError,
    file_sha256,
    load_scenario_config,
    screening_path,
)
from src.occlusion_screening import (  # noqa: E402
    WEIGHTS_SIDECAR_NAME,
    base_entry,
    build_screening,
    clip_dir_for,
    ensure_screening_clip,
    plan_group_candidates,
    screen_candidate,
    tracks_log_for,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "occlusion_scenarios_v2.yaml"
Runner = Callable[[list[str]], int]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--group", required=True, help="Nhom (dev | heldout)")
    parser.add_argument("--detector", type=parse_detector, action="append", required=True,
                        help="ten=duong_dan_weights, MOI detector trong target_lock.required_detectors")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Chi N ung vien dau (do thoi gian; screening danh dau partial)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Mac dinh <output_root>/<group>/screening.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chi in lenh + so lan goi du kien, khong chay, khong ghi file")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit phai >= 1")
    return args


def _sidecar_problem(log: Path, weights: str, sha: Optional[str]) -> Optional[str]:
    sidecar = log.parent / WEIGHTS_SIDECAR_NAME
    if not sidecar.is_file():
        return f"thieu {sidecar.name} canh log cu"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if data.get("weights_sha256") != sha:
        return f"log cu tao boi weights khac (sha256 {str(data.get('weights_sha256'))[:12]})"
    return None


def run_screening(args: argparse.Namespace, runner: Runner = default_runner) -> tuple[dict, dict]:
    """Tra ve (screening dict hoac {} khi dry-run, bao cao chay)."""
    cfg = load_scenario_config(args.config)
    if not cfg.uses_screening:
        raise SystemExit(f"{args.config} khong co target_lock - screening chi dung cho config v2")
    names = [n for n, _ in args.detector]
    if sorted(names) != sorted(cfg.required_detectors) or len(set(names)) != len(names):
        raise SystemExit(f"--detector phai dung moi ten trong required_detectors {list(cfg.required_detectors)}, "
                         f"nhan {names}")
    weights = dict(args.detector)

    detectors: dict[str, dict] = {}
    for name in cfg.required_detectors:
        path = Path(weights[name])
        if not path.is_file() and not args.dry_run:
            raise SystemExit(f"Khong thay weights {path} cua {name}")
        detectors[name] = {"weights": weights[name],
                           "weights_sha256": file_sha256(path) if path.is_file() else None}

    plans = plan_group_candidates(cfg, args.group)
    selected = plans if args.limit is None else plans[: args.limit]
    report: dict = {"candidates_total": len(plans), "candidates_selected": len(selected),
                    "tracked": 0, "reused": 0, "commands": [], "timings": {}, "errors": []}

    ranks: dict[str, int] = {}
    entries = []
    for k, (plan, data) in enumerate(plans):
        seq_rank = ranks.setdefault(plan.sequence, 0)
        ranks[plan.sequence] = seq_rank + 1
        if k >= len(selected):
            entries.append({**base_entry(plan, seq_rank, cfg), "status": "not_run", "eligible": None})
            continue
        clip_dir = clip_dir_for(cfg, args.group, plan)
        if not args.dry_run:
            ensure_screening_clip(data, plan, cfg, clip_dir)
        records: dict[str, list] = {}
        shas: dict[str, str] = {}
        failed = False
        for name in cfg.required_detectors:
            log = tracks_log_for(clip_dir, name)
            label = f"{clip_dir.name}/{name}"
            if args.dry_run:
                if not log.is_file():
                    report["commands"].append(tracker_command(weights[name], clip_dir, log, args))
                    print(" ".join(report["commands"][-1]))
                continue
            if log.is_file():
                problem = _sidecar_problem(log, weights[name], detectors[name]["weights_sha256"])
                problems = [problem] if problem else tracks_log_problems(log, clip_dir, cfg.pre_occlusion_frames)
                if problems:
                    report["errors"].append(f"{label}: log track cu khong dung duoc - xu ly tay: {problems[0]}")
                    failed = True
                    continue
                report["reused"] += 1
            else:
                cmd = tracker_command(weights[name], clip_dir, log, args)
                report["commands"].append(cmd)
                start = time.perf_counter()
                code = runner(cmd)
                report["timings"].setdefault("track", []).append(time.perf_counter() - start)
                problems = (["src.track loi"] if code != 0
                            else tracks_log_problems(log, clip_dir, cfg.pre_occlusion_frames))
                if problems:
                    report["errors"].append(f"{label}: {problems[0]}")
                    failed = True
                    continue
                (log.parent / WEIGHTS_SIDECAR_NAME).write_text(json.dumps(
                    {"weights": weights[name], "weights_sha256": detectors[name]["weights_sha256"]}, indent=2
                ) + "\n", encoding="utf-8")
                report["tracked"] += 1
            from scripts.run_pipeline import load_replay

            replay = load_replay(log)
            records[name] = replay["records"]
            shas[name] = replay["sha256"]
        if args.dry_run:
            continue
        if failed:
            entries.append({**base_entry(plan, seq_rank, cfg), "status": "error", "eligible": None})
            continue
        entries.append(screen_candidate(plan, seq_rank, cfg, clip_dir, records, shas))

    if args.dry_run:
        return {}, report
    from src.utils.run_context import git_info, package_versions

    screening = build_screening(
        cfg, args.group, entries, detectors, limit=args.limit, errors=report["errors"],
        git=git_info(REPO_ROOT), versions=package_versions(),
    )
    out = args.output or screening_path(cfg, args.group)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(screening, indent=2) + "\n", encoding="utf-8")
    report["output"] = out
    return screening, report


def print_summary(screening: dict, report: dict, args: argparse.Namespace) -> None:
    s = screening["summary"]
    print(f"Nhom {args.group}: {s['screened']}/{s['candidates']} ung vien da screening, "
          f"{s['eligible']} dat (moi detector deu khoa), chon toi da {s['selected_targets']} muc tieu")
    for name, d in s["per_detector"].items():
        rate = f"{d['pass_rate']:.2f}" if d["pass_rate"] is not None else "-"
        print(f"  {name}: khoa {d['locked']}/{d['screened']} (ti le {rate}), {d['by_status']}")
    for seq, row in s["per_sequence"].items():
        print(f"  {seq}: {row['eligible']}/{row['screened']} dat, chon {row['selected']}")


def print_timings(report: dict) -> None:
    values = report["timings"].get("track")
    if not values:
        return
    per = sum(values) / len(values)
    total = report["candidates_total"] * len(report["detectors"])
    print(f"Thoi gian src.track: {len(values)} lan, tong {sum(values):.1f} s, trung binh {per:.2f} s/lan")
    print(f"Uoc tinh ca nhom ({total} lan src.track): ~{per * total / 60:.1f} phut")


def main(argv: Optional[list[str]] = None, runner: Runner = default_runner) -> int:
    args = parse_args(argv)
    try:
        screening, report = run_screening(args, runner)
    except ScreeningError as exc:
        print(f"LOI: {exc}")
        return 1
    report["detectors"] = [n for n, _ in args.detector]
    if args.dry_run:
        n_det = len(args.detector)
        print(f"[dry-run] Nhom {args.group}: {report['candidates_selected']}/{report['candidates_total']} "
              f"ung vien x {n_det} detector; du kien src.track {len(report['commands'])} lan "
              f"(log da co duoc dung lai). Khong chay gi, khong ghi file.")
        return 0
    print(f"src.track: chay {report['tracked']}, dung lai log {report['reused']}, loi {len(report['errors'])}")
    for err in report["errors"]:
        print(f"  LOI: {err}")
    print_summary(screening, report, args)
    print_timings(report)
    print(f"Da ghi: {report['output']}")
    if screening["partial"]:
        print("CHU Y: screening partial (--limit hoac co loi) -> generator se tu choi; KHONG dung de sinh kich ban.")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
