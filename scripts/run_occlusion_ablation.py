"""Chay ablation co/khong Motion Memory tren moi kich ban occlusion cua mot nhom.

Voi moi detector (--detector ten=duong_dan_weights) va moi kich ban trong
<scenario-root>/<group>/index.json:
    1. python -m src.track: chay YOLO + BoT-SORT MOT LAN, ghi log toan bo track
       (tracks.jsonl + tracks.meta.json); kiem tra so frame va thu tu anh;
    2. scripts/run_pipeline.py --replay-tracks (baseline) va ... --motion-memory
       tren CUNG log do -> hai cau hinh thay chinh xac cung dau vao tracker;
       muc tieu chon theo GT (--gt-target-id + annotations.txt cua kich ban);
    3. python -m src.evaluate occlusion -> metrics.json cua kich ban.
Sau do, voi moi experiment DU kich ban: `aggregate`; moi detector: `compare`
(tu choi neu hai lan chay khong cung tracks_sha256).

Ten experiment (CLAUDE.md muc 14)::

    experiments/<detector>_botsort_tracks_occl_<group>_<tag>/ (log tracker dung chung)
        config.yaml
        scenarios/<kich ban>/logs/tracks.jsonl, tracks.meta.json
    experiments/<detector>_botsort_occl_<group>_<tag>/        (baseline)
    experiments/<detector>_botsort_mm_occl_<group>_<tag>/     (+ Motion Memory)
        config.yaml                     cau hinh experiment
        metrics.json                    ket qua gop
        scenarios/<kich ban>/           run_context.json, target_selection.json,
                                        metrics.json, logs/

Khong ghi de: kich ban da co metrics.json bi bo qua; log track hop le duoc dung
lai. Log (track hoac pipeline) hong/chua du (lan chay bi ngat) -> bao de
xu ly tay (KHONG xoa, KHONG chay lai len log cu vi log duoc ghi noi). Log
pipeline day du nhung chua co metrics -> chi danh gia lai.

--dry-run: in lenh, so kich ban va so lan goi du kien theo loai, khong chay gi.
--limit-scenarios N: chi lay N kich ban dau cua index (do thoi gian / smoke
test). Ket qua bi danh dau trong config.yaml (khong tron voi lan chay du) va
`aggregate` chay voi --allow-incomplete (complete=false) -> KHONG phai benchmark.
Cuoi moi lan chay in thoi gian tung loai lenh va uoc tinh cho ca nhom.

Kich ban v2 (configs/occlusion_scenarios_v2.yaml, index co `screening`) +
metrics v2 (mac dinh --metrics-config configs/occlusion_metrics_v2.yaml):
    - moi --detector phai co trong screening cua kich ban VA cung weights
      (sha256) - neu khong, tap muc tieu da loc khong hop le -> tu choi;
    - run_pipeline nhan --gt-match-iou / --gt-max-search-frames tu
      manifest.target_lock (khong tu configs/eval_visdrone.yaml);
    - evaluate dung metrics v2: kich ban chua khoa duoc dem rieng.
Kich ban v1 (lan chay thu "timing"): --metrics-config configs/occlusion_metrics.yaml.
Metrics v2 tren kich ban v1 (va nguoc lai) -> tu choi.

Vi du (tren cloud)::

    python scripts/run_occlusion_ablation.py --group dev --tag v2_001 \\
        --scenario-root data/occlusion_test_scenarios/v2 \\
        --detector yolo11=weights/yolo11_best.pt --detector yolo26=weights/yolo26_best.pt \\
        --device 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCENARIO_ROOT = REPO_ROOT / "data" / "occlusion_test_scenarios"
DEFAULT_MM_CONFIG = REPO_ROOT / "configs" / "motion_memory.yaml"
# Metrics v2 (mac dinh) chi dung cho kich ban v2 (index co screening); kich ban
# v1 (lan chay thu "timing") dung --metrics-config configs/occlusion_metrics.yaml.
DEFAULT_METRICS_CONFIG = REPO_ROOT / "configs" / "occlusion_metrics_v2.yaml"

Runner = Callable[[list[str]], int]


def experiment_id(detector: str, motion_memory: bool, group: str, tag: str) -> str:
    return f"{detector}_botsort_{'mm_' if motion_memory else ''}occl_{group}_{tag}"


def tracks_experiment_id(detector: str, group: str, tag: str) -> str:
    return f"{detector}_botsort_tracks_occl_{group}_{tag}"


TRACKS_LOG = Path("logs") / "tracks.jsonl"


@dataclass
class RunSpec:
    detector: str
    weights: str
    motion_memory: bool
    exp_id: str
    exp_dir: Path


@dataclass
class Report:
    tracked: list[str] = field(default_factory=list)
    tracks_reused: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    aggregated: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    timings: dict[str, list[float]] = field(default_factory=dict)
    scenarios_total: int = 0
    scenarios_selected: int = 0


# Thu tu in cac loai lenh.
COMMAND_KINDS = ("track", "pipeline", "occlusion", "aggregate", "compare")
KIND_LABELS = {
    "track": "src.track",
    "pipeline": "run_pipeline",
    "occlusion": "evaluate occlusion",
    "aggregate": "evaluate aggregate",
    "compare": "evaluate compare",
}


def command_kind(cmd: list[str]) -> str:
    if cmd[1:3] == ["-m", "src.track"]:
        return "track"
    if cmd[1].endswith("run_pipeline.py"):
        return "pipeline"
    if cmd[1:3] == ["-m", "src.evaluate"] and len(cmd) > 3:
        return cmd[3]
    return "other"


def count_kinds(commands: list[list[str]]) -> dict[str, int]:
    counts = {k: 0 for k in COMMAND_KINDS}
    for cmd in commands:
        kind = command_kind(cmd)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def parse_detector(value: str) -> tuple[str, str]:
    name, sep, weights = value.partition("=")
    if not sep or not name or not weights:
        raise argparse.ArgumentTypeError(f"--detector can dang ten=duong_dan_weights, nhan {value!r}")
    return name, weights


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="Nhom kich ban (dev | heldout)")
    parser.add_argument("--tag", required=True, help="Hau to experiment ID, vd 001")
    parser.add_argument("--detector", type=parse_detector, action="append", required=True,
                        help="ten=duong_dan_weights (lap lai cho nhieu detector)")
    parser.add_argument("--scenario-root", type=Path, default=DEFAULT_SCENARIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--motion-memory-config", type=Path, default=DEFAULT_MM_CONFIG)
    parser.add_argument("--metrics-config", type=Path, default=DEFAULT_METRICS_CONFIG,
                        help="Dinh nghia do do (mac dinh v2; kich ban v1: configs/occlusion_metrics.yaml)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-partial-index", action="store_true",
                        help="Cho phep index danh dau partial (smoke test, KHONG phai benchmark)")
    parser.add_argument("--limit-scenarios", type=int, default=None, metavar="N",
                        help="Chi chay N kich ban dau cua index (do thoi gian; KHONG phai benchmark)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chi in lenh + so kich ban/so lan goi du kien, khong chay, khong ghi file")
    args = parser.parse_args(argv)
    names = [n for n, _ in args.detector]
    if len(set(names)) != len(names):
        parser.error("ten detector bi trung")
    if args.limit_scenarios is not None and args.limit_scenarios < 1:
        parser.error("--limit-scenarios phai >= 1")
    return args


def load_index(scenario_root: Path, group: str, allow_partial: bool) -> tuple[Path, dict]:
    index_path = scenario_root / group / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Khong thay {index_path} - chay scripts/generate_occlusion_scenarios.py truoc")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("group") != group:
        raise ValueError(f"{index_path}: group={index.get('group')!r} khac {group!r}")
    if index.get("partial") and not allow_partial:
        raise ValueError(f"{index_path} la index partial (smoke test); dung --allow-partial-index neu chu y")
    if not index.get("scenarios"):
        raise ValueError(f"{index_path}: khong co kich ban nao")
    return index_path, index


def _scenarios_config(args: argparse.Namespace, index_path: Path, index: dict) -> dict:
    out = {
        "group": args.group,
        "index": str(index_path),
        "config_sha256": index.get("config_sha256"),
        "seed": index.get("seed"),
        "count": len(index["scenarios"]),
        "partial": bool(index.get("partial")),
        # Khac None -> chi N kich ban dau (do thoi gian), khong phai benchmark.
        "limit_scenarios": args.limit_scenarios,
    }
    if index.get("screening"):  # kich ban v2
        out["config_version"] = index.get("config_version")
        out["screening_sha256"] = index["screening"].get("sha256")
        out["target_lock"] = index.get("target_lock")
    return out


def metrics_config_summary(args: argparse.Namespace) -> dict:
    """Version + sha256 cua dinh nghia do do (ghi vao config.yaml cua experiment)."""
    from src.evaluate import load_occlusion_metrics_config

    cfg = load_occlusion_metrics_config(args.metrics_config)
    return {"path": str(args.metrics_config), "version": cfg["version"], "config_sha256": cfg["config_sha256"]}


def check_scenarios_and_detectors(args: argparse.Namespace, index: dict, metrics: dict) -> None:
    """Metrics v2 <=> kich ban v2; detector + weights sha256 phai khop screening."""
    screening = index.get("screening")
    if metrics["version"] >= 2 and not screening:
        raise ValueError(
            f"Metrics v{metrics['version']} chi dung cho kich ban v2 (index co screening); kich ban v1: "
            "dung --metrics-config configs/occlusion_metrics.yaml"
        )
    if metrics["version"] < 2 and screening:
        raise ValueError("Kich ban v2 phai danh gia bang metrics v2 (configs/occlusion_metrics_v2.yaml)")
    if not screening:
        return
    from src.occlusion_scenarios import file_sha256

    recorded = screening.get("detectors") or {}
    for name, weights in args.detector:
        if name not in recorded:
            raise ValueError(f"Detector {name!r} khong co trong screening cua kich ban (co: {sorted(recorded)})")
        if not Path(weights).is_file():
            raise ValueError(f"Khong thay weights {weights} cua {name}")
        if file_sha256(weights) != recorded[name].get("weights_sha256"):
            raise ValueError(
                f"Weights {weights} cua {name} khac weights da dung khi screening (sha256 khac) - tap muc "
                "tieu khong con hop le cho detector nay"
            )


def experiment_config(
    spec: RunSpec, args: argparse.Namespace, index_path: Path, index: dict, metrics: Optional[dict] = None
) -> dict:
    return {
        "experiment_id": spec.exp_id,
        "purpose": "Ablation Motion Memory tren kich ban occlusion nhan tao (CLAUDE.md muc 12)",
        "detector": {"name": spec.detector, "weights": spec.weights},
        "tracker": {"type": "botsort", "config": "configs/botsort.yaml"},
        "tracker_input": {
            "mode": "replay",
            "tracks_experiment": tracks_experiment_id(spec.detector, args.group, args.tag),
        },
        "motion_memory": {
            "enabled": spec.motion_memory,
            "config": str(args.motion_memory_config) if spec.motion_memory else None,
        },
        "scenarios": _scenarios_config(args, index_path, index),
        "metrics_config": metrics if metrics is not None else metrics_config_summary(args),
    }


def tracks_config(detector: str, weights: str, args: argparse.Namespace, index_path: Path, index: dict) -> dict:
    return {
        "experiment_id": tracks_experiment_id(detector, args.group, args.tag),
        "purpose": "Log YOLO + BoT-SORT dung chung cho baseline va Motion Memory (replay)",
        "detector": {"name": detector, "weights": weights},
        "tracker": {"type": "botsort", "config": "configs/botsort.yaml"},
        "device": args.device,
        "scenarios": _scenarios_config(args, index_path, index),
    }


def ensure_experiment_config(exp_dir: Path, config: dict) -> None:
    """Ghi config.yaml cua experiment; da co ma khac -> tu choi (khong tron cau hinh)."""
    import yaml

    path = exp_dir / "config.yaml"
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing != config:
            raise ValueError(
                f"{path} da ton tai voi cau hinh khac - dung --tag moi thay vi tron hai cau hinh"
            )
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Sinh boi scripts/run_occlusion_ablation.py - khong sua sau khi da co ket qua.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def tracker_command(weights: str, scen_dir: Path, tracks_log: Path, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.track",
        "--weights", weights,
        "--source", str(scen_dir / "images"),
        "--output", str(tracks_log),
    ]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    return cmd


def tracks_log_problems(tracks_log: Path, scen_dir: Path, frames: int) -> list[str]:
    """Log track dung duoc de replay? Rong = OK (du frame, dung thu tu anh, co meta)."""
    from scripts.run_pipeline import ReplayError, load_replay
    from src.occlusion_scenarios import frame_order_problems

    try:
        replay = load_replay(tracks_log)
    except ReplayError as exc:
        return [str(exc)]
    records = replay["records"]
    if len(records) != frames:
        return [f"log track co {len(records)} frame, manifest co {frames}"]
    return frame_order_problems([r.get("path") for r in records], scen_dir)


def pipeline_command(
    spec: RunSpec, scen_dir: Path, manifest: dict, tracks_log: Path, args: argparse.Namespace
) -> list[str]:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--weights", spec.weights,
        "--source", str(scen_dir / "images"),
        "--replay-tracks", str(tracks_log),
        "--experiment-id", scen_dir.name,
        "--output-root", str(spec.exp_dir / "scenarios"),
        "--gt-target-id", str(manifest["target"]["gt_target_id"]),
        "--gt-annotation", str(scen_dir / "annotations.txt"),
        "--gt-first-frame", str(manifest.get("first_frame", 1)),
    ]
    lock = manifest.get("target_lock")
    if lock:  # kich ban v2: tham so chon muc tieu lay tu manifest (co version), khong tu eval_visdrone.yaml
        cmd += ["--gt-match-iou", str(lock["match_iou"]),
                "--gt-max-search-frames", str(lock["max_search_frames"])]
    if spec.motion_memory:
        cmd += ["--motion-memory", "--motion-memory-config", str(args.motion_memory_config)]
    return cmd


def evaluate_command(scen_dir: Path, run_dir: Path, metrics_config: Optional[Path] = None) -> list[str]:
    cmd = [sys.executable, "-m", "src.evaluate", "occlusion",
           "--scenario-dir", str(scen_dir), "--run-dir", str(run_dir)]
    if metrics_config is not None:
        cmd += ["--metrics-config", str(metrics_config)]
    return cmd


def default_runner(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def run_ablation(args: argparse.Namespace, runner: Runner = default_runner) -> Report:
    index_path, index = load_index(args.scenario_root, args.group, args.allow_partial_index)
    metrics = metrics_config_summary(args)
    check_scenarios_and_detectors(args, index, metrics)
    group_dir = args.scenario_root / args.group
    report = Report()

    scenarios = list(index["scenarios"])
    report.scenarios_total = len(scenarios)
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]
    report.scenarios_selected = len(scenarios)
    incomplete_ok = args.allow_partial_index or args.limit_scenarios is not None

    def call(cmd: list[str]) -> int:
        report.commands.append(cmd)
        if args.dry_run:
            print(" ".join(cmd))
            return 0
        start = time.perf_counter()
        code = runner(cmd)
        report.timings.setdefault(command_kind(cmd), []).append(time.perf_counter() - start)
        return code

    detectors: list[tuple[str, str, Path, list[RunSpec]]] = []
    for name, weights in args.detector:
        specs = []
        for mm in (False, True):
            exp = experiment_id(name, mm, args.group, args.tag)
            specs.append(RunSpec(name, weights, mm, exp, args.output_root / exp))
        tracks_dir = args.output_root / tracks_experiment_id(name, args.group, args.tag)
        detectors.append((name, weights, tracks_dir, specs))
    if not args.dry_run:
        for name, weights, tracks_dir, specs in detectors:
            ensure_experiment_config(tracks_dir, tracks_config(name, weights, args, index_path, index))
            for spec in specs:
                ensure_experiment_config(spec.exp_dir, experiment_config(spec, args, index_path, index, metrics))

    for name, weights, tracks_dir, specs in detectors:
        complete = {spec.exp_id: True for spec in specs}
        for scen_name in scenarios:
            scen_dir = group_dir / scen_name
            manifest = json.loads((scen_dir / "manifest.json").read_text(encoding="utf-8"))
            pending = [
                spec for spec in specs
                if not (spec.exp_dir / "scenarios" / scen_name / "metrics.json").is_file()
            ]
            for spec in specs:
                if spec not in pending:
                    report.skipped.append(f"{spec.exp_id}/{scen_name}")
            if not pending:
                continue

            # 1. Log tracker dung chung (chay YOLO + BoT-SORT mot lan).
            tracks_log = tracks_dir / "scenarios" / scen_name / TRACKS_LOG
            track_label = f"{tracks_dir.name}/{scen_name}"
            tracks_ok = True
            if args.dry_run:
                if not tracks_log.is_file():
                    call(tracker_command(weights, scen_dir, tracks_log, args))
            elif tracks_log.is_file():
                problems = tracks_log_problems(tracks_log, scen_dir, int(manifest["frames"]))
                if problems:
                    report.failed.append(f"{track_label} (log track cu khong hop le - can xu ly tay: "
                                         f"{problems[0]})")
                    tracks_ok = False
                else:
                    report.tracks_reused.append(track_label)
            elif call(tracker_command(weights, scen_dir, tracks_log, args)) != 0:
                report.failed.append(f"{track_label} (src.track loi)")
                tracks_ok = False
            else:
                problems = tracks_log_problems(tracks_log, scen_dir, int(manifest["frames"]))
                if problems:
                    report.failed.append(f"{track_label} (log track khong hop le: {problems[0]})")
                    tracks_ok = False
                else:
                    report.tracked.append(track_label)
            if not tracks_ok:
                for spec in pending:
                    complete[spec.exp_id] = False
                continue

            # 2. Hau ky baseline va Motion Memory tren CUNG log track.
            for spec in pending:
                run_dir = spec.exp_dir / "scenarios" / scen_name
                label = f"{spec.exp_id}/{scen_name}"
                if (run_dir / "logs" / "pipeline.jsonl").is_file():
                    # Log da co (lan truoc bi ngat sau khi chay xong?) - chi danh gia lai.
                    if call(evaluate_command(scen_dir, run_dir, args.metrics_config)) == 0:
                        report.recovered.append(label)
                    else:
                        report.failed.append(f"{label} (log cu khong hop le - can xu ly tay)")
                        complete[spec.exp_id] = False
                    continue
                if call(pipeline_command(spec, scen_dir, manifest, tracks_log, args)) != 0:
                    report.failed.append(f"{label} (run_pipeline loi)")
                    complete[spec.exp_id] = False
                    continue
                if call(evaluate_command(scen_dir, run_dir, args.metrics_config)) != 0:
                    report.failed.append(f"{label} (evaluate loi)")
                    complete[spec.exp_id] = False
                    continue
                report.ran.append(label)

        for spec in specs:
            if not complete[spec.exp_id]:
                continue
            cmd = [sys.executable, "-m", "src.evaluate", "aggregate",
                   "--exp-dir", str(spec.exp_dir), "--index", str(index_path)]
            if incomplete_ok:
                cmd.append("--allow-incomplete")
            if call(cmd) == 0:
                report.aggregated.append(spec.exp_id)
            else:
                report.failed.append(f"{spec.exp_id} (aggregate loi)")

        base, mem = specs[0].exp_id, specs[1].exp_id
        if base in report.aggregated and mem in report.aggregated:
            out = args.output_root / mem / f"ablation_vs_{base}.json"
            cmd = [sys.executable, "-m", "src.evaluate", "compare",
                   "--baseline", str(args.output_root / base),
                   "--memory", str(args.output_root / mem), "--output", str(out)]
            if call(cmd) == 0:
                report.compared.append(f"{base} vs {mem}")
            else:
                report.failed.append(f"{base} vs {mem} (compare loi)")
    return report


def print_plan(report: Report, args: argparse.Namespace) -> None:
    limit = f" (--limit-scenarios {args.limit_scenarios})" if args.limit_scenarios else ""
    print(f"[dry-run] Nhom {args.group}: {report.scenarios_selected}/{report.scenarios_total} "
          f"kich ban{limit}, {len(args.detector)} detector")
    counts = count_kinds(report.commands)
    parts = ", ".join(f"{KIND_LABELS.get(k, k)} {counts[k]}" for k in counts)
    print(f"[dry-run] Du kien: {parts} (tong {len(report.commands)} lenh); "
          f"bo qua vi da co metrics: {len(report.skipped)} lan chay kich ban")
    print("[dry-run] Khong chay gi, khong ghi file.")


def print_timings(report: Report, args: argparse.Namespace) -> None:
    if not report.timings:
        return
    print("Thoi gian (giay):")
    for kind in list(COMMAND_KINDS) + sorted(set(report.timings) - set(COMMAND_KINDS)):
        values = report.timings.get(kind)
        if values:
            print(f"  {KIND_LABELS.get(kind, kind)}: {len(values)} lan, tong {sum(values):.1f}, "
                  f"trung binh {sum(values) / len(values):.2f}")
    # Uoc tinh: moi cap (detector, kich ban) chay moi = 1 src.track + 2 run_pipeline + 2 evaluate.
    units = len(report.timings.get("track", []))
    if units:
        per_unit = sum(sum(report.timings.get(k, [])) for k in ("track", "pipeline", "occlusion")) / units
        total_units = report.scenarios_total * len(args.detector)
        print(f"Uoc tinh ca nhom ({report.scenarios_total} kich ban x {len(args.detector)} detector): "
              f"~{per_unit * total_units / 60:.1f} phut (dua tren {units} cap detector x kich ban "
              f"vua chay, {per_unit:.1f} s/cap)")


def main(argv: Optional[list[str]] = None, runner: Runner = default_runner) -> int:
    args = parse_args(argv)
    report = run_ablation(args, runner)
    if args.dry_run:
        print_plan(report, args)
        return 0
    print(f"Tracker: chay {len(report.tracked)}, dung lai log {len(report.tracks_reused)}")
    print(f"Hau ky: chay moi {len(report.ran)}, bo qua (da co): {len(report.skipped)}, "
          f"danh gia lai: {len(report.recovered)}, loi: {len(report.failed)}")
    for item in report.failed:
        print(f"  LOI: {item}")
    print(f"Da gop: {report.aggregated}")
    print(f"Da so sanh: {report.compared}")
    print_timings(report, args)
    if args.limit_scenarios is not None:
        print(f"CHU Y: --limit-scenarios {args.limit_scenarios} -> ket qua gop danh dau incomplete, "
              "KHONG phai benchmark.")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
