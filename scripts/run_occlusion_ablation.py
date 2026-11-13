"""Chay ablation co/khong Motion Memory tren moi kich ban occlusion cua mot nhom.

Voi moi detector (--detector ten=duong_dan_weights) va moi kich ban trong
<scenario-root>/<group>/index.json:
    1. scripts/run_pipeline.py (baseline) va scripts/run_pipeline.py --motion-memory,
       muc tieu chon theo GT (--gt-target-id + annotations.txt cua kich ban);
    2. python -m src.evaluate occlusion -> metrics.json cua kich ban.
Sau do, voi moi experiment DU kich ban: `aggregate`; moi detector: `compare`.

Ten experiment (CLAUDE.md muc 14)::

    experiments/<detector>_botsort_occl_<group>_<tag>/        (baseline)
    experiments/<detector>_botsort_mm_occl_<group>_<tag>/     (+ Motion Memory)
        config.yaml                     cau hinh experiment (commit)
        metrics.json                    ket qua gop (commit)
        scenarios/<kich ban>/           run_context.json, target_selection.json,
                                        metrics.json, logs/ (gitignore)

Khong ghi de: kich ban da co metrics.json bi bo qua. Kich ban co log nhung
chua co metrics (lan chay bi ngat) -> thu danh gia lai; that bai thi bao de xu
ly tay (KHONG xoa, KHONG chay lai len log cu vi log duoc ghi noi).

Vi du (tren cloud)::

    python scripts/run_occlusion_ablation.py --group dev --tag 001 \\
        --detector yolo11=weights/yolo11_best.pt --detector yolo26=weights/yolo26_best.pt \\
        --device 0 --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCENARIO_ROOT = REPO_ROOT / "data" / "occlusion_test_scenarios"
DEFAULT_MM_CONFIG = REPO_ROOT / "configs" / "motion_memory.yaml"

Runner = Callable[[list[str]], int]


def experiment_id(detector: str, motion_memory: bool, group: str, tag: str) -> str:
    return f"{detector}_botsort_{'mm_' if motion_memory else ''}occl_{group}_{tag}"


@dataclass
class RunSpec:
    detector: str
    weights: str
    motion_memory: bool
    exp_id: str
    exp_dir: Path


@dataclass
class Report:
    ran: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    aggregated: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)


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
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-partial-index", action="store_true",
                        help="Cho phep index danh dau partial (smoke test, KHONG phai benchmark)")
    parser.add_argument("--dry-run", action="store_true", help="Chi in lenh, khong chay, khong ghi file")
    args = parser.parse_args(argv)
    names = [n for n, _ in args.detector]
    if len(set(names)) != len(names):
        parser.error("ten detector bi trung")
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


def experiment_config(spec: RunSpec, args: argparse.Namespace, index_path: Path, index: dict) -> dict:
    return {
        "experiment_id": spec.exp_id,
        "purpose": "Ablation Motion Memory tren kich ban occlusion nhan tao (CLAUDE.md muc 12)",
        "detector": {"name": spec.detector, "weights": spec.weights},
        "tracker": {"type": "botsort", "config": "configs/botsort.yaml"},
        "motion_memory": {
            "enabled": spec.motion_memory,
            "config": str(args.motion_memory_config) if spec.motion_memory else None,
        },
        "scenarios": {
            "group": args.group,
            "index": str(index_path),
            "config_sha256": index.get("config_sha256"),
            "seed": index.get("seed"),
            "count": len(index["scenarios"]),
            "partial": bool(index.get("partial")),
        },
        "metrics_config": "configs/occlusion_metrics.yaml",
    }


def ensure_experiment_config(spec: RunSpec, config: dict) -> None:
    """Ghi config.yaml cua experiment; da co ma khac -> tu choi (khong tron cau hinh)."""
    import yaml

    path = spec.exp_dir / "config.yaml"
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing != config:
            raise ValueError(
                f"{path} da ton tai voi cau hinh khac - dung --tag moi thay vi tron hai cau hinh"
            )
        return
    spec.exp_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Sinh boi scripts/run_occlusion_ablation.py - khong sua sau khi da co ket qua.\n"
        + yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def pipeline_command(spec: RunSpec, scen_dir: Path, manifest: dict, args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--weights", spec.weights,
        "--source", str(scen_dir / "images"),
        "--experiment-id", scen_dir.name,
        "--output-root", str(spec.exp_dir / "scenarios"),
        "--gt-target-id", str(manifest["target"]["gt_target_id"]),
        "--gt-annotation", str(scen_dir / "annotations.txt"),
        "--gt-first-frame", str(manifest.get("first_frame", 1)),
    ]
    if args.device is not None:
        cmd += ["--device", str(args.device)]
    if spec.motion_memory:
        cmd += ["--motion-memory", "--motion-memory-config", str(args.motion_memory_config)]
    return cmd


def evaluate_command(scen_dir: Path, run_dir: Path) -> list[str]:
    return [sys.executable, "-m", "src.evaluate", "occlusion",
            "--scenario-dir", str(scen_dir), "--run-dir", str(run_dir)]


def default_runner(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def run_ablation(args: argparse.Namespace, runner: Runner = default_runner) -> Report:
    index_path, index = load_index(args.scenario_root, args.group, args.allow_partial_index)
    group_dir = args.scenario_root / args.group
    report = Report()

    def call(cmd: list[str]) -> int:
        report.commands.append(cmd)
        if args.dry_run:
            print(" ".join(cmd))
            return 0
        return runner(cmd)

    specs: list[RunSpec] = []
    for name, weights in args.detector:
        for mm in (False, True):
            exp = experiment_id(name, mm, args.group, args.tag)
            specs.append(RunSpec(name, weights, mm, exp, args.output_root / exp))
    if not args.dry_run:
        for spec in specs:
            ensure_experiment_config(spec, experiment_config(spec, args, index_path, index))

    for spec in specs:
        complete = True
        for scen_name in index["scenarios"]:
            scen_dir = group_dir / scen_name
            manifest = json.loads((scen_dir / "manifest.json").read_text(encoding="utf-8"))
            run_dir = spec.exp_dir / "scenarios" / scen_name
            label = f"{spec.exp_id}/{scen_name}"
            if (run_dir / "metrics.json").is_file():
                report.skipped.append(label)
                continue
            if (run_dir / "logs" / "pipeline.jsonl").is_file():
                # Log da co (lan truoc bi ngat sau khi chay xong?) - chi danh gia lai.
                if call(evaluate_command(scen_dir, run_dir)) == 0:
                    report.recovered.append(label)
                else:
                    report.failed.append(f"{label} (log cu khong hop le - can xu ly tay)")
                    complete = False
                continue
            if call(pipeline_command(spec, scen_dir, manifest, args)) != 0:
                report.failed.append(f"{label} (run_pipeline loi)")
                complete = False
                continue
            if call(evaluate_command(scen_dir, run_dir)) != 0:
                report.failed.append(f"{label} (evaluate loi)")
                complete = False
                continue
            report.ran.append(label)
        if complete:
            cmd = [sys.executable, "-m", "src.evaluate", "aggregate",
                   "--exp-dir", str(spec.exp_dir), "--index", str(index_path)]
            if args.allow_partial_index:
                cmd.append("--allow-incomplete")
            if call(cmd) == 0:
                report.aggregated.append(spec.exp_id)
            else:
                report.failed.append(f"{spec.exp_id} (aggregate loi)")

    for name, _ in args.detector:
        base = experiment_id(name, False, args.group, args.tag)
        mem = experiment_id(name, True, args.group, args.tag)
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


def main(argv: Optional[list[str]] = None, runner: Runner = default_runner) -> int:
    args = parse_args(argv)
    report = run_ablation(args, runner)
    if args.dry_run:
        print(f"[dry-run] {len(report.commands)} lenh, khong chay gi.")
        return 0
    print(f"Chay moi: {len(report.ran)}, bo qua (da co): {len(report.skipped)}, "
          f"danh gia lai: {len(report.recovered)}, loi: {len(report.failed)}")
    for item in report.failed:
        print(f"  LOI: {item}")
    print(f"Da gop: {report.aggregated}")
    print(f"Da so sanh: {report.compared}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
