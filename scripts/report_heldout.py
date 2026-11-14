"""Bao cao heldout theo ke hoach khai bao truoc (configs/heldout_report_plan.yaml).

Chi DOC ket qua da co (khong chay YOLO/replay). Voi MOI cot da khai bao x detector:
    - bang mo ta (overall + moi duration; src/mm_variant_eval.describe: cpr, box_precision,
      occluded_frames_with_output, mean_position_error_px, re-ID, re-ID ID moi, time-to-reid, gap)
      + gop hai detector (cong count) + so kich ban/muc tieu/sequence, per_sequence;
    - cot QUYET DINH: dung ba dieu kien (i)-(iii) cua file decision.rules_from (sha256 phai khop
      plan), tung detector -> "DAT" / "KHONG DAT" (bao ca khi khong dat);
    - bootstrap theo sequence (src/occlusion_report.bootstrap, thong so tu plan) - MO TA;
    - phan loai frame bi che co box (scripts/diagnose_mm_hijack.py) cho cac cot MM cua plan;
    - moi truong (experiment log tracker), commit git, sha256 plan.
Thieu cot / khac plan -> report van ghi, status = incomplete, liet ke van de (khong bo cot).

    python scripts/report_heldout.py --plan configs/heldout_report_plan.yaml
    -> experiments/heldout_report_<tag>/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.diagnose_mm_hijack as dg  # noqa: E402
from src import mm_variant_eval as ve  # noqa: E402
from src import occlusion_report as rep  # noqa: E402
from src.heldout_plan import HeldoutPlan  # noqa: E402

KIND = "heldout_report"
STATEMENTS = [
    "Heldout chay dung mot lan theo ke hoach khai bao truoc; khong tune lai, khong chon lai cau hinh.",
    "Cot quyet dinh dung dung ba dieu kien (i)-(iii) cua decision.rules_from, tung detector; bao ca khi khong dat.",
    "Cac cot mo ta (vd mm_tuned, mm_anchor_lost25), box_precision, phan loai frame, bootstrap KHONG dung de chon "
    "cau hinh sau khi thay ket qua.",
    "Motion Memory la 'motion-only + ID anchor', KHONG co appearance Re-ID; re-ID o d60 bi gioi han boi "
    "track_buffer=30 cua BoT-SORT va thiet ke motion-only.",
    "Selection bias: chi muc tieu ma ca hai detector khoa duoc truoc khi che; khong suy rong cho muc tieu kho.",
    "Bao ca chieu xau: re-ID giam, box_precision thap, box cua track khac (tracking_other_id).",
    "mean_position_error_px do tren tap frame CO BOX cua tung cot (khong cung tap); mean_time_to_reidentify_frames "
    "la trung binh CO DIEU KIEN; reidentified_with_new_track_id cua no_mm = 0 do dinh nghia.",
    rep.BOOTSTRAP_NOTE,
]


def _repo_file(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_report(plan_path: Path, output_root: Path, scenario_root: Path, tag: Optional[str]) -> dict[str, Any]:
    from src.evaluate import load_scenario_metrics
    from src.occlusion_scenarios import config_sha256

    raw = Path(plan_path).read_bytes()
    plan = HeldoutPlan.from_mapping(__import__("yaml").safe_load(raw.decode("utf-8")) or {})
    plan_sha = config_sha256(raw)
    tag = tag or plan.tag
    dets = plan.detectors
    problems: list[str] = []
    for path, sha in plan.referenced_files().items():
        f = _repo_file(path)
        if not f.is_file() or config_sha256(f.read_bytes()) != sha:
            problems.append(f"file plan tham chieu thieu/khac sha256: {path}")

    aggs: dict[str, dict[str, dict]] = {}
    experiments: dict[str, dict[str, str]] = {}
    for col in plan.columns:
        aggs[col.name], experiments[col.name] = {}, {}
        for det in dets:
            exp = plan.experiment_id(col.name, det, tag)
            experiments[col.name][det] = exp
            exp_dir = output_root / exp
            if not (exp_dir / "metrics.json").is_file():
                problems.append(f"thieu metrics gop: {exp}")
                continue
            conf = _load_yaml(exp_dir / "config.yaml") if (exp_dir / "config.yaml").is_file() else {}
            if ((conf.get("heldout") or {}).get("plan") or {}).get("sha256") != plan_sha:
                problems.append(f"{exp}: config.yaml khong ghi dung sha256 plan")
            agg = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
            if not (agg.get("completeness") or {}).get("complete"):
                problems.append(f"{exp}: metrics gop chua day du")
            aggs[col.name][det] = agg

    def complete_col(name: str) -> bool:
        return all(det in aggs[name] for det in dets)

    tables = {}
    for col in plan.columns:
        row: dict[str, Any] = {"role": col.role, "experiments": experiments[col.name],
                               "per_detector": ve.describe(aggs[col.name], dets)}
        if complete_col(col.name):
            row["pooled_detectors"] = rep.pooled_across_detectors(aggs[col.name], dets)
            row["counts"] = {det: {k: aggs[col.name][det]["overall"].get(k) for k in
                                   ("scenarios_total", "scenarios_not_locked", "independent_targets",
                                    "independent_sequences", "per_sequence")} for det in dets}
        tables[col.name] = row

    decision: dict[str, Any]
    rules_ref = plan.decision["rules_from"]
    try:
        rules_file = _repo_file(rules_ref["path"])
        if config_sha256(rules_file.read_bytes()) != rules_ref["sha256"]:
            raise ve.VariantPlanError(f"{rules_ref['path']} khac sha256 trong plan")
        rules = ve.VariantPlan.from_mapping(_load_yaml(rules_file))
        refs = {ref: aggs[col] for ref, col in plan.decision["reference_columns"].items()}
        decision = {"status": "complete", "column": plan.decision["column"],
                    "rules_from": rules_ref, **ve.decide(rules, aggs[plan.decision["column"]], refs)}
        decision["verdict"] = "DAT" if decision["passed"] else "KHONG DAT"
    except ve.VariantPlanError as exc:
        decision = {"status": "not_available", "column": plan.decision["column"], "reason": str(exc)}

    bootstrap: dict[str, Any]
    try:
        columns = {}
        for col in plan.columns:
            columns[col.name] = {}
            for det in dets:
                exp_dir = output_root / experiments[col.name][det]
                columns[col.name][det] = list(load_scenario_metrics(exp_dir).values())
        durations = sorted({int(it["scenario"]["duration_frames"]) for c in columns.values()
                            for items in c.values() for it in items})
        b = plan.bootstrap
        bootstrap = {"status": "complete", **rep.bootstrap(
            columns, dets, resamples=b["resamples"], seed=b["seed"], ci=b["ci"], paired_vs=b["paired_vs"],
            metrics=b["metrics"], durations=durations)}
    except Exception as exc:   # thieu file tung kich ban / khac tap -> ghi ly do, khong bo report
        bootstrap = {"status": "not_available", "reason": f"{type(exc).__name__}: {exc}"}

    diagnosis: dict[str, Any] = {}
    import src.evaluate as ev

    correct_iou = float(ev.load_occlusion_metrics_config(_repo_file(plan.metrics_config["path"]))["correct_iou"])
    for name in plan.report.get("diagnosis_columns") or []:
        diagnosis[name] = {}
        for det in dets:
            try:
                diagnosis[name][det] = dg.diagnose_experiment(output_root / experiments[name][det], scenario_root,
                                                              correct_iou)
            except Exception as exc:
                diagnosis[name][det] = {"status": "not_available", "reason": f"{type(exc).__name__}: {exc}"}

    environment = {}
    for det in dets:
        tconf = output_root / f"{det}_botsort_tracks_occl_{plan.group}_{tag}" / "config.yaml"
        environment[det] = (_load_yaml(tconf).get("heldout") or {}) if tconf.is_file() else None

    status = "complete" if not problems and decision["status"] == "complete" else "incomplete"
    from src.utils.run_context import git_info

    return {
        "kind": KIND,
        "status": status,
        "problems": problems,
        "plan": {"path": str(plan_path), "sha256": plan_sha, "version": plan.version, "tag": tag},
        "columns_declared": [c.name for c in plan.columns],
        "decision": decision,
        "tables": tables,
        "bootstrap": bootstrap,
        "diagnosis": diagnosis,
        "run_environment": environment,
        "statements": STATEMENTS,
        "git": git_info(REPO_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--scenario-root", type=Path, default=REPO_ROOT / "data" / "occlusion_test_scenarios" / "v2")
    parser.add_argument("--tag", default=None, help="Mac dinh tag cua plan")
    parser.add_argument("--output", type=Path, default=None, help="Mac dinh experiments/heldout_report_<tag>/report.json")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    from src.utils.logging_utils import write_json

    args = parse_args(argv)
    report = build_report(args.plan, args.output_root, args.scenario_root, args.tag)
    out = args.output or args.output_root / f"heldout_report_{report['plan']['tag']}" / "report.json"
    write_json(out, report)
    print(f"Da ghi: {out} (status {report['status']})")
    for p in report["problems"]:
        print(f"  VAN DE: {p}")
    dec = report["decision"]
    if dec["status"] == "complete":
        for det, row in dec["per_detector"].items():
            for c in row["conditions"]:
                print(f"  {det}: {c['condition']}: {c['variant']['fraction']} vs {c['reference']['fraction']} "
                      f"-> {'dat' if c['ok'] else 'TRUOT'}")
        print(f"COT QUYET DINH {dec['column']}: {dec['verdict']}")
    else:
        print(f"Chua quyet dinh duoc: {dec['reason']}")
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
