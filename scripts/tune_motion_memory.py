"""Tune Motion Memory tren nhom dev theo quy tac DANG KY TRUOC.

Quy tac: configs/motion_memory_tuning_dev.yaml (version + sha256 ghi vao moi
ket qua); logic chon: src/motion_memory_tuning.py (ham thuan, co test).

Chi REPLAY log tracker cua ablation nguon (`source_tag`, vd v2_001) - KHONG
chay YOLO, chay duoc tren CPU. Voi moi cau hinh Motion Memory va moi detector,
dung lai ham cua scripts/run_occlusion_ablation.py:
    run_pipeline --replay-tracks --motion-memory (MM config cua cau hinh)
    -> evaluate occlusion (metrics v2) -> aggregate -> compare voi baseline
    nguon (<det>_botsort_occl_<nhom>_<source_tag>; tu choi neu khac log track
    hoac khac tap kich ban da khoa). So sanh voi cau hinh `default` (diem, hop
    le, chenh lech) nam trong summary.json - `evaluate compare` chi nhan cap
    baseline (khong MM) vs MM; moi cau hinh deu khop baseline nguon tung kich
    ban nen cung log track/tap da khoa voi nhau.

Thu muc::

    experiments/<det>_botsort_mm_occl_<nhom>_tune_<id cau hinh>/   (gitignore scenarios/)
        config.yaml, metrics.json, ablation_vs_<baseline nguon>.json
        scenarios/<kich ban>/...
    experiments/motion_memory_tuning_<nhom>/                       (commit)
        config.yaml          cau hinh lan tuning (sha256 quy tac, nguon, kich ban)
        summary.json         MOI cau hinh, metrics gop, per_sequence, vong 1/2,
                             cau hinh chon + ket luan, sha256 quy tac + log tracker
        mm_configs/<id>.yaml tham so Motion Memory day du cua tung cau hinh

Buoc (chay theo thu tu)::

    python scripts/tune_motion_memory.py --round 1 --dry-run
    python scripts/tune_motion_memory.py --round 1 --jobs 16
    python scripts/tune_motion_memory.py --round 2 --jobs 16
    python scripts/tune_motion_memory.py --select

--round 2 tu choi khi vong 1 chua du (moi cau hinh x detector co metrics gop
day du). Cau hinh ket hop trung cau hinh da chay -> khong chay lai.
--select: chon cau hinh cuoi; CHI KHI chon duoc moi ghi
configs/motion_memory_tuned.yaml (khong ghi de configs/motion_memory.yaml).

Resume: kich ban da co metrics.json bi bo qua; log pipeline co ma chua co
metrics -> chi danh gia lai; loi -> bao, khong xoa.
--limit-scenarios N: chi N kich ban dau (thu nghiem), tag rieng
(mac dinh tune_limit<N>), ket qua partial - KHONG dung de chon cau hinh,
khong chay duoc vong 2/--select.
--jobs N: so tien trinh song song (moi lenh la mot subprocess Python rieng;
mac dinh so nhan CPU - 2). Moi subprocess bi gioi han 1 luong BLAS/OpenMP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
from src import motion_memory_tuning as mt  # noqa: E402

DEFAULT_TUNING_CONFIG = REPO_ROOT / "configs" / "motion_memory_tuning_dev.yaml"
DEFAULT_SCENARIO_ROOT = REPO_ROOT / "data" / "occlusion_test_scenarios" / "v2"
DEFAULT_TUNED_OUTPUT = REPO_ROOT / "configs" / "motion_memory_tuned.yaml"
PROTECTED_MM_CONFIG = REPO_ROOT / "configs" / "motion_memory.yaml"
DEFAULT_TAG = "tune"
SUMMARY_KIND = "motion_memory_tuning"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

Runner = Callable[[list[str]], int]


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


def subprocess_runner(cmd: list[str]) -> int:
    """Chay mot lenh; output giu lai, chi in khi loi (tranh tron output song song)."""
    env = {**os.environ, **{k: "1" for k in THREAD_ENV}}
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        print(f"LOI ({proc.returncode}): {' '.join(cmd[:4])} ...\n    " + "\n    ".join(tail), flush=True)
    return proc.returncode


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--round", type=int, choices=(1, 2), help="Chay vong 1 hoac 2")
    step.add_argument("--select", action="store_true", help="Chon cau hinh cuoi tu ket qua vong 1 + 2")
    parser.add_argument("--config", type=Path, default=DEFAULT_TUNING_CONFIG, help="Quy tac tuning")
    parser.add_argument("--scenario-root", type=Path, default=DEFAULT_SCENARIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--tuned-output", type=Path, default=DEFAULT_TUNED_OUTPUT,
                        help="Noi ghi cau hinh da tune (chi khi chon duoc)")
    parser.add_argument("--jobs", type=int, default=default_jobs(),
                        help=f"So lenh chay song song (mac dinh {default_jobs()} = so nhan CPU - 2)")
    parser.add_argument("--tag", default=None,
                        help=f"Tien to tag experiment (mac dinh {DEFAULT_TAG}; voi --limit-scenarios: "
                             f"{DEFAULT_TAG}_limit<N>)")
    parser.add_argument("--limit-scenarios", type=int, default=None, metavar="N",
                        help="Chi N kich ban dau (thu nghiem; partial, KHONG dung de chon)")
    parser.add_argument("--dry-run", action="store_true", help="Chi in ke hoach, khong chay, khong ghi file")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs phai >= 1")
    if args.limit_scenarios is not None and args.limit_scenarios < 1:
        parser.error("--limit-scenarios phai >= 1")
    if args.tag is None:
        args.tag = DEFAULT_TAG if args.limit_scenarios is None else f"{DEFAULT_TAG}_limit{args.limit_scenarios}"
    elif args.limit_scenarios is not None and args.tag == DEFAULT_TAG:
        parser.error(f"--limit-scenarios khong dung tag '{DEFAULT_TAG}' cua lan chay du")
    if args.select and args.limit_scenarios is not None:
        parser.error("--select khong dung voi --limit-scenarios (ket qua partial khong dung de chon)")
    if args.round == 2 and args.limit_scenarios is not None:
        parser.error("--round 2 can vong 1 day du, khong dung voi --limit-scenarios")
    return args


def _repo_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


# ---------------- trang thai lan tuning ----------------

@dataclass
class Context:
    args: argparse.Namespace
    rules: mt.TuningRules
    tuning: dict                 # path, version, sha256
    metrics: dict                # path, version, config_sha256 (abl.metrics_config_summary)
    base_mm: dict                # path, sha256, values
    index_path: Path
    index: dict
    scenarios: list[str]
    sources: dict[str, dict]     # det -> weights, tracks_dir, baseline_dir, source_mm_dir
    base_ns: argparse.Namespace  # Namespace cho ham cua run_occlusion_ablation
    manifests: dict[str, dict] = field(default_factory=dict)

    @property
    def summary_dir(self) -> Path:
        suffix = "" if self.args.tag == DEFAULT_TAG else f"_{self.args.tag}"
        return self.args.output_root / f"motion_memory_tuning_{self.rules.group}{suffix}"

    @property
    def partial(self) -> bool:
        return self.args.limit_scenarios is not None

    def mm_config_path(self, cfg_id: str) -> Path:
        return self.summary_dir / "mm_configs" / f"{cfg_id}.yaml"

    def exp_id(self, det: str, cfg_id: str) -> str:
        return abl.experiment_id(det, True, self.rules.group, f"{self.args.tag}_{cfg_id}")

    def exp_dir(self, det: str, cfg_id: str) -> Path:
        return self.args.output_root / self.exp_id(det, cfg_id)

    def manifest(self, scen: str) -> dict:
        if scen not in self.manifests:
            path = self.args.scenario_root / self.rules.group / scen / "manifest.json"
            self.manifests[scen] = json.loads(path.read_text(encoding="utf-8"))
        return self.manifests[scen]

    def tracks_log(self, det: str, scen: str) -> Path:
        return self.sources[det]["tracks_dir"] / "scenarios" / scen / abl.TRACKS_LOG


def load_tuning(path: Path) -> tuple[mt.TuningRules, dict, dict]:
    import yaml

    from src.occlusion_scenarios import config_sha256

    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    rules = mt.TuningRules.from_mapping(data)
    return rules, data, {"path": str(path), "version": rules.version, "sha256": config_sha256(raw)}


def _read_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_context(args: argparse.Namespace) -> Context:
    from src.occlusion_scenarios import file_sha256
    from src.utils.motion_memory_settings import resolve_motion_memory_config

    rules, data, tuning = load_tuning(args.config)
    metrics_path = _repo_path(data["metrics_config"])
    base_path = _repo_path(data["base_motion_memory_config"])
    base_values = asdict(resolve_motion_memory_config(None, base_path).config)
    mt.check_defaults(rules, base_values)

    index_path, index = abl.load_index(args.scenario_root, rules.group, allow_partial=False)
    scenarios = list(index["scenarios"])
    if args.limit_scenarios is not None:
        scenarios = scenarios[: args.limit_scenarios]

    group, src_tag = rules.group, rules.source_tag
    sources: dict[str, dict] = {}
    problems: list[str] = []
    for det in rules.detectors:
        tracks_dir = args.output_root / abl.tracks_experiment_id(det, group, src_tag)
        baseline_dir = args.output_root / abl.experiment_id(det, False, group, src_tag)
        cfg_path = tracks_dir / "config.yaml"
        if not cfg_path.is_file():
            problems.append(f"Khong thay {cfg_path} (ablation nguon {src_tag} chua chay?)")
            continue
        tcfg = _read_yaml(cfg_path)
        scen_cfg = tcfg.get("scenarios") or {}
        if scen_cfg.get("limit_scenarios") is not None:
            problems.append(f"{cfg_path}: log tracker nguon la lan chay --limit-scenarios")
        if scen_cfg.get("config_sha256") != index.get("config_sha256"):
            problems.append(f"{cfg_path}: config_sha256 kich ban khac index hien tai")
        bmetrics = baseline_dir / "metrics.json"
        if not bmetrics.is_file():
            problems.append(f"Khong thay {bmetrics} (baseline nguon chua gop)")
        elif not (json.loads(bmetrics.read_text(encoding="utf-8")).get("completeness") or {}).get("complete"):
            problems.append(f"{bmetrics}: baseline nguon chua day du (complete=false)")
        sources[det] = {
            "weights": str(tcfg["detector"]["weights"]),
            "tracks_dir": tracks_dir,
            "baseline_dir": baseline_dir,
            "source_mm_dir": args.output_root / abl.experiment_id(det, True, group, src_tag),
        }
    for det in sources:
        for scen in scenarios:
            log = args.output_root / abl.tracks_experiment_id(det, group, src_tag) / "scenarios" / scen / abl.TRACKS_LOG
            if not log.is_file():
                problems.append(f"Thieu log tracker {log}")
    if problems:
        raise SystemExit("Khong tune duoc - nguon chua san sang:\n  " + "\n  ".join(problems[:20])
                         + (f"\n  ... (+{len(problems) - 20})" if len(problems) > 20 else ""))

    base_ns = argparse.Namespace(
        group=group, tag=src_tag,   # tag nguon: tracker_input.tracks_experiment tro ve log v2_001
        detector=[(d, sources[d]["weights"]) for d in rules.detectors],
        scenario_root=args.scenario_root, output_root=args.output_root,
        motion_memory_config=base_path, metrics_config=metrics_path,
        device=None, allow_partial_index=False, limit_scenarios=args.limit_scenarios, dry_run=args.dry_run,
    )
    metrics = abl.metrics_config_summary(base_ns)
    if metrics["version"] != rules.metrics_config_version:
        raise SystemExit(f"Metrics config version {metrics['version']} khac quy tac "
                         f"({rules.metrics_config_version})")
    abl.check_scenarios_and_detectors(base_ns, index, metrics)
    ctx = Context(
        args=args, rules=rules, tuning=tuning, metrics=metrics,
        base_mm={"path": str(base_path), "sha256": file_sha256(base_path), "values": base_values},
        index_path=index_path, index=index, scenarios=scenarios, sources=sources, base_ns=base_ns,
    )
    if not args.dry_run:
        problems = []
        for det in rules.detectors:
            for scen in scenarios:
                found = abl.tracks_log_problems(ctx.tracks_log(det, scen),
                                                args.scenario_root / group / scen, int(ctx.manifest(scen)["frames"]))
                if found:
                    problems.append(f"{det}/{scen}: {found[0]}")
        if problems:
            raise SystemExit("Log tracker nguon khong hop le (khong xoa, xu ly tay):\n  " + "\n  ".join(problems[:20]))
    return ctx


def run_config(ctx: Context) -> dict:
    """config.yaml cua lan tuning (khong doi giua cac vong)."""
    return {
        "kind": SUMMARY_KIND,
        "purpose": "Tune Motion Memory tren dev theo quy tac dang ky truoc (CLAUDE.md muc 13)",
        "tuning_config": ctx.tuning,
        "group": ctx.rules.group,
        "tag": ctx.args.tag,
        "source_tag": ctx.rules.source_tag,
        "detectors": {d: {"weights": s["weights"], "tracks_experiment": s["tracks_dir"].name,
                          "baseline_experiment": s["baseline_dir"].name} for d, s in ctx.sources.items()},
        "scenarios": {"index": str(ctx.index_path), "config_sha256": ctx.index.get("config_sha256"),
                      "screening_sha256": (ctx.index.get("screening") or {}).get("sha256"),
                      "count": len(ctx.index["scenarios"]), "limit_scenarios": ctx.args.limit_scenarios},
        "metrics_config": ctx.metrics,
        "base_motion_memory_config": {"path": ctx.base_mm["path"], "sha256": ctx.base_mm["sha256"]},
    }


# ---------------- cau hinh + file MM ----------------

def mm_config_text(ctx: Context, cfg: dict, values: dict) -> str:
    import yaml

    return (
        f"# Sinh boi scripts/tune_motion_memory.py - cau hinh {cfg['id']} (vong {cfg['round']}).\n"
        f"# Quy tac: {Path(ctx.tuning['path']).name} version {ctx.tuning['version']} "
        f"sha256 {ctx.tuning['sha256']}\n"
        f"# Khac mac dinh: {json.dumps(cfg['overrides'], sort_keys=True)}\n"
        + yaml.safe_dump({"motion_memory": values}, sort_keys=False)
    )


def ensure_mm_config(ctx: Context, cfg: dict) -> tuple[Path, dict]:
    values = mt.config_values(ctx.base_mm["values"], cfg["overrides"])
    path = ctx.mm_config_path(cfg["id"])
    text = mm_config_text(ctx, cfg, values)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} da ton tai voi noi dung khac - khong ghi de (doi quy tac -> tag moi)")
    elif not ctx.args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path, values


def config_ns(ctx: Context, mm_path: Path) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(ctx.base_ns), "motion_memory_config": mm_path})


def spec_for(ctx: Context, det: str, cfg_id: str) -> abl.RunSpec:
    return abl.RunSpec(det, ctx.sources[det]["weights"], True, ctx.exp_id(det, cfg_id), ctx.exp_dir(det, cfg_id))


def experiment_config(ctx: Context, spec: abl.RunSpec, cfg: dict, mm_path: Path) -> dict:
    conf = abl.experiment_config(spec, config_ns(ctx, mm_path), ctx.index_path, ctx.index, ctx.metrics)
    conf["purpose"] = "Tune Motion Memory (dev, replay log tracker nguon) - mot cau hinh cua luoi"
    conf["tuning"] = {
        "config": ctx.tuning,
        "config_id": cfg["id"],
        "round": cfg["round"],
        "overrides": cfg["overrides"],
        "mm_config_sha256": hashlib.sha256(Path(mm_path).read_bytes()).hexdigest() if Path(mm_path).is_file()
        else None,
    }
    return conf


# ---------------- chay ----------------

@dataclass
class Report:
    ran: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    aggregated: list[str] = field(default_factory=list)
    compared: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    timings: dict[str, list[float]] = field(default_factory=dict)
    wall_seconds: float = 0.0


def run_configs(ctx: Context, configs: list[dict], runner: Runner) -> Report:
    args, report, lock = ctx.args, Report(), threading.Lock()
    group_dir = args.scenario_root / ctx.rules.group

    def call(cmd: list[str]) -> int:
        with lock:
            report.commands.append(cmd)
        if args.dry_run:
            return 0
        start = time.perf_counter()
        code = runner(cmd)
        with lock:
            report.timings.setdefault(abl.command_kind(cmd), []).append(time.perf_counter() - start)
        return code

    prepared = []
    for cfg in configs:
        mm_path, _ = ensure_mm_config(ctx, cfg)
        ns = config_ns(ctx, mm_path)
        for det in ctx.rules.detectors:
            spec = spec_for(ctx, det, cfg["id"])
            if not args.dry_run:
                abl.ensure_experiment_config(spec.exp_dir, experiment_config(ctx, spec, cfg, mm_path))
            prepared.append((cfg, det, spec, ns))

    units = []
    for cfg, det, spec, ns in prepared:
        for scen in ctx.scenarios:
            if (spec.exp_dir / "scenarios" / scen / "metrics.json").is_file():
                report.skipped += 1
            else:
                units.append((spec, ns, group_dir / scen, scen, det))

    def do_unit(unit) -> tuple[str, str, str]:
        spec, ns, scen_dir, scen, det = unit
        status, label = abl.postprocess_scenario(spec, scen_dir, ctx.manifest(scen), ctx.tracks_log(det, scen),
                                                 ns, call)
        return spec.exp_id, status, label

    start = time.perf_counter()
    incomplete: set[str] = set()
    for scen in ctx.scenarios:          # nap manifest truoc (khong doc song song cung dict)
        ctx.manifest(scen)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for exp_id, status, label in pool.map(do_unit, units):
            getattr(report, status).append(label)
            if status == "failed":
                incomplete.add(exp_id)

    # Gop + so sanh moi (cau hinh, detector) du kich ban.
    def do_aggregate(item) -> tuple[str, bool]:
        cfg, det, spec, _ = item
        return spec.exp_id, call(abl.aggregate_command(spec.exp_dir, ctx.index_path, ctx.partial)) == 0

    todo = [p for p in prepared if p[2].exp_id not in incomplete]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for exp_id, ok in pool.map(do_aggregate, todo):
            (report.aggregated.append(exp_id) if ok else report.failed.append(f"{exp_id} (aggregate loi)"))

    if not ctx.partial:   # compare can CUNG tap kich ban voi baseline nguon (du nhom)
        def do_compare(job) -> tuple[str, bool]:
            base_dir, mem_dir = job
            out = mem_dir / f"ablation_vs_{base_dir.name}.json"
            return f"{base_dir.name} vs {mem_dir.name}", call(abl.compare_command(base_dir, mem_dir, out)) == 0

        jobs = []
        for cfg, det, spec, _ in prepared:
            if spec.exp_id not in report.aggregated and not args.dry_run:
                continue
            jobs.append((ctx.sources[det]["baseline_dir"], spec.exp_dir))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for label, ok in pool.map(do_compare, jobs):
                (report.compared.append(label) if ok else report.failed.append(f"{label} (compare loi)"))
    report.wall_seconds = time.perf_counter() - start
    return report


# ---------------- ket qua + summary ----------------

def load_entry(ctx: Context, cfg_id: str) -> tuple[Optional[dict], dict]:
    """(entry cho motion_memory_tuning hoac None neu chua co gi, chi tiet tung detector)."""
    details: dict[str, dict] = {}
    complete = True
    any_found = False
    for det in ctx.rules.detectors:
        exp_dir = ctx.exp_dir(det, cfg_id)
        agg_path = exp_dir / "metrics.json"
        row: dict[str, Any] = {"experiment": exp_dir.name, "aggregated": agg_path.is_file()}
        if not agg_path.is_file():
            complete = False
            details[det] = row
            continue
        any_found = True
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        cmp_base = exp_dir / f"ablation_vs_{ctx.sources[det]['baseline_dir'].name}.json"
        det_complete = bool((agg.get("completeness") or {}).get("complete")) and not ctx.partial \
            and cmp_base.is_file()
        complete = complete and det_complete
        row.update({
            "complete": det_complete,
            "completeness": agg.get("completeness"),
            "motion_memory_config": agg.get("motion_memory_config"),
            "provenance": agg.get("provenance"),
            "overall": agg.get("overall"),
            "by_duration_frames": agg.get("by_duration_frames"),
            "by_occluder": agg.get("by_occluder"),
            "compare_vs_source_baseline": cmp_base.name if cmp_base.is_file() else None,
        })
        details[det] = row
    if not any_found:
        return None, details
    entry = {"complete": complete,
             "detectors": {d: r["overall"] for d, r in details.items() if r.get("overall") is not None}}
    return entry, details


def tracks_logs_sha256(ctx: Context, det: str) -> dict:
    lines = []
    for scen in ctx.scenarios:
        log = ctx.tracks_log(det, scen)
        lines.append(f"{scen}\t{hashlib.sha256(log.read_bytes()).hexdigest()}\n")
    return {"tracks_experiment": ctx.sources[det]["tracks_dir"].name, "logs": len(lines),
            "combined_sha256": hashlib.sha256("".join(lines).encode()).hexdigest()}


def _read_overall(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("overall")


def build_summary(ctx: Context) -> dict:
    from src.utils.run_context import git_info

    rules = ctx.rules
    r1 = mt.round1_configs(rules)
    results: dict[str, dict] = {}
    configs_out = []
    for cfg in r1:
        entry, details = load_entry(ctx, cfg["id"])
        if entry is not None:
            results[cfg["id"]] = entry
        configs_out.append({**cfg, "detectors": details})

    round1: dict[str, Any] = {"configs": [c["id"] for c in r1]}
    winners = None
    try:
        winners = mt.round1_winners(rules, results)
        round1.update({"status": "complete", "winners": winners})
    except mt.TuningError as exc:
        round1.update({"status": "incomplete", "reason": str(exc)})

    round2: dict[str, Any] = {"status": "waiting_for_round1"}
    if winners is not None:
        r2 = mt.round2_config(rules, winners)
        round2 = {"config": r2}
        if r2["same_as"] is not None:
            round2["status"] = f"not_run_same_as_{r2['same_as']}"
        else:
            entry, details = load_entry(ctx, mt.ROUND2_ID)
            if entry is not None:
                results[mt.ROUND2_ID] = entry
            round2["status"] = "complete" if entry and entry["complete"] else "pending"
            configs_out.append({**r2, "detectors": details})

    default = results.get(mt.DEFAULT_ID)
    for c in configs_out:
        entry = results.get(c["id"])
        try:
            c["assessment"] = {k: v for k, v in mt.assess(rules, c["id"], entry, default).items()
                               if not k.startswith("_")}
        except mt.TuningError as exc:
            c["assessment"] = {"error": str(exc)}
        mm_path = ctx.mm_config_path(c["id"])
        c["mm_config_file"] = str(mm_path.relative_to(ctx.summary_dir)) if mm_path.is_file() else None
        c["mm_config_sha256"] = hashlib.sha256(mm_path.read_bytes()).hexdigest() if mm_path.is_file() else None

    selection: dict[str, Any] = {"status": "not_available"}
    if winners is not None and not ctx.partial:
        try:
            selection = {"status": "complete", **mt.final_selection(rules, results, winners)}
        except mt.TuningError as exc:
            selection = {"status": "not_available", "reason": str(exc)}

    reproduce = {}
    for det in rules.detectors:
        src = _read_overall(ctx.sources[det]["source_mm_dir"] / "metrics.json")
        mine = (results.get(mt.DEFAULT_ID) or {}).get("detectors", {}).get(det)
        reproduce[det] = None if src is None or mine is None or ctx.partial else (src == mine)

    return {
        "kind": SUMMARY_KIND,
        "partial": ctx.partial,
        "partial_note": "--limit-scenarios: KHONG dung de chon cau hinh" if ctx.partial else None,
        "tuning_config": ctx.tuning,
        "rules": {"grid": {p: list(v) for p, v in rules.grid}, "expected_defaults": rules.defaults,
                  "min_improvement": float(rules.min_improvement), "max_rounds": rules.max_rounds},
        "metrics_config": ctx.metrics,
        "base_motion_memory_config": ctx.base_mm,
        "group": rules.group,
        "tag": ctx.args.tag,
        "scenarios": {"index": str(ctx.index_path), "config_sha256": ctx.index.get("config_sha256"),
                      "screening_sha256": (ctx.index.get("screening") or {}).get("sha256"),
                      "count": len(ctx.index["scenarios"]), "selected": len(ctx.scenarios),
                      "limit_scenarios": ctx.args.limit_scenarios},
        "source": {
            "tag": rules.source_tag,
            "tracks": {d: tracks_logs_sha256(ctx, d) for d in rules.detectors},
            "baseline": {d: {"experiment": ctx.sources[d]["baseline_dir"].name,
                             "overall": _read_overall(ctx.sources[d]["baseline_dir"] / "metrics.json")}
                         for d in rules.detectors},
        },
        "default_reproduces_source_motion_memory": reproduce,
        "configs": configs_out,
        "round1": round1,
        "round2": round2,
        "selection": selection,
        "conclusion": _conclusion_text(selection),
        "note": "metrics v2: chi kich ban da khoa truoc khi che; ket qua co dieu kien tren muc tieu ca hai "
                "detector khoa duoc (thien lech chon mau). Diem gop ca hai detector.",
        "git": git_info(REPO_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _conclusion_text(selection: dict) -> Optional[str]:
    if selection.get("status") != "complete":
        return None
    if selection["conclusion"] == mt.CONCLUSION_KEEP_DEFAULT:
        return f"GIU CAU HINH MAC DINH: {selection['reason']}"
    return (f"CHON {selection['selected_config_id']} (khac mac dinh: {selection['overrides']}); "
            f"{selection['reason']}")


def write_summary(ctx: Context, summary: dict) -> Path:
    from src.utils.logging_utils import write_json

    path = ctx.summary_dir / "summary.json"
    write_json(path, summary)
    return path


def tuned_config_text(ctx: Context, summary: dict) -> str:
    import yaml

    sel = summary["selection"]
    values = mt.config_values(ctx.base_mm["values"], sel["overrides"])
    return (
        "# Cau hinh Motion Memory DA TUNE tren dev - sinh boi scripts/tune_motion_memory.py --select.\n"
        f"# Quy tac: {Path(ctx.tuning['path']).name} version {ctx.tuning['version']} sha256 {ctx.tuning['sha256']}\n"
        f"# Cau hinh chon: {sel['selected_config_id']}; khac mac dinh: {json.dumps(sel['overrides'], sort_keys=True)}\n"
        f"# Chi tiet: {ctx.summary_dir.name}/summary.json. KHONG tinh chinh them tren heldout.\n"
        "# Dung cho heldout: scripts/run_occlusion_ablation.py --motion-memory-config <file nay>\n"
        + yaml.safe_dump({"motion_memory": values}, sort_keys=False)
    )


def write_tuned_config(ctx: Context, summary: dict) -> Optional[Path]:
    """Chi ghi khi chon duoc cau hinh; khong ghi de configs/motion_memory.yaml / file khac noi dung."""
    if summary["selection"].get("conclusion") != mt.CONCLUSION_TUNED:
        return None
    out = Path(ctx.args.tuned_output)
    if out.resolve() == PROTECTED_MM_CONFIG.resolve():
        raise SystemExit(f"Khong ghi de {PROTECTED_MM_CONFIG}")
    text = tuned_config_text(ctx, summary)
    if out.is_file() and out.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{out} da ton tai voi noi dung khac - khong ghi de")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


# ---------------- in ----------------

def print_report(ctx: Context, report: Report, configs: list[dict]) -> None:
    from collections import Counter

    counts = Counter(abl.command_kind(c) for c in report.commands)
    prefix = "[dry-run] " if ctx.args.dry_run else ""
    print(f"{prefix}{len(configs)} cau hinh x {len(ctx.rules.detectors)} detector x {len(ctx.scenarios)} "
          f"kich ban (tong nhom {len(ctx.index['scenarios'])}), --jobs {ctx.args.jobs}")
    print(f"{prefix}Lenh: " + ", ".join(f"{abl.KIND_LABELS.get(k, k)} {n}" for k, n in sorted(counts.items()))
          + f"; bo qua (da co metrics): {report.skipped}")
    if ctx.args.dry_run:
        for cfg in configs:
            print(f"  {cfg['id']}: {json.dumps(cfg['overrides'], sort_keys=True)} -> "
                  + ", ".join(ctx.exp_id(d, cfg['id']) for d in ctx.rules.detectors))
        if report.commands:
            print("  Vi du lenh: " + " ".join(report.commands[0]))
        print("[dry-run] Khong chay gi, khong ghi file.")
        return
    print(f"Hau ky: chay moi {len(report.ran)}, danh gia lai {len(report.recovered)}, loi {len(report.failed)}; "
          f"gop {len(report.aggregated)}, so sanh {len(report.compared)}")
    for item in report.failed[:30]:
        print(f"  LOI: {item}")
    if report.timings:
        print(f"Thoi gian thuc: {report.wall_seconds:.1f} s ({ctx.args.jobs} luong). Tung lenh (giay):")
        for kind, values in sorted(report.timings.items()):
            print(f"  {abl.KIND_LABELS.get(kind, kind)}: {len(values)} lan, trung binh {sum(values) / len(values):.2f}")
        per_unit = sum(sum(report.timings.get(k, [])) for k in ("pipeline", "occlusion"))
        n_units = len(report.timings.get("pipeline", [])) or 1
        full = len(ctx.index["scenarios"]) * len(ctx.rules.detectors)
        print(f"Uoc tinh moi cau hinh du nhom ({full} cap detector x kich ban): "
              f"~{per_unit / n_units * full / ctx.args.jobs / 60:.1f} phut voi {ctx.args.jobs} luong")


def print_summary(summary: dict, path: Path) -> None:
    print(f"Da ghi: {path}")
    for c in summary["configs"]:
        a = c.get("assessment") or {}
        if "score" in a:
            print(f"  {c['id']:<32} diem {a['score']:.4f} (delta {a['delta_vs_default']:+.4f}) "
                  f"hop le={a['valid']}")
    if summary["round1"].get("status") == "complete":
        for p, w in summary["round1"]["winners"].items():
            print(f"  thang {p}: {w['value'] if w['value'] is not None else 'giu mac dinh'} ({w['reason']})")
    print(f"Vong 2: {summary['round2'].get('status')}")
    if summary["conclusion"]:
        print(f"KET LUAN: {summary['conclusion']}")
    if summary["partial"]:
        print("CHU Y: ket qua partial (--limit-scenarios) - KHONG dung de chon cau hinh.")


# ---------------- main ----------------

def main(argv: Optional[list[str]] = None, runner: Runner = subprocess_runner) -> int:
    args = parse_args(argv)
    ctx = build_context(args)
    if not args.dry_run:
        abl.ensure_experiment_config(ctx.summary_dir, run_config(ctx))

    configs: list[dict] = []
    if args.round == 1:
        configs = mt.round1_configs(ctx.rules)
    elif args.round == 2:
        summary = build_summary(ctx)
        if summary["round1"]["status"] != "complete":
            raise SystemExit(f"Vong 1 chua du - khong chay vong 2: {summary['round1']['reason']}")
        r2 = summary["round2"]["config"]
        if r2["same_as"] is not None:
            print(f"Vong 2: cau hinh ket hop {r2['overrides']} trung {r2['same_as']} (da chay) - khong chay lai.")
        else:
            configs = [r2]

    failed = False
    if configs:
        report = run_configs(ctx, configs, runner)
        print_report(ctx, report, configs)
        failed = bool(report.failed)
    if args.dry_run:
        return 0
    summary = build_summary(ctx)
    path = write_summary(ctx, summary)
    print_summary(summary, path)
    if args.select:
        if summary["selection"].get("status") != "complete":
            print(f"Chua chon duoc: {summary['selection'].get('reason') or summary['round1'].get('reason')}")
            return 1
        out = write_tuned_config(ctx, summary)
        print(f"Da ghi cau hinh da tune: {out}" if out else
              "Khong ghi configs/motion_memory_tuned.yaml (ket luan: giu cau hinh mac dinh).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
