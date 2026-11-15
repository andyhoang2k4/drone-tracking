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

Che do bien the (--variant <file khai bao>, vd configs/motion_memory_anchor_dev.yaml):
chay cac bien the Motion Memory co dinh trong file khai bao (cung log tracker
nguon, cung ha tang: tham so tu meta, pre-flight, dung som, song song), roi
kiem tra hoi quy + ap quy tac quyet dinh cua file (src/mm_variant_eval.py) ->
experiments/<name>/{config.yaml,summary.json,mm_configs/}::

    python scripts/tune_motion_memory.py --variant configs/motion_memory_anchor_dev.yaml --dry-run
    python scripts/tune_motion_memory.py --variant configs/motion_memory_anchor_dev.yaml --jobs 16

Code tu 2026-11-15 co them tham so anchor_recovery trong MotionMemoryConfig:
khong chay lai vong tuning cu (tag tune) bang code moi (mm_configs/tuned se
khac noi dung -> bi tu choi); ket qua tuning cu chi doc.

Resume: kich ban da co metrics.json bi bo qua; log pipeline co ma chua co
metrics -> chi danh gia lai; loi -> bao, khong xoa.
--limit-scenarios N: chi N kich ban dau (thu nghiem), tag rieng tu danh so
tune_limit<N>_run<k> (k nho nhat chua dung - khong dung vao thu muc cua lan
chay truoc; resume: --tag <tag do>; tag dinh dang cu tune_limit<N> bi tu
choi), ket qua partial - KHONG dung de chon cau hinh,
khong chay duoc vong 2/--select.
--jobs N: so tien trinh song song (moi lenh la mot subprocess Python rieng;
mac dinh so nhan CPU - 2). Moi subprocess: 1 luong BLAS/OpenMP, khong thay GPU
(CUDA_VISIBLE_DEVICES rong) - run_context cua lan replay ghi khong co GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
from src import mm_variant_eval as ve  # noqa: E402
from src import motion_memory_tuning as mt  # noqa: E402

DEFAULT_TUNING_CONFIG = REPO_ROOT / "configs" / "motion_memory_tuning_dev.yaml"
DEFAULT_SCENARIO_ROOT = REPO_ROOT / "data" / "occlusion_test_scenarios" / "v2"
DEFAULT_TUNED_OUTPUT = REPO_ROOT / "configs" / "motion_memory_tuned.yaml"
PROTECTED_MM_CONFIG = REPO_ROOT / "configs" / "motion_memory.yaml"
DEFAULT_TAG = "tune"
SUMMARY_KIND = "motion_memory_tuning"
VARIANT_KIND = "motion_memory_variant"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Runner tra ve ma thoat, hoac (ma thoat, output) - subprocess_runner tra output de bao loi day du.
Runner = Callable[[list[str]], Any]


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 3) - 2)


def subprocess_env() -> dict[str, str]:
    """Moi subprocess: 1 luong BLAS/OpenMP, AN GPU (replay chi can CPU; tranh moi
    tien trinh song song tao mot CUDA context khi run_context hoi thong tin GPU)."""
    return {**os.environ, **{k: "1" for k in THREAD_ENV}, "CUDA_VISIBLE_DEVICES": ""}


def subprocess_runner(cmd: list[str]) -> tuple[int, str]:
    """Chay mot lenh (cwd = goc repo); tra ve (ma thoat, stdout + stderr) de bao loi day du."""
    env = subprocess_env()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def run_cmd(runner: "Runner", cmd: list[str]) -> tuple[int, str]:
    """Runner tra ve ma thoat (int) hoac (ma thoat, output)."""
    result = runner(cmd)
    if isinstance(result, tuple):
        return int(result[0]), str(result[1] or "")
    return int(result), ""


def output_tail(output: str, lines: int = 60) -> str:
    rows = output.strip().splitlines()
    return "\n".join(("    " + r) for r in rows[-lines:]) if rows else "    (khong co output)"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    step = parser.add_mutually_exclusive_group(required=True)
    step.add_argument("--round", type=int, choices=(1, 2), help="Chay vong 1 hoac 2")
    step.add_argument("--select", action="store_true", help="Chon cau hinh cuoi tu ket qua vong 1 + 2")
    step.add_argument("--variant", type=Path, default=None,
                      help="File khai bao bien the (vd configs/motion_memory_anchor_dev.yaml)")
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
    full_tag = load_variant(args.variant)[0].tag if args.variant is not None else DEFAULT_TAG
    if args.tag is None:
        args.tag = full_tag if args.limit_scenarios is None else next_limit_tag(
            args.output_root, args.limit_scenarios, full_tag)
    elif args.limit_scenarios is not None and args.tag == full_tag:
        parser.error(f"--limit-scenarios khong dung tag '{full_tag}' cua lan chay du")
    elif re.fullmatch(rf"{DEFAULT_TAG}_limit\d+", args.tag):
        parser.error(f"tag '{args.tag}' la dinh dang cu (truoc ban sua 2026-11-14), danh cho lan chay loi da co - "
                     f"khong dung lai; bo --tag de tu danh so ({DEFAULT_TAG}_limit<N>_run<k>)")
    if args.select and args.limit_scenarios is not None:
        parser.error("--select khong dung voi --limit-scenarios (ket qua partial khong dung de chon)")
    if args.round == 2 and args.limit_scenarios is not None:
        parser.error("--round 2 can vong 1 day du, khong dung voi --limit-scenarios")
    return args


def tag_in_use(output_root: Path, tag: str) -> bool:
    """Da co thu muc nao cua tag (summary hoac experiment tung cau hinh, moi nhom)?"""
    root = Path(output_root)
    return any(root.glob(f"*_{tag}")) or any(root.glob(f"*_occl_*_{tag}_*"))


def next_limit_tag(output_root: Path, limit: int, prefix: str = DEFAULT_TAG) -> str:
    """Tag tu dong cho lan chay thu --limit-scenarios: <tune>_limit<N>_run<k>, k nho nhat CHUA
    dung -> khong bao gio ghi vao thu muc cua lan chay truoc (ke ca lan chay loi; dinh dang cu
    <tune>_limit<N> khong bao gio duoc sinh lai). Muon resume mot lan chay thu: --tag <tag do>."""
    k = 1
    while tag_in_use(output_root, f"{prefix}_limit{limit}_run{k}"):
        k += 1
    return f"{prefix}_limit{limit}_run{k}"


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
    replay: dict[tuple[str, str], dict] = field(default_factory=dict)
    plan: Optional[ve.VariantPlan] = None   # che do --variant (rules = plan)

    @property
    def summary_dir(self) -> Path:
        if self.plan is not None:
            suffix = "" if self.args.tag == self.plan.tag else f"_{self.args.tag}"
            return self.args.output_root / f"{self.plan.name}{suffix}"
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

    def replay_inputs(self, det: str, scen: str) -> dict:
        """Tham so replay lay tu CHINH tracks.meta.json cua log tracker nguon.

        run_pipeline --replay-tracks doi chieu weights/source/tracker voi meta
        (apply_replay_meta), va `evaluate compare` doi chieu CHUOI weights/source/
        tracker voi baseline nguon (cung chuoi da truyen cho src.track). Vi vay
        dung nguyen chuoi cua meta, khong tu dung duong dan moi (vd tu
        --scenario-root tuyet doi khi ablation nguon chay voi duong dan tuong doi).
        """
        key = (det, scen)
        if key not in self.replay:
            self.replay[key] = _replay_inputs_from_meta(self, det, scen)
        return self.replay[key]


def _resolve_run_path(value: str) -> Path:
    """Duong dan trong meta/CLI cua lenh con (cwd = goc repo) -> duong dan tuyet doi."""
    p = Path(value)
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def _replay_inputs_from_meta(ctx: "Context", det: str, scen: str) -> dict:
    from scripts.run_pipeline import REPLAY_PATH_KEYS, replay_meta_path

    log = ctx.tracks_log(det, scen)
    meta_path = replay_meta_path(log)
    if not meta_path.is_file():
        raise ValueError(f"Khong thay {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [k for k in REPLAY_PATH_KEYS if not meta.get(k)]
    if missing:
        raise ValueError(f"{meta_path}: thieu {missing}")
    source = Path(str(meta["source"]))
    if source.name != "images" or source.parent.name != scen:
        raise ValueError(f"{meta_path}: source={meta['source']!r} khong phai <...>/{scen}/images")
    expected = (ctx.args.scenario_root / ctx.rules.group / scen / "images").resolve()
    if _resolve_run_path(str(source)) != expected:
        raise ValueError(f"{meta_path}: source={meta['source']!r} khong tro toi {expected} (--scenario-root khac "
                         "noi ablation nguon da dung?)")
    if Path(str(meta["weights"])) != Path(ctx.sources[det]["weights"]):
        raise ValueError(f"{meta_path}: weights={meta['weights']!r} khac config.yaml cua log tracker "
                         f"({ctx.sources[det]['weights']!r})")
    return {"scen_dir": source.parent, "weights": str(meta["weights"]), "meta": meta}


def load_variant(path: Path) -> tuple[ve.VariantPlan, dict, dict]:
    """File khai bao bien the -> (plan, du lieu, {path, version, sha256})."""
    import yaml

    from src.occlusion_scenarios import config_sha256

    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    plan = ve.VariantPlan.from_mapping(data)
    return plan, data, {"path": str(path), "version": plan.version, "sha256": config_sha256(raw)}


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

    plan: Optional[ve.VariantPlan] = None
    if getattr(args, "variant", None) is not None:
        plan, data, tuning = load_variant(args.variant)
        rules = plan   # cung thuoc tinh group/detectors/source_tag/metrics_config_version
    else:
        rules, data, tuning = load_tuning(args.config)
    metrics_path = _repo_path(data["metrics_config"])
    base_path = _repo_path(data["base_motion_memory_config"])
    base_values = asdict(resolve_motion_memory_config(None, base_path).config)
    if plan is not None:
        ve.check_defaults(plan, base_values)
    else:
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
        index_path=index_path, index=index, scenarios=scenarios, sources=sources, base_ns=base_ns, plan=plan,
    )
    problems = []
    for det in rules.detectors:
        for scen in scenarios:
            try:
                ctx.replay_inputs(det, scen)       # nap truoc (khong doc song song)
            except ValueError as exc:
                problems.append(str(exc))
    if problems:
        raise SystemExit("Meta cua log tracker nguon khong dung duoc:\n  " + "\n  ".join(problems[:20]))
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
        "kind": VARIANT_KIND if ctx.plan is not None else SUMMARY_KIND,
        "purpose": ("Danh gia bien the Motion Memory tren dev theo file khai bao truoc (CLAUDE.md muc 13)"
                    if ctx.plan is not None else
                    "Tune Motion Memory tren dev theo quy tac dang ky truoc (CLAUDE.md muc 13)"),
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


def unit_inputs(ctx: Context, spec: abl.RunSpec, det: str, scen: str) -> tuple[abl.RunSpec, Path]:
    """(RunSpec voi weights cua meta, thu muc kich ban theo source cua meta) cho mot lan replay."""
    from dataclasses import replace

    inputs = ctx.replay_inputs(det, scen)
    return replace(spec, weights=inputs["weights"]), inputs["scen_dir"]


def unit_pipeline_command(ctx: Context, det: str, cfg_id: str, scen: str, mm_path: Path,
                          exp_dir: Optional[Path] = None) -> list[str]:
    """Lenh run_pipeline replay dung nhu do_unit se chay (exp_dir khac -> chay thu o cho khac)."""
    spec = spec_for(ctx, det, cfg_id)
    if exp_dir is not None:
        spec = abl.RunSpec(spec.detector, spec.weights, True, spec.exp_id, exp_dir)
    replay_spec, scen_dir = unit_inputs(ctx, spec, det, scen)
    return abl.pipeline_command(replay_spec, scen_dir, ctx.manifest(scen), ctx.tracks_log(det, scen),
                                config_ns(ctx, mm_path))


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
    cancelled: int = 0
    stopped: Optional[dict] = None   # dung som: dot, thong bao, so lenh, vi du lenh + output


# Dung som: trong mot dot (hau ky / gop / so sanh), khi da chay >= min(N, so muc
# cua dot) lenh va > 50% lenh da chay loi CUNG mot thong bao -> dung dot, khong
# chay tiep cac dot sau.
EARLY_STOP_MIN_COMMANDS = 20
EARLY_STOP_FRACTION = 0.5
_EXC_LINE = re.compile(r"^[\w.]*(Error|Exception|Exit)\b")


def error_signature(output: str, code: int) -> str:
    """Thong bao dai dien cua mot lenh loi: dong exception cuoi (hoac dong cuoi), bo
    phan rieng tung kich ban (chuoi trong dau nhay, so) de cac lan loi cung loai trung nhau."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    line = next((x for x in reversed(lines) if _EXC_LINE.match(x)), lines[-1] if lines else "")
    if not line:
        return f"ma thoat {code}, khong co output"
    line = re.sub(r"'[^']*'", "'...'", line)
    line = re.sub(r'"[^"]*"', '"..."', line)
    return re.sub(r"\d+", "N", line)


class BatchMonitor:
    """Dem lenh da chay / loi theo thong bao trong mot dot; quyet dinh dung som."""

    def __init__(self, name: str, items: int):
        self.name = name
        self.min_commands = max(1, min(EARLY_STOP_MIN_COMMANDS, items))
        self.lock = threading.Lock()
        self.done = 0
        self.counts: Counter = Counter()
        self.examples: dict[str, tuple[list[str], str]] = {}
        self.stop = threading.Event()
        self.stopped: Optional[dict] = None

    def record(self, cmd: list[str], code: int, output: str) -> None:
        with self.lock:
            self.done += 1
            if code == 0:
                return
            sig = error_signature(output, code)
            self.counts[sig] += 1
            self.examples.setdefault(sig, (cmd, output))
            if (self.stopped is None and self.done >= self.min_commands
                    and self.counts[sig] > EARLY_STOP_FRACTION * self.done):
                ex_cmd, ex_out = self.examples[sig]
                self.stopped = {"batch": self.name, "signature": sig, "failed": self.counts[sig],
                                "commands_done": self.done, "example_command": ex_cmd, "example_output": ex_out}
                self.stop.set()


def preflight_static(ctx: Context, configs: list[dict]) -> None:
    """Moi lenh replay cua vong phai qua dung kiem tra cua run_pipeline (parse_args +
    apply_replay_meta) voi meta cua log - khong chay gi, chay ca khi --dry-run.
    Giua cac cau hinh chi --motion-memory-config khac nhau (khong bi doi chieu voi
    meta) nen kiem tren cau hinh dau cua vong."""
    from scripts.run_pipeline import ReplayError, apply_replay_meta
    from scripts.run_pipeline import parse_args as pipeline_parse_args

    cfg = configs[0]
    problems = []
    for det in ctx.rules.detectors:
        for scen in ctx.scenarios:
            cmd = unit_pipeline_command(ctx, det, cfg["id"], scen, ctx.mm_config_path(cfg["id"]))
            try:
                apply_replay_meta(pipeline_parse_args(cmd[2:]), dict(ctx.replay_inputs(det, scen)["meta"]))
            except (ReplayError, SystemExit) as exc:
                problems.append((det, scen, cmd, str(exc)))
    if problems:
        det, scen, cmd, msg = problems[0]
        raise SystemExit(
            f"Pre-flight: {len(problems)}/{len(ctx.rules.detectors) * len(ctx.scenarios)} lenh replay KHONG khop "
            f"meta cua log tracker - khong chay.\nVi du {det}/{scen}:\n{msg}\nLenh: {' '.join(cmd)}"
        )


def preflight_live(ctx: Context, configs: list[dict], runner: Runner) -> None:
    """Chay THAT dung mot lenh replay (kich ban dau, cau hinh dau cua vong) cho moi detector,
    ghi vao thu muc tam (khong dung toi experiment). Loi -> dung ngay, in day du."""
    import tempfile

    cfg, scen = configs[0], ctx.scenarios[0]
    with tempfile.TemporaryDirectory(prefix="tune_preflight_") as tmp:
        for det in ctx.rules.detectors:
            cmd = unit_pipeline_command(ctx, det, cfg["id"], scen, ctx.mm_config_path(cfg["id"]),
                                        exp_dir=Path(tmp) / det)
            start = time.perf_counter()
            code, output = run_cmd(runner, cmd)
            if code != 0:
                raise SystemExit(
                    f"Pre-flight THAT BAI ({det}, {cfg['id']}, {scen}): run_pipeline replay thoat ma {code} - "
                    f"dung truoc khi chay ca vong.\nLenh: {' '.join(cmd)}\nOutput:\n{output_tail(output)}"
                )
            print(f"Pre-flight {det}: replay {scen} OK ({time.perf_counter() - start:.1f} s)", flush=True)


def run_configs(ctx: Context, configs: list[dict], runner: Runner) -> Report:
    args, report, lock = ctx.args, Report(), threading.Lock()
    group_dir = args.scenario_root / ctx.rules.group

    monitor = {"current": BatchMonitor("", 1)}

    def call(cmd: list[str]) -> int:
        with lock:
            report.commands.append(cmd)
        if args.dry_run:
            return 0
        start = time.perf_counter()
        code, output = run_cmd(runner, cmd)
        with lock:
            report.timings.setdefault(abl.command_kind(cmd), []).append(time.perf_counter() - start)
        if code != 0:
            print(f"LOI ({code}): {' '.join(cmd[:4])} ...\n{output_tail(output, 10)}", flush=True)
        monitor["current"].record(cmd, code, output)
        return code

    def run_batch(name: str, items: list, work: Callable) -> list:
        """Chay `work` song song tren items; dung som theo BatchMonitor. Tra ve ket qua
        cac muc da chay (muc bi bo do dung som khong co trong ket qua)."""
        mon = BatchMonitor(name, len(items))
        monitor["current"] = mon

        def guarded(item):
            return None if mon.stop.is_set() else work(item)

        results = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for fut in as_completed([pool.submit(guarded, it) for it in items]):
                res = fut.result()
                if res is None:
                    report.cancelled += 1
                else:
                    results.append(res)
        if mon.stopped is not None:
            report.stopped = mon.stopped
        return results

    prepared = []
    for cfg in configs:
        mm_path, _ = ensure_mm_config(ctx, cfg)
        ns = config_ns(ctx, mm_path)
        for det in ctx.rules.detectors:
            spec = spec_for(ctx, det, cfg["id"])
            if not args.dry_run:
                abl.ensure_experiment_config(spec.exp_dir, experiment_config(ctx, spec, cfg, mm_path))
            prepared.append((cfg, det, spec, ns))

    preflight_static(ctx, configs)
    if not args.dry_run:
        preflight_live(ctx, configs, runner)

    units = []
    for cfg, det, spec, ns in prepared:
        for scen in ctx.scenarios:
            if (spec.exp_dir / "scenarios" / scen / "metrics.json").is_file():
                report.skipped += 1
            else:
                units.append((spec, ns, group_dir / scen, scen, det))

    def do_unit(unit) -> tuple[str, str, str]:
        spec, ns, _, scen, det = unit
        replay_spec, scen_dir = unit_inputs(ctx, spec, det, scen)
        status, label = abl.postprocess_scenario(replay_spec, scen_dir, ctx.manifest(scen),
                                                 ctx.tracks_log(det, scen), ns, call)
        return spec.exp_id, status, label

    start = time.perf_counter()
    incomplete: set[str] = set()
    for scen in ctx.scenarios:          # nap manifest truoc (khong doc song song cung dict)
        ctx.manifest(scen)
    for exp_id, status, label in run_batch("hau ky (run_pipeline + evaluate occlusion)", units, do_unit):
        getattr(report, status).append(label)
        if status == "failed":
            incomplete.add(exp_id)
    if report.stopped is not None:
        report.wall_seconds = time.perf_counter() - start
        return report

    # Gop + so sanh moi (cau hinh, detector) du kich ban.
    def do_aggregate(item) -> tuple[str, bool]:
        cfg, det, spec, _ = item
        return spec.exp_id, call(abl.aggregate_command(spec.exp_dir, ctx.index_path, ctx.partial)) == 0

    todo = [p for p in prepared if p[2].exp_id not in incomplete]
    for exp_id, ok in run_batch("evaluate aggregate", todo, do_aggregate):
        (report.aggregated.append(exp_id) if ok else report.failed.append(f"{exp_id} (aggregate loi)"))

    if not ctx.partial and report.stopped is None:   # compare can CUNG tap kich ban voi baseline nguon (du nhom)
        def do_compare(job) -> tuple[str, bool]:
            base_dir, mem_dir = job
            out = mem_dir / f"ablation_vs_{base_dir.name}.json"
            return f"{base_dir.name} vs {mem_dir.name}", call(abl.compare_command(base_dir, mem_dir, out)) == 0

        jobs = []
        for cfg, det, spec, _ in prepared:
            if spec.exp_id not in report.aggregated and not args.dry_run:
                continue
            jobs.append((ctx.sources[det]["baseline_dir"], spec.exp_dir))
        for label, ok in run_batch("evaluate compare", jobs, do_compare):
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


# ---------------- che do bien the (--variant) ----------------

def variant_configs(plan: ve.VariantPlan) -> list[dict]:
    return [{"id": v.id, "round": "variant", "role": v.role, "param": None, "value": None,
             "overrides": dict(v.overrides)} for v in plan.variants]


def verify_check_files(ctx: Context) -> dict[str, dict]:
    """Bien the co check_file (vd anchor_lost25 vs configs/motion_memory_tuned.yaml): tham so
    file (da resolve) + check_extra phai trung tham so bien the; ghi sha256 file. Lech/thieu -> dung."""
    from src.occlusion_scenarios import file_sha256
    from src.utils.motion_memory_settings import resolve_motion_memory_config

    out = {}
    for v in ctx.plan.variants:
        if not v.check_file:
            continue
        path = _repo_path(v.check_file)
        if not path.is_file():
            raise SystemExit(f"Bien the {v.id}: khong thay {path} (check_file) - khong chay")
        file_values = asdict(resolve_motion_memory_config(None, path).config)
        try:
            ve.check_file_values(v, mt.config_values(ctx.base_mm["values"], v.overrides), file_values)
        except ve.VariantPlanError as exc:
            raise SystemExit(str(exc)) from exc
        out[v.id] = {"path": str(v.check_file), "sha256": file_sha256(path)}
    return out


def _read_agg(exp_dir: Path) -> Optional[dict]:
    path = exp_dir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def variant_aggs(ctx: Context, vid: str) -> dict[str, dict]:
    """{det: metrics.json gop} cua mot bien the; complete chi khi aggregate du + co compare voi
    baseline nguon + khong partial (ghi de completeness.complete cua ban sao)."""
    out = {}
    for det in ctx.rules.detectors:
        exp_dir = ctx.exp_dir(det, vid)
        agg = _read_agg(exp_dir)
        if agg is None:
            continue
        cmp_ok = (exp_dir / f"ablation_vs_{ctx.sources[det]['baseline_dir'].name}.json").is_file()
        complete = bool((agg.get("completeness") or {}).get("complete")) and cmp_ok and not ctx.partial
        out[det] = {**agg, "completeness": {**(agg.get("completeness") or {}), "complete": complete}}
    return out


def reference_aggs(ctx: Context) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for name in ctx.plan.references:
        out[name] = {}
        for det in ctx.rules.detectors:
            agg = _read_agg(ctx.args.output_root / ctx.plan.reference_experiment(name, det))
            if agg is not None:
                out[name][det] = agg
    return out


def recover_reason_counts(exp_dir: Path) -> Optional[dict]:
    """So kich ban DA KHOA co recover_reason anchor_id / motion_gate (metrics tung kich ban), theo duration."""
    from src.evaluate import load_scenario_metrics

    if not (exp_dir / "scenarios").is_dir():
        return None
    out: dict[str, dict[str, int]] = {}
    for m in load_scenario_metrics(exp_dir).values():
        metrics = m.get("metrics") or {}
        if not metrics.get("target_locked_at_occlusion"):
            continue
        for key in ("overall", str(m["scenario"]["duration_frames"])):
            row = out.setdefault(key, {"scenarios": 0, "anchor_id": 0, "motion_gate": 0})
            row["scenarios"] += 1
            for reason in ("anchor_id", "motion_gate"):
                row[reason] += reason in (metrics.get("recover_reasons") or [])
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "overall", int(kv[0]) if kv[0].isdigit() else 0)))


def build_variant_summary(ctx: Context, check_files: dict) -> dict:
    from src.utils.run_context import git_info

    plan = ctx.plan
    dets = plan.detectors
    variants = {v.id: variant_aggs(ctx, v.id) for v in plan.variants}
    refs = reference_aggs(ctx)
    status, regression, decision, reason = "complete", None, None, None
    reg_v, reg_ref = plan.regression["variant"], plan.regression["reference"]
    try:
        ve.check_complete(reg_v, variants[reg_v], dets)
        ve.check_complete(reg_ref, refs.get(reg_ref) or {}, dets)
        regression = ve.regression_check(plan, variants[reg_v], refs[reg_ref])
        if not regression["ok"]:
            status = "blocked_by_regression"
            reason = (f"{reg_v} KHONG trung {reg_ref} ({regression['n_differences']} khac biet) - DUNG, khong "
                      "dien giai ket qua bien the")
        else:
            decision = ve.decide(plan, variants[plan.decision["variant"]], refs)
    except ve.VariantPlanError as exc:
        status, reason = "not_available", str(exc)
    columns = {**{f"ref:{n}": refs.get(n) or {} for n in plan.references},
               **{f"variant:{v}": a for v, a in variants.items()}}
    reasons = {f"ref:{n}": {d: recover_reason_counts(ctx.args.output_root / plan.reference_experiment(n, d))
                            for d in dets} for n in plan.references}
    reasons.update({f"variant:{v.id}": {d: recover_reason_counts(ctx.exp_dir(d, v.id)) for d in dets}
                    for v in plan.variants})
    return {
        "kind": VARIANT_KIND,
        "status": status,
        "reason": reason,
        "partial": ctx.partial,
        "plan": ctx.tuning,
        "variants": [{"id": v.id, "role": v.role, "overrides": v.overrides,
                      "experiments": {d: ctx.exp_id(d, v.id) for d in dets},
                      "mm_config_file": (str(ctx.mm_config_path(v.id).relative_to(ctx.summary_dir))
                                         if ctx.mm_config_path(v.id).is_file() else None),
                      "mm_config_sha256": (hashlib.sha256(ctx.mm_config_path(v.id).read_bytes()).hexdigest()
                                           if ctx.mm_config_path(v.id).is_file() else None)}
                     for v in plan.variants],
        "check_files": check_files,
        "references": {n: {d: plan.reference_experiment(n, d) for d in dets} for n in plan.references},
        "regression": regression,
        "decision": decision,
        "descriptive": {col: ve.describe(aggs, dets) for col, aggs in columns.items()},
        "recover_reason_scenarios": reasons,
        "metrics_config": ctx.metrics,
        "base_motion_memory_config": ctx.base_mm,
        "scenarios": {"index": str(ctx.index_path), "config_sha256": ctx.index.get("config_sha256"),
                      "count": len(ctx.index["scenarios"]), "selected": len(ctx.scenarios),
                      "limit_scenarios": ctx.args.limit_scenarios},
        "source": {"tag": plan.source_tag, "tracks": {d: tracks_logs_sha256(ctx, d) for d in dets}},
        "note": "metrics v2: chi kich ban da khoa truoc khi che; thien lech chon mau (chi muc tieu ca hai "
                "detector khoa duoc). Cot mo ta khong dung de chon cau hinh.",
        "git": git_info(REPO_ROOT),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def print_variant_summary(summary: dict, path: Path) -> None:
    print(f"Da ghi: {path}")
    if summary["partial"]:
        print("CHU Y: ket qua partial (--limit-scenarios) - KHONG dung de quyet dinh.")
    if summary["status"] == "blocked_by_regression":
        print(f"LOI HOI QUY: {summary['reason']}")
        for d in summary["regression"]["differences"][:10]:
            print(f"  {d}")
        return
    if summary["status"] != "complete":
        print(f"Chua quyet dinh duoc: {summary['reason']}")
        return
    print("Hoi quy: default_regression trung tham chieu (overall + by_duration_frames).")
    dec = summary["decision"]
    for det, row in dec["per_detector"].items():
        for c in row["conditions"]:
            print(f"  {det}: {c['condition']}: {c['variant']['fraction']} vs {c['reference']['fraction']} "
                  f"-> {'dat' if c['ok'] else 'TRUOT'}")
    print(f"KET LUAN: {dec['conclusion']}")


# ---------------- main ----------------

def main(argv: Optional[list[str]] = None, runner: Runner = subprocess_runner) -> int:
    args = parse_args(argv)
    ctx = build_context(args)
    if args.limit_scenarios is not None:
        print(f"Tag lan chay thu: {args.tag} (resume: --tag {args.tag}); ket qua o {ctx.summary_dir.name}/")
    if not args.dry_run:
        abl.ensure_experiment_config(ctx.summary_dir, run_config(ctx))

    configs: list[dict] = []
    if ctx.plan is not None:
        check_files = verify_check_files(ctx)
        configs = variant_configs(ctx.plan)
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
        failed = bool(report.failed) or report.stopped is not None
        if report.stopped is not None:
            st = report.stopped
            print(f"DUNG SOM (dot {st['batch']}): {st['failed']}/{st['commands_done']} lenh da chay loi cung thong "
                  f"bao (> {EARLY_STOP_FRACTION:.0%}); bo {report.cancelled} muc chua chay.\n"
                  f"Thong bao: {st['signature']}\nVi du lenh: {' '.join(st['example_command'])}\n"
                  f"Output:\n{output_tail(st['example_output'])}", flush=True)
    if args.dry_run:
        return 0
    if ctx.plan is not None:
        summary = build_variant_summary(ctx, check_files)
        path = write_summary(ctx, summary)
        print_variant_summary(summary, path)
        # partial (--limit-scenarios): khong quyet dinh la dung thiet ke -> chi loi lenh moi la loi
        return 1 if failed or (summary["status"] != "complete" and not ctx.partial) else 0
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
