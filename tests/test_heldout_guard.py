"""Test guard heldout cua scripts/run_occlusion_ablation.py (bo chay GIA, khong YOLO).

Ke hoach la BAN SAO cua configs/heldout_report_plan.yaml, chi doi: scenario_config + config cac
cot phu -> file tam (sha256 tinh lai), dev_evidence rong, tag 'v2'. Trang thai git duoc thay
bang ham gia (cay tam ngoai repo); dinh nghia cay sach co test rieng (test_run_context.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_occlusion_ablation as abl  # noqa: E402
import scripts.run_pipeline as rp  # noqa: E402
import src.track as tr  # noqa: E402
from src.occlusion_scenarios import config_sha256  # noqa: E402
from src.utils import run_context as rc  # noqa: E402
from tests.test_run_occlusion_ablation import FakeRunner, fake_tracks_v2, make_group_v2  # noqa: E402

DETS = ("yolo11", "yolo26")
EXTRAS = {"tuned": "lost_timeout_frames: 25", "anchor": "anchor_recovery: true",
          "anchor_lost25": "lost_timeout_frames: 25\n  anchor_recovery: true"}


@pytest.fixture
def heldout(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "iter_tracks", lambda w, s, **k: iter(fake_tracks_v2(s)))

    def no_yolo(*a, **k):
        raise AssertionError("khong chay YOLO khi replay")

    monkeypatch.setattr(rp, "iter_tracks", no_yolo)
    state = {"changes": [], "untracked": []}
    monkeypatch.setattr(rc, "git_tracked_changes", lambda root=None: list(state["changes"]))
    monkeypatch.setattr(rc, "git_untracked_paths", lambda paths, root=None: list(state["untracked"]))
    weights = {}
    for d in DETS:
        weights[d] = tmp_path / f"{d}.pt"
        weights[d].write_bytes(f"weights-{d}".encode())
    scn = tmp_path / "scn"
    make_group_v2(scn, weights, group="dev")
    make_group_v2(scn, weights, group="heldout")
    scen_cfg = tmp_path / "occlusion_scenarios_v2.yaml"
    scen_cfg.write_text("kich ban gia\n", encoding="utf-8")
    idx_path = scn / "heldout" / "index.json"
    idx = json.loads(idx_path.read_text())
    idx["config_sha256"] = config_sha256(scen_cfg.read_bytes())
    idx_path.write_text(json.dumps(idx))

    plan = yaml.safe_load((REPO_ROOT / "configs" / "heldout_report_plan.yaml").read_text(encoding="utf-8"))
    plan["tag"] = "v2"
    plan["scenario_config"] = {"path": str(scen_cfg), "sha256": config_sha256(scen_cfg.read_bytes())}
    plan["dev_evidence"] = []
    extra_args = []
    for col in plan["columns"]:
        if col.get("kind") == "extra":
            f = tmp_path / f"mm_{col['extra_name']}.yaml"
            f.write_text(f"motion_memory:\n  {EXTRAS[col['extra_name']]}\n", encoding="utf-8")
            col["config"], col["sha256"] = str(f), config_sha256(f.read_bytes())
            extra_args += ["--extra-motion-memory-config", f"{col['extra_name']}={f}"]
    plan_path = tmp_path / "heldout_plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    base = ["--group", "heldout", "--tag", "v2", "--scenario-root", str(scn),
            "--output-root", str(tmp_path / "experiments"),
            "--detector", f"yolo11={weights['yolo11']}", "--detector", f"yolo26={weights['yolo26']}"]
    return {"tmp": tmp_path, "base": base, "plan": plan_path, "extras": extra_args, "state": state,
            "exps": tmp_path / "experiments"}


def args_of(h, *extra):
    return h["base"] + ["--report-plan", str(h["plan"])] + h["extras"] + list(extra)


def test_heldout_requires_report_plan(heldout):
    with pytest.raises(ValueError, match="bat buoc --report-plan"):
        abl.main(heldout["base"] + heldout["extras"] + ["--dry-run"], FakeRunner())


def test_full_heldout_run_records_plan_columns_environment(heldout, capsys):
    runner = FakeRunner()
    assert abl.main(args_of(heldout), runner) == 0
    assert "Guard heldout: OK" in capsys.readouterr().out
    assert runner.count("track") == 2 * 3                                # tracker mot lan / detector x kich ban
    exps = heldout["exps"]
    plan_sha = config_sha256(heldout["plan"].read_bytes())
    cols = {"yolo11_botsort_occl_heldout_v2": ("no_mm", "reference"),
            "yolo11_botsort_mm_occl_heldout_v2": ("mm_default", "reference"),
            "yolo11_botsort_mm_tuned_occl_heldout_v2": ("mm_tuned", "descriptive"),
            "yolo11_botsort_mm_anchor_occl_heldout_v2": ("mm_anchor", "decision"),
            "yolo11_botsort_mm_anchor_lost25_occl_heldout_v2": ("mm_anchor_lost25", "descriptive")}
    for exp, (col, role) in cols.items():
        conf = yaml.safe_load((exps / exp / "config.yaml").read_text())
        h = conf["heldout"]
        assert (h["column"], h["role"]) == (col, role)
        assert h["plan"]["sha256"] == plan_sha and h["git"]["clean"] is True
        assert "untracked" in h["git"]["clean_definition"] or "File untracked" in h["git"]["clean_definition"]
        assert set(h["weights_sha256"]) == set(DETS)
        assert (exps / exp / "metrics.json").is_file()
        if col != "no_mm":
            assert (exps / exp / "ablation_vs_yolo11_botsort_occl_heldout_v2.json").is_file()
    tconf = yaml.safe_load((exps / "yolo26_botsort_tracks_occl_heldout_v2" / "config.yaml").read_text())
    env = tconf["heldout"]["environment"]
    assert env["python"] and "hardware" in env and "versions" in env and "CPU" in tconf["heldout"]["note"]

    with pytest.raises(ValueError, match="da co metrics.json gop"):             # chi chay mot lan
        abl.main(args_of(heldout, "--dry-run"), FakeRunner())
    new_tag = [a if a != "v2" else "v2_rerun" for a in args_of(heldout, "--dry-run")]
    with pytest.raises(ValueError, match="heldout-rerun-note"):
        abl.main(new_tag, FakeRunner())
    assert abl.main(new_tag + ["--heldout-rerun-note", "lan chay moi vi ..."], FakeRunner()) == 0


@pytest.mark.parametrize("break_it,match", [
    (lambda h: (h["tmp"] / "mm_anchor.yaml").write_text("motion_memory:\n  anchor_recovery: false\n"), "sha256 cua"),
    (lambda h: h["extras"].__delitem__(slice(-2, None)), "cot phu"),
    (lambda h: h["state"].update(changes=[" M src/motion_memory.py"]), "KHONG sach"),
    (lambda h: h["state"].update(untracked=["configs/heldout_report_plan.yaml"]), "chua commit"),
    (lambda h: h["extras"].extend(["--limit-scenarios", "1"]), "limit-scenarios"),
    (lambda h: h["tmp"].joinpath("yolo26.pt").write_bytes(b"weights khac"), "weights"),
])
def test_guard_refusals(heldout, break_it, match):
    break_it(heldout)
    with pytest.raises(ValueError, match=match):
        abl.main(args_of(heldout, "--dry-run"), FakeRunner())


class FailOnce(FakeRunner):
    """run_pipeline cua cot mm_anchor o yolo26 loi lan dau (gia lap lan chay bi ngat)."""

    def __init__(self):
        super().__init__()
        self.failed = False

    def __call__(self, cmd):
        if (not self.failed and cmd[1].endswith("run_pipeline.py")
                and "yolo26_botsort_mm_anchor_occl" in " ".join(cmd)):
            self.failed = True
            self.calls.append(cmd)
            return 1
        return super().__call__(cmd)


def test_interrupted_heldout_defers_aggregate_and_resumes_only_with_flag(heldout):
    assert abl.main(args_of(heldout), FailOnce()) == 1
    exps = heldout["exps"]
    assert not list(exps.glob("*/metrics.json"))                          # chua gop o BAT KY cot nao
    with pytest.raises(ValueError, match="--resume-incomplete"):
        abl.main(args_of(heldout), FakeRunner())
    runner = FakeRunner()
    assert abl.main(args_of(heldout, "--resume-incomplete"), runner) == 0
    assert runner.count("track") == 0 and runner.count("pipeline") == 1    # chi kich ban loi
    conf = yaml.safe_load((exps / "yolo11_botsort_mm_occl_heldout_v2" / "config.yaml").read_text())
    assert len(conf["heldout_resumes"]) == 1 and conf["heldout_resumes"][0]["git_commit"] is not None
    assert len(list(exps.glob("*_occl_heldout_v2/metrics.json"))) == 2 * 5
    with pytest.raises(ValueError, match="da co metrics.json gop"):
        abl.main(args_of(heldout, "--resume-incomplete"), FakeRunner())


def test_resume_flag_without_partial_run_refused(heldout):
    with pytest.raises(ValueError, match="chua co ket qua dang do"):
        abl.main(args_of(heldout, "--resume-incomplete", "--dry-run"), FakeRunner())
