"""Test scripts/report_heldout.py tren ket qua heldout GIA (chay bang guard + runner that, YOLO gia).

Dieu kien (i) cua file khai bao dev dung d15/d30; kich ban gia chi co d5 nen test dung BAN SAO
cua configs/motion_memory_anchor_dev.yaml voi durations [5] (sha256 tinh lai trong plan tam).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.report_heldout as rh  # noqa: E402
import scripts.run_occlusion_ablation as abl  # noqa: E402
from src.occlusion_scenarios import config_sha256  # noqa: E402
from tests.test_heldout_guard import args_of, heldout  # noqa: E402,F401
from tests.test_run_occlusion_ablation import FakeRunner  # noqa: E402


def use_d5_rules(h) -> None:
    dev = yaml.safe_load((REPO_ROOT / "configs" / "motion_memory_anchor_dev.yaml").read_text(encoding="utf-8"))
    dev["decision"]["reid_not_below"][0]["durations"] = [5]
    rules = h["tmp"] / "anchor_rules_d5.yaml"
    rules.write_text(yaml.safe_dump(dev, sort_keys=False), encoding="utf-8")
    plan = yaml.safe_load(h["plan"].read_text(encoding="utf-8"))
    plan["decision"]["rules_from"] = {"path": str(rules), "sha256": config_sha256(rules.read_bytes())}
    h["plan"].write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def test_report_covers_all_columns_decision_bootstrap_diagnosis(heldout, capsys):
    use_d5_rules(heldout)
    assert abl.main(args_of(heldout), FakeRunner()) == 0
    out = heldout["tmp"] / "report.json"
    code = rh.main(["--plan", str(heldout["plan"]), "--output-root", str(heldout["exps"]),
                    "--scenario-root", str(heldout["tmp"] / "scn"), "--tag", "v2", "--output", str(out)])
    r = json.loads(out.read_text())
    assert code == 0 and r["status"] == "complete" and r["problems"] == []
    assert r["columns_declared"] == ["no_mm", "mm_default", "mm_tuned", "mm_anchor", "mm_anchor_lost25"]
    assert set(r["tables"]) == set(r["columns_declared"])
    t = r["tables"]["mm_anchor"]
    assert t["role"] == "decision"
    assert t["per_detector"]["yolo11"]["overall"]["box_precision"] is not None
    assert "5" in t["per_detector"]["yolo26"]["by_duration_frames"]
    assert t["pooled_detectors"]["overall"]["scenarios_evaluated"] == 4             # 2 da khoa x 2 detector
    assert t["counts"]["yolo11"]["independent_sequences"] == 1
    d = r["decision"]
    assert d["status"] == "complete" and d["column"] == "mm_anchor" and d["verdict"] in ("DAT", "KHONG DAT")
    assert len(d["per_detector"]["yolo11"]["conditions"]) == 3                       # (i) d5, (iii) d5, (ii) cpr
    b = r["bootstrap"]
    assert b["status"] == "complete" and b["settings"]["seed"] == 20261115 and b["settings"]["resamples"] == 2000
    assert "mm_anchor - no_mm" in b["paired_differences"]["pooled"]["overall"]
    assert "mm_anchor - mm_default" in b["paired_differences"]["yolo11"]["d5"]
    assert set(r["diagnosis"]) == {"mm_default", "mm_tuned", "mm_anchor", "mm_anchor_lost25"}
    assert "tracking_other_id" in r["diagnosis"]["mm_anchor"]["yolo11"]["overall"]
    assert r["run_environment"]["yolo11"]["environment"]["python"]
    assert any("motion-only" in s for s in r["statements"]) and any("Selection bias" in s for s in r["statements"])
    assert "COT QUYET DINH mm_anchor" in capsys.readouterr().out


def test_report_lists_missing_columns_instead_of_dropping(heldout):
    use_d5_rules(heldout)
    assert abl.main(args_of(heldout), FakeRunner()) == 0
    import shutil

    shutil.rmtree(heldout["exps"] / "yolo26_botsort_mm_tuned_occl_heldout_v2")
    r = rh.build_report(heldout["plan"], heldout["exps"], heldout["tmp"] / "scn", "v2")
    assert r["status"] == "incomplete"
    assert any("yolo26_botsort_mm_tuned_occl_heldout_v2" in p for p in r["problems"])
    assert "mm_tuned" in r["tables"] and r["tables"]["mm_tuned"]["per_detector"]["yolo26"] is None
    assert r["bootstrap"]["status"] == "not_available"
