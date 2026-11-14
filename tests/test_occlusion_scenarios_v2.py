"""Test sinh kich ban occlusion v2 (ung vien + loc theo screening).

Anh + annotation VisDrone GIA rat nho trong tmp_path; screening.json gia lap
(khong chay YOLO). Weights la file byte tuy y (chi de tinh sha256).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from src.occlusion_scenarios import (  # noqa: E402
    SCREENING_KIND,
    ScenarioConfigError,
    ScreeningError,
    candidate_eligible,
    candidate_id,
    file_sha256,
    generate_group,
    list_sequences,
    load_manifest,
    load_scenario_config,
    load_sequence,
    plan_candidates,
    plan_sequence,
)

N_FRAMES, W, H = 14, 96, 64
DATASET_CONFIG = (REPO_ROOT / "configs" / "dataset_visdrone.yaml").as_posix()
N_TARGETS = 5


def make_sequence(split_dir: Path, name: str, n_targets: int = N_TARGETS) -> None:
    seq = split_dir / "sequences" / name
    seq.mkdir(parents=True)
    rng = np.random.default_rng(sum(map(ord, name)))
    for f in range(1, N_FRAMES + 1):
        cv2.imwrite(str(seq / f"{f:07d}.png"), rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8))
    lines = []
    for f in range(1, N_FRAMES + 1):
        for t in range(1, n_targets + 1):  # muc tieu hop le 8x6, moi muc tieu mot hang
            lines.append(f"{f},{t},{4 + 2 * (f - 1)},{2 + 11 * (t - 1)},8,6,1,4,0,0")
    (split_dir / "annotations").mkdir(exist_ok=True)
    (split_dir / "annotations" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


TEMPLATE = """
version: {version}
seed: {seed}
raw_root: {raw}
dataset_config: {dataset}
output_root: {out}
groups:
  dev: {{split_dir: VAL}}
  heldout: {{split_dir: TEST}}
target_rules:
  classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  min_box_side_px: 4
  max_gt_truncation: 0
  max_gt_occlusion: 1
  max_ignore_ioa: 0.0
  pre_occlusion_frames: 3
  post_occlusion_frames: 2
{extra_rules}
{lock}
durations_frames: [1, 3]
start_policy: seeded_shared
occluders:
  - {{name: full_solid, coverage: full, texture: solid}}
  - {{name: partial_noise, coverage: partial, texture: noise}}
occluder_params:
  margin_ratio: 0.25
  partial_fraction: 0.5
  solid_color_bgr: [10, 200, 30]
image:
  format: png
"""

LOCK = """target_lock:
  match_iou: 0.5
  max_search_frames: 2
  lock_confirm_frames: 2
  lock_iou: 0.5
  required_detectors: {dets}
screening:
  work_root: {work}
  size_bins_px: [4, 8, 16]
"""


def write_cfg(tmp_path, raw, *, name="v2", version=2, seed=7, cands=3, targets=2, lock=True,
              dets="[a, b]", extra_rules=None):
    if extra_rules is None:
        extra_rules = (f"  candidates_per_sequence: {cands}\n" if lock else "") + f"  targets_per_sequence: {targets}"
    lock_text = LOCK.format(dets=dets, work=(tmp_path / f"work_{name}").as_posix()) if lock else ""
    path = tmp_path / f"{name}.yaml"
    path.write_text(TEMPLATE.format(version=version, seed=seed, raw=raw.as_posix(), dataset=DATASET_CONFIG,
                                    out=(tmp_path / f"out_{name}").as_posix(), extra_rules=extra_rules,
                                    lock=lock_text), encoding="utf-8")
    return path


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    make_sequence(root / "VAL", "seqA")
    make_sequence(root / "VAL", "seqB")
    make_sequence(root / "TEST", "seqC")
    return root


@pytest.fixture
def weights(tmp_path):
    out = {}
    for name in ("a", "b"):
        p = tmp_path / f"{name}.pt"
        p.write_bytes(f"weights-{name}".encode())
        out[name] = p
    return out


def all_candidates(cfg, group="dev"):
    split = cfg.raw_root / cfg.groups[group]
    return {seq: plan_candidates(load_sequence(split, seq, cfg.category_map), cfg)
            for seq in list_sequences(split)}


def make_screening(cfg, weights, *, group="dev", statuses=None, **overrides):
    """screening.json gia: mac dinh moi ung vien 'locked' voi ca hai detector.
    statuses: {candidate_id: {detector: status}} de ghi de."""
    statuses = statuses or {}
    entries = []
    for seq, plans in sorted(all_candidates(cfg, group).items()):
        for rank, p in enumerate(plans):
            cid = candidate_id(seq, p.target_id)
            dets = {d: {"status": statuses.get(cid, {}).get(d, "locked")} for d in cfg.required_detectors}
            entry = {"candidate_id": cid, "sequence": seq, "gt_target_id": p.target_id, "rank": rank,
                     "occlusion_start": p.occlusion_start, "status": "screened", "detectors": dets}
            entry["eligible"] = candidate_eligible(entry, cfg.required_detectors)
            entries.append(entry)
    data = {
        "kind": SCREENING_KIND, "group": group, "config_sha256": cfg.config_sha256,
        "partial": False, "errors": [],
        "detectors": {d: {"weights": str(weights[d]), "weights_sha256": file_sha256(weights[d])}
                      for d in cfg.required_detectors},
        "candidates": entries,
    }
    data.update(overrides)
    path = cfg.output_root / group / "screening.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path, data


def shas(weights):
    return {k: file_sha256(v) for k, v in weights.items()}


# ---------------- config ----------------


def test_repo_configs_v1_unchanged_v2_new():
    v1 = load_scenario_config(REPO_ROOT / "configs" / "occlusion_scenarios.yaml")
    v2 = load_scenario_config(REPO_ROOT / "configs" / "occlusion_scenarios_v2.yaml")
    assert v1.version == 1 and not v1.uses_screening and v1.targets_per_sequence == 2
    assert v2.version == 2 and v2.uses_screening
    assert (v2.candidates_per_sequence, v2.targets_per_sequence) == (12, 4)
    assert v2.required_detectors == ("yolo11", "yolo26")
    assert v2.target_lock.as_dict() == {"match_iou": 0.5, "max_search_frames": 10,
                                        "lock_confirm_frames": 3, "lock_iou": 0.5}
    assert v2.output_root != v1.output_root and v2.output_root.parts[-2:] == ("occlusion_test_scenarios", "v2")
    # quy tac GT, seed, thoi luong, kieu che giong v1
    for key in ("seed", "classes", "min_box_side_px", "max_gt_truncation", "max_gt_occlusion",
                "max_ignore_ioa", "pre_occlusion_frames", "post_occlusion_frames", "durations_frames",
                "occluders", "margin_ratio", "partial_fraction", "solid_color_bgr"):
        assert getattr(v1, key) == getattr(v2, key), key
    assert v1.config_sha256 != v2.config_sha256


@pytest.mark.parametrize("kw, match", [
    ({"cands": 1, "targets": 2}, "candidates_per_sequence"),
    ({"version": 1}, "version 2"),
    ({"dets": "[a, a]"}, "required_detectors"),
    ({"lock": False, "extra_rules": "  candidates_per_sequence: 3\n  targets_per_sequence: 2"}, "target_lock"),
])
def test_invalid_v2_configs(tmp_path, raw, kw, match):
    with pytest.raises(ScenarioConfigError, match=match):
        load_scenario_config(write_cfg(tmp_path, raw, **kw))


# ---------------- ung vien ----------------


def test_candidates_deterministic_capped_and_ranked(tmp_path, raw):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    c1 = all_candidates(cfg)
    c2 = all_candidates(load_scenario_config(write_cfg(tmp_path, raw, name="again")))
    assert {s: [p.target_id for p in v] for s, v in c1.items()} == {s: [p.target_id for p in v] for s, v in c2.items()}
    assert all(len(v) == 3 for v in c1.values())               # cap 3 trong 5 muc tieu hop le
    other = all_candidates(load_scenario_config(write_cfg(tmp_path, raw, name="s99", seed=99)))
    assert [p.target_id for p in other["seqA"]] != [p.target_id for p in c1["seqA"]] or \
        [p.target_id for p in other["seqB"]] != [p.target_id for p in c1["seqB"]]


def test_candidate_start_uses_same_formula_as_v1(tmp_path, raw):
    v2 = load_scenario_config(write_cfg(tmp_path, raw))
    v1 = load_scenario_config(write_cfg(tmp_path, raw, name="v1", version=1, lock=False, targets=N_TARGETS))
    data = load_sequence(raw / "VAL", "seqA", v2.category_map)
    starts_v1 = {p.target_id: p.occlusion_start for p in plan_sequence(data, v1)}
    for p in plan_candidates(data, v2):
        assert p.occlusion_start == starts_v1[p.target_id]


# ---------------- tu choi screening ----------------


def test_refuses_without_screening(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    with pytest.raises(ScreeningError, match="screen_occlusion_targets"):
        generate_group(cfg, "dev", weights_sha256=shas(weights), dry_run=True)


@pytest.mark.parametrize("overrides, match", [
    ({"partial": True}, "partial"),
    ({"errors": ["seqA__t1 yolo loi"]}, "loi"),
    ({"config_sha256": "khac"}, "config_sha256"),
    ({"group": "heldout"}, "group"),
    ({"kind": "khac"}, "kind"),
])
def test_refuses_bad_screening_header(tmp_path, raw, weights, overrides, match):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    make_screening(cfg, weights, **overrides)
    with pytest.raises(ScreeningError, match=match):
        generate_group(cfg, "dev", weights_sha256=shas(weights))


def test_refuses_candidate_list_mismatch(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    _, data = make_screening(cfg, weights)
    make_screening(cfg, weights, candidates=data["candidates"][1:])
    with pytest.raises(ScreeningError, match="danh sach ung vien"):
        generate_group(cfg, "dev", weights_sha256=shas(weights))


def test_refuses_screening_of_other_config(tmp_path, raw, weights):
    """Doi config (vd candidates_per_sequence) sau screening -> sha + danh sach ung vien khac."""
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    path, _ = make_screening(cfg, weights)
    cfg4 = load_scenario_config(write_cfg(tmp_path, raw, name="v2b", cands=4))
    with pytest.raises(ScreeningError, match="config_sha256"):
        generate_group(cfg4, "dev", weights_sha256=shas(weights), screening_file=path)


def test_refuses_weights_sha_mismatch_or_missing(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    make_screening(cfg, weights)
    bad = dict(shas(weights), b="0" * 64)
    with pytest.raises(ScreeningError, match="weights sha256 cua b"):
        generate_group(cfg, "dev", weights_sha256=bad)
    with pytest.raises(ScreeningError, match="can --detector"):
        generate_group(cfg, "dev", weights_sha256={"a": shas(weights)["a"]})


def test_refuses_inconsistent_eligible_flag(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    _, data = make_screening(cfg, weights)
    data["candidates"][0]["detectors"]["a"]["status"] = "not_selected"   # eligible van True
    make_screening(cfg, weights, candidates=data["candidates"])
    with pytest.raises(ScreeningError, match="eligible"):
        generate_group(cfg, "dev", weights_sha256=shas(weights))


def test_refuses_unfinished_candidate(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    _, data = make_screening(cfg, weights)
    data["candidates"][2]["status"] = "error"
    make_screening(cfg, weights, candidates=data["candidates"])
    with pytest.raises(ScreeningError, match="status"):
        generate_group(cfg, "dev", weights_sha256=shas(weights))


# ---------------- chon muc tieu + ghi kich ban ----------------


def test_selects_first_eligible_in_rank_order(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    cands = all_candidates(cfg)
    a0, a1, a2 = (p.target_id for p in cands["seqA"])
    b0, b1, b2 = (p.target_id for p in cands["seqB"])
    statuses = {
        candidate_id("seqA", a0): {"b": "not_selected"},             # chi mot detector khoa -> loai
        candidate_id("seqB", b0): {"a": "lost_before_occlusion"},
        candidate_id("seqB", b2): {"a": "not_selected", "b": "not_selected"},
    }
    make_screening(cfg, weights, statuses=statuses)
    index = generate_group(cfg, "dev", weights_sha256=shas(weights))
    assert index["selected_targets"] == {"seqA": [a1, a2], "seqB": [b1]}
    assert index["candidates_per_sequence"] == {"seqA": 3, "seqB": 3}
    assert index["eligible_per_sequence"] == {"seqA": 2, "seqB": 1}
    assert len(index["scenarios"]) == 3 * 2 * 2                  # 3 muc tieu x 2 thoi luong x 2 kieu che
    assert not any(f"__t{a0}__" in n and n.startswith("seqA") for n in index["scenarios"])
    stored = json.loads((cfg.output_root / "dev" / "index.json").read_text())
    assert stored["screening"]["sha256"] and stored["config_version"] == 2
    assert stored["target_lock"]["required_detectors"] == ["a", "b"]

    m = load_manifest(cfg.output_root / "dev" / index["scenarios"][0])
    assert m["config_version"] == 2
    assert m["target_lock"]["occlusion_start_frame_idx"] == m["occlusion"]["start_frame_idx"] == 3
    assert m["target_lock"]["max_search_frames"] == 2 and m["target_lock"]["required_detectors"] == ["a", "b"]
    assert m["screening"]["candidate_rank"] in (1, 2)
    assert m["screening"]["detectors"]["a"]["weights_sha256"] == file_sha256(weights["a"])


def test_cap_targets_per_sequence(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw, cands=4, targets=2))
    make_screening(cfg, weights)                                 # moi ung vien deu dat
    index = generate_group(cfg, "dev", weights_sha256=shas(weights), dry_run=True)
    assert all(len(v) == 2 for v in index["selected_targets"].values())
    assert index["eligible_per_sequence"] == {"seqA": 4, "seqB": 4}


def test_sequence_without_eligible_target(tmp_path, raw, weights):
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    statuses = {candidate_id("seqB", p.target_id): {"a": "not_selected"} for p in all_candidates(cfg)["seqB"]}
    make_screening(cfg, weights, statuses=statuses)
    index = generate_group(cfg, "dev", weights_sha256=shas(weights), dry_run=True)
    assert index["sequences_without_target"] == ["seqB"] and index["selected_targets"]["seqB"] == []


def test_rerun_skips_and_v1_output_untouched(tmp_path, raw, weights):
    v1 = load_scenario_config(write_cfg(tmp_path, raw, name="v1", version=1, lock=False))
    idx1 = generate_group(v1, "dev")
    v1_index = (v1.output_root / "dev" / "index.json").read_bytes()
    cfg = load_scenario_config(write_cfg(tmp_path, raw))
    make_screening(cfg, weights)
    generate_group(cfg, "dev", weights_sha256=shas(weights))
    again = generate_group(cfg, "dev", weights_sha256=shas(weights))
    assert sorted(again["skipped_existing"]) == sorted(again["scenarios"])
    assert (v1.output_root / "dev" / "index.json").read_bytes() == v1_index
    assert all((v1.output_root / "dev" / n / "manifest.json").is_file() for n in idx1["scenarios"])


def test_script_requires_detectors_for_v2(tmp_path, raw, weights, capsys):
    import scripts.generate_occlusion_scenarios as gen

    path = write_cfg(tmp_path, raw)
    cfg = load_scenario_config(path)
    make_screening(cfg, weights)
    with pytest.raises(ScreeningError, match="can --detector"):
        gen.main(["--config", str(path), "--group", "dev", "--dry-run"])
    assert gen.main(["--config", str(path), "--group", "dev", "--dry-run",
                     "--detector", f"a={weights['a']}", "--detector", f"b={weights['b']}"]) == 0
    assert "ung vien" in capsys.readouterr().out
    v1 = write_cfg(tmp_path, raw, name="v1", version=1, lock=False)
    with pytest.raises(SystemExit):
        gen.main(["--config", str(v1), "--group", "dev", "--detector", f"a={weights['a']}"])
