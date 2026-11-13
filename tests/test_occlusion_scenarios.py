"""Test sinh kich ban occlusion (src/occlusion_scenarios.py).

Dung anh + annotation VisDrone GIA rat nho (PNG 64x48) trong tmp_path; khong
dung dataset that, khong chay YOLO.
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

from src.motion_memory import BBox  # noqa: E402
from src.occlusion_scenarios import (  # noqa: E402
    ScenarioConfigError,
    generate_group,
    load_manifest,
    load_scenario_config,
    manifest_ground_truth,
    occluded_flags,
    scenario_name,
)
from src.utils.gt_target import load_target_boxes  # noqa: E402

N_FRAMES = 14
W, H = 64, 48
DATASET_CONFIG = (REPO_ROOT / "configs" / "dataset_visdrone.yaml").as_posix()
CATEGORY_MAP = {c: c - 1 for c in range(1, 11)}


def target1_box(frame: int) -> tuple[int, int, int, int]:
    """Muc tieu hop le: 12x10, di sang phai 2 px/frame."""
    return 10 + 2 * (frame - 1), 20, 12, 10


def make_sequence(split_dir: Path, name: str) -> None:
    seq = split_dir / "sequences" / name
    seq.mkdir(parents=True)
    rng = np.random.default_rng(sum(map(ord, name)))  # noi dung anh tuy y, co dinh
    for f in range(1, N_FRAMES + 1):
        img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
        cv2.imwrite(str(seq / f"{f:07d}.png"), img)
    lines = []
    for f in range(1, N_FRAMES + 1):
        x, y, w, h = target1_box(f)
        lines.append(f"{f},1,{x},{y},{w},{h},1,4,0,0")          # car, hop le
        lines.append(f"{f},2,40,5,3,3,1,1,0,0")                 # qua nho
        lines.append(f"{f},3,2,2,10,10,1,1,0,0")                # nam trong vung bo qua
        lines.append(f"{f},5,40,30,10,10,1,1,1,0")              # bi cat (truncation=1)
        lines.append(f"{f},0,0,0,14,14,0,0,0,0")                # vung bo qua
    (split_dir / "annotations").mkdir(exist_ok=True)
    (split_dir / "annotations" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


CONFIG_TEMPLATE = """
version: 1
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
  pre_occlusion_frames: 2
  post_occlusion_frames: 2
  targets_per_sequence: 3
durations_frames: [1, 3]
start_policy: seeded_shared
occluders:
  - {{name: full_solid, coverage: full, texture: solid}}
  - {{name: full_noise, coverage: full, texture: noise}}
  - {{name: partial_solid, coverage: partial, texture: solid}}
occluder_params:
  margin_ratio: 0.25
  partial_fraction: 0.5
  solid_color_bgr: [10, 200, 30]
image:
  format: png
"""


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    make_sequence(root / "VAL", "seqA")
    make_sequence(root / "VAL", "seqB")
    make_sequence(root / "TEST", "seqC")
    return root


def write_config(tmp_path: Path, raw: Path, out: Path, seed: int = 7, name: str = "occ.yaml") -> Path:
    path = tmp_path / name
    path.write_text(CONFIG_TEMPLATE.format(seed=seed, raw=raw.as_posix(), dataset=DATASET_CONFIG,
                                           out=out.as_posix()), encoding="utf-8")
    return path


def generate(tmp_path, raw, out_name="out", seed=7, group="dev", **kw):
    out = tmp_path / out_name
    cfg = load_scenario_config(write_config(tmp_path, raw, out, seed, name=f"{out_name}.yaml"))
    return cfg, generate_group(cfg, group, **kw), out


# ---------------- chon muc tieu ----------------


def test_only_valid_target_selected(tmp_path, raw):
    _, index, _ = generate(tmp_path, raw)
    # chi muc tieu 1 hop le: 2 sequence x 2 thoi luong x 3 kieu che
    assert len(index["scenarios"]) == 2 * 2 * 3
    assert all("__t1__" in n for n in index["scenarios"])
    assert index["sequences"] == ["seqA", "seqB"]


# ---------------- ground truth ----------------


def test_ground_truth_every_frame(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    for name in index["scenarios"]:
        m = load_manifest(out / "dev" / name)
        dur = m["occlusion"]["duration_frames"]
        assert m["frames"] == 2 + dur + 2
        gt = manifest_ground_truth(m)
        assert len(gt) == m["frames"]
        for row, box in zip(m["ground_truth"], gt):
            x, y, w, h = target1_box(row["source_frame"])
            assert box == BBox(x, y, x + w, y + h)
        flags = occluded_flags(m)
        lo, hi = m["occlusion"]["start_frame_idx"], m["occlusion"]["end_frame_idx"]
        assert lo == 2 and hi == 2 + dur - 1
        assert flags == [lo <= k <= hi for k in range(m["frames"])]
        images = sorted((out / "dev" / name / "images").iterdir())
        assert [p.stem for p in images] == [f"{k + 1:07d}" for k in range(m["frames"])]


def test_annotations_renumbered_and_usable_for_gt_selection(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    scen = out / "dev" / index["scenarios"][0]
    m = load_manifest(scen)
    boxes = load_target_boxes(scen / "annotations.txt", m["target"]["gt_target_id"], CATEGORY_MAP)
    assert sorted(boxes) == list(range(1, m["frames"] + 1))
    for row in m["ground_truth"]:
        assert boxes[row["frame_idx"] + 1].as_tuple() == tuple(row["bbox"])


def test_manifest_records_source_and_seed(tmp_path, raw):
    cfg, index, out = generate(tmp_path, raw)
    m = load_manifest(out / "dev" / index["scenarios"][0])
    assert m["group"] == "dev" and m["source"]["split_dir"] == "VAL"
    assert m["source"]["sequence"] in {"seqA", "seqB"}
    assert m["source"]["last_source_frame"] - m["source"]["first_source_frame"] + 1 == m["frames"]
    assert m["seed"] == 7 and m["config_sha256"] == cfg.config_sha256
    assert m["target"] == {"gt_target_id": 1, "class": 3, "category": 4}


# ---------------- vung che ----------------


def _src_image(raw: Path, seq: str, frame: int):
    return cv2.imread(str(raw / "VAL" / "sequences" / seq / f"{frame:07d}.png"))


def _rect(box):
    import math
    x1, y1, x2, y2 = box
    return math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)


def test_full_solid_occluder_in_right_place(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    name = next(n for n in index["scenarios"] if n.endswith("d003__full_solid"))
    scen = out / "dev" / name
    m = load_manifest(scen)
    for row in m["ground_truth"]:
        img = cv2.imread(str(scen / "images" / f"{row['frame_idx'] + 1:07d}.png"))
        src = _src_image(raw, m["source"]["sequence"], row["source_frame"])
        if not row["occluded"]:
            assert row["occluder_bbox"] is None
            assert np.array_equal(img, src)
            continue
        x1, y1, x2, y2 = _rect(row["occluder_bbox"])
        gx1, gy1, gx2, gy2 = row["bbox"]
        assert x1 <= gx1 and y1 <= gy1 and x2 >= gx2 and y2 >= gy2  # che kin bbox GT
        assert (img[y1:y2, x1:x2] == [10, 200, 30]).all()
        mask = np.ones(img.shape[:2], bool)
        mask[y1:y2, x1:x2] = False
        assert np.array_equal(img[mask], src[mask])  # ngoai vat che giu nguyen


def test_partial_occluder_leaves_one_side_visible(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    name = next(n for n in index["scenarios"] if n.endswith("d003__partial_solid"))
    m = load_manifest(out / "dev" / name)
    side = m["occlusion"]["partial_side"]
    assert side in {"left", "right"}
    for row in m["ground_truth"]:
        if not row["occluded"]:
            continue
        ox1, _, ox2, _ = row["occluder_bbox"]
        gx1, _, gx2, _ = row["bbox"]
        covered = min(ox2, gx2) - max(ox1, gx1)
        assert covered == pytest.approx(0.5 * (gx2 - gx1))
        if side == "left":
            assert ox1 <= gx1 and ox2 < gx2
        else:
            assert ox2 >= gx2 and ox1 > gx1


def test_noise_occluder_is_textured(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    name = next(n for n in index["scenarios"] if n.endswith("d003__full_noise"))
    scen = out / "dev" / name
    m = load_manifest(scen)
    row = next(r for r in m["ground_truth"] if r["occluded"])
    img = cv2.imread(str(scen / "images" / f"{row['frame_idx'] + 1:07d}.png"))
    src = _src_image(raw, m["source"]["sequence"], row["source_frame"])
    x1, y1, x2, y2 = _rect(row["occluder_bbox"])
    patch = img[y1:y2, x1:x2]
    assert len(np.unique(patch.reshape(-1, 3), axis=0)) > 10
    assert not np.array_equal(patch, src[y1:y2, x1:x2])


def test_occluder_clipped_to_image(tmp_path, raw):
    from src.occlusion_scenarios import OccluderSpec, occluder_box

    cfg = load_scenario_config(write_config(tmp_path, raw, tmp_path / "o"))
    box = occluder_box(BBox(-2, 40, 10, 50), OccluderSpec("f", "full", "solid"), cfg, "left", W, H)
    assert box.x1 == 0.0 and box.y2 == float(H)


# ---------------- tai lap theo seed ----------------


def _snapshot(group_dir: Path) -> dict[str, bytes]:
    return {str(p.relative_to(group_dir)): p.read_bytes()
            for p in sorted(group_dir.rglob("*")) if p.is_file() and p.name != "index.json"}


def test_reproducible_with_same_seed(tmp_path, raw):
    _, _, out1 = generate(tmp_path, raw, out_name="o1")
    _, _, out2 = generate(tmp_path, raw, out_name="o2")
    a, b = _snapshot(out1 / "dev"), _snapshot(out2 / "dev")
    # manifest chua config_sha256 cua file config (khac duong dan output) -> so sanh rieng
    for key in a:
        if key.endswith("manifest.json"):
            ma, mb = json.loads(a[key]), json.loads(b[key])
            ma.pop("config_sha256"); mb.pop("config_sha256")
            assert ma == mb
        else:
            assert a[key] == b[key], key
    assert a.keys() == b.keys()


def test_different_seed_changes_noise(tmp_path, raw):
    _, idx1, out1 = generate(tmp_path, raw, out_name="s1", seed=1)
    _, idx2, out2 = generate(tmp_path, raw, out_name="s2", seed=2)
    n1 = next(n for n in idx1["scenarios"] if n.endswith("full_noise"))
    files1 = _snapshot(out1 / "dev" / n1)
    n2 = next(n for n in idx2["scenarios"] if n.endswith("full_noise"))
    files2 = _snapshot(out2 / "dev" / n2)
    assert files1 != files2


# ---------------- nhom dev / held-out ----------------


def test_dev_and_heldout_disjoint(tmp_path, raw):
    _, dev, _ = generate(tmp_path, raw, out_name="g")
    _, held, _ = generate(tmp_path, raw, out_name="g2", group="heldout")
    assert set(dev["sequences"]).isdisjoint(held["sequences"])
    assert held["sequences"] == ["seqC"]


def test_overlapping_groups_rejected(tmp_path, raw):
    make_sequence(raw / "TEST", "seqA")  # trung ten voi nhom dev
    with pytest.raises(ScenarioConfigError):
        generate(tmp_path, raw)


def test_same_split_dir_rejected(tmp_path, raw):
    path = write_config(tmp_path, raw, tmp_path / "o")
    path.write_text(path.read_text().replace("split_dir: TEST", "split_dir: VAL"), encoding="utf-8")
    with pytest.raises(ScenarioConfigError):
        load_scenario_config(path)


# ---------------- ghi de / dry-run ----------------


def test_existing_scenarios_skipped(tmp_path, raw):
    cfg, first, out = generate(tmp_path, raw)
    second = generate_group(cfg, "dev")
    assert sorted(second["skipped_existing"]) == sorted(first["scenarios"])


def test_changed_config_requires_overwrite(tmp_path, raw):
    _, index, out = generate(tmp_path, raw)
    path = write_config(tmp_path, raw, out, name="out.yaml")
    path.write_text(path.read_text().replace("margin_ratio: 0.25", "margin_ratio: 0.3"), encoding="utf-8")
    cfg2 = load_scenario_config(path)
    with pytest.raises(FileExistsError):
        generate_group(cfg2, "dev")
    index2 = generate_group(cfg2, "dev", overwrite=True)
    m = load_manifest(out / "dev" / index2["scenarios"][0])
    assert m["occlusion"]["margin_ratio"] == 0.3


def test_dry_run_writes_nothing(tmp_path, raw):
    _, index, out = generate(tmp_path, raw, dry_run=True)
    assert index["scenarios"] and not out.exists()


def test_max_sequences_marks_partial(tmp_path, raw):
    _, index, _ = generate(tmp_path, raw, max_sequences=1)
    assert index["partial"] is True and index["sequences"] == ["seqA"]


def test_script_dry_run(tmp_path, raw, capsys):
    import scripts.generate_occlusion_scenarios as gen

    cfg_path = write_config(tmp_path, raw, tmp_path / "o")
    assert gen.main(["--config", str(cfg_path), "--group", "dev", "--dry-run"]) == 0
    assert scenario_name("seqA", 1, 1, "full_solid") in capsys.readouterr().out


def test_repo_config_loads():
    cfg = load_scenario_config(REPO_ROOT / "configs" / "occlusion_scenarios.yaml")
    assert set(cfg.groups) == {"dev", "heldout"}
    assert cfg.durations_frames == (5, 15, 30, 60)
    assert len(cfg.occluders) == 4
