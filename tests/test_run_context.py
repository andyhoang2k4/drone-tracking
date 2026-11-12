"""Unit test cho src/utils/run_context.py (phan dataset_info)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.run_context import build_run_context, dataset_info  # noqa: E402


def _write_dataset(tmp_path: Path, path_value: str, manifest: dict | None) -> Path:
    dataset_dir = tmp_path / "labeled" / "visdrone_mot_a_v2"
    dataset_dir.mkdir(parents=True)
    if manifest is not None:
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(f"path: {path_value}\ntrain: images/train\n", encoding="utf-8")
    return yaml_path


def test_dataset_info_reads_name_version_from_manifest(tmp_path: Path):
    dataset_dir = tmp_path / "labeled" / "visdrone_mot_a_v2"
    yaml_path = _write_dataset(
        tmp_path, str(dataset_dir), {"dataset": {"name": "visdrone_mot_a", "version": 2}}
    )
    info = dataset_info(yaml_path)
    assert info["name"] == "visdrone_mot_a"
    assert info["version"] == 2
    assert info["yaml"] == str(yaml_path)
    assert info["manifest"] == str(dataset_dir / "manifest.json")


def test_dataset_info_full_run_not_partial(tmp_path: Path):
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2",
        {"dataset": {"name": "d", "version": 2}, "max_sequences": None,
         "masks": {"splits": {}, "max_images": None}},
    )
    info = dataset_info(yaml_path)
    assert info["partial"] is False
    assert info["partial_reasons"] == []


def test_dataset_info_smoke_prepare_is_partial(tmp_path: Path):
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2",
        {"dataset": {"name": "d", "version": 2}, "max_sequences": 1},
    )
    info = dataset_info(yaml_path)
    assert info["partial"] is True
    assert info["partial_reasons"] == ["prepare_visdrone.py --max-sequences 1"]


def test_dataset_info_smoke_masks_is_partial(tmp_path: Path):
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2",
        {"dataset": {"name": "d", "version": 2}, "max_sequences": None,
         "masks": {"max_images": 5}},
    )
    info = dataset_info(yaml_path)
    assert info["partial"] is True
    assert info["partial_reasons"] == ["generate_masks.py --max-images 5"]


def test_dataset_info_manifest_without_key_is_unknown(tmp_path: Path):
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2", {"dataset": {"name": "d", "version": 2}}
    )
    assert dataset_info(yaml_path)["partial"] is None


def test_dataset_info_relative_path_resolves_from_yaml_dir(tmp_path: Path):
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2", {"dataset": {"name": "d", "version": 1}}
    )
    assert dataset_info(yaml_path)["version"] == 1


def test_dataset_info_without_manifest_returns_none(tmp_path: Path):
    yaml_path = _write_dataset(tmp_path, str(tmp_path / "labeled" / "visdrone_mot_a_v2"), None)
    info = dataset_info(yaml_path)
    assert info["name"] is None and info["version"] is None and info["manifest"] is None
    assert info["partial"] is None


def test_dataset_info_missing_yaml(tmp_path: Path):
    info = dataset_info(tmp_path / "khong_co.yaml")
    assert info["name"] is None and info["path"] is None


def test_build_run_context_keeps_dataset_mapping(tmp_path: Path):
    ctx = build_run_context(
        experiment_id="x",
        config={},
        dataset={"name": "visdrone_mot_a", "version": 2},
        repo_root=tmp_path,
    )
    assert ctx["dataset"] == {"name": "visdrone_mot_a", "version": 2}


# ---------------- chan train tren dataset chay thu ----------------

import re  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402

from src.utils.run_context import partial_dataset_error  # noqa: E402


def test_partial_dataset_error():
    partial = {"name": "d", "version": 2, "partial": True,
               "partial_reasons": ["prepare_visdrone.py --max-sequences 1"], "manifest": "m"}
    msg = partial_dataset_error(partial, allow_partial=False)
    assert msg is not None and "--max-sequences 1" in msg and "--allow-partial" in msg
    assert partial_dataset_error(partial, allow_partial=True) is None
    assert partial_dataset_error({"partial": False}, allow_partial=False) is None
    # Khong xac dinh (khong co manifest) -> khong chan, script chi canh bao.
    assert partial_dataset_error({"partial": None}, allow_partial=False) is None


def _train_heredoc(script: Path) -> str:
    match = re.search(r"<<'PY'\n(.*?)\nPY\n", script.read_text(encoding="utf-8"), re.S)
    assert match, f"Khong tim thay doan Python trong {script}"
    return match.group(1)


@pytest.mark.parametrize("model", ["yolo11", "yolo26"])
def test_train_script_stops_on_partial_dataset(tmp_path: Path, model: str):
    """Chay doan Python cua train_*.sh voi dataset chay thu: phai dung truoc
    khi tao thu muc experiment va truoc khi import ultralytics (khong train)."""
    yaml_path = _write_dataset(
        tmp_path, "labeled/visdrone_mot_a_v2",
        {"dataset": {"name": "visdrone_mot_a", "version": 2}, "max_sequences": 1},
    )
    config = tmp_path / "train_config.yaml"
    config.write_text(
        f"data: {yaml_path}\n"
        f"common: {{epochs: 1}}\n"
        f"weights_dir: {tmp_path / 'weights'}\n"
        f"models: {{m: {{weights: x.pt}}}}\n"
        f"output_root: {tmp_path / 'experiments'}\n",
        encoding="utf-8",
    )
    code = _train_heredoc(REPO_ROOT / "scripts" / f"train_{model}.sh")
    result = subprocess.run(
        [sys.executable, "-c", code, str(config), "m", "exp_001"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "DUNG" in result.stderr and "--max-sequences 1" in result.stderr
    assert not (tmp_path / "experiments" / "exp_001").exists()
