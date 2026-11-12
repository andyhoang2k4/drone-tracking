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
