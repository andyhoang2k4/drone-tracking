"""Unit test cho src/utils/inference_settings.py (thu tu uu tien imgsz)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.inference_settings import (  # noqa: E402
    DEFAULT_TRAIN_CONFIG,
    find_train_run_context,
    resolve_imgsz,
)


def _train_config(tmp_path: Path, imgsz=960) -> Path:
    path = tmp_path / "train_config.yaml"
    value = "null" if imgsz is None else imgsz
    path.write_text(f"common:\n  imgsz: {value}\n", encoding="utf-8")
    return path


def _trained_weights(tmp_path: Path, imgsz=960) -> Path:
    """experiments/<id>/train/weights/best.pt + run_context.json cua lan train."""
    exp = tmp_path / "experiments" / "yolo11_baseline_001"
    weights = exp / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")
    (exp / "run_context.json").write_text(
        json.dumps({"config": {"imgsz": imgsz, "epochs": 1}}), encoding="utf-8"
    )
    return weights


def _experiment_dir(tmp_path: Path, imgsz) -> Path:
    exp = tmp_path / "experiments" / "yolo11_botsort_001"
    exp.mkdir(parents=True)
    value = "null" if imgsz is None else imgsz
    (exp / "config.yaml").write_text(f"detector:\n  imgsz: {value}\n", encoding="utf-8")
    return exp


def test_cli_has_highest_priority(tmp_path: Path):
    choice = resolve_imgsz(
        1280,
        experiment_dir=_experiment_dir(tmp_path, 800),
        weights=_trained_weights(tmp_path, 960),
        train_config=_train_config(tmp_path, 640),
    )
    assert (choice.imgsz, choice.source) == (1280, "cli")
    # Khac imgsz luc train -> co canh bao
    assert len(choice.warnings) == 1 and "960" in choice.warnings[0]


def test_experiment_config_before_train_run_context(tmp_path: Path):
    exp = _experiment_dir(tmp_path, 800)
    choice = resolve_imgsz(
        None, experiment_dir=exp, weights=_trained_weights(tmp_path, 960),
        train_config=_train_config(tmp_path, 640),
    )
    assert choice.imgsz == 800
    assert choice.source == f"experiment_config:{exp / 'config.yaml'}"


def test_train_run_context_when_experiment_config_empty(tmp_path: Path):
    weights = _trained_weights(tmp_path, 960)
    choice = resolve_imgsz(
        None, experiment_dir=_experiment_dir(tmp_path, None), weights=weights,
        train_config=_train_config(tmp_path, 640),
    )
    assert choice.imgsz == 960
    assert choice.source.startswith("train_run_context:")
    assert choice.warnings == []


def test_train_config_is_last_resort(tmp_path: Path):
    train_config = _train_config(tmp_path, 960)
    choice = resolve_imgsz(None, weights="yolo11n-seg.pt", train_config=train_config)
    assert choice.imgsz == 960
    assert choice.source == f"train_config:{train_config}"


def test_nothing_found_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="--imgsz"):
        resolve_imgsz(None, train_config=_train_config(tmp_path, None))


@pytest.mark.parametrize("bad", [0, -1, "960", 9.5, True])
def test_invalid_cli_imgsz(bad, tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_imgsz(bad, train_config=_train_config(tmp_path))


def test_find_train_run_context_limited_to_three_levels(tmp_path: Path):
    weights = _trained_weights(tmp_path)
    assert find_train_run_context(weights) == weights.parents[2] / "run_context.json"
    deep = weights.parent / "a" / "b" / "best.pt"
    deep.parent.mkdir(parents=True)
    assert find_train_run_context(deep) is None
    assert find_train_run_context(None) is None


def test_repo_train_config_has_imgsz():
    """configs/train_config.yaml cua repo phai co common.imgsz (bac uu tien 3)."""
    choice = resolve_imgsz(None, train_config=DEFAULT_TRAIN_CONFIG)
    assert choice.imgsz > 0 and choice.source.startswith("train_config:")
