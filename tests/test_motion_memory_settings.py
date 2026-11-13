"""Test resolve tham so Motion Memory: CLI > file > dataclass."""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.motion_memory import MotionMemoryConfig  # noqa: E402
from src.utils.motion_memory_settings import (  # noqa: E402
    DEFAULT_MOTION_MEMORY_CONFIG,
    resolve_motion_memory_config,
)


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mm.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_repo_config_matches_dataclass_defaults():
    """configs/motion_memory.yaml khop mac dinh dataclass (gia tri khoi dau)."""
    choice = resolve_motion_memory_config(None, DEFAULT_MOTION_MEMORY_CONFIG)
    assert asdict(choice.config) == asdict(MotionMemoryConfig())
    assert set(choice.sources.values()) == {"file"}


def test_priority_cli_over_file_over_default(tmp_path: Path):
    path = write(tmp_path, "motion_memory:\n  lost_timeout_frames: 10\n  reid_timeout_frames: 20\n")
    choice = resolve_motion_memory_config({"lost_timeout_frames": 7, "reid_timeout_frames": None}, path)
    assert choice.config.lost_timeout_frames == 7 and choice.sources["lost_timeout_frames"] == "cli"
    assert choice.config.reid_timeout_frames == 20 and choice.sources["reid_timeout_frames"] == "file"
    assert choice.config.velocity_decay == MotionMemoryConfig().velocity_decay
    assert choice.sources["velocity_decay"] == "default"


def test_null_in_file_means_default(tmp_path: Path):
    path = write(tmp_path, "motion_memory:\n  match_iou: null\n")
    choice = resolve_motion_memory_config(None, path)
    assert choice.sources["match_iou"] == "default"


def test_as_dict_is_json_ready(tmp_path: Path):
    import json

    path = write(tmp_path, "motion_memory:\n  velocity_decay: 1\n")
    payload = resolve_motion_memory_config(None, path).as_dict()
    json.dumps(payload)
    assert payload["config"]["velocity_decay"] == 1.0
    assert payload["config_path"] == str(path)


@pytest.mark.parametrize(
    "body",
    [
        "motion_memory:\n  lost_timeout: 10\n",            # sai ten
        "motion_memory:\n  lost_timeout_frames: 1.5\n",    # sai kieu
        "motion_memory:\n  match_iou: true\n",             # bool
        "motion_memory:\n  velocity_decay: 2.0\n",         # ngoai khoang
        "other: 1\n",                                      # thieu khoa
    ],
)
def test_invalid_file_rejected(tmp_path: Path, body: str):
    with pytest.raises(ValueError):
        resolve_motion_memory_config(None, write(tmp_path, body))


def test_missing_file_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_motion_memory_config(None, tmp_path / "khong_co.yaml")


def test_unknown_cli_key_rejected():
    with pytest.raises(ValueError):
        resolve_motion_memory_config({"bogus": 1}, DEFAULT_MOTION_MEMORY_CONFIG)
