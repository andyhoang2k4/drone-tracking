"""Unit test cho cac ham thuan cua scripts/check_env.py."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_env import (  # noqa: E402
    FAIL,
    OK,
    WARN,
    base_version,
    compare_keys,
    compare_version,
    parse_pinned_requirements,
)


def test_parse_pinned_requirements_skips_comments_and_unpinned():
    text = (
        "# comment\n"
        "ultralytics==8.3.0\n"
        "\n"
        "Torch==2.5.1  # inline comment\n"
        "numpy>=1.26\n"
        "opencv-python==4.10.0.84\n"
    )
    assert parse_pinned_requirements(text) == {
        "ultralytics": "8.3.0",
        "torch": "2.5.1",
        "opencv-python": "4.10.0.84",
    }


def test_base_version_strips_local_suffix():
    assert base_version("2.7.1+cu128") == "2.7.1"
    assert base_version("8.3.0") == "8.3.0"


def test_compare_version_statuses():
    assert compare_version("2.5.1", "2.5.1+cu121")[0] == OK
    assert compare_version("2.5.1", "2.7.1")[0] == WARN
    assert compare_version("2.5.1", None)[0] == FAIL


def test_compare_keys():
    missing, unknown = compare_keys({"a", "b", "x"}, {"a", "b", "c"})
    assert missing == ["c"]
    assert unknown == ["x"]
