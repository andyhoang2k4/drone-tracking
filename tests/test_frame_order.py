"""Test kiem tra thu tu anh: path trong log run_pipeline vs ten anh da sap xep."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.occlusion_scenarios import frame_order_problems, scenario_image_names  # noqa: E402


@pytest.fixture
def scen(tmp_path):
    d = tmp_path / "dev" / "seqA__t1__d005__full_solid"
    (d / "images").mkdir(parents=True)
    # Tao theo thu tu nguoc de chac chan ket qua khong phu thuoc thu tu tao file.
    for k in (3, 1, 2):
        (d / "images" / f"{k:07d}.jpg").write_bytes(b"")
    (d / "images" / "notes.txt").write_text("x")  # khong phai anh -> bo qua
    return d


def paths(scen, names):
    return [str(scen / "images" / n) for n in names]


def test_sorted_names(scen):
    assert scenario_image_names(scen) == ["0000001.jpg", "0000002.jpg", "0000003.jpg"]


def test_matching_order_ok(scen):
    assert frame_order_problems(paths(scen, ["0000001.jpg", "0000002.jpg", "0000003.jpg"]), scen) == []


def test_other_machine_absolute_prefix_ok(scen):
    rows = [f"/home/u/x/dev/{scen.name}/images/{k:07d}.jpg" for k in (1, 2, 3)]
    assert frame_order_problems(rows, scen) == []


def test_swapped_order_reported(scen):
    probs = frame_order_problems(paths(scen, ["0000001.jpg", "0000003.jpg", "0000002.jpg"]), scen)
    assert len(probs) == 2 and "frame_idx 1" in probs[0]


def test_missing_frame_reported(scen):
    probs = frame_order_problems(paths(scen, ["0000001.jpg", "0000003.jpg"]), scen)
    assert any("2 frame" in p and "3 anh" in p for p in probs)
    assert any("frame_idx 1" in p for p in probs)


def test_extra_frame_reported(scen):
    probs = frame_order_problems(paths(scen, ["0000001.jpg", "0000002.jpg", "0000003.jpg", "0000004.jpg"]), scen)
    assert probs and "4 frame" in probs[0]


def test_missing_path_reported(scen):
    probs = frame_order_problems([None] * 3, scen)
    assert len(probs) == 3 and "khong co path" in probs[0]


def test_other_scenario_dir_reported(scen):
    rows = [f"/x/other_scen/images/{k:07d}.jpg" for k in (1, 2, 3)]
    assert "khong thuoc" in frame_order_problems(rows, scen)[0]


def test_report_capped(tmp_path):
    d = tmp_path / "s"
    (d / "images").mkdir(parents=True)
    for k in range(1, 21):
        (d / "images" / f"{k:07d}.jpg").write_bytes(b"")
    probs = frame_order_problems([None] * 20, d, max_report=3)
    assert len(probs) == 4 and "20 frame" in probs[-1]


def test_no_images_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        scenario_image_names(tmp_path)
